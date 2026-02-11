"""
骨骼映射模块
实现从AMASS SMPL 24关节到项目22关节标准的映射转换

主要功能：
- SMPL到目标骨骼的关节映射
- 融合映射（如脊柱融合）
- 新建关节生成（如脚趾关节）
- BVH文件生成
- 训练数据标准化
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import bvhio

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SkeletonMapper:
    """骨骼映射器"""
    
    def __init__(self):
        """初始化骨骼映射器"""
        # SMPL源骨架定义 (24关节)
        self.SMPL_JOINTS = 24
        self.smpl_joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
        ]
        
        # 项目目标骨架定义 (22关节) - 严格遵循Unity Humanoid标准
        self.TARGET_JOINTS = 22
        self.target_joint_names = [
            'Hips', 'LeftUpperLeg', 'RightUpperLeg', 'Spine', 'LeftLowerLeg', 'RightLowerLeg',
            'Spine1', 'LeftFoot', 'RightFoot', 'Spine2', 'LeftToes', 'RightToes',
            'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftUpperArm', 'RightUpperArm',
            'LeftLowerArm', 'RightLowerArm', 'LeftHand', 'RightHand'
        ]
        
        # 映射关系定义 - 按照《映射规范》
        self.joint_mapping = {
            # 直接一对一映射
            'Hips': 'pelvis',           # 0 -> 0
            'LeftUpperLeg': 'left_hip', # 1 -> 1
            'RightUpperLeg': 'right_hip', # 2 -> 2
            'Spine': 'spine1',          # 3 -> 3
            'LeftLowerLeg': 'left_knee', # 4 -> 4
            'RightLowerLeg': 'right_knee', # 5 -> 5
            'Spine1': 'spine2',         # 6 -> 6
            'LeftFoot': 'left_ankle',   # 7 -> 7
            'RightFoot': 'right_ankle', # 8 -> 8
            'Neck': 'neck',             # 12 -> 12
            'Head': 'head',             # 15 -> 15
            'LeftUpperArm': 'left_shoulder', # 16 -> 16
            'RightUpperArm': 'right_shoulder', # 17 -> 17
            'LeftLowerArm': 'left_elbow', # 18 -> 18
            'RightLowerArm': 'right_elbow', # 19 -> 19
            'LeftHand': 'left_wrist',   # 20 -> 20
            'RightHand': 'right_wrist', # 21 -> 21
            
            # 一对多/合并映射
            'Spine2': ['spine3', 'neck'], # 9, 12 -> 融合
            
            # 肩膀映射
            'LeftShoulder': 'left_collar',  # 13 -> 13
            'RightShoulder': 'right_collar', # 14 -> 14
            
            # 新建关节（AMASS中不存在，需生成）
            'LeftToes': 'left_foot',    # 基于left_foot生成
            'RightToes': 'right_foot',  # 基于right_foot生成
        }
        
        # 骨骼层级结构 (父节点索引，-1表示根节点)
        self.target_hierarchy = {
            'Hips': -1,
            'LeftUpperLeg': 0,
            'RightUpperLeg': 0,
            'Spine': 0,
            'LeftLowerLeg': 1,
            'RightLowerLeg': 2,
            'Spine1': 3,
            'LeftFoot': 4,
            'RightFoot': 5,
            'Spine2': 3,  # 父节点为Spine(3)，不是Spine1(6)
            'LeftToes': 7,
            'RightToes': 8,
            'Neck': 9,
            'LeftShoulder': 9,
            'RightShoulder': 9,
            'Head': 12,
            'LeftUpperArm': 13,
            'RightUpperArm': 14,
            'LeftLowerArm': 16,
            'RightLowerArm': 17,
            'LeftHand': 18,
            'RightHand': 19
        }
        
        # 初始偏移量定义 (单位: 米) - 基于典型人体尺寸
        self.initial_offsets = {
            'Hips': [0.0, 0.0, 0.0],
            'LeftUpperLeg': [0.08, -0.02, 0.0],
            'RightUpperLeg': [-0.08, -0.02, 0.0],
            'Spine': [0.0, 0.1, 0.0],
            'LeftLowerLeg': [0.0, -0.42, 0.0],
            'RightLowerLeg': [0.0, -0.42, 0.0],
            'Spine1': [0.0, 0.12, 0.0],
            'LeftFoot': [0.0, -0.43, 0.0],
            'RightFoot': [0.0, -0.43, 0.0],
            'Spine2': [0.0, 0.15, 0.0],
            'LeftToes': [0.0, 0.0, 0.1],  # 脚尖方向
            'RightToes': [0.0, 0.0, 0.1], # 脚尖方向
            'Neck': [0.0, 0.15, 0.0],
            'LeftShoulder': [0.05, 0.1, 0.0],
            'RightShoulder': [-0.05, 0.1, 0.0],
            'Head': [0.0, 0.17, 0.0],
            'LeftUpperArm': [0.15, 0.0, 0.0],
            'RightUpperArm': [-0.15, 0.0, 0.0],
            'LeftLowerArm': [0.25, 0.0, 0.0],
            'RightLowerArm': [-0.25, 0.0, 0.0],
            'LeftHand': [0.25, 0.0, 0.0],
            'RightHand': [-0.25, 0.0, 0.0]
        }
        
        logger.info("SkeletonMapper初始化完成")
        logger.info(f"源关节数: {self.SMPL_JOINTS}")
        logger.info(f"目标关节数: {self.TARGET_JOINTS}")
    
    def map_joints(self, smpl_rotations: np.ndarray) -> np.ndarray:
        """
        执行关节映射：SMPL 24关节 → 项目22关节
        
        Args:
            smpl_rotations: SMPL旋转矩阵 (frames, 24, 3, 3)
            
        Returns:
            目标旋转矩阵 (frames, 22, 3, 3)
        """
        logger.info("执行关节映射...")
        
        num_frames = smpl_rotations.shape[0]
        target_rotations = np.zeros((num_frames, self.TARGET_JOINTS, 3, 3))
        
        # 创建关节名称到索引的映射
        smpl_name_to_idx = {name: idx for idx, name in enumerate(self.smpl_joint_names)}
        
        # 执行映射
        for target_idx, target_name in enumerate(self.target_joint_names):
            source_name = self.joint_mapping[target_name]
            
            if isinstance(source_name, list):  # 融合映射 (如Spine2)
                if target_name == 'Spine2':
                    spine3_idx = smpl_name_to_idx['spine3']
                    neck_idx = smpl_name_to_idx['neck']
                    
                    # 使用球面线性插值融合两个关节的旋转
                    for frame in range(num_frames):
                        r1 = Rotation.from_matrix(smpl_rotations[frame, spine3_idx])
                        r2 = Rotation.from_matrix(smpl_rotations[frame, neck_idx])
                        
                        # 0.5权重的球面插值
                        slerp = Slerp([0, 1], Rotation.from_quat([r1.as_quat(), r2.as_quat()]))
                        interpolated = slerp(0.5)
                        target_rotations[frame, target_idx] = interpolated.as_matrix()
                        
            elif target_name in ['LeftToes', 'RightToes']:  # 新建关节
                # 基于脚部旋转生成脚趾旋转（简单继承策略）
                source_idx = smpl_name_to_idx[source_name]
                target_rotations[:, target_idx] = smpl_rotations[:, source_idx].copy()
                # 可以在这里添加更复杂的脚趾旋转生成逻辑
                
            else:  # 直接一对一映射
                source_idx = smpl_name_to_idx[source_name]
                target_rotations[:, target_idx] = smpl_rotations[:, source_idx].copy()
                
        logger.info("关节映射完成")
        return target_rotations
    
    def compute_local_rotations(self, global_rotations: np.ndarray) -> np.ndarray:
        """
        将全局旋转转换为局部旋转
        
        Args:
            global_rotations: 全局旋转矩阵 (frames, 22, 3, 3)
            
        Returns:
            局部旋转矩阵 (frames, 22, 3, 3)
        """
        logger.info("计算局部旋转...")
        
        frames, joints = global_rotations.shape[:2]
        local_rotations = np.zeros_like(global_rotations)
        
        for frame in range(frames):
            for joint_idx, joint_name in enumerate(self.target_joint_names):
                parent_idx = self.target_hierarchy[joint_name]
                
                if parent_idx == -1:  # 根节点
                    local_rotations[frame, joint_idx] = global_rotations[frame, joint_idx]
                else:  # 子关节
                    # 局部旋转 = 父关节逆矩阵 × 当前关节全局矩阵
                    parent_global_inv = np.transpose(global_rotations[frame, parent_idx])  # 逆矩阵 = 转置
                    local_rotations[frame, joint_idx] = np.dot(
                        parent_global_inv, 
                        global_rotations[frame, joint_idx]
                    )
                    
        logger.info("局部旋转计算完成")
        return local_rotations
    
    def normalize_training_data(self, local_rotations: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        标准化训练数据
        
        Args:
            local_rotations: 局部旋转矩阵 (frames, 22, 3, 3)
            
        Returns:
            标准化数据和归一化参数
        """
        logger.info("标准化训练数据...")
        
        # 转换为轴角表示用于标准化
        frames, joints = local_rotations.shape[:2]
        axis_angles = np.zeros((frames, joints, 3))
        
        for frame in range(frames):
            for joint in range(joints):
                rotation = Rotation.from_matrix(local_rotations[frame, joint])
                axis_angles[frame, joint] = rotation.as_rotvec()
                
        # 计算均值和标准差（按关节分别计算）
        mean_per_joint = np.mean(axis_angles, axis=0)  # (22, 3)
        std_per_joint = np.std(axis_angles, axis=0)    # (22, 3)
        
        # 避免除零
        std_per_joint = np.maximum(std_per_joint, 1e-8)
        
        # 标准化
        normalized_data = (axis_angles - mean_per_joint) / std_per_joint
        
        normalization_params = {
            'mean': mean_per_joint,
            'std': std_per_joint,
            'joints': self.target_joint_names
        }
        
        logger.info("训练数据标准化完成")
        return normalized_data, normalization_params
    
    def generate_bvh_structure(self) -> bvhio.BVH:
        """
        生成BVH骨架结构
        
        Returns:
            BVH对象
        """
        logger.info("生成BVH骨架结构...")
        
        # 创建根关节
        root = bvhio.BVH(name='Hips', offset=self.initial_offsets['Hips'])
        
        # 按层级创建关节树
        joint_objects = {'Hips': root}
        
        # 按拓扑排序创建关节（确保父节点先创建）
        creation_order = [
            'Hips', 'LeftUpperLeg', 'RightUpperLeg', 'Spine',
            'LeftLowerLeg', 'RightLowerLeg', 'Spine1',
            'LeftFoot', 'RightFoot', 'Spine2',
            'LeftToes', 'RightToes', 'Neck',
            'LeftShoulder', 'RightShoulder', 'Head',
            'LeftUpperArm', 'RightUpperArm',
            'LeftLowerArm', 'RightLowerArm',
            'LeftHand', 'RightHand'
        ]
        
        for joint_name in creation_order[1:]:  # 跳过根节点
            parent_idx = self.target_hierarchy[joint_name]
            parent_name = self.target_joint_names[parent_idx]
            parent_obj = joint_objects[parent_name]
            
            # 创建新关节
            new_joint = bvhio.BVH(
                name=joint_name,
                offset=self.initial_offsets[joint_name],
                parent=parent_obj
            )
            joint_objects[joint_name] = new_joint
            
        logger.info("BVH骨架结构生成完成")
        return root
    
    def save_bvh_file(self, 
                     euler_angles: np.ndarray, 
                     root_translations: np.ndarray,
                     output_path: str,
                     frame_rate: float = 120.0):
        """
        保存为BVH文件
        
        Args:
            euler_angles: 欧拉角数据 (frames, 22, 3) [Z, X, Y]
            root_translations: 根节点位移 (frames, 3)
            output_path: 输出文件路径
            frame_rate: 帧率
        """
        logger.info(f"保存BVH文件: {output_path}")
        
        # 生成骨架结构
        root_joint = self.generate_bvh_structure()
        
        # 准备帧数据
        num_frames = euler_angles.shape[0]
        frame_time = 1.0 / frame_rate
        
        # 构造帧数据：[root_pos(3) + root_rot(3) + joint_rot(3)*21]
        frames_data = []
        for frame in range(num_frames):
            # 根节点数据：位置(3) + 旋转(3)
            root_data = list(root_translations[frame]) + list(euler_angles[frame, 0])
            
            # 其他关节数据：每个3个旋转值
            joint_data = []
            for joint_idx in range(1, self.TARGET_JOINTS):  # 跳过根节点
                joint_data.extend(euler_angles[frame, joint_idx])
                
            frame_data = root_data + joint_data
            frames_data.append(frame_data)
            
        # 创建BVH对象
        bvh = bvhio.BVH(
            root=root_joint,
            frames=frames_data,
            frame_time=frame_time
        )
        
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        bvhio.write(bvh, output_path)
        logger.info(f"BVH文件保存完成: {output_path}")
    
    def save_training_data(self, 
                          normalized_data: np.ndarray,
                          normalization_params: Dict[str, Any],
                          trans_data: np.ndarray,
                          output_path: str,
                          metadata: Optional[Dict[str, Any]] = None):
        """
        保存训练数据
        
        Args:
            normalized_data: 标准化数据
            normalization_params: 归一化参数
            trans_data: 位移数据
            output_path: 输出路径
            metadata: 额外元数据
        """
        logger.info(f"保存训练数据: {output_path}")
        
        training_data = {
            'poses': normalized_data,           # 标准化姿态数据
            'translations': trans_data,         # 位移数据
            'normalization_mean': normalization_params['mean'],
            'normalization_std': normalization_params['std'],
            'joint_names': normalization_params['joints']
        }
        
        if metadata:
            training_data.update(metadata)
            
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
            
        np.savez_compressed(output_path, **training_data)
        logger.info(f"训练数据保存完成: {output_path}")
    
    def process_skeleton_mapping(self, 
                               smpl_rotations: np.ndarray,
                               root_translations: np.ndarray,
                               output_bvh: Optional[str] = None,
                               output_training: Optional[str] = None,
                               frame_rate: float = 120.0,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        完整的骨骼映射处理流程
        
        Args:
            smpl_rotations: SMPL旋转矩阵 (frames, 24, 3, 3)
            root_translations: 根节点位移 (frames, 3)
            output_bvh: 输出BVH文件路径（可选）
            output_training: 输出训练数据路径（可选）
            frame_rate: 帧率
            metadata: 元数据
            
        Returns:
            处理结果信息
        """
        logger.info("="*60)
        logger.info("开始骨骼映射处理流程")
        logger.info("="*60)
        
        # 1. 关节映射
        target_rotations = self.map_joints(smpl_rotations)
        
        # 2. 转换为欧拉角（用于BVH）
        euler_angles = self.rotation_matrix_to_euler_zxy_batch(target_rotations)
        
        # 3. 生成BVH文件
        if output_bvh:
            self.save_bvh_file(
                euler_angles=euler_angles,
                root_translations=root_translations,
                output_path=output_bvh,
                frame_rate=frame_rate
            )
            
        # 4. 计算局部旋转（用于训练）
        local_rotations = self.compute_local_rotations(target_rotations)
        
        # 5. 标准化训练数据
        normalized_data, norm_params = self.normalize_training_data(local_rotations)
        
        # 6. 保存训练数据
        if output_training:
            self.save_training_data(
                normalized_data=normalized_data,
                normalization_params=norm_params,
                trans_data=root_translations,
                output_path=output_training,
                metadata=metadata
            )
            
        # 7. 返回处理结果
        result = {
            'target_shape': target_rotations.shape,
            'frames_processed': smpl_rotations.shape[0],
            'output_bvh': output_bvh,
            'output_training': output_training,
            'success': True
        }
        
        logger.info("="*60)
        logger.info("骨骼映射处理完成")
        logger.info("="*60)
        
        return result
    
    def rotation_matrix_to_euler_zxy_batch(self, rotation_matrices: np.ndarray) -> np.ndarray:
        """
        批量将旋转矩阵转换为ZXY欧拉角
        
        Args:
            rotation_matrices: 旋转矩阵 (frames, joints, 3, 3)
            
        Returns:
            欧拉角 (frames, joints, 3) [Z, X, Y] 单位：度
        """
        frames, joints = rotation_matrices.shape[:2]
        euler_angles = np.zeros((frames, joints, 3))
        
        for frame in range(frames):
            for joint in range(joints):
                euler_angles[frame, joint] = self._rotation_matrix_to_euler_zxy(
                    rotation_matrices[frame, joint]
                )
                
        return euler_angles
    
    def _rotation_matrix_to_euler_zxy(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        旋转矩阵转ZXY欧拉角（度）
        
        Args:
            rotation_matrix: 旋转矩阵 (3, 3)
            
        Returns:
            欧拉角 (3,) [Z, X, Y] 单位：度
        """
        # 提取矩阵元素
        r00, r01, r02 = rotation_matrix[0]
        r10, r11, r12 = rotation_matrix[1]
        r20, r21, r22 = rotation_matrix[2]
        
        # 处理万向锁情况
        if abs(r20) >= 1:
            # 万向锁：y = ±90°
            y = np.sign(r20) * np.pi / 2
            z = 0
            x = np.arctan2(r01, r11)
        else:
            # 正常情况
            y = -np.arcsin(r20)
            x = np.arctan2(r21, r22)
            z = np.arctan2(r10, r00)
            
        # 转换为度
        return np.degrees([z, x, y])  # ZXY顺序

def main():
    """主函数 - 演示基本功能"""
    mapper = SkeletonMapper()
    
    # 创建测试数据
    test_frames = 100
    smpl_rotations = np.tile(np.eye(3), (test_frames, 24, 1, 1))  # 单位矩阵
    root_translations = np.random.randn(test_frames, 3) * 0.01  # 小幅位移
    
    print("=== 骨骼映射器演示 ===")
    
    # 执行完整映射流程
    result = mapper.process_skeleton_mapping(
        smpl_rotations=smpl_rotations,
        root_translations=root_translations,
        output_bvh="./test_output/test.bvh",
        output_training="./test_output/test_training.npz",
        metadata={'test': 'data'}
    )
    
    print(f"处理帧数: {result['frames_processed']}")
    print(f"目标形状: {result['target_shape']}")
    print(f"BVH输出: {result['output_bvh']}")
    print(f"训练数据输出: {result['output_training']}")

if __name__ == "__main__":
    main()