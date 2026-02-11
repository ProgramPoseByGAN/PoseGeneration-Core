"""
姿态转换模块
负责AMASS姿态数据的各种表示形式转换

主要功能：
- 轴角 ↔ 旋转矩阵 ↔ 欧拉角转换
- 姿态数据提取和重塑
- 旋转表示标准化
- 姿态数据分析
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PoseConverter:
    """姿态数据转换器"""
    
    def __init__(self):
        """初始化姿态转换器"""
        # SMPL模型参数
        self.SMPL_JOINTS = 24
        self.JOINT_DIM = 3
        self.BODY_DIM = self.SMPL_JOINTS * self.JOINT_DIM
        
        logger.info("PoseConverter初始化完成")
        logger.info(f"SMPL关节数: {self.SMPL_JOINTS}")
        logger.info(f"身体姿态维度: {self.BODY_DIM}")
    
    def extract_body_poses(self, poses_data: np.ndarray) -> np.ndarray:
        """
        从完整的poses数据中提取身体部分（前72维）
        
        AMASS数据中的poses字段通常包含156维数据：
        - 前72维：身体关节姿态（24关节×3维）
        - 后续维度：手部等其他部位
        
        Args:
            poses_data: 完整的姿态数据，形状为 (frames, 156)
            
        Returns:
            身体姿态数据，形状为 (frames, 72)
            
        Raises:
            ValueError: 数据维度不足
        """
        logger.info("提取身体姿态数据...")
        
        # 验证输入数据维度
        if poses_data.ndim != 2:
            raise ValueError(f"期望2维数组，得到{poses_data.ndim}维")
            
        if poses_data.shape[1] < self.BODY_DIM:
            raise ValueError(f"姿态数据维度不足，需要至少{self.BODY_DIM}维，实际只有{poses_data.shape[1]}维")
            
        # 提取前BODY_DIM维作为身体姿态数据
        body_poses = poses_data[:, :self.BODY_DIM]
        
        logger.info(f"成功提取身体姿态数据: {body_poses.shape}")
        return body_poses
    
    def axis_angle_to_rotation_matrix(self, axis_angle: np.ndarray) -> np.ndarray:
        """
        轴角向量转旋转矩阵
        
        使用scipy.spatial.transform进行专业转换
        
        Args:
            axis_angle: 轴角向量 (3,)
            
        Returns:
            旋转矩阵 (3, 3)
        """
        # 处理零向量的特殊情况
        if np.linalg.norm(axis_angle) < 1e-10:
            return np.eye(3)
            
        # 使用scipy进行转换
        rotation = Rotation.from_rotvec(axis_angle)
        return rotation.as_matrix()
    
    def rotation_matrix_to_axis_angle(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        旋转矩阵转轴角向量
        
        Args:
            rotation_matrix: 旋转矩阵 (3, 3)
            
        Returns:
            轴角向量 (3,)
        """
        # 使用scipy进行转换
        rotation = Rotation.from_matrix(rotation_matrix)
        return rotation.as_rotvec()
    
    def poses_to_rotation_matrices(self, body_poses: np.ndarray) -> np.ndarray:
        """
        将身体姿态转换为旋转矩阵表示
        
        Args:
            body_poses: 身体姿态数据 (frames, 72)
            
        Returns:
            旋转矩阵 (frames, 24, 3, 3)
        """
        logger.info("转换姿态数据为旋转矩阵...")
        
        num_frames = body_poses.shape[0]
        rot_mats = np.zeros((num_frames, self.SMPL_JOINTS, 3, 3))
        
        # 逐帧逐关节进行转换
        for frame in range(num_frames):
            for joint in range(self.SMPL_JOINTS):
                # 计算当前关节在数据中的索引范围
                start_idx = joint * self.JOINT_DIM
                end_idx = start_idx + self.JOINT_DIM
                
                # 提取当前关节的轴角数据
                axis_angle = body_poses[frame, start_idx:end_idx]
                
                # 转换为旋转矩阵
                rot_mats[frame, joint] = self.axis_angle_to_rotation_matrix(axis_angle)
                
        logger.info(f"转换完成: {rot_mats.shape}")
        return rot_mats
    
    def rotation_matrices_to_poses(self, rot_mats: np.ndarray) -> np.ndarray:
        """
        将旋转矩阵转换回轴角表示
        
        Args:
            rot_mats: 旋转矩阵 (frames, joints, 3, 3)
            
        Returns:
            轴角表示的姿态数据 (frames, joints*3)
        """
        logger.info("转换旋转矩阵为轴角表示...")
        
        frames, joints = rot_mats.shape[:2]
        poses = np.zeros((frames, joints * 3))
        
        # 逐帧逐关节进行转换
        for frame in range(frames):
            for joint in range(joints):
                # 转换为轴角
                axis_angle = self.rotation_matrix_to_axis_angle(rot_mats[frame, joint])
                
                # 存储到结果数组
                start_idx = joint * 3
                end_idx = start_idx + 3
                poses[frame, start_idx:end_idx] = axis_angle
                
        logger.info(f"转换完成: {poses.shape}")
        return poses
    
    def rotation_matrix_to_euler_xyz(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        旋转矩阵转XYZ欧拉角（弧度）
        
        Args:
            rotation_matrix: 旋转矩阵 (3, 3)
            
        Returns:
            欧拉角 (3,) [X, Y, Z] 单位：弧度
        """
        # 提取矩阵元素
        r00, r01, r02 = rotation_matrix[0]
        r10, r11, r12 = rotation_matrix[1]
        r20, r21, r22 = rotation_matrix[2]
        
        # 处理万向锁情况
        if abs(r20) >= 1:
            # 万向锁：y = ±90°
            y = np.sign(r20) * np.pi / 2
            x = 0
            z = np.arctan2(-r01, r11)
        else:
            # 正常情况
            y = -np.arcsin(r20)
            x = np.arctan2(r21, r22)
            z = np.arctan2(r10, r00)
            
        return np.array([x, y, z])
    
    def rotation_matrix_to_euler_zxy(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        旋转矩阵转ZXY欧拉角（度）
        按照BVH标准的ZXY旋转顺序
        
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
    
    def poses_to_euler_angles(self, body_poses: np.ndarray, 
                            order: str = 'zxy') -> np.ndarray:
        """
        将身体姿态直接转换为欧拉角
        
        Args:
            body_poses: 身体姿态数据 (frames, 72)
            order: 欧拉角顺序 ('xyz' 或 'zxy')
            
        Returns:
            欧拉角 (frames, 24, 3)
        """
        logger.info(f"转换姿态数据为{order.upper()}欧拉角...")
        
        # 先转换为旋转矩阵
        rot_mats = self.poses_to_rotation_matrices(body_poses)
        
        frames, joints = rot_mats.shape[:2]
        euler_angles = np.zeros((frames, joints, 3))
        
        # 选择转换函数
        if order.lower() == 'xyz':
            converter_func = self.rotation_matrix_to_euler_xyz
        elif order.lower() == 'zxy':
            converter_func = self.rotation_matrix_to_euler_zxy
        else:
            raise ValueError(f"不支持的欧拉角顺序: {order}")
        
        # 逐帧逐关节转换
        for frame in range(frames):
            for joint in range(joints):
                euler_angles[frame, joint] = converter_func(rot_mats[frame, joint])
                
        logger.info(f"转换完成: {euler_angles.shape}")
        return euler_angles
    
    def normalize_rotations(self, rot_mats: np.ndarray) -> np.ndarray:
        """
        标准化旋转矩阵（确保正交性）
        
        Args:
            rot_mats: 旋转矩阵数组 (..., 3, 3)
            
        Returns:
            标准化后的旋转矩阵
        """
        logger.info("标准化旋转矩阵...")
        
        # 使用奇异值分解进行正交化
        u, _, vt = np.linalg.svd(rot_mats)
        normalized = np.matmul(u, vt)
        
        # 确保行列式为1（旋转矩阵性质）
        det = np.linalg.det(normalized)
        if np.any(det < 0):
            # 如果行列式为负，翻转最后一列
            normalized[..., :, -1] *= -1
            
        logger.info("标准化完成")
        return normalized
    
    def compute_angular_velocity(self, poses: np.ndarray, 
                               framerate: float = 120.0) -> np.ndarray:
        """
        计算角速度
        
        Args:
            poses: 姿态数据 (frames, joints*3)
            framerate: 帧率
            
        Returns:
            角速度 (frames-1, joints*3)
        """
        logger.info("计算角速度...")
        
        if len(poses) < 2:
            logger.warning("姿态数据帧数不足，无法计算角速度")
            return np.array([])
            
        # 计算帧间差分
        diff = np.diff(poses, axis=0)
        
        # 转换为角速度（弧度/秒）
        time_delta = 1.0 / framerate
        angular_velocity = diff / time_delta
        
        logger.info(f"角速度计算完成: {angular_velocity.shape}")
        return angular_velocity
    
    def analyze_pose_statistics(self, poses_data: np.ndarray) -> Dict[str, Any]:
        """
        分析姿态数据统计信息
        
        Args:
            poses_data: 完整的姿态数据 (frames, 156)
            
        Returns:
            统计信息字典
        """
        logger.info("分析姿态数据统计信息...")
        
        # 提取身体姿态
        body_poses = self.extract_body_poses(poses_data)
        frames = body_poses.shape[0]
        
        # 重塑为关节格式便于分析
        joint_poses = body_poses.reshape(frames, self.SMPL_JOINTS, 3)
        
        # 计算统计信息
        stats = {
            'num_frames': frames,
            'total_joints': self.SMPL_JOINTS,
            'joint_stats': {}
        }
        
        # 逐关节计算统计
        joint_names = self.get_smpl_joint_names()
        for i, joint_name in enumerate(joint_names):
            joint_data = joint_poses[:, i, :]
            
            stats['joint_stats'][joint_name] = {
                'mean': np.mean(joint_data, axis=0).tolist(),
                'std': np.std(joint_data, axis=0).tolist(),
                'min': np.min(joint_data, axis=0).tolist(),
                'max': np.max(joint_data, axis=0).tolist(),
                'range': (np.max(joint_data, axis=0) - np.min(joint_data, axis=0)).tolist()
            }
            
        # 计算整体统计
        all_data = body_poses.flatten()
        stats['overall_stats'] = {
            'mean': float(np.mean(all_data)),
            'std': float(np.std(all_data)),
            'min': float(np.min(all_data)),
            'max': float(np.max(all_data))
        }
        
        logger.info("统计分析完成")
        return stats
    
    def get_smpl_joint_names(self) -> List[str]:
        """
        获取SMPL关节名称列表
        
        Returns:
            关节名称列表
        """
        return [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
        ]
    
    def validate_pose_data(self, poses_data: np.ndarray) -> Dict[str, Any]:
        """
        验证姿态数据有效性
        
        Args:
            poses_data: 姿态数据
            
        Returns:
            验证结果字典
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # 检查数据类型
            if not isinstance(poses_data, np.ndarray):
                result['errors'].append("数据不是numpy数组")
                result['is_valid'] = False
                return result
                
            # 检查维度
            if poses_data.ndim != 2:
                result['errors'].append(f"数据维度应该是2维，当前是{poses_data.ndim}维")
                result['is_valid'] = False
                return result
                
            # 检查最小维度
            if poses_data.shape[1] < self.BODY_DIM:
                result['errors'].append(f"数据维度不足，至少需要{self.BODY_DIM}维")
                result['is_valid'] = False
                return result
                
            # 检查NaN值
            if np.isnan(poses_data).any():
                result['warnings'].append("数据包含NaN值")
                
            # 检查无穷值
            if np.isinf(poses_data).any():
                result['warnings'].append("数据包含无穷值")
                
            # 检查数值范围合理性
            abs_max = np.max(np.abs(poses_data))
            if abs_max > 10:  # 轴角值通常不会超过这个范围
                result['warnings'].append(f"数据值过大 (最大绝对值: {abs_max})")
                
        except Exception as e:
            result['errors'].append(f"验证过程出错: {str(e)}")
            result['is_valid'] = False
            
        return result

def main():
    """主函数 - 演示基本功能"""
    converter = PoseConverter()
    
    # 创建测试数据
    test_frames = 100
    test_poses = np.random.randn(test_frames, 156) * 0.1  # 小幅随机数据
    
    print("=== 姿态转换器演示 ===")
    
    # 1. 提取身体姿态
    body_poses = converter.extract_body_poses(test_poses)
    print(f"身体姿态形状: {body_poses.shape}")
    
    # 2. 转换为旋转矩阵
    rot_mats = converter.poses_to_rotation_matrices(body_poses)
    print(f"旋转矩阵形状: {rot_mats.shape}")
    
    # 3. 转换为欧拉角
    euler_angles = converter.poses_to_euler_angles(body_poses, order='zxy')
    print(f"欧拉角形状: {euler_angles.shape}")
    
    # 4. 数据验证
    validation = converter.validate_pose_data(test_poses)
    print(f"数据有效性: {'✓' if validation['is_valid'] else '✗'}")
    if validation['warnings']:
        print(f"警告: {validation['warnings']}")
    if validation['errors']:
        print(f"错误: {validation['errors']}")
    
    # 5. 统计分析
    stats = converter.analyze_pose_statistics(test_poses)
    print(f"帧数: {stats['num_frames']}")
    print(f"关节数: {stats['total_joints']}")
    print(f"整体均值: {stats['overall_stats']['mean']:.4f}")

if __name__ == "__main__":
    main()