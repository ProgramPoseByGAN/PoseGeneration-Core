"""
AMASS数据集核心处理工具
整合数据读取、姿态转换、统计分析等功能

主要功能：
- 数据集信息查询和文件管理
- SMPL姿态数据解析和转换
- 支持轴角、旋转矩阵、欧拉角等多种表示形式
- 姿态数据统计分析
- 批量处理和单文件快速处理
"""

import logging
import os
from typing import Dict, List, Any, Optional

import numpy as np

# 配置日志
# 设置日志级别为INFO，格式包含时间、级别和消息内容
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AMASSProcessor:
    """AMASS数据集综合处理器
    
    主要负责AMASS数据集的读取、处理和转换工作。
    支持多种姿态表示形式的相互转换，提供数据统计分析功能。
    """

    def __init__(self, dataset_root: str = r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS"):
        """
        初始化处理器
        
        Args:
            dataset_root: 数据集根目录路径
        """
        # 存储数据集根目录路径
        self.dataset_root = dataset_root
        # SMPL模型关节数量（24个标准关节点）
        self.SMPL_JOINTS = 24
        # 每个关节的维度（轴角表示需要3个参数）
        self.JOINT_DIM = 3
        # 身体姿态总维度（24关节 × 3维 = 72维）
        self.BODY_DIM = self.SMPL_JOINTS * self.JOINT_DIM

        # 记录初始化信息
        logger.info(f"初始化AMASS处理器")
        logger.info(f"数据集根目录: {self.dataset_root}")
        logger.info(f"SMPL关节数: {self.SMPL_JOINTS}")
        logger.info(f"身体姿态维度: {self.BODY_DIM}")

    # ==================== 数据集管理功能 ====================

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        获取数据集基本信息
        
        扫描数据集目录，统计文件数量和子集分布情况
        
        Returns:
            包含数据集统计信息的字典
        """
        # 检查数据集目录是否存在
        if not os.path.exists(self.dataset_root):
            return {"error": "数据集目录不存在"}

        # 初始化统计变量
        total_files = 0
        subsets_info = {}

        # 遍历目录树，统计所有.npz文件
        for root, dirs, files in os.walk(self.dataset_root):
            # 统计当前目录下的.npz文件数量
            npz_count = len([f for f in files if f.endswith('.npz')])
            if npz_count > 0:
                # 记录相对路径和文件数量
                rel_path = os.path.relpath(root, self.dataset_root)
                subsets_info[rel_path] = npz_count
                total_files += npz_count

        # 返回统计结果
        return {
            "root_dir": self.dataset_root,
            "total_files": total_files,
            "subsets": subsets_info
        }

    def find_npz_files(self, max_files: Optional[int] = None) -> List[str]:
        """
        查找所有.npz文件
        
        Args:
            max_files: 最大返回文件数，None表示不限制
            
        Returns:
            .npz文件路径列表
        """
        # 存储找到的.npz文件路径
        npz_files = []
        # 递归遍历数据集目录
        for root, dirs, files in os.walk(self.dataset_root):
            for file in files:
                # 筛选.npz文件
                if file.endswith('.npz'):
                    npz_files.append(os.path.join(root, file))

        # 如果指定了最大文件数，则截取前max_files个文件
        if max_files:
            npz_files = npz_files[:max_files]

        # 记录找到的文件数量
        logger.info(f"找到 {len(npz_files)} 个.npz文件")
        return npz_files

    def load_npz_file(self, file_path: str) -> Dict[str, Any]:
        """
        加载单个.npz文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含所有数据键值的字典
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            # 使用numpy加载.npz文件
            data = np.load(file_path)
            result = {}
            # 提取所有数据键值对
            for key in data.files:
                result[key] = data[key]

            # 记录加载成功的日志信息
            logger.info(f"成功加载文件: {os.path.basename(file_path)}")
            logger.info(f"包含键: {list(result.keys())}")
            return result
        except Exception as e:
            # 记录错误并重新抛出异常
            logger.error(f"加载文件失败: {e}")
            raise

    # ==================== 姿态数据处理功能 ====================

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
        """
        # 检查输入数据维度
        if poses_data.ndim != 2:
            raise ValueError(f"期望2维数组，得到{poses_data.ndim}维")

        # 检查数据维度是否足够
        if poses_data.shape[1] < self.BODY_DIM:
            raise ValueError(f"姿态数据维度不足，需要至少{self.BODY_DIM}维，实际只有{poses_data.shape[1]}维")

        # 提取前BODY_DIM维作为身体姿态数据
        body_poses = poses_data[:, :self.BODY_DIM]
        logger.info(f"成功提取身体姿态数据: {body_poses.shape}")
        return body_poses

    def axis_angle_to_rotation_matrix(self, axis_angle: np.ndarray) -> np.ndarray:
        """
        手动实现轴角到旋转矩阵的转换
        
        使用罗德里格斯公式(Rodrigues' formula)将轴角表示转换为旋转矩阵
        
        Args:
            axis_angle: 轴角向量，形状为 (3,)
            
        Returns:
            旋转矩阵，形状为 (3, 3)
        """
        # 计算旋转角度（向量的模长）
        angle = np.linalg.norm(axis_angle)

        # 处理零旋转的特殊情况
        if angle < 1e-10:
            return np.eye(3)

        # 计算单位旋转轴
        axis = axis_angle / angle
        # 计算三角函数值
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # 构造反对称矩阵
        skew_symmetric = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # 应用罗德里格斯公式计算旋转矩阵
        rotation_matrix = (
                cos_angle * np.eye(3) +
                sin_angle * skew_symmetric +
                (1 - cos_angle) * np.outer(axis, axis)
        )

        return rotation_matrix

    def convert_poses_to_rotation_matrices(self, body_poses: np.ndarray) -> np.ndarray:
        """
        将身体姿态转换为旋转矩阵
        
        将每个关节的轴角表示转换为3×3旋转矩阵
        
        Args:
            body_poses: 身体姿态数据，形状为 (frames, 72)
            
        Returns:
            旋转矩阵，形状为 (frames, 24, 3, 3)
        """

        # 获取帧数
        num_frames = body_poses.shape[0]
        # 初始化旋转矩阵数组
        rot_mats = np.zeros((num_frames, self.SMPL_JOINTS, 3, 3))

        # 逐帧逐关节进行转换
        for frame in range(num_frames):
            for joint in range(self.SMPL_JOINTS):
                # 计算当前关节在数据中的索引范围
                start_idx = joint * self.JOINT_DIM
                end_idx = start_idx + self.JOINT_DIM
                # 提取当前关节的轴角数据
                axis_angle = body_poses[frame, start_idx:end_idx]
                # 转换为旋转矩阵并存储
                rot_mats[frame, joint] = self.axis_angle_to_rotation_matrix(axis_angle)

        # 记录转换结果
        logger.info(f"成功转换为旋转矩阵: {rot_mats.shape}")
        return rot_mats

    def convert_poses_to_euler_angles(self, body_poses: np.ndarray) -> np.ndarray:
        """
        将身体姿态转换为欧拉角
        
        先转换为旋转矩阵，再从中提取欧拉角（XYZ顺序）
        
        Args:
            body_poses: 身体姿态数据，形状为 (frames, 72)
            
        Returns:
            欧拉角（度），形状为 (frames, 24, 3)
        """
        # 获取帧数
        num_frames = body_poses.shape[0]
        # 初始化欧拉角数组
        euler_angles = np.zeros((num_frames, self.SMPL_JOINTS, 3))

        # 先转换为旋转矩阵
        rot_mats = self.convert_poses_to_rotation_matrices(body_poses)

        # 逐帧逐关节提取欧拉角
        for frame in range(num_frames):
            for joint in range(self.SMPL_JOINTS):
                # 获取当前关节的旋转矩阵
                R = rot_mats[frame, joint]

                # 使用标准的XYZ顺序提取欧拉角
                # 处理万向锁(gimbal lock)的特殊情况
                if R[2, 0] < 1:
                    if R[2, 0] > -1:
                        # 正常情况
                        x = np.arctan2(R[2, 1], R[2, 2])
                        y = np.arcsin(-R[2, 0])
                        z = np.arctan2(R[1, 0], R[0, 0])
                    else:
                        # y = -π/2 的特殊情况
                        x = -np.arctan2(-R[1, 2], R[1, 1])
                        y = np.pi / 2
                        z = 0
                else:
                    # y = π/2 的特殊情况
                    x = np.arctan2(-R[1, 2], R[1, 1])
                    y = -np.pi / 2
                    z = 0

                # 转换为度数并存储
                euler_angles[frame, joint] = np.degrees([x, y, z])

        # 记录转换结果
        logger.info(f"成功转换为欧拉角: {euler_angles.shape}")
        return euler_angles

    def get_joint_names(self) -> List[str]:
        """
        获取SMPL模型的关节名称
        
        Returns:
            关节名称列表，按照标准SMPL关节顺序排列
        """
        # 返回24个标准SMPL关节名称
        # 按照身体层次结构排列：躯干→下肢→上肢
        return [
            'Pelvis',  # 骨盆
            'L_Hip',  # 左髋关节
            'R_Hip',  # 右髋关节
            'Spine1',  # 脊柱第1节
            'L_Knee',  # 左膝关节
            'R_Knee',  # 右膝关节
            'Spine2',  # 脊柱第2节
            'L_Ankle',  # 左踝关节
            'R_Ankle',  # 右踝关节
            'Spine3',  # 脊柱第3节
            'L_Foot',  # 左脚
            'R_Foot',  # 右脚
            'Neck',  # 脖子
            'L_Collar',  # 左锁骨
            'R_Collar',  # 右锁骨
            'Head',  # 头部
            'L_Shoulder',  # 左肩关节
            'R_Shoulder',  # 右肩关节
            'L_Elbow',  # 左肘关节
            'R_Elbow',  # 右肘关节
            'L_Wrist',  # 左腕关节
            'R_Wrist',  # 右腕关节
            'L_Hand',  # 左手
            'R_Hand'  # 右手
        ]

    def batch_convert_poses(self, poses_data: np.ndarray,
                            target_formats: List[str] = ['rotation_matrix']) -> Dict[str, np.ndarray]:
        """
        批量转换姿态数据到多种格式
        
        支持同时转换为多种表示形式，提高处理效率
        
        Args:
            poses_data: 完整的姿态数据，形状为 (frames, 156)
            target_formats: 目标格式列表 ['rotation_matrix', 'euler_angles']
            
        Returns:
            包含各种格式的字典
        """

        # 记录开始转换的日志
        logger.info("开始批量转换姿态数据...")

        # 提取身体姿态数据
        body_poses = self.extract_body_poses(poses_data)
        # 存储转换结果
        results = {}

        # 逐个格式进行转换
        for format_name in target_formats:
            try:
                # 根据目标格式调用相应的转换函数
                if format_name == 'rotation_matrix':
                    results['rotation_matrices'] = self.convert_poses_to_rotation_matrices(body_poses)
                elif format_name == 'euler_angles':
                    results['euler_angles'] = self.convert_poses_to_euler_angles(body_poses)
                else:
                    logger.warning(f"未知的格式: {format_name}")
                    continue

                # 记录转换成功的日志
                logger.info(f"{format_name} 转换完成")

            except Exception as e:
                # 记录转换失败的日志并继续处理其他格式
                logger.error(f"{format_name} 转换失败: {e}")
                continue

        # 记录最终转换结果
        logger.info(f"批量转换完成，共生成 {len(results)} 种格式")
        return results

    # ==================== 统计分析功能 ====================

    def analyze_pose_statistics(self, poses_data: np.ndarray) -> Dict[str, Any]:
        """
        分析姿态数据的统计信息
        
        计算每个关节在所有帧中的统计特征，包括均值、标准差、最值等
        
        Args:
            poses_data: 完整的姿态数据，形状为 (frames, 156)
            
        Returns:
            统计信息字典
        """

        # 提取身体姿态数据并重塑为关节格式
        body_poses = self.extract_body_poses(poses_data)
        num_frames = body_poses.shape[0]
        # 重塑为 (帧数, 关节数, 维度) 的格式便于分析
        joint_angles = body_poses.reshape(num_frames, self.SMPL_JOINTS, 3)

        # 初始化统计结果字典
        stats = {
            'num_frames': num_frames,  # 总帧数
            'joint_stats': {}  # 各关节统计信息
        }

        # 获取关节名称列表
        joint_names = self.get_joint_names()

        # 逐个关节计算统计信息
        for i, joint_name in enumerate(joint_names):
            # 提取当前关节的所有帧数据
            joint_data = joint_angles[:, i, :]

            # 计算该关节的各项统计指标
            stats['joint_stats'][joint_name] = {
                'mean': np.mean(joint_data, axis=0).tolist(),  # 均值
                'std': np.std(joint_data, axis=0).tolist(),  # 标准差
                'min': np.min(joint_data, axis=0).tolist(),  # 最小值
                'max': np.max(joint_data, axis=0).tolist(),  # 最大值
                'range': (np.max(joint_data, axis=0) - np.min(joint_data, axis=0)).tolist()  # 范围
            }

        # 记录统计完成日志
        logger.info("姿态统计数据计算完成")
        return stats

    # ==================== 便捷方法 ====================

    def process_single_file(self, file_path: str,
                            target_formats: List[str] = ['rotation_matrix']) -> Dict[str, Any]:
        """
        处理单个AMASS文件的完整流程
        
        集成数据加载、姿态转换、统计分析的完整处理流程
        
        Args:
            file_path: .npz文件路径
            target_formats: 目标转换格式
            
        Returns:
            包含原始数据和转换结果的完整字典
        """

        # 第一步：加载原始数据
        data = self.load_npz_file(file_path)

        # 初始化结果字典结构
        result = {
            'metadata': {  # 元数据信息
                'file_path': file_path,  # 文件完整路径
                'file_name': os.path.basename(file_path),  # 文件名
                'original_keys': list(data.keys())  # 原始数据键名
            },
            'raw_data': data,  # 原始加载的数据
            'processed_data': {}  # 处理后的数据
        }

        # 第二步：如果包含姿态数据，则进行处理
        if 'poses' in data:
            # 批量转换姿态数据到指定格式
            converted = self.batch_convert_poses(data['poses'], target_formats)
            result['processed_data']['converted_poses'] = converted

            # 第三步：进行统计分析
            stats = self.analyze_pose_statistics(data['poses'])
            result['processed_data']['statistics'] = stats

        return result


# ==================== 便捷函数 ====================
def create_processor(dataset_root: str = r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS") -> AMASSProcessor:
    """
    创建AMASS处理器实例
    
    工厂函数模式，提供统一的处理器创建接口
    
    Args:
        dataset_root: 数据集根目录
        
    Returns:
        AMASSProcessor实例
    """
    # 实例化并返回AMASS处理器
    return AMASSProcessor(dataset_root)


def quick_process_file(file_path: str,
                       dataset_root: str = r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS") -> Dict[str, Any]:
    """
    快速处理单个文件的便捷函数
    
    封装了完整的文件处理流程，适合快速使用场景
    
    Args:
        file_path: 文件路径
        dataset_root: 数据集根目录
        
    Returns:
        处理结果字典
    """
    # 创建处理器实例
    processor = create_processor(dataset_root)
    # 调用单文件处理方法并返回结果
    return processor.process_single_file(file_path)
