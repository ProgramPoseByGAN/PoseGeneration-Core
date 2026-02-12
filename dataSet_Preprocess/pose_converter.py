"""
姿态转换模块
负责AMASS姿态数据的各种数学表示形式之间的转换
该模块提供了完整的姿态数据处理工具链，支持多种旋转表示方法的相互转换

主要功能：
- 轴角 ↔ 旋转矩阵 ↔ 欧拉角转换：实现三种主流旋转表示方法的相互转换
- 姿态数据提取和重塑：从完整AMASS数据中提取身体姿态部分并重新组织
- 旋转表示标准化：确保旋转矩阵的数值稳定性和正交性
- 姿态数据分析：提供姿态数据的统计分析和验证功能
"""

# 第三方库导入
import logging                                  # 日志记录模块
from typing import Dict, List, Any, Optional, Tuple  # 类型提示支持
import numpy as np                              # 数值计算核心库
from scipy.spatial.transform import Rotation   # 科学计算库的旋转变换模块

# 配置日志系统 - 设置统一的日志格式和级别
logging.basicConfig(
    level=logging.INFO,                                    # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s'    # 定义日志输出格式
)
logger = logging.getLogger(__name__)  # 创建模块专用日志记录器

class PoseConverter:
    """姿态数据转换器
    提供AMASS姿态数据在不同数学表示形式之间的转换功能
    支持轴角、旋转矩阵、欧拉角等多种旋转表示方法的相互转换
    """
    
    def __init__(self):
        """初始化姿态转换器实例
        设置SMPL人体模型的关键参数，为后续姿态转换操作做准备
        """
        # SMPL人体模型参数定义
        self.SMPL_JOINTS = 24      # SMPL模型关节数量
        self.JOINT_DIM = 3         # 每个关节的维度（3D空间）
        self.BODY_DIM = self.SMPL_JOINTS * self.JOINT_DIM  # 身体姿态总维度
        
        # 记录初始化完成信息
        logger.info("PoseConverter初始化完成")
        logger.info(f"SMPL关节数: {self.SMPL_JOINTS}")
        logger.info(f"身体姿态维度: {self.BODY_DIM}")
    
    def extract_body_poses(self, poses_data: np.ndarray) -> np.ndarray:
        """
        从完整的AMASS poses数据中提取身体关节姿态部分（前72维）
        AMASS数据采用扩展的SMPL模型，除了基本的身体关节外还包含手部等附加部位
        该方法专注于提取核心的身体姿态信息，便于后续的标准处理
        
        AMASS数据中的poses字段通常包含156维数据结构：
        - 前72维：身体关节姿态（24关节×3维轴角表示）
        - 后续维度：手部、面部等其他部位的精细姿态信息
        
        Args:
            poses_data: 完整的姿态数据数组，预期形状为 (frames, 156)
            
        Returns:
            提取的身体姿态数据，形状为 (frames, 72)
            只包含24个身体关节的基础姿态信息
            
        Raises:
            ValueError: 当输入数据维度不足72维时抛出异常
        """
        # 记录身体姿态提取开始日志
        logger.info("提取身体姿态数据...")
        
        # 验证输入数据维度的正确性
        if poses_data.ndim != 2:
            raise ValueError(f"期望2维数组，得到{poses_data.ndim}维")
            
        # 检查数据维度是否满足最低要求
        if poses_data.shape[1] < self.BODY_DIM:
            raise ValueError(f"姿态数据维度不足，需要至少{self.BODY_DIM}维，实际只有{poses_data.shape[1]}维")
            
        # 提取前BODY_DIM维作为身体姿态数据（核心的身体关节信息）
        body_poses = poses_data[:, :self.BODY_DIM]
        
        # 记录提取成功信息
        logger.info(f"成功提取身体姿态数据: {body_poses.shape}")
        return body_poses
    
    def axis_angle_to_rotation_matrix(self, axis_angle: np.ndarray) -> np.ndarray:
        """
        将轴角向量转换为对应的3×3旋转矩阵
        使用scipy.spatial.transform库提供的专业旋转变换功能，确保数值精度和稳定性
        轴角表示法通过一个三维向量同时描述旋转轴和旋转角度
        
        Args:
            axis_angle: 轴角向量，形状为 (3,)，其中向量方向表示旋转轴，模长表示旋转角度（弧度）
            
        Returns:
            对应的3×3旋转矩阵，满足正交性和行列式为1的数学性质
        """
        # 处理零向量的特殊情况：当旋转角度接近0时，返回单位矩阵
        if np.linalg.norm(axis_angle) < 1e-10:
            return np.eye(3)
            
        # 使用scipy.spatial.transform进行专业的轴角到旋转矩阵转换
        rotation = Rotation.from_rotvec(axis_angle)
        return rotation.as_matrix()
    
    def rotation_matrix_to_axis_angle(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        将3×3旋转矩阵转换回轴角向量表示
        这是axis_angle_to_rotation_matrix方法的逆变换
        
        Args:
            rotation_matrix: 3×3正交旋转矩阵
            
        Returns:
            对应的轴角向量 (3,)，向量方向表示旋转轴，模长表示旋转角度
        """
        # 使用scipy.spatial.transform进行专业的旋转矩阵到轴角转换
        rotation = Rotation.from_matrix(rotation_matrix)
        return rotation.as_rotvec()
    
    def poses_to_rotation_matrices(self, body_poses: np.ndarray) -> np.ndarray:
        """
        将身体姿态数据（轴角表示）批量转换为旋转矩阵表示
        该方法对每一帧的每个关节执行轴角到旋转矩阵的转换
        
        Args:
            body_poses: 身体姿态数据，形状为 (frames, 72)
                       每帧包含24个关节，每个关节用3维轴角向量表示
            
        Returns:
            旋转矩阵表示的姿态数据，形状为 (frames, 24, 3, 3)
            每个关节对应一个3×3的旋转矩阵
        """
        # 记录转换开始日志
        logger.info("转换姿态数据为旋转矩阵...")
        
        # 获取帧数并初始化结果数组
        num_frames = body_poses.shape[0]
        rot_mats = np.zeros((num_frames, self.SMPL_JOINTS, 3, 3))
        
        # 逐帧逐关节进行批量转换处理
        for frame in range(num_frames):
            for joint in range(self.SMPL_JOINTS):
                # 计算当前关节在数据数组中的索引范围
                start_idx = joint * self.JOINT_DIM
                end_idx = start_idx + self.JOINT_DIM
                
                # 提取当前关节的轴角数据向量
                axis_angle = body_poses[frame, start_idx:end_idx]
                
                # 执行单个关节的轴角到旋转矩阵转换
                rot_mats[frame, joint] = self.axis_angle_to_rotation_matrix(axis_angle)
                
        # 记录转换完成信息
        logger.info(f"转换完成: {rot_mats.shape}")
        return rot_mats
    
    def rotation_matrices_to_poses(self, rot_mats: np.ndarray) -> np.ndarray:
        """
        将旋转矩阵表示的姿态数据转换回轴角向量表示
        这是poses_to_rotation_matrices方法的逆向操作，用于数据格式的相互转换
        
        Args:
            rot_mats: 旋转矩阵表示的姿态数据，形状为 (frames, joints, 3, 3)
                     每个关节对应一个3×3的旋转矩阵
            
        Returns:
            轴角表示的姿态数据，形状为 (frames, joints*3)
            每个关节用3维轴角向量表示，展平后连续存储
        """
        # 记录反向转换开始日志
        logger.info("转换旋转矩阵为轴角表示...")
        
        # 获取数据维度信息并初始化结果数组
        frames, joints = rot_mats.shape[:2]
        poses = np.zeros((frames, joints * 3))
        
        # 逐帧逐关节执行旋转矩阵到轴角的批量转换
        for frame in range(frames):
            for joint in range(joints):
                # 对单个关节执行旋转矩阵到轴角的转换
                axis_angle = self.rotation_matrix_to_axis_angle(rot_mats[frame, joint])
                
                # 将转换结果存储到对应位置的结果数组中
                start_idx = joint * 3
                end_idx = start_idx + 3
                poses[frame, start_idx:end_idx] = axis_angle
                
        # 记录转换完成信息
        logger.info(f"转换完成: {poses.shape}")
        return poses
    
    def rotation_matrix_to_euler_xyz(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        将3×3旋转矩阵转换为XYZ顺序的欧拉角表示（弧度制）
        该方法手动实现了旋转矩阵到欧拉角的转换算法，处理了万向锁等特殊情况
        XYZ顺序表示依次绕X轴、Y轴、Z轴进行旋转
        
        Args:
            rotation_matrix: 3×3正交旋转矩阵
            
        Returns:
            XYZ顺序的欧拉角数组 (3,)，单位为弧度
            格式：[绕X轴旋转角度, 绕Y轴旋转角度, 绕Z轴旋转角度]
        """
        # 提取矩阵元素
        r00, r01, r02 = rotation_matrix[0]
        r10, r11, r12 = rotation_matrix[1]
        r20, r21, r22 = rotation_matrix[2]
        
        # 处理万向锁特殊情况：当sin(Y) = ±1时会出现奇异性
        if abs(r20) >= 1:
            # 万向锁情况：绕Y轴旋转±90度
            y = np.sign(r20) * np.pi / 2
            x = 0  # X轴旋转设为0（奇异性处理）
            z = np.arctan2(-r01, r11)  # 通过其他元素计算Z轴旋转
        else:
            # 正常情况：使用标准反三角函数计算
            y = -np.arcsin(r20)        # Y轴旋转
            x = np.arctan2(r21, r22)   # X轴旋转
            z = np.arctan2(r10, r00)   # Z轴旋转
            
        # 返回XYZ顺序的欧拉角数组
        return np.array([x, y, z])
    
    def rotation_matrix_to_euler_zxy(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        将3×3旋转矩阵转换为ZXY顺序的欧拉角表示（角度制）
        严格按照BVH动画文件标准的ZXY旋转顺序实现，用于BVH文件生成
        ZXY顺序表示依次绕Z轴、X轴、Y轴进行旋转
        
        Args:
            rotation_matrix: 3×3正交旋转矩阵
            
        Returns:
            ZXY顺序的欧拉角数组 (3,)，单位为度
            格式：[绕Z轴旋转角度, 绕X轴旋转角度, 绕Y轴旋转角度]
        """
        # 提取旋转矩阵元素用于欧拉角计算
        r00, r01, r02 = rotation_matrix[0]
        r10, r11, r12 = rotation_matrix[1]
        r20, r21, r22 = rotation_matrix[2]
        
        # 处理万向锁特殊情况（当sin(X) = ±1时）
        if abs(r20) >= 1:
            # 万向锁情况：绕X轴旋转±90度
            y = np.sign(r20) * np.pi / 2
            z = 0  # Z轴旋转设为0（奇异性处理）
            x = np.arctan2(r01, r11)  # 通过其他元素计算X轴旋转
        else:
            # 正常情况：使用标准反三角函数计算各轴旋转角度
            y = -np.arcsin(r20)        # Y轴旋转
            x = np.arctan2(r21, r22)   # X轴旋转
            z = np.arctan2(r10, r00)   # Z轴旋转
            
        # 转换为角度制（BVH标准使用角度而非弧度）
        return np.degrees([z, x, y])  # ZXY顺序
    
    def poses_to_euler_angles(self, body_poses: np.ndarray, 
                            order: str = 'zxy') -> np.ndarray:
        """
        将身体姿态数据（轴角表示）批量转换为指定顺序的欧拉角表示
        该方法提供了一站式的姿态转换服务，内部自动处理中间的旋转矩阵转换
        
        Args:
            body_poses: 身体姿态数据，形状为 (frames, 72)
                       包含多帧的24个关节轴角数据
            order: 欧拉角旋转顺序，支持'xyz'或'zxy'两种标准顺序
            
        Returns:
            欧拉角表示的姿态数据，形状为 (frames, 24, 3)
            每个关节对应三个欧拉角分量
        """
        # 记录转换开始日志，显示使用的欧拉角顺序
        logger.info(f"转换姿态数据为{order.upper()}欧拉角...")
        
        # 第一步：将轴角表示转换为中间的旋转矩阵表示
        rot_mats = self.poses_to_rotation_matrices(body_poses)
        
        # 获取数据维度并初始化欧拉角结果数组
        frames, joints = rot_mats.shape[:2]
        euler_angles = np.zeros((frames, joints, 3))
        
        # 根据指定顺序选择相应的转换函数
        if order.lower() == 'xyz':
            converter_func = self.rotation_matrix_to_euler_xyz  # XYZ顺序转换函数
        elif order.lower() == 'zxy':
            converter_func = self.rotation_matrix_to_euler_zxy  # ZXY顺序转换函数
        else:
            raise ValueError(f"不支持的欧拉角顺序: {order}")
        
        # 逐帧逐关节执行批量转换
        for frame in range(frames):
            for joint in range(joints):
                # 对每个关节调用选定的欧拉角转换函数
                euler_angles[frame, joint] = converter_func(rot_mats[frame, joint])
                
        # 记录转换完成信息
        logger.info(f"转换完成: {euler_angles.shape}")
        return euler_angles
    
    def normalize_rotations(self, rot_mats: np.ndarray) -> np.ndarray:
        """
        对旋转矩阵数组进行标准化处理，确保每个矩阵都满足严格的正交性要求
        该方法使用奇异值分解(SVD)技术来修正数值误差导致的矩阵偏差
        是保证旋转矩阵数学性质的重要数值稳定化步骤
        
        Args:
            rot_mats: 任意形状的旋转矩阵数组，最后两维必须是(3, 3)
                     可以是单个矩阵或批量矩阵
            
        Returns:
            标准化后的旋转矩阵数组，保持原有形状但满足正交性约束
        """
        # 记录标准化处理开始日志
        logger.info("标准化旋转矩阵...")
        
        # 使用奇异值分解(SVD)进行矩阵正交化
        # SVD分解：rot_mats = U @ Σ @ V^T，其中U和V是正交矩阵
        u, _, vt = np.linalg.svd(rot_mats)
        # 重构为正交矩阵：U @ V^T（忽略奇异值Σ）
        normalized = np.matmul(u, vt)
        
        # 确保行列式为+1（真正的旋转矩阵性质，排除镜像变换）
        det = np.linalg.det(normalized)
        if np.any(det < 0):
            # 如果行列式为负，说明包含了镜像变换，需要修正
            # 通过翻转最后一列来改变行列式的符号
            normalized[..., :, -1] *= -1
            
        # 记录标准化完成信息
        logger.info("标准化完成")
        return normalized
    
    def compute_angular_velocity(self, poses: np.ndarray, 
                               framerate: float = 120.0) -> np.ndarray:
        """
        通过有限差分方法计算姿态数据的角速度
        该方法基于相邻帧间的姿态变化率来估算角速度，是运动分析的基础工具
        
        Args:
            poses: 姿态数据，形状为 (frames, joints*3)
                  通常为轴角表示的姿态数据
            framerate: 数据采集帧率（Hz），用于时间间隔计算
            
        Returns:
            角速度数据，形状为 (frames-1, joints*3)
            结果比输入少一帧，因为使用了前后帧差分
        """
        # 记录角速度计算开始日志
        logger.info("计算角速度...")
        
        # 检查数据是否足够进行差分计算
        if len(poses) < 2:
            logger.warning("姿态数据帧数不足，无法计算角速度")
            return np.array([])
            
        # 计算相邻帧间的差分（有限差分法）
        diff = np.diff(poses, axis=0)
        
        # 转换为真实的角速度单位（弧度/秒）
        time_delta = 1.0 / framerate  # 时间间隔（秒）
        angular_velocity = diff / time_delta  # 差分除以时间间隔得到角速度
        
        # 记录计算完成信息
        logger.info(f"角速度计算完成: {angular_velocity.shape}")
        return angular_velocity
    
    def analyze_pose_statistics(self, poses_data: np.ndarray) -> Dict[str, Any]:
        """
        对姿态数据进行全面的统计分析，包括各关节的角度分布、数据范围等信息
        该方法为数据质量评估和异常检测提供重要的统计基础
        
        Args:
            poses_data: 完整的姿态数据，形状为 (frames, 156)
                       包含所有关节和附加部位的姿态信息
            
        Returns:
            包含详细统计信息的字典，结构如下：
            {
                'num_frames': int,           # 帧数
                'total_joints': int,         # 关节数
                'joint_stats': dict,         # 各关节详细统计
                'overall_stats': dict        # 整体统计信息
            }
        """
        # 记录统计分析开始日志
        logger.info("分析姿态数据统计信息...")
        
        # 提取核心的身体姿态数据（前72维）
        body_poses = self.extract_body_poses(poses_data)
        frames = body_poses.shape[0]
        
        # 重塑数据格式为关节组织形式，便于按关节进行统计分析
        joint_poses = body_poses.reshape(frames, self.SMPL_JOINTS, 3)
        
        # 初始化统计信息结构
        stats = {
            'num_frames': frames,              # 总帧数
            'total_joints': self.SMPL_JOINTS,  # 总关节数
            'joint_stats': {}                  # 各关节统计信息容器
        }
        
        # 逐关节计算详细的统计信息
        joint_names = self.get_smpl_joint_names()
        for i, joint_name in enumerate(joint_names):
            # 提取当前关节的所有帧数据
            joint_data = joint_poses[:, i, :]
            
            # 计算当前关节在三个轴向上的统计信息
            stats['joint_stats'][joint_name] = {
                'mean': np.mean(joint_data, axis=0).tolist(),      # 均值
                'std': np.std(joint_data, axis=0).tolist(),        # 标准差
                'min': np.min(joint_data, axis=0).tolist(),        # 最小值
                'max': np.max(joint_data, axis=0).tolist(),        # 最大值
                'range': (np.max(joint_data, axis=0) - np.min(joint_data, axis=0)).tolist()  # 范围
            }
            
        # 计算整体数据的统计信息
        all_data = body_poses.flatten()  # 将所有数据展平为一维数组
        stats['overall_stats'] = {
            'mean': float(np.mean(all_data)),    # 整体均值
            'std': float(np.std(all_data)),      # 整体标准差
            'min': float(np.min(all_data)),      # 整体最小值
            'max': float(np.max(all_data))       # 整体最大值
        }
        
        # 记录统计分析完成信息
        logger.info("统计分析完成")
        return stats
    
    def get_smpl_joint_names(self) -> List[str]:
        """
        获取SMPL人体模型的标准关节名称列表
        按照SMPL模型的固定顺序返回所有24个关节的名称
        该列表用于关节数据的语义标识和索引映射
        
        Returns:
            包含24个关节名称的字符串列表，按SMPL标准顺序排列
        """
        return [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
        ]
    
    def validate_pose_data(self, poses_data: np.ndarray) -> Dict[str, Any]:
        """
        对输入的姿态数据进行全面的有效性验证
        检查数据类型、维度、数值范围等多个方面，确保数据质量和可用性
        
        Args:
            poses_data: 待验证的姿态数据数组
            
        Returns:
            包含验证结果的字典，结构如下：
            {
                'is_valid': bool,      # 数据是否有效
                'errors': list,        # 错误信息列表
                'warnings': list       # 警告信息列表
            }
        """
        # 初始化验证结果结构
        result = {
            'is_valid': True,     # 默认假设数据有效
            'errors': [],         # 错误信息收集列表
            'warnings': []        # 警告信息收集列表
        }
        
        try:
            # 检查数据类型是否为numpy数组
            if not isinstance(poses_data, np.ndarray):
                result['errors'].append("数据不是numpy数组")
                result['is_valid'] = False
                return result
                
            # 检查数组维度是否符合要求
            if poses_data.ndim != 2:
                result['errors'].append(f"数据维度应该是2维，当前是{poses_data.ndim}维")
                result['is_valid'] = False
                return result
                
            # 检查数据维度是否满足最低要求
            if poses_data.shape[1] < self.BODY_DIM:
                result['errors'].append(f"数据维度不足，至少需要{self.BODY_DIM}维")
                result['is_valid'] = False
                return result
                
            # 检查是否存在NaN（Not a Number）值
            if np.isnan(poses_data).any():
                result['warnings'].append("数据包含NaN值")
                
            # 检查是否存在无穷大值
            if np.isinf(poses_data).any():
                result['warnings'].append("数据包含无穷值")
                
            # 检查数值范围的合理性（轴角值通常在合理范围内）
            abs_max = np.max(np.abs(poses_data))
            if abs_max > 10:  # 轴角值通常不会超过这个范围
                result['warnings'].append(f"数据值过大 (最大绝对值: {abs_max})")
                
        except Exception as e:
            # 捕获并记录验证过程中出现的任何异常
            result['errors'].append(f"验证过程出错: {str(e)}")
            result['is_valid'] = False
            
        return result

def main():
    """主函数 - 演示姿态转换器的核心功能和使用方法
    该函数通过创建测试数据来展示各类转换功能的效果和正确性
    """
    # 创建姿态转换器实例
    converter = PoseConverter()
    
    # 创建测试用的姿态数据
    test_frames = 100
    # 生成小幅随机数据模拟真实的姿态数据分布
    test_poses = np.random.randn(test_frames, 156) * 0.1
    
    print("=== 姿态转换器演示 ===")
    
    # 1. 提取身体姿态（从完整数据中分离出核心身体关节数据）
    body_poses = converter.extract_body_poses(test_poses)
    print(f"身体姿态形状: {body_poses.shape}")
    
    # 2. 将轴角表示转换为旋转矩阵表示
    rot_mats = converter.poses_to_rotation_matrices(body_poses)
    print(f"旋转矩阵形状: {rot_mats.shape}")
    
    # 3. 将姿态数据转换为ZXY顺序的欧拉角表示
    euler_angles = converter.poses_to_euler_angles(body_poses, order='zxy')
    print(f"欧拉角形状: {euler_angles.shape}")
    
    # 4. 验证输入数据的有效性
    validation = converter.validate_pose_data(test_poses)
    print(f"数据有效性: {'✓' if validation['is_valid'] else '✗'}")
    # 显示验证过程中发现的警告信息
    if validation['warnings']:
        print(f"警告: {validation['warnings']}")
    # 显示验证过程中发现的错误信息
    if validation['errors']:
        print(f"错误: {validation['errors']}")
    
    # 5. 对姿态数据进行统计分析
    stats = converter.analyze_pose_statistics(test_poses)
    print(f"帧数: {stats['num_frames']}")
    print(f"关节数: {stats['total_joints']}")
    print(f"整体均值: {stats['overall_stats']['mean']:.4f}")

# 程序入口点：当脚本直接运行时执行演示功能
if __name__ == "__main__":
    main()