"""
AMASS数据读取模块
负责AMASS数据集的文件扫描、加载和基本信息提取
该模块提供了完整的AMASS数据集访问接口，支持高效的数据检索和基础信息提取功能

主要功能：
- 递归扫描.npz文件：自动遍历数据集目录结构，识别所有AMASS格式文件
- 加载AMASS数据文件：安全可靠地读取.npz格式的姿态数据文件
- 提取数据集统计信息：计算文件数量、大小分布等关键统计数据
- 验证文件完整性：检查数据文件的基本结构和必要字段完整性
"""

# 系统和第三方库导入
import os                    # 操作系统接口，用于文件系统操作
import logging              # 日志记录模块
from typing import Dict, List, Any, Optional, Tuple  # 类型提示支持
from pathlib import Path    # 现代路径操作库
import numpy as np          # 数值计算库，用于数据处理

# 配置日志系统 - 设置统一的日志格式和级别
logging.basicConfig(
    level=logging.INFO,                                    # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s'    # 定义日志输出格式：时间 - 级别 - 消息
)
logger = logging.getLogger(__name__)  # 创建模块专用日志记录器

# 数据集根目录配置 - 定义AMASS数据集的默认存储位置
DATASET_DIR = r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS"

class AMASSDataReader:
    """AMASS数据读取器
    提供AMASS数据集的完整访问接口，包括文件扫描、数据加载、统计信息提取等功能
    该类封装了所有与AMASS数据集交互的核心逻辑
    """
    
    def __init__(self, dataset_root: str = DATASET_DIR):
        """
        初始化数据读取器实例
        设置数据集根目录并验证其存在性，为后续数据操作做好准备
        
        Args:
            dataset_root: 数据集根目录路径，默认使用预定义的DATASET_DIR
        """
        # 存储数据集根目录路径（转换为Path对象以便跨平台兼容）
        self.dataset_root = Path(dataset_root)
        # 定义支持的文件类型列表，目前仅支持.npz格式
        self.supported_file_types = ['.npz']
        
        # 记录初始化完成信息
        logger.info(f"AMASSDataReader初始化完成")
        logger.info(f"数据集根目录: {self.dataset_root}")
        
        # 验证数据集目录是否存在，如果不存在则发出警告
        if not self.dataset_root.exists():
            logger.warning(f"数据集目录不存在: {self.dataset_root}")
    
    def scan_dataset(self, max_files: Optional[int] = None, 
                    file_pattern: str = "*.npz") -> List[Dict[str, Any]]:
        """
        扫描数据集目录，递归遍历所有子目录获取符合条件的文件信息
        该方法支持文件数量限制和自定义文件模式匹配，提高扫描效率
        
        Args:
            max_files: 最大返回文件数，None表示不限制，用于控制内存使用
            file_pattern: 文件匹配模式，默认匹配所有.npz文件
            
        Returns:
            文件信息列表，每个元素包含文件路径、大小、修改时间等详细信息
            格式：[{path: str, name: str, size: int, modified_time: float, relative_path: str}, ...]
        """
        # 记录扫描开始日志
        logger.info("开始扫描数据集...")
        
        # 初始化文件信息列表，用于存储扫描结果
        file_infos = []
        
        # 递归遍历目录结构，os.walk提供深度优先的目录遍历
        for root, dirs, files in os.walk(self.dataset_root):
            for file in files:
                # 检查文件扩展名是否在支持的类型列表中
                if file.endswith(tuple(self.supported_file_types)):
                    # 构造完整的文件路径
                    file_path = Path(root) / file
                    
                    try:
                        # 获取文件系统级别的详细信息
                        stat = file_path.stat()
                        # 构造文件信息字典
                        file_info = {
                            'path': str(file_path),                    # 完整文件路径
                            'name': file,                              # 文件名
                            'size': stat.st_size,                      # 文件大小（字节）
                            'modified_time': stat.st_mtime,            # 最后修改时间戳
                            'relative_path': str(file_path.relative_to(self.dataset_root))  # 相对于数据集根目录的路径
                        }
                        
                        # 将文件信息添加到结果列表
                        file_infos.append(file_info)
                        
                        # 检查是否达到最大文件数限制，用于控制资源使用
                        if max_files and len(file_infos) >= max_files:
                            logger.info(f"达到最大文件数限制: {max_files}")
                            break
                            
                    except Exception as e:
                        # 处理文件信息获取过程中的异常（如权限问题、文件损坏等）
                        logger.warning(f"获取文件信息失败 {file_path}: {e}")
                        continue
                        
            # 外层循环检查：如果已达到文件数量限制，则提前退出目录遍历
            if max_files and len(file_infos) >= max_files:
                break
                
        # 记录扫描完成状态和找到的文件总数
        logger.info(f"扫描完成，找到 {len(file_infos)} 个文件")
        return file_infos
    
    def load_amass_file(self, file_path: str) -> Dict[str, Any]:
        """
        加载单个AMASS .npz文件，提取其中的所有数据字段并进行基本验证
        该方法支持相对路径和绝对路径两种输入方式，提供灵活的文件访问机制
        
        Args:
            file_path: 文件路径，可以是绝对路径或相对于数据集根目录的相对路径
            
        Returns:
            包含所有数据字段的字典，额外添加_metadata字段包含文件元信息
            返回格式：{'poses': ndarray, 'trans': ndarray, ..., '_metadata': {...}}
            
        Raises:
            FileNotFoundError: 当指定文件不存在时抛出
            ValueError: 当文件格式不正确或缺少必要字段时抛出
        """
        # 处理相对路径转换：如果是相对路径则拼接数据集根目录
        if not os.path.isabs(file_path):
            full_path = self.dataset_root / file_path
        else:
            # 如果已经是绝对路径则直接使用
            full_path = Path(file_path)
            
        # 验证文件存在性，确保后续操作的安全性
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {full_path}")
            
        # 记录文件加载开始日志
        logger.info(f"加载文件: {full_path.name}")
        
        try:
            # 使用numpy加载.npz压缩文件格式
            data = np.load(full_path)
            
            # 提取.npz文件中的所有数据字段到字典中
            result = {}
            for key in data.files:
                result[key] = data[key]
                
            # 添加文件元信息，便于后续处理和调试
            stat = full_path.stat()
            result['_metadata'] = {
                'file_path': str(full_path),      # 完整文件路径
                'file_name': full_path.name,      # 文件名
                'file_size': stat.st_size,        # 文件大小
                'modified_time': stat.st_mtime,   # 修改时间
                'data_keys': list(data.files)     # 包含的数据键列表
            }
            
            # 验证必要字段完整性，确保数据可用性
            required_fields = ['poses', 'trans']  # AMASS数据必须包含姿态和位移字段
            missing_fields = [field for field in required_fields if field not in data.files]
            if missing_fields:
                raise ValueError(f"缺少必要字段: {missing_fields}")
                
            # 记录加载成功信息和关键数据统计
            logger.info(f"文件加载成功: {full_path.name}")
            logger.info(f"  数据键: {list(data.files)}")
            logger.info(f"  姿态数据形状: {data['poses'].shape}")
            logger.info(f"  位移数据形状: {data['trans'].shape}")
            
            return result
            
        except Exception as e:
            # 记录加载失败的详细错误信息
            logger.error(f"加载文件失败 {full_path}: {e}")
            raise
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        获取数据集的全面统计信息，包括文件数量、大小分布、子集组织结构等
        该方法通过扫描整个数据集来计算各种统计指标，为数据管理提供参考
        
        Returns:
            包含数据集统计信息的字典，格式如下：
            {
                'total_files': int,           # 总文件数
                'total_size': int,            # 总大小（字节）
                'total_size_mb': float,       # 总大小（MB）
                'subsets': dict,              # 各子集统计信息
                'dataset_root': str           # 数据集根目录路径
            }
        """
        # 记录统计计算开始日志
        logger.info("计算数据集统计信息...")
        
        # 扫描所有文件以获取完整文件列表
        all_files = self.scan_dataset()
        
        # 检查是否有文件存在，如果没有则返回错误信息
        if not all_files:
            return {"error": "未找到任何数据文件"}
            
        # 按子目录组织结构进行统计分类
        subset_stats = {}  # 存储各子集的统计信息
        total_size = 0     # 累计总文件大小
        
        # 遍历所有文件信息进行分类统计
        for file_info in all_files:
            rel_path = Path(file_info['relative_path'])
            # 获取子目录名称（通常是第一个目录层级）
            if len(rel_path.parts) > 1:
                subset = rel_path.parts[0]  # 取第一级子目录名
            else:
                subset = "root"  # 如果没有子目录则归类为root
                
            # 初始化子集统计信息结构
            if subset not in subset_stats:
                subset_stats[subset] = {
                    'file_count': 0,      # 该子集文件数量
                    'total_size': 0,      # 该子集总大小
                    'files': []           # 该子集文件列表
                }
                
            # 更新子集统计信息
            subset_stats[subset]['file_count'] += 1
            subset_stats[subset]['total_size'] += file_info['size']
            subset_stats[subset]['files'].append(file_info['name'])
            
            # 累计总大小
            total_size += file_info['size']
        
        # 计算并组装总体统计信息
        stats = {
            'total_files': len(all_files),                          # 总文件数量
            'total_size': total_size,                               # 总大小（字节）
            'total_size_mb': round(total_size / (1024 * 1024), 2),  # 总大小（MB，保留2位小数）
            'subsets': subset_stats,                                # 各子集详细统计
            'dataset_root': str(self.dataset_root)                  # 数据集根目录路径
        }
        
        # 记录统计结果摘要信息
        logger.info(f"数据集统计完成:")
        logger.info(f"  总文件数: {stats['total_files']}")
        logger.info(f"  总大小: {stats['total_size_mb']} MB")
        logger.info(f"  子集数量: {len(subset_stats)}")
        
        return stats
    
    def get_sample_files(self, count: int = 5, 
                        include_stageii_only: bool = True) -> List[str]:
        """
        获取指定数量的样本文件路径列表，用于测试和演示目的
        该方法支持过滤条件，可以根据需要筛选特定类型的文件
        
        Args:
            count: 需要获取的样本文件数量
            include_stageii_only: 是否只包含stageii文件（通常质量更高）
            
        Returns:
            符合条件的文件路径字符串列表
        """
        # 记录样本获取开始日志
        logger.info(f"获取 {count} 个样本文件...")
        
        # 扫描文件，预留更多文件以防过滤后数量不足
        file_infos = self.scan_dataset(max_files=count * 2)  # 多扫描一些以防过滤后不够
        
        # 初始化样本路径列表
        sample_paths = []
        # 遍历文件信息进行筛选
        for file_info in file_infos:
            file_path = file_info['path']
            
            # 根据条件筛选文件：如果要求stageii且文件名不包含stageii则跳过
            if include_stageii_only and 'stageii' not in file_path.lower():
                continue
                
            # 将符合条件的文件路径添加到样本列表
            sample_paths.append(file_path)
            
            # 达到所需数量时提前结束
            if len(sample_paths) >= count:
                break
                
        # 记录获取结果
        logger.info(f"获取到 {len(sample_paths)} 个样本文件")
        return sample_paths
    
    def validate_file_integrity(self, file_path: str) -> Dict[str, Any]:
        """
        验证指定文件的完整性和基本结构，检查是否存在关键数据字段
        该方法不仅检查文件是否存在，还会尝试加载并验证数据结构的正确性
        
        Args:
            file_path: 需要验证的文件路径
            
        Returns:
            验证结果字典，包含以下字段：
            {
                'file_path': str,      # 验证的文件路径
                'is_valid': bool,      # 文件是否有效
                'errors': list,        # 错误信息列表
                'warnings': list       # 警告信息列表
            }
        """
        # 初始化验证结果结构
        result = {
            'file_path': file_path,    # 记录被验证的文件路径
            'is_valid': False,         # 默认设置为无效
            'errors': [],              # 错误信息收集列表
            'warnings': []             # 警告信息收集列表
        }
        
        try:
            # 检查文件物理存在性
            if not os.path.exists(file_path):
                result['errors'].append("文件不存在")
                return result
                
            # 尝试加载文件以进行深入验证
            data = self.load_amass_file(file_path)
            
            # 检查必要数据字段的存在性和有效性
            required_fields = ['poses', 'trans']
            for field in required_fields:
                if field not in data:
                    result['errors'].append(f"缺少必要字段: {field}")
                else:
                    # 检查数据维度和非空性
                    field_data = data[field]
                    if field_data.size == 0:
                        result['errors'].append(f"字段 {field} 为空")
                    elif len(field_data.shape) < 1:
                        result['errors'].append(f"字段 {field} 维度不正确")
                        
            # 检查数据内容的合理性
            poses = data.get('poses', np.array([]))
            if poses.size > 0:
                # 检查姿态数据维度是否满足基本要求（至少72维身体数据）
                if poses.shape[1] < 72:
                    result['warnings'].append(f"姿态数据维度不足 (当前: {poses.shape[1]}, 建议: ≥72)")
                    
            # 如果没有任何错误，则标记文件为有效
            if not result['errors']:
                result['is_valid'] = True
                
        except Exception as e:
            # 捕获并记录加载过程中的任何异常
            result['errors'].append(f"文件加载失败: {str(e)}")
            
        return result

def main():
    """主函数 - 演示AMASS数据读取器的基本功能和使用方法
    该函数展示了数据读取模块的核心功能，包括数据集统计、样本获取和文件验证
    """
    # 创建数据读取器实例
    reader = AMASSDataReader()
    
    # 1. 显示数据集统计信息
    print("=== 数据集统计信息 ===")
    stats = reader.get_dataset_statistics()
    if 'error' not in stats:
        print(f"总文件数: {stats['total_files']}")
        print(f"总大小: {stats['total_size_mb']} MB")
        print(f"子集分布:")
        # 显示前5个子集的详细信息
        for subset, info in list(stats['subsets'].items())[:5]:
            print(f"  {subset}: {info['file_count']} 文件, {round(info['total_size']/(1024*1024), 2)} MB")
    
    # 2. 获取样本文件用于演示
    print("\n=== 样本文件 ===")
    samples = reader.get_sample_files(count=3)
    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample}")
        
        # 验证文件完整性，检查数据结构是否正确
        validation = reader.validate_file_integrity(sample)
        print(f"   有效性: {'✓' if validation['is_valid'] else '✗'}")
        # 显示具体的错误信息
        if validation['errors']:
            print(f"   错误: {validation['errors']}")
        # 显示警告信息
        if validation['warnings']:
            print(f"   警告: {validation['warnings']}")

# 程序入口点：当脚本直接运行时执行main函数
if __name__ == "__main__":
    main()