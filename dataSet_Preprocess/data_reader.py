"""
AMASS数据读取模块
负责AMASS数据集的文件扫描、加载和基本信息提取

主要功能：
- 递归扫描.npz文件
- 加载AMASS数据文件
- 提取数据集统计信息
- 验证文件完整性
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据集根目录配置
DATASET_DIR = r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS"

class AMASSDataReader:
    """AMASS数据读取器"""
    
    def __init__(self, dataset_root: str = DATASET_DIR):
        """
        初始化数据读取器
        
        Args:
            dataset_root: 数据集根目录路径
        """
        self.dataset_root = Path(dataset_root)
        self.supported_file_types = ['.npz']
        
        logger.info(f"AMASSDataReader初始化完成")
        logger.info(f"数据集根目录: {self.dataset_root}")
        
        # 验证数据集目录是否存在
        if not self.dataset_root.exists():
            logger.warning(f"数据集目录不存在: {self.dataset_root}")
    
    def scan_dataset(self, max_files: Optional[int] = None, 
                    file_pattern: str = "*.npz") -> List[Dict[str, Any]]:
        """
        扫描数据集目录，获取文件信息列表
        
        Args:
            max_files: 最大返回文件数，None表示不限制
            file_pattern: 文件匹配模式
            
        Returns:
            文件信息列表，每个元素包含文件路径、大小、修改时间等信息
        """
        logger.info("开始扫描数据集...")
        
        file_infos = []
        
        # 递归遍历目录
        for root, dirs, files in os.walk(self.dataset_root):
            for file in files:
                if file.endswith(tuple(self.supported_file_types)):
                    file_path = Path(root) / file
                    
                    try:
                        # 获取文件信息
                        stat = file_path.stat()
                        file_info = {
                            'path': str(file_path),
                            'name': file,
                            'size': stat.st_size,
                            'modified_time': stat.st_mtime,
                            'relative_path': str(file_path.relative_to(self.dataset_root))
                        }
                        
                        file_infos.append(file_info)
                        
                        # 检查是否达到最大文件数限制
                        if max_files and len(file_infos) >= max_files:
                            logger.info(f"达到最大文件数限制: {max_files}")
                            break
                            
                    except Exception as e:
                        logger.warning(f"获取文件信息失败 {file_path}: {e}")
                        continue
                        
            # 检查是否达到最大文件数限制
            if max_files and len(file_infos) >= max_files:
                break
                
        logger.info(f"扫描完成，找到 {len(file_infos)} 个文件")
        return file_infos
    
    def load_amass_file(self, file_path: str) -> Dict[str, Any]:
        """
        加载单个AMASS .npz文件
        
        Args:
            file_path: 文件路径（绝对路径或相对于数据集根目录的路径）
            
        Returns:
            包含所有数据字段的字典
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不正确或缺少必要字段
        """
        # 处理相对路径
        if not os.path.isabs(file_path):
            full_path = self.dataset_root / file_path
        else:
            full_path = Path(file_path)
            
        # 验证文件存在性
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {full_path}")
            
        logger.info(f"加载文件: {full_path.name}")
        
        try:
            # 加载.npz文件
            data = np.load(full_path)
            
            # 提取所有数据字段
            result = {}
            for key in data.files:
                result[key] = data[key]
                
            # 添加文件元信息
            stat = full_path.stat()
            result['_metadata'] = {
                'file_path': str(full_path),
                'file_name': full_path.name,
                'file_size': stat.st_size,
                'modified_time': stat.st_mtime,
                'data_keys': list(data.files)
            }
            
            # 验证必要字段
            required_fields = ['poses', 'trans']
            missing_fields = [field for field in required_fields if field not in data.files]
            if missing_fields:
                raise ValueError(f"缺少必要字段: {missing_fields}")
                
            logger.info(f"文件加载成功: {full_path.name}")
            logger.info(f"  数据键: {list(data.files)}")
            logger.info(f"  姿态数据形状: {data['poses'].shape}")
            logger.info(f"  位移数据形状: {data['trans'].shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"加载文件失败 {full_path}: {e}")
            raise
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            包含数据集统计信息的字典
        """
        logger.info("计算数据集统计信息...")
        
        # 扫描所有文件
        all_files = self.scan_dataset()
        
        if not all_files:
            return {"error": "未找到任何数据文件"}
            
        # 按子目录统计
        subset_stats = {}
        total_size = 0
        
        for file_info in all_files:
            rel_path = Path(file_info['relative_path'])
            # 获取子目录名称（通常是第一个目录）
            if len(rel_path.parts) > 1:
                subset = rel_path.parts[0]
            else:
                subset = "root"
                
            if subset not in subset_stats:
                subset_stats[subset] = {
                    'file_count': 0,
                    'total_size': 0,
                    'files': []
                }
                
            subset_stats[subset]['file_count'] += 1
            subset_stats[subset]['total_size'] += file_info['size']
            subset_stats[subset]['files'].append(file_info['name'])
            
            total_size += file_info['size']
        
        # 计算总体统计
        stats = {
            'total_files': len(all_files),
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'subsets': subset_stats,
            'dataset_root': str(self.dataset_root)
        }
        
        logger.info(f"数据集统计完成:")
        logger.info(f"  总文件数: {stats['total_files']}")
        logger.info(f"  总大小: {stats['total_size_mb']} MB")
        logger.info(f"  子集数量: {len(subset_stats)}")
        
        return stats
    
    def get_sample_files(self, count: int = 5, 
                        include_stageii_only: bool = True) -> List[str]:
        """
        获取样本文件路径列表
        
        Args:
            count: 样本数量
            include_stageii_only: 是否只包含stageii文件
            
        Returns:
            文件路径列表
        """
        logger.info(f"获取 {count} 个样本文件...")
        
        file_infos = self.scan_dataset(max_files=count * 2)  # 多扫描一些以防过滤后不够
        
        sample_paths = []
        for file_info in file_infos:
            file_path = file_info['path']
            
            # 根据条件筛选文件
            if include_stageii_only and 'stageii' not in file_path.lower():
                continue
                
            sample_paths.append(file_path)
            
            if len(sample_paths) >= count:
                break
                
        logger.info(f"获取到 {len(sample_paths)} 个样本文件")
        return sample_paths
    
    def validate_file_integrity(self, file_path: str) -> Dict[str, Any]:
        """
        验证文件完整性
        
        Args:
            file_path: 文件路径
            
        Returns:
            验证结果字典
        """
        result = {
            'file_path': file_path,
            'is_valid': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                result['errors'].append("文件不存在")
                return result
                
            # 尝试加载文件
            data = self.load_amass_file(file_path)
            
            # 检查必要字段
            required_fields = ['poses', 'trans']
            for field in required_fields:
                if field not in data:
                    result['errors'].append(f"缺少必要字段: {field}")
                else:
                    # 检查数据维度
                    field_data = data[field]
                    if field_data.size == 0:
                        result['errors'].append(f"字段 {field} 为空")
                    elif len(field_data.shape) < 1:
                        result['errors'].append(f"字段 {field} 维度不正确")
                        
            # 检查数据合理性
            poses = data.get('poses', np.array([]))
            if poses.size > 0:
                if poses.shape[1] < 72:
                    result['warnings'].append(f"姿态数据维度不足 (当前: {poses.shape[1]}, 建议: ≥72)")
                    
            # 如果没有错误，则认为文件有效
            if not result['errors']:
                result['is_valid'] = True
                
        except Exception as e:
            result['errors'].append(f"文件加载失败: {str(e)}")
            
        return result

def main():
    """主函数 - 演示基本功能"""
    reader = AMASSDataReader()
    
    # 1. 显示数据集统计
    print("=== 数据集统计信息 ===")
    stats = reader.get_dataset_statistics()
    if 'error' not in stats:
        print(f"总文件数: {stats['total_files']}")
        print(f"总大小: {stats['total_size_mb']} MB")
        print(f"子集分布:")
        for subset, info in list(stats['subsets'].items())[:5]:
            print(f"  {subset}: {info['file_count']} 文件, {round(info['total_size']/(1024*1024), 2)} MB")
    
    # 2. 获取样本文件
    print("\n=== 样本文件 ===")
    samples = reader.get_sample_files(count=3)
    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample}")
        
        # 验证文件完整性
        validation = reader.validate_file_integrity(sample)
        print(f"   有效性: {'✓' if validation['is_valid'] else '✗'}")
        if validation['errors']:
            print(f"   错误: {validation['errors']}")
        if validation['warnings']:
            print(f"   警告: {validation['warnings']}")

if __name__ == "__main__":
    main()