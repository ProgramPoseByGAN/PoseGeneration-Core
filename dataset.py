import numpy as np
import os
from typing import Dict, List, Union
from pathlib import Path

class PoseDataset:
    """
    姿态数据集类，用于读取和处理AMASS数据集中的.npz文件
    """
    
    # 全局变量：数据集根目录
    dataset_dir = r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS"
    
    def __init__(self, data_dir: str = None):
        """
        初始化数据集类
        
        Args:
            data_dir (str): 数据集目录路径，默认使用全局变量dataset_dir
        """
        self.data_dir = data_dir if data_dir else self.dataset_dir
        self.file_list: List[str] = []
        self.loaded_data: Dict[str, np.ndarray] = {}
        
        # 验证目录是否存在
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据集目录不存在: {self.data_dir}")
            
        # 获取所有.npz文件路径
        self._scan_npz_files()
        
    def _scan_npz_files(self):
        """扫描目录下所有的.npz文件"""
        self.file_list = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.npz'):
                    full_path = os.path.join(root, file)
                    self.file_list.append(full_path)
        
        print(f"找到 {len(self.file_list)} 个.npz文件")
        
    def load_single_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        加载单个.npz文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            Dict[str, np.ndarray]: 包含文件中所有数组的字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        try:
            # 加载.npz文件
            data = np.load(file_path)
            
            # 转换为字典格式
            result_dict = {}
            for key in data.files:
                result_dict[key] = data[key]
                
            print(f"成功加载文件: {file_path}")
            print(f"包含的键: {list(result_dict.keys())}")
            
            # 打印每个数组的基本信息
            for key, array in result_dict.items():
                print(f"  {key}: shape={array.shape}, dtype={array.dtype}")
                
            return result_dict
            
        except Exception as e:
            print(f"加载文件失败 {file_path}: {str(e)}")
            return {}
    
    def load_all_files(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        加载所有.npz文件
        
        Returns:
            Dict[str, Dict[str, np.ndarray]]: 以文件路径为键，数据字典为值的嵌套字典
        """
        all_data = {}
        
        print("开始加载所有.npz文件...")
        for i, file_path in enumerate(self.file_list, 1):
            print(f"\n[{i}/{len(self.file_list)}] 正在处理: {os.path.basename(file_path)}")
            data = self.load_single_file(file_path)
            if data:  # 只保存成功加载的数据
                all_data[file_path] = data
                
        print(f"\n总共成功加载 {len(all_data)} 个文件")
        return all_data
    
    def load_sample_files(self, num_samples: int = 5) -> Dict[str, Dict[str, np.ndarray]]:
        """
        加载指定数量的样本文件（用于测试）
        
        Args:
            num_samples (int): 要加载的样本数量
            
        Returns:
            Dict[str, Dict[str, np.ndarray]]: 样本数据
        """
        sample_files = self.file_list[:num_samples] if num_samples > 0 else self.file_list
        sample_data = {}
        
        print(f"加载前 {min(num_samples, len(self.file_list))} 个样本文件...")
        for i, file_path in enumerate(sample_files, 1):
            print(f"\n[{i}/{len(sample_files)}] 正在处理: {os.path.basename(file_path)}")
            data = self.load_single_file(file_path)
            if data:
                sample_data[file_path] = data
                
        return sample_data
    
    def get_statistics(self) -> Dict[str, Union[int, List[str]]]:
        """
        获取数据集统计信息
        
        Returns:
            Dict: 包含统计信息的字典
        """
        stats = {
            'total_files': len(self.file_list),
            'sample_files': self.file_list[:3],  # 显示前3个文件作为示例
            'data_dir': self.data_dir
        }
        return stats
    
    def print_dataset_info(self):
        """打印数据集基本信息"""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("数据集信息:")
        print("="*50)
        print(f"数据集目录: {stats['data_dir']}")
        print(f"总文件数: {stats['total_files']}")
        print("示例文件:")
        for i, file_path in enumerate(stats['sample_files'], 1):
            print(f"  {i}. {os.path.basename(file_path)}")
        print("="*50)

# 使用示例
if __name__ == "__main__":
    # 创建数据集实例
    dataset = PoseDataset()
    
    # 打印数据集基本信息
    dataset.print_dataset_info()
    
    # 加载几个样本文件进行测试
    print("\n开始加载样本数据...")
    sample_data = dataset.load_sample_files(num_samples=3)
    
    # 展示加载结果
    print(f"\n成功加载 {len(sample_data)} 个样本文件")
    
    # 展示第一个文件的详细内容
    if sample_data:
        first_file_path = list(sample_data.keys())[0]
        first_file_data = sample_data[first_file_path]
        print(f"\n第一个文件 '{os.path.basename(first_file_path)}' 的详细内容:")
        for key, array in first_file_data.items():
            print(f"  {key}: shape={array.shape}, dtype={array.dtype}")
            # 显示部分数据内容
            if array.size > 0:
                print(f"    前5个元素: {array.flat[:min(5, array.size)]}")