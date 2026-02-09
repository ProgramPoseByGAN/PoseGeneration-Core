import os
import numpy as np
from typing import Dict, List, Any

# 全局变量：数据集目录
dataset_dir = r"/PoseGeneration-Core/dataset/AMASS"

# ==================== 配置管理功能 ====================
def get_dataset_info() -> Dict[str, Any]:
    """获取数据集基本信息"""
    if not os.path.exists(dataset_dir):
        return {"error": "数据集目录不存在"}
    
    # 统计文件数量
    total_files = 0
    subsets_info = {}
    
    for root, dirs, files in os.walk(dataset_dir):
        npz_count = len([f for f in files if f.endswith('.npz')])
        if npz_count > 0:
            rel_path = os.path.relpath(root, dataset_dir)
            subsets_info[rel_path] = npz_count
            total_files += npz_count
    
    return {
        "root_dir": dataset_dir,
        "total_files": total_files,
        "subsets": subsets_info
    }

def print_dataset_overview():
    """打印数据集概览"""
    info = get_dataset_info()
    if "error" in info:
        print(f"{info['error']}")
        return
        
    print("=== 数据集概览 ===")
    print(f"根目录: {info['root_dir']}")
    print(f"总文件数: {info['total_files']} 个.npz文件")
    print("\n子目录分布:")
    for subset, count in list(info['subsets'].items())[:5]:  # 显示前5个
        print(f"  {subset}: {count} 个文件")
    if len(info['subsets']) > 5:
        print(f"  ... 还有 {len(info['subsets']) - 5} 个子目录")

# ==================== 核心读取功能 ====================
def read_npz_files(data_dir: str = dataset_dir, max_files: int = 5) -> List[Dict[str, Any]]:
    """
    读取指定目录下的.npz文件，返回包含数据的列表
    
    Args:
        data_dir (str): 数据集目录路径
        max_files (int): 最大读取文件数
        
    Returns:
        List[Dict[str, Any]]: 包含所有文件数据的列表
    """
    # 验证目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")
    
    # 查找所有.npz文件
    npz_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    
    print(f"找到 {len(npz_files)} 个.npz文件")
    
    # 读取文件数据（限制数量）
    actual_read = min(max_files, len(npz_files))
    print(f"正在读取前 {actual_read} 个文件...")
    
    data_list = []
    for i, file_path in enumerate(npz_files[:actual_read], 1):
        print(f"\n[{i}/{actual_read}] 读取文件: {os.path.basename(file_path)}")
        
        try:
            # 加载.npz文件
            data = np.load(file_path)
            
            # 转换为字典格式
            file_data = {}
            for key in data.files:
                file_data[key] = data[key]
                print(f"  键 '{key}': shape={data[key].shape}, dtype={data[key].dtype}")
            
            # 添加文件信息
            file_data['_metadata'] = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path)
            }
            
            data_list.append(file_data)
            
        except Exception as e:
            print(f"   读取失败: {str(e)}")
    
    return data_list

def print_data_summary(data_list: List[Dict[str, Any]]):
    """
    打印数据摘要信息
    
    Args:
        data_list (List[Dict[str, Any]]): 数据列表
    """
    print("\n" + "="*60)
    print(" 数据集摘要")
    print("="*60)
    print(f" 成功读取文件数: {len(data_list)}")
    
    if data_list:
        print(f"\n 第一个文件详细信息:")
        first_data = data_list[0]
        metadata = first_data['_metadata']
        print(f"文件名: {metadata['file_name']}")
        print(f"文件大小: {metadata['file_size']} bytes")
        print(f"完整路径: {metadata['file_path']}")
        
        print(f"\n 数据键值:")
        for key in first_data:
            if not key.startswith('_'):  # 跳过内部字段
                data = first_data[key]
                print(f"  {key}:")
                print(f"    类型: {type(data).__name__}")
                print(f"    形状: {data.shape}")
                print(f"    数据类型: {data.dtype}")
                # 显示部分数据
                flat_data = data.flat[:min(3, data.size)]
                print(f"    示例数据: {flat_data}")

# ==================== 高级功能 ====================
def read_specific_file(file_path: str, base_dir: str = dataset_dir) -> Dict[str, Any]:
    """
    读取指定的单个文件
    
    Args:
        file_path (str): 文件路径（相对或绝对）
        base_dir (str): 基础目录
        
    Returns:
        Dict[str, Any]: 文件数据
    """
    # 处理相对路径
    if not os.path.isabs(file_path):
        full_path = os.path.join(base_dir, file_path)
    else:
        full_path = file_path
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"文件不存在: {full_path}")
    
    print(f" 读取文件: {os.path.basename(full_path)}")
    
    try:
        data = np.load(full_path)
        result = {}
        for key in data.files:
            result[key] = data[key]
            print(f"  '{key}': shape={data[key].shape}")
        return result
    except Exception as e:
        print(f" 读取失败: {str(e)}")
        return {}

# ==================== 主程序入口 ====================
def main():
    """主程序 - 演示所有功能"""
    print("姿态数据集读取工具")
    print("="*50)
    
    # 1. 显示数据集概览
    print_dataset_overview()
    
    # 2. 读取样本数据
    print("\n" + "="*50)
    print("开始读取样本数据...")
    dataset_data = read_npz_files(max_files=3)
    
    # 3. 打印数据摘要
    if dataset_data:
        print_data_summary(dataset_data)
        
        # 4. 演示单文件读取
        print("\n" + "="*50)
        print("单文件读取演示:")
        sample_file = dataset_data[0]['_metadata']['file_path']
        relative_path = os.path.relpath(sample_file, dataset_dir)
        single_file_data = read_specific_file(relative_path)
        
    print(f"\n程序执行完成！")

# 便捷函数导出
__all__ = [
    'dataset_dir',
    'get_dataset_info', 
    'print_dataset_overview',
    'read_npz_files',
    'print_data_summary',
    'read_specific_file',
    'main'
]

if __name__ == "__main__":
    main()