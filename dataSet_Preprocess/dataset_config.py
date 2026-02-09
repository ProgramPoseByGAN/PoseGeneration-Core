# 数据集配置文件

import os

# 全局变量：数据集根目录
DATASET_DIR = r"/PoseGeneration-Core/dataset/AMASS"

# 数据集子目录映射
DATASET_SUBSETS = {
    'ACCAD': 'ACCAD', 
    'CMU': 'CMU',
    'EKUT': 'EKUT',
    'HumanEva': 'HumanEva',
    'SFU': 'SFU',
    'TotalCapture': 'TotalCapture'
}

# 常用文件模式
COMMON_FILE_PATTERNS = {
    'standing': '*Stand*',
    'walking': '*Walk*',
    'running': '*Run*',
    'gestures': '*Gesture*'
}

def get_dataset_path(subset: str = '') -> str:
    """
    获取数据集路径
    
    Args:
        subset (str): 数据集子集名称
        
    Returns:
        str: 完整路径
    """
    if subset and subset in DATASET_SUBSETS:
        return os.path.join(DATASET_DIR, DATASET_SUBSETS[subset])
    return DATASET_DIR

def count_npz_files(directory: str) -> int:
    """递归计算目录中.npz文件的数量"""
    count = 0
    try:
        for root, dirs, files in os.walk(directory):
            count += len([f for f in files if f.endswith('.npz')])
    except:
        pass
    return count

def print_dataset_structure():
    """打印数据集结构信息"""
    print("=== 数据集结构信息 ===")
    print(f"根目录: {DATASET_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        print("错误: 数据集根目录不存在!")
        return
    
    # 统计总文件数
    total_files = count_npz_files(DATASET_DIR)
    print(f"\n总文件数: {total_files} 个.npz文件")
    
    print("\n子目录结构:")
    for name in DATASET_SUBSETS.keys():
        subset_path = os.path.join(DATASET_DIR, name, name)  # 实际路径结构是 dataset/AMASS/ACCAD/ACCAD/
        if os.path.exists(subset_path):
            file_count = count_npz_files(subset_path)
            print(f"  {name}: {subset_path}")
            print(f"    文件数: {file_count}")
            
            # 显示该目录下的子目录
            try:
                subdirs = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
                if subdirs:
                    print(f"    子目录: {', '.join(subdirs[:3])}" + ("..." if len(subdirs) > 3 else ""))
            except:
                pass
        else:
            print(f"  {name}: 目录不存在")

def get_sample_files(count: int = 5) -> list:
    """
    获取样本文件路径
    
    Args:
        count (int): 样本数量
        
    Returns:
        list: 文件路径列表
    """
    sample_files = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith('.npz'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, DATASET_DIR)
                sample_files.append(rel_path)
                if len(sample_files) >= count:
                    return sample_files
    return sample_files

if __name__ == "__main__":
    print_dataset_structure()
    
    print("\n=== 样本文件 ===")
    samples = get_sample_files(5)
    for i, file_path in enumerate(samples, 1):
        print(f"{i}. {file_path}")