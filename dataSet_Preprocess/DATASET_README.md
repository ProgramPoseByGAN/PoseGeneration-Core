# 数据集读取工具说明文档

## 概述
本项目提供了一套完整的AMASS数据集读取和处理工具，专门用于处理.pose数据集中的.npz文件。工具集成了配置管理、数据读取、统计分析等核心功能。

## 📁 当前文件结构

```
dataSet_Preprocess/
├── DATASET_README.md      # 本文档
├── dataset.py            # 核心数据处理工具 ⭐
└── dataset_config.py     # 配置管理工具 ⭐
```

## 🎯 核心功能

### 1. 全局变量配置
```python
# 所有工具共享的全局变量
dataset_dir = r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS"
```

### 2. 主要模块介绍

#### `dataset.py` - 核心数据处理工具 ⭐⭐⭐⭐⭐
**功能特点：**
- 集成配置管理、数据读取、摘要显示于一体的综合工具
- 支持批量文件扫描和限制读取数量
- 提供详细的数据结构信息和统计功能
- 包含完善的错误处理机制

**核心函数：**
- `get_dataset_info()` - 获取数据集基本信息统计
- `print_dataset_overview()` - 打印数据集概览
- `read_npz_files()` - 读取.npz文件（可限制数量）
- `print_data_summary()` - 打印数据摘要信息
- `read_specific_file()` - 读取指定单个文件

#### `dataset_config.py` - 配置管理工具 ⭐⭐⭐⭐
**功能特点：**
- 专业的数据集路径和结构管理
- 提供子目录映射和常用文件模式定义
- 支持数据集结构统计和样本文件获取

**核心函数：**
- `get_dataset_path()` - 获取指定子集路径
- `count_npz_files()` - 统计目录中.npz文件数量
- `print_dataset_structure()` - 打印完整的数据集结构
- `get_sample_files()` - 获取样本文件路径列表

## 🚀 使用示例

### 基础使用
```python
# 导入核心模块
from dataset import (
    read_npz_files, 
    print_data_summary, 
    print_dataset_overview
)

# 1. 查看数据集概览
print_dataset_overview()

# 2. 读取数据（默认读取前5个文件）
data_list = read_npz_files(max_files=10)

# 3. 打印数据摘要
print_data_summary(data_list)
```

### 高级使用
```python
# 使用配置管理工具
from dataset_config import (
    print_dataset_structure, 
    get_sample_files,
    get_dataset_path
)

# 1. 查看完整数据集结构
print_dataset_structure()

# 2. 获取样本文件列表
samples = get_sample_files(count=3)
for i, file_path in enumerate(samples, 1):
    print(f"{i}. {file_path}")

# 3. 获取特定子集路径
accad_path = get_dataset_path('ACCAD')
print(f"ACCAD数据集路径: {accad_path}")
```

### 单文件读取
```python
from dataset import read_specific_file

# 读取指定文件
file_data = read_specific_file("ACCAD/ACCAD/Female1General_c3d/A1 - Stand_poses.npz")
if file_data:
    for key, array in file_data.items():
        print(f"{key}: shape={array.shape}, dtype={array.dtype}")
```

## 📊 输出数据格式

所有工具都会输出标准化的数据结构：

```python
{
    'trans': numpy.ndarray,      # 位移数据 (frames, 3)
    'gender': numpy.ndarray,     # 性别信息 ()
    'mocap_framerate': numpy.ndarray,  # 帧率 ()
    'betas': numpy.ndarray,      # 身体形状参数 (16,)
    'dmpls': numpy.ndarray,      # 动态形状混合参数 (frames, 8)
    'poses': numpy.ndarray,      # 姿态参数 (frames, 156)
    '_metadata': {               # 文件元信息（仅dataset.py提供）
        'file_path': str,
        'file_name': str, 
        'file_size': int
    }
}
```

## ▶️ 运行方式

### 直接运行脚本
```bash
# 运行核心数据处理工具（推荐）
python dataset.py

# 运行配置管理工具
python dataset_config.py
```

### 作为模块导入
```python
# 在其他Python文件中使用
import sys
sys.path.append('D:/LAB/Pose/PoseGeneration-Core/dataSet_Preprocess')

from dataset import read_npz_files, print_data_summary
```

## 📈 数据集统计信息

根据实际扫描结果：
- **总文件数**：2798个.npz文件
- **主要子集分布**：
  - ACCAD: 252个文件
  - CMU: 2088个文件  
  - EKUT: 349个文件
  - HumanEva: 28个文件
  - SFU: 44个文件
  - TotalCapture: 37个文件
- **数据维度**：每帧包含156维姿态参数 + 3维位移 + 其他元数据

## ⚠️ 注意事项

1. **路径配置**：确保 `dataset_dir` 变量指向正确的数据集目录
2. **内存管理**：大批量读取时建议限制文件数量（使用 `max_files` 参数）
3. **首次运行**：可能需要较长时间扫描所有文件
4. **错误处理**：工具包含完善的异常处理，读取失败会显示具体错误信息
5. **依赖要求**：需要安装 numpy 库（`pip install numpy`）

## 🔧 性能优化建议

```python
# 1. 限制读取数量（推荐）
data = read_npz_files(max_files=10)  # 只读取前10个文件

# 2. 分批处理大数据集
for batch_start in range(0, total_files, batch_size):
    batch_files = all_files[batch_start:batch_start + batch_size]
    # 处理批次数据

# 3. 使用生成器模式（可自定义实现）
def read_files_generator(file_list, batch_size=5):
    for i in range(0, len(file_list), batch_size):
        yield read_npz_files_from_list(file_list[i:i+batch_size])
```

## 📝 开发建议

1. **功能扩展**：可在 `dataset.py` 中添加新的数据处理功能
2. **配置定制**：在 `dataset_config.py` 中调整子目录映射关系
3. **日志记录**：建议添加日志模块用于生产环境
4. **单元测试**：为关键函数编写测试用例确保稳定性

## 🆘 常见问题

**Q: 找不到数据集文件怎么办？**
A: 检查 `dataset_dir` 路径是否正确，确认数据集文件确实存在于指定目录

**Q: 内存不足如何处理？**
A: 使用 `max_files` 参数限制读取数量，或分批处理数据

**Q: 如何读取特定类型的文件？**
A: 可以结合 `dataset_config.py` 中的文件模式定义筛选特定文件