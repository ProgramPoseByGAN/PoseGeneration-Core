# AMASS数据预处理与映射工具使用说明

## 概述

本工具实现了从AMASS数据集(.npz文件)到项目22关节标准的完整转换流程，严格按照《骨骼约束规范》和《映射规范》开发。

## 功能特性

### 核心功能模块

1. **数据解析与校验**
   - 自动加载AMASS .npz文件
   - 验证必要字段(poses, trans)
   - 提取身体姿态数据(前72维)

2. **核心骨骼映射 (24→22关节)**
   - 严格按照映射规范执行关节对应
   - 支持一对一映射、融合映射、新建关节
   - 使用球面线性插值(Slerp)处理脊柱融合

3. **新建关节数据处理**
   - 自动生成LeftToes/RightToes关节数据
   - 基于脚部旋转智能生成脚趾朝向

4. **BVH文件生成**
   - 生成标准BVH格式文件
   - 严格遵循Unity Humanoid骨骼命名
   - 使用ZXY欧拉角顺序
   - 包含完整的骨架层级和初始偏移

5. **训练数据预处理**
   - 转换为局部旋转表示
   - 按关节进行数据标准化
   - 生成包含归一化参数的训练数据

## 文件结构

```
dataSet_Preprocess/
├── pose_mapper.py          # 核心映射处理器
├── process_examples.py     # 使用示例和批量处理
├── dataset.py             # 数据集读取工具
├── amass_processor.py     # AMASS数据处理基础类
├── dataset_config.py      # 数据集配置
└── POSE_MAPPER_USAGE.md   # 本使用说明
```

## 安装依赖

```bash
pip install numpy scipy bvhio
```

## 快速开始

### 1. 单文件处理演示

```python
from pose_mapper import PoseMapper
import os

# 创建映射器
mapper = PoseMapper()

# 设置文件路径
input_file = r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS\CMU\01\01_01_stageii.npz"
output_dir = r"D:\LAB\Pose\PoseGeneration-Core\output"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 执行处理
result = mapper.process_file(
    input_file=input_file,
    output_bvh=os.path.join(output_dir, "output.bvh"),
    output_training=os.path.join(output_dir, "training_data.npz")
)

print(f"处理完成: {result['frames_processed']} 帧")
```

### 2. 批量处理

```bash
# 使用命令行工具
python process_examples.py --mode batch --max-files 100

# 或在Python中调用
from process_examples import batch_process

batch_process(
    dataset_root=r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS",
    output_dir=r"D:\LAB\Pose\PoseGeneration-Core\processed_data",
    max_files=100,
    skip_existing=True
)
```

### 3. 验证输出文件

```bash
python process_examples.py --mode validate --output-dir ./processed_data
```

## 详细API说明

### PoseMapper类

#### 初始化
```python
mapper = PoseMapper(dataset_root="path/to/amass/dataset")
```

#### 核心方法

**process_file()** - 完整处理流程
```python
result = mapper.process_file(
    input_file="input.npz",           # 输入AMASS文件
    output_bvh="output.bvh",          # 输出BVH文件（可选）
    output_training="training.npz"    # 输出训练数据（可选）
)
```

返回结果：
```python
{
    'input_file': str,        # 输入文件路径
    'frames_processed': int,  # 处理的帧数
    'output_bvh': str,        # BVH输出路径
    'output_training': str,   # 训练数据输出路径
    'target_shape': tuple,    # 目标数据形状
    'success': bool          # 是否成功
}
```

#### 单独功能方法

**load_amass_file()** - 加载AMASS文件
```python
data = mapper.load_amass_file("file.npz")
# 返回: {'poses': array, 'trans': array, 'betas': array, ...}
```

**map_joints()** - 执行关节映射
```python
target_rotations = mapper.map_joints(smpl_rotations)
# 输入: (frames, 24, 3, 3) SMPL旋转矩阵
# 输出: (frames, 22, 3, 3) 目标旋转矩阵
```

## 映射规范详解

### 关节对应关系

| 目标关节 | 源关节 | 映射类型 | 说明 |
|---------|--------|---------|------|
| Hips | pelvis | 直接映射 | 根节点，包含位移数据 |
| LeftUpperLeg | left_hip | 直接映射 | 左大腿 |
| RightUpperLeg | right_hip | 直接映射 | 右大腿 |
| Spine | spine1 | 直接映射 | 腰椎下部 |
| LeftLowerLeg | left_knee | 直接映射 | 左小腿 |
| RightLowerLeg | right_knee | 直接映射 | 右小腿 |
| Spine1 | spine2 | 直接映射 | 胸椎下部 |
| LeftFoot | left_ankle | 直接映射 | 左脚踝 |
| RightFoot | right_ankle | 直接映射 | 右脚踝 |
| Spine2 | spine3+neck | 融合映射 | 球面插值α=0.5 |
| LeftShoulder | left_collar | 直接映射 | 左锁骨 |
| RightShoulder | right_collar | 直接映射 | 右锁骨 |
| Neck | neck | 直接映射 | 颈部 |
| Head | head | 直接映射 | 头部 |
| LeftUpperArm | left_shoulder | 直接映射 | 左上臂 |
| RightUpperArm | right_shoulder | 直接映射 | 右上臂 |
| LeftLowerArm | left_elbow | 直接映射 | 左下臂 |
| RightLowerArm | right_elbow | 直接映射 | 右下臂 |
| LeftHand | left_wrist | 直接映射 | 左手腕（末端）|
| RightHand | right_wrist | 直接映射 | 右手腕（末端）|
| LeftToes | left_foot | 新建关节 | 基于脚部旋转生成 |
| RightToes | right_foot | 新建关节 | 基于脚部旋转生成 |

### 特殊处理说明

1. **Spine2融合映射**：
   - 使用球面线性插值(Slerp)融合spine3和neck的旋转
   - 权重α=0.5，使胸颈连接更加自然平滑

2. **Toes关节生成**：
   - 基于对应脚部关节的旋转数据
   - 可扩展为更复杂的脚趾动力学模型

3. **末端关节处理**：
   - 舍弃AMASS中的left_hand(22)和right_hand(23)
   - 在wrist处终止，简化控制

## 输出文件格式

### BVH文件
- **格式**：标准BVH动画文件
- **骨骼**：22关节，严格遵循Unity Humanoid命名
- **旋转顺序**：ZXY欧拉角
- **单位**：角度（度）
- **坐标系**：右手坐标系（Y-up）

### 训练数据 (.npz)
```python
{
    'poses': array,              # 标准化姿态数据 (frames, 22, 3)
    'translations': array,       # 位移数据 (frames, 3)
    'normalization_mean': array, # 每关节均值 (22, 3)
    'normalization_std': array,  # 每关节标准差 (22, 3)
    'joint_names': list,         # 关节名称列表
    'source_file': str,          # 源文件名
    'frame_rate': float,         # 帧率
    'gender': str               # 性别信息
}
```

## 验证与测试

### 1. 编程验证
```python
# 检查关键关节旋转连续性
import numpy as np

data = np.load("training_data.npz")
poses = data['poses']

# 检查相邻帧差异
diffs = np.diff(poses, axis=0)
max_diff = np.max(np.abs(diffs))
print(f"最大帧间差异: {max_diff}")
```

### 2. 可视化验证
- **Blender导入**：检查动画流畅性，无骨骼扭曲
- **Unity测试**：创建Humanoid Avatar，验证动画驱动效果

### 3. 自动化测试
```bash
# 运行验证脚本
python process_examples.py --mode validate --output-dir ./test_output
```

## 常见问题与解决方案

### Q1: 处理速度慢怎么办？
**A**: 可以考虑：
- 减少处理的文件数量
- 使用更小的帧率采样
- 并行处理多个文件

### Q2: BVH文件在Unity中显示异常？
**A**: 检查：
- Unity导入设置：Forward Axis设为-Z，Up Axis设为Y
- 骨骼命名是否完全匹配Unity Humanoid标准
- 旋转顺序是否为ZXY

### Q3: 训练数据质量不佳？
**A**: 
- 检查源数据质量
- 调整标准化参数
- 增加数据过滤步骤

## 性能优化建议

1. **批处理优化**：
   ```python
   # 使用多进程处理
   from multiprocessing import Pool
   
   def process_wrapper(args):
       mapper, file_path, output_dir = args
       return process_single_file(mapper, file_path, output_dir)
   
   with Pool(processes=4) as pool:
       results = pool.map(process_wrapper, file_args)
   ```

2. **内存管理**：
   - 及时释放不需要的大数组
   - 使用生成器处理大数据集
   - 分批次处理大量文件

## 贡献与扩展

### 扩展功能建议

1. **支持更多数据格式**：
   - 添加对其他动作捕捉格式的支持
   - 实现数据格式间的相互转换

2. **高级映射策略**：
   - 实现更复杂的脚趾动力学
   - 添加手指关节支持
   - 支持面部表情映射

3. **质量控制**：
   - 添加生物力学合理性检查
   - 实现异常姿态检测
   - 提供数据清洗功能

### 代码结构改进

```
# 建议的模块划分
pose_mapper/
├── core/                 # 核心映射逻辑
│   ├── mapper.py        # 主映射器
│   ├── joint_mapping.py # 关节映射规则
│   └── validation.py    # 验证功能
├── io/                  # 输入输出处理
│   ├── bvh_writer.py   # BVH生成
│   ├── npz_handler.py  # NPZ处理
│   └── file_utils.py   # 文件操作
├── utils/               # 工具函数
│   ├── math_utils.py   # 数学计算
│   ├── log_utils.py    # 日志处理
│   └── config.py       # 配置管理
└── cli/                # 命令行接口
    └── main.py         # 主程序入口
```

## 版本历史

- **v1.0.0** (2024-02-10)：初始版本
  - 实现完整的24→22关节映射
  - 支持BVH文件生成
  - 提供训练数据预处理
  - 符合骨骼约束规范和映射规范

## 许可证

本工具仅供学术研究和非商业用途使用，遵循AMASS数据集的许可协议。