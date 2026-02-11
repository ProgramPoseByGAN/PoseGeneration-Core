# AMASS数据预处理模块文档

## 概述

本项目提供了一个完整的AMASS数据预处理流水线，实现了从原始.npz文件到可用于深度学习训练的标准化数据的完整转换流程。系统采用模块化设计，包含四个核心处理阶段：

1. **数据读取** - 扫描和加载AMASS数据文件
2. **姿态转换** - 姿态表示形式转换（轴角↔旋转矩阵↔欧拉角）
3. **骨骼映射** - SMPL 24关节到项目22关节标准的映射转换
4. **数据清洗** - 数据质量检测与异常处理

## 模块架构

```
dataSet_Preprocess/
├── data_reader.py              # 数据读取模块
├── pose_converter.py           # 姿态转换模块
├── skeleton_mapper.py          # 骨骼映射模块
├── data_cleaner.py             # 数据清洗模块
├── preprocessing_pipeline.py   # 主流程控制模块
├── __init__.py                 # 包初始化文件
└── DATA_PREPROCESSING_DOCUMENTATION.md  # 本文档
```

## 核心模块详细介绍

### 1. 数据读取模块 (data_reader.py)

**功能职责**：
- 递归扫描AMASS数据集目录
- 加载.npz文件并验证完整性
- 提取数据集统计信息
- 文件完整性验证

**主要类和方法**：

```python
class AMASSDataReader:
    def __init__(self, dataset_root: str)
    def scan_dataset(self, max_files: Optional[int], file_pattern: str) -> List[Dict]
    def load_amass_file(self, file_path: str) -> Dict[str, Any]
    def get_dataset_statistics(self) -> Dict[str, Any]
    def get_sample_files(self, count: int) -> List[str]
    def validate_file_integrity(self, file_path: str) -> Dict[str, Any]
```

**使用示例**：
```python
from data_reader import AMASSDataReader

reader = AMASSDataReader("/path/to/amass/dataset")

# 扫描数据集
files = reader.scan_dataset(max_files=100)

# 加载单个文件
data = reader.load_amass_file("CMU/01/01_01_poses.npz")

# 获取统计信息
stats = reader.get_dataset_statistics()
```

### 2. 姿态转换模块 (pose_converter.py)

**功能职责**：
- 轴角、旋转矩阵、欧拉角之间的相互转换
- 姿态数据提取和重塑
- 旋转表示标准化
- 姿态数据统计分析

**主要类和方法**：

```python
class PoseConverter:
    def __init__(self)
    def extract_body_poses(self, poses_data: np.ndarray) -> np.ndarray
    def axis_angle_to_rotation_matrix(self, axis_angle: np.ndarray) -> np.ndarray
    def poses_to_rotation_matrices(self, body_poses: np.ndarray) -> np.ndarray
    def poses_to_euler_angles(self, body_poses: np.ndarray, order: str) -> np.ndarray
    def normalize_rotations(self, rot_mats: np.ndarray) -> np.ndarray
    def compute_angular_velocity(self, poses: np.ndarray, framerate: float) -> np.ndarray
    def analyze_pose_statistics(self, poses_data: np.ndarray) -> Dict[str, Any]
```

**使用示例**：
```python
from pose_converter import PoseConverter
import numpy as np

converter = PoseConverter()

# 提取身体姿态（前72维）
body_poses = converter.extract_body_poses(poses_data)  # (frames, 72)

# 转换为旋转矩阵
rot_matrices = converter.poses_to_rotation_matrices(body_poses)  # (frames, 24, 3, 3)

# 转换为欧拉角
euler_angles = converter.poses_to_euler_angles(body_poses, order='zxy')  # (frames, 24, 3)
```

### 3. 骨骼映射模块 (skeleton_mapper.py)

**功能职责**：
- SMPL 24关节到项目22关节的标准映射
- 融合映射（如脊柱关节融合）
- 新建关节生成（如脚趾关节）
- BVH文件生成
- 训练数据标准化

**映射关系**：

| 目标关节 | 源关节 | 映射类型 |
|---------|--------|---------|
| Hips | pelvis | 直接映射 |
| LeftUpperLeg | left_hip | 直接映射 |
| RightUpperLeg | right_hip | 直接映射 |
| Spine | spine1 | 直接映射 |
| LeftLowerLeg | left_knee | 直接映射 |
| RightLowerLeg | right_knee | 直接映射 |
| Spine1 | spine2 | 直接映射 |
| LeftFoot | left_ankle | 直接映射 |
| RightFoot | right_ankle | 直接映射 |
| Spine2 | spine3+neck | 融合映射 |
| LeftShoulder | left_collar | 直接映射 |
| RightShoulder | right_collar | 直接映射 |
| Neck | neck | 直接映射 |
| Head | head | 直接映射 |
| LeftUpperArm | left_shoulder | 直接映射 |
| RightUpperArm | right_shoulder | 直接映射 |
| LeftLowerArm | left_elbow | 直接映射 |
| RightLowerArm | right_elbow | 直接映射 |
| LeftHand | left_wrist | 直接映射 |
| RightHand | right_wrist | 直接映射 |
| LeftToes | left_foot | 新建关节 |
| RightToes | right_foot | 新建关节 |

**主要类和方法**：

```python
class SkeletonMapper:
    def __init__(self)
    def map_joints(self, smpl_rotations: np.ndarray) -> np.ndarray
    def compute_local_rotations(self, global_rotations: np.ndarray) -> np.ndarray
    def normalize_training_data(self, local_rotations: np.ndarray) -> Tuple[np.ndarray, Dict]
    def save_bvh_file(self, euler_angles: np.ndarray, root_translations: np.ndarray, output_path: str)
    def save_training_data(self, normalized_data: np.ndarray, normalization_params: Dict, trans_data: np.ndarray, output_path: str)
    def process_skeleton_mapping(self, smpl_rotations: np.ndarray, root_translations: np.ndarray, ...) -> Dict[str, Any]
```

**使用示例**：
```python
from skeleton_mapper import SkeletonMapper

mapper = SkeletonMapper()

# 执行完整骨骼映射流程
result = mapper.process_skeleton_mapping(
    smpl_rotations=smpl_rotations,      # (frames, 24, 3, 3)
    root_translations=root_translations, # (frames, 3)
    output_bvh="./output/result.bvh",
    output_training="./output/training_data.npz"
)
```

### 4. 数据清洗模块 (data_cleaner.py)

**功能职责**：
- 数据有效性检查（缺失值、静态数据检测）
- 生物力学合理性检查（关节角极限、肢体扭曲检测）
- 运动学质量检查（过度抖动、足部滑步检测）
- 质量评分与报告生成

**质量检测维度**：

1. **数据有效性检查**
   - NaN/Inf值检测
   - 静态数据检测（根节点位移方差、旋转变化）
   - 数据完整性验证

2. **生物力学合理性检查**
   - 膝关节屈伸范围检查 [0°, 150°]
   - 肘关节屈伸范围检查 [0°, 150°]
   - 脊柱旋转安全阈值检查 45°

3. **运动学质量检查**
   - 角速度限制检查（720°/秒）
   - 足部滑步检测
   - 运动连续性检查

**主要类和方法**：

```python
class DataCleaner:
    def __init__(self, thresholds: Optional[Dict] = None)
    def check_data_validity(self, poses: np.ndarray, trans: np.ndarray) -> Dict[str, Any]
    def check_biomechanical_reasonableness(self, poses: np.ndarray) -> Dict[str, Any]
    def check_motion_quality(self, poses: np.ndarray, trans: np.ndarray, framerate: float) -> Dict[str, Any]
    def generate_quality_score(self, validity_report: Dict, biomechanical_report: Dict, motion_report: Dict) -> float
    def clean_single_file(self, file_data: Dict[str, Any], output_dir: Optional[str]) -> DataQualityReport

@dataclass
class DataQualityReport:
    file_path: str
    file_name: str
    total_frames: int
    quality_score: float
    recommendation: str
    # ... 其他字段
```

**使用示例**：
```python
from data_cleaner import DataCleaner

cleaner = DataCleaner()

# 清洗单个文件
report = cleaner.clean_single_file(file_data, "./quality_reports")

print(f"质量评分: {report.quality_score}/100")
print(f"处理建议: {report.recommendation}")
```

### 5. 主流程控制模块 (preprocessing_pipeline.py)

**功能职责**：
- 整合所有处理步骤的完整流水线
- 支持单文件、批量、数据集级别的处理
- 处理结果统计和报告生成
- 命令行接口支持

**主要类和方法**：

```python
class PreprocessingPipeline:
    def __init__(self, dataset_root: str)
    def process_single_file(self, input_file: str, output_dir: str, ...) -> Dict[str, Any]
    def process_batch(self, input_files: List[str], output_dir: str, ...) -> List[Dict[str, Any]]
    def process_dataset(self, output_dir: str, max_files: int, ...) -> List[Dict[str, Any]]
    def get_dataset_overview(self) -> Dict[str, Any]
```

**使用示例**：
```python
from preprocessing_pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline("/path/to/amass/dataset")

# 单文件处理
result = pipeline.process_single_file(
    input_file="CMU/01/01_01_poses.npz",
    output_dir="./processed_output"
)

# 批量处理
results = pipeline.process_batch(
    input_files=["file1.npz", "file2.npz"],
    output_dir="./batch_output"
)

# 数据集处理
results = pipeline.process_dataset(
    output_dir="./dataset_output",
    max_files=100
)
```

## 安装依赖

```bash
pip install numpy scipy bvhio
```

## 命令行使用

```bash
# 查看数据集概览
python preprocessing_pipeline.py --mode overview

# 处理单个文件
python preprocessing_pipeline.py --mode single --input-file CMU/01/01_01_poses.npz --output-dir ./output

# 批量处理样本文件
python preprocessing_pipeline.py --mode batch --max-files 10 --output-dir ./batch_output

# 处理整个数据集（前100个文件）
python preprocessing_pipeline.py --mode dataset --max-files 100 --output-dir ./dataset_output

# 不执行数据清洗
python preprocessing_pipeline.py --mode single --input-file file.npz --no-cleaning
```

## API使用示例

### 完整处理流程

```python
from preprocessing_pipeline import PreprocessingPipeline

# 创建处理流水线
pipeline = PreprocessingPipeline(dataset_root="/path/to/amass")

# 处理单个文件
result = pipeline.process_single_file(
    input_file="CMU/01/01_01_poses.npz",
    output_dir="./output",
    generate_bvh=True,           # 生成BVH文件
    generate_training_data=True, # 生成训练数据
    perform_cleaning=True        # 执行数据清洗
)

if result['success']:
    print(f"处理成功! 帧数: {result['frames_processed']}")
    print(f"质量评分: {result['quality_report']['quality_score']}/100")
else:
    print(f"处理失败: {result['error']}")
```

### 模块化使用

```python
# 分步处理示例
from data_reader import AMASSDataReader
from pose_converter import PoseConverter
from skeleton_mapper import SkeletonMapper
from data_cleaner import DataCleaner

# 1. 数据读取
reader = AMASSDataReader()
file_data = reader.load_amass_file("input.npz")

# 2. 数据清洗
cleaner = DataCleaner()
quality_report = cleaner.clean_single_file(file_data)

# 3. 姿态转换
converter = PoseConverter()
body_poses = converter.extract_body_poses(file_data['poses'])
rot_matrices = converter.poses_to_rotation_matrices(body_poses)

# 4. 骨骼映射
mapper = SkeletonMapper()
mapping_result = mapper.process_skeleton_mapping(
    smpl_rotations=rot_matrices,
    root_translations=file_data['trans'],
    output_bvh="./output/result.bvh",
    output_training="./output/training.npz"
)
```

## 输出文件格式

### BVH文件
- 标准BVH动画文件格式
- 22关节完整骨架结构
- ZXY欧拉角旋转顺序
- 包含根节点位移信息

### 训练数据 (.npz)
```python
{
    'poses': 标准化姿态数据 (frames, 22, 3),           # 局部旋转的轴角表示
    'translations': 位移数据 (frames, 3),               # 根节点位移
    'normalization_mean': 每关节均值 (22, 3),          # 标准化参数
    'normalization_std': 每关节标准差 (22, 3),         # 标准化参数
    'joint_names': 关节名称列表,                        # 22个关节名称
    'source_file': 源文件路径,                          # 元数据
    'quality_score': 数据质量评分,                      # 质量信息
    'processing_time': 处理耗时                        # 性能信息
}
```

### 质量报告 (JSON)
```json
{
    "file_path": "文件路径",
    "file_name": "文件名",
    "total_frames": 1000,
    "quality_score": 85.5,
    "recommendation": "数据质量良好，建议轻微后处理",
    "validity_issues": {...},
    "biomechanical_violations": {...},
    "motion_quality_issues": {...}
}
```

## 质量评分标准

- **90-100分**：数据质量优秀，可直接用于训练
- **70-89分**：数据质量良好，建议轻微后处理
- **50-69分**：数据质量一般，需要进行数据清洗和修复
- **30-49分**：数据质量较差，建议丢弃或大幅修正
- **0-29分**：数据质量很差，强烈建议丢弃

## 性能优化建议

1. **批量处理**：使用`process_batch`或`process_dataset`方法处理多个文件
2. **并行处理**：可扩展为多进程并行处理（当前为串行）
3. **内存管理**：及时释放大数组，避免内存溢出
4. **文件缓存**：对于重复处理的文件可实现缓存机制

## 常见问题解答

**Q: 处理速度慢怎么办？**
A: 可以减少处理文件数量，或者调整数据清洗的严格程度。

**Q: BVH文件在Unity中显示异常？**
A: 检查Unity导入设置：Forward Axis设为-Z，Up Axis设为Y。

**Q: 如何自定义质量检测阈值？**
A: 在创建DataCleaner时传入自定义阈值字典。

**Q: 支持其他数据格式吗？**
A: 当前专门针对AMASS .npz格式，可扩展支持其他格式。

## 版本历史

- **v2.0.0** (2024-02-11)：重大重构版本
  - 采用模块化架构设计
  - 完整的数据清洗功能
  - 标准化的处理流程
  - 完善的文档和示例

- **v1.0.0** (2024-02-10)：初始版本
  - 基础的映射转换功能
  - 简单的数据处理能力

## 许可证

本工具仅供学术研究和非商业用途使用，遵循AMASS数据集的许可协议。