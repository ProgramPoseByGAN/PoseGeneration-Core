"""
AMASS数据预处理工具包
整合数据读取、姿态转换、骨骼映射、数据清洗的完整处理流程
该模块提供了一个完整的AMASS数据预处理解决方案，包含从原始数据读取到标准化输出的全流程处理功能
"""

# 核心处理模块导入
# 数据读取模块：负责AMASS数据集的文件扫描、加载和基本信息提取
from .data_reader import AMASSDataReader
# 姿态转换模块：负责AMASS姿态数据的各种表示形式转换
from .pose_converter import PoseConverter
# 骨骼映射模块：实现从AMASS SMPL 24关节到项目22关节标准的映射转换
from .skeleton_mapper import SkeletonMapper
# 数据清洗模块：根据骨骼约束规范和功能需求实现完整的数据质量检测与清洗功能
from .data_cleaner import DataCleaner, DataQualityReport
# 主流程控制模块：整合上述模块形成完整的处理流水线
from .preprocessing_pipeline import PreprocessingPipeline

# 版本信息定义
__version__ = "2.0.0"  # 当前工具包版本号
__author__ = "PoseGeneration Team"  # 开发团队信息

# 公开接口定义 - 控制模块对外暴露的类和函数
__all__ = [
    # 核心模块
    'AMASSDataReader',      # 数据读取器类
    'PoseConverter',        # 姿态转换器类
    'SkeletonMapper',       # 骨骼映射器类
    'DataCleaner',          # 数据清洗器类
    'DataQualityReport',    # 数据质量报告数据类
    'PreprocessingPipeline' # 预处理流水线类
]