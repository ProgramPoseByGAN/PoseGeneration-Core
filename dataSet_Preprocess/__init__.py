"""
AMASS数据预处理工具包
整合数据读取、姿态转换、骨骼映射、数据清洗的完整处理流程
"""

# 核心处理模块
from .data_reader import AMASSDataReader
from .pose_converter import PoseConverter
from .skeleton_mapper import SkeletonMapper
from .data_cleaner import DataCleaner, DataQualityReport
from .preprocessing_pipeline import PreprocessingPipeline

__version__ = "2.0.0"
__author__ = "PoseGeneration Team"

__all__ = [
    # 核心模块
    'AMASSDataReader',
    'PoseConverter',
    'SkeletonMapper',
    'DataCleaner',
    'DataQualityReport',
    'PreprocessingPipeline'
]