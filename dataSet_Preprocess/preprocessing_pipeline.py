"""
数据预处理主流程控制模块
整合数据读取、姿态转换、骨骼映射、数据清洗的完整处理流程
该模块提供了一个统一的接口来协调各个子模块的工作，实现端到端的数据预处理解决方案

处理流程：
1. 数据读取：从AMASS数据集中加载原始姿态数据
2. 姿态转换：将原始轴角表示转换为旋转矩阵等中间表示
3. 骨骼映射：将SMPL 24关节体系映射到项目22关节标准
4. 数据清洗与异常处理：检测并处理数据质量问题
"""

# 系统和第三方库导入
import os                                      # 操作系统接口
import logging                                # 日志记录模块
from typing import Dict, List, Any, Optional, Tuple  # 类型提示支持
from pathlib import Path                      # 现代路径操作库
from datetime import datetime                 # 日期时间处理
import numpy as np                            # 数值计算核心库
import argparse                              # 命令行参数解析

# 导入各个功能子模块
from data_reader import AMASSDataReader      # 数据读取模块
from pose_converter import PoseConverter     # 姿态转换模块
from skeleton_mapper import SkeletonMapper   # 骨骼映射模块
from data_cleaner import DataCleaner, DataQualityReport  # 数据清洗模块

# 配置日志系统 - 设置统一的日志格式和级别
logging.basicConfig(
    level=logging.INFO,                                    # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s'    # 定义日志输出格式
)
logger = logging.getLogger(__name__)  # 创建模块专用日志记录器

class PreprocessingPipeline:
    """数据预处理流水线
    协调各个处理模块形成完整的端到端数据预处理流程
    提供统一的API接口支持单文件处理、批量处理和数据集级处理
    """
    
    def __init__(self, dataset_root: str = r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS"):
        """
        初始化预处理流水线实例
        创建并配置所有必要的处理子模块
        
        Args:
            dataset_root: AMASS数据集的根目录路径
        """
        # 存储数据集根目录路径
        self.dataset_root = Path(dataset_root)
        
        # 初始化各个处理子模块实例
        self.data_reader = AMASSDataReader(str(self.dataset_root))  # 数据读取器
        self.pose_converter = PoseConverter()                       # 姿态转换器
        self.skeleton_mapper = SkeletonMapper()                     # 骨骼映射器
        self.data_cleaner = DataCleaner()                           # 数据清洗器
        
        # 记录初始化完成信息
        logger.info("PreprocessingPipeline初始化完成")
        logger.info(f"数据集根目录: {self.dataset_root}")
    
    def process_single_file(self, 
                          input_file: str,
                          output_dir: str,
                          generate_bvh: bool = True,
                          generate_training_data: bool = True,
                          perform_cleaning: bool = True) -> Dict[str, Any]:
        """
        处理单个AMASS文件的完整预处理流程
        按照标准流程依次执行数据读取、清洗、转换、映射等步骤
        
        Args:
            input_file: 待处理的AMASS .npz文件路径
            output_dir: 处理结果的输出目录
            generate_bvh: 是否生成BVH动画文件（用于可视化）
            generate_training_data: 是否生成标准化的训练数据
            perform_cleaning: 是否执行数据质量检查和清洗
            
        Returns:
            包含处理结果和状态信息的字典
        """
        logger.info("="*80)
        logger.info(f"开始处理文件: {input_file}")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        try:
            # 确保输出目录存在
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 1. 数据读取
            logger.info("步骤1: 数据读取")
            file_data = self.data_reader.load_amass_file(input_file)
            
            # 2. 数据清洗（可选）
            quality_report = None
            if perform_cleaning:
                logger.info("步骤2: 数据清洗与质量检测")
                quality_report = self.data_cleaner.clean_single_file(file_data, str(output_path))
                
                # 根据质量评分决定是否继续处理
                if quality_report.quality_score < 30:
                    logger.warning(f"数据质量评分过低 ({quality_report.quality_score})，跳过后续处理")
                    return {
                        'success': False,
                        'reason': 'low_quality',
                        'quality_report': quality_report.to_dict() if quality_report else None
                    }
            
            # 3. 姿态转换
            logger.info("步骤3: 姿态转换")
            body_poses = self.pose_converter.extract_body_poses(file_data['poses'])
            smpl_rotations = self.pose_converter.poses_to_rotation_matrices(body_poses)
            
            # 4. 骨骼映射
            logger.info("步骤4: 骨骼映射")
            
            # 生成输出文件路径
            input_path = Path(input_file)
            base_name = input_path.stem
            bvh_output = str(output_path / f"{base_name}.bvh") if generate_bvh else None
            training_output = str(output_path / f"{base_name}_training.npz") if generate_training_data else None
            
            # 执行骨骼映射
            mapping_result = self.skeleton_mapper.process_skeleton_mapping(
                smpl_rotations=smpl_rotations,
                root_translations=file_data['trans'],
                output_bvh=bvh_output,
                output_training=training_output,
                frame_rate=file_data.get('mocap_framerate', 120.0),
                metadata={
                    'source_file': input_file,
                    'quality_score': quality_report.quality_score if quality_report else None,
                    'processing_time': str(datetime.now() - start_time)
                }
            )
            
            # 5. 返回处理结果
            processing_time = str(datetime.now() - start_time)
            
            result = {
                'success': True,
                'input_file': input_file,
                'processing_time': processing_time,
                'frames_processed': body_poses.shape[0],
                'quality_report': quality_report.to_dict() if quality_report else None,
                'mapping_result': mapping_result
            }
            
            logger.info("="*80)
            logger.info("文件处理完成!")
            logger.info(f"处理时间: {processing_time}")
            logger.info(f"处理帧数: {result['frames_processed']}")
            if quality_report:
                logger.info(f"数据质量评分: {quality_report.quality_score:.1f}/100")
            logger.info("="*80)
            
            return result
            
        except Exception as e:
            logger.error(f"处理文件失败: {input_file}")
            logger.error(f"错误详情: {str(e)}")
            
            return {
                'success': False,
                'input_file': input_file,
                'error': str(e),
                'processing_time': str(datetime.now() - start_time)
            }
    
    def process_batch(self, 
                     input_files: List[str],
                     output_dir: str,
                     max_workers: int = 1,
                     **kwargs) -> List[Dict[str, Any]]:
        """
        批量处理多个文件
        
        Args:
            input_files: 输入文件路径列表
            output_dir: 输出目录
            max_workers: 最大并发工作数
            **kwargs: 传递给process_single_file的参数
            
        Returns:
            处理结果列表
        """
        logger.info(f"开始批量处理 {len(input_files)} 个文件")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"并发数: {max_workers}")
        
        results = []
        
        # 串行处理（后续可扩展为并行处理）
        for i, file_path in enumerate(input_files, 1):
            logger.info(f"[{i}/{len(input_files)}] 处理文件: {Path(file_path).name}")
            
            result = self.process_single_file(
                input_file=file_path,
                output_dir=str(Path(output_dir) / f"processed_{i:04d}"),
                **kwargs
            )
            
            results.append(result)
            
            # 记录处理统计
            if result['success']:
                logger.info(f"✓ 处理成功")
            else:
                logger.error(f"✗ 处理失败: {result.get('error', 'Unknown error')}")
        
        # 生成批处理统计报告
        success_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - success_count
        
        logger.info("="*80)
        logger.info("批量处理完成!")
        logger.info(f"成功: {success_count}")
        logger.info(f"失败: {failed_count}")
        logger.info(f"成功率: {success_count/len(results)*100:.1f}%")
        logger.info("="*80)
        
        return results
    
    def process_dataset(self, 
                       output_dir: str,
                       max_files: int = 100,
                       file_pattern: str = "*.npz",
                       **kwargs) -> List[Dict[str, Any]]:
        """
        处理整个数据集
        
        Args:
            output_dir: 输出目录
            max_files: 最大处理文件数
            file_pattern: 文件匹配模式
            **kwargs: 传递给process_single_file的参数
            
        Returns:
            处理结果列表
        """
        logger.info("开始处理整个数据集...")
        
        # 扫描数据集文件
        file_infos = self.data_reader.scan_dataset(
            max_files=max_files, 
            file_pattern=file_pattern
        )
        
        if not file_infos:
            logger.error("未找到任何文件")
            return []
        
        # 提取文件路径
        file_paths = [info['path'] for info in file_infos]
        
        # 执行批量处理
        results = self.process_batch(
            input_files=file_paths,
            output_dir=output_dir,
            **kwargs
        )
        
        return results
    
    def get_dataset_overview(self) -> Dict[str, Any]:
        """
        获取数据集概览信息
        
        Returns:
            数据集统计信息
        """
        logger.info("获取数据集概览...")
        
        # 获取数据集统计
        stats = self.data_reader.get_dataset_statistics()
        
        # 获取样本文件用于快速测试
        sample_files = self.data_reader.get_sample_files(count=5)
        
        overview = {
            'dataset_stats': stats,
            'sample_files': sample_files,
            'module_versions': {
                'data_reader': '1.0',
                'pose_converter': '1.0', 
                'skeleton_mapper': '1.0',
                'data_cleaner': '1.0'
            }
        }
        
        logger.info("数据集概览:")
        logger.info(f"  总文件数: {stats.get('total_files', 0)}")
        logger.info(f"  总大小: {stats.get('total_size_mb', 0)} MB")
        logger.info(f"  样本文件数: {len(sample_files)}")
        
        return overview

def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description="AMASS数据预处理流水线")
    
    # 基本参数
    parser.add_argument("--mode", choices=["single", "batch", "dataset", "overview"], 
                       default="dataset", help="处理模式")
    parser.add_argument("--dataset-root", default=r"D:\LAB\Pose\PoseGeneration-Core\dataset\AMASS",
                       help="数据集根目录")
    parser.add_argument("--output-dir", default="./preprocessing_output",
                       help="输出目录")
    
    # 文件参数
    parser.add_argument("--input-file", help="单文件处理时的输入文件路径")
    parser.add_argument("--max-files", type=int, default=10,
                       help="最大处理文件数")
    
    # 处理选项
    parser.add_argument("--no-bvh", action="store_true", 
                       help="不生成BVH文件")
    parser.add_argument("--no-training-data", action="store_true",
                       help="不生成训练数据")
    parser.add_argument("--no-cleaning", action="store_true",
                       help="跳过数据清洗步骤")
    
    args = parser.parse_args()
    
    # 创建处理流水线
    pipeline = PreprocessingPipeline(dataset_root=args.dataset_root)
    
    if args.mode == "overview":
        # 显示数据集概览
        overview = pipeline.get_dataset_overview()
        print("\n=== 数据集概览 ===")
        stats = overview['dataset_stats']
        if 'error' not in stats:
            print(f"总文件数: {stats['total_files']}")
            print(f"总大小: {stats['total_size_mb']} MB")
            print(f"子集数量: {len(stats['subsets'])}")
            print("\n样本文件:")
            for i, sample in enumerate(overview['sample_files'], 1):
                print(f"  {i}. {sample}")
    
    elif args.mode == "single":
        # 单文件处理
        if not args.input_file:
            print("错误: 单文件处理模式需要指定 --input-file 参数")
            return
            
        result = pipeline.process_single_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            generate_bvh=not args.no_bvh,
            generate_training_data=not args.no_training_data,
            perform_cleaning=not args.no_cleaning
        )
        
        print("\n=== 处理结果 ===")
        if result['success']:
            print("✓ 处理成功!")
            print(f"处理时间: {result['processing_time']}")
            print(f"帧数: {result['frames_processed']}")
            if result['quality_report']:
                print(f"质量评分: {result['quality_report']['quality_score']:.1f}/100")
        else:
            print("✗ 处理失败!")
            print(f"错误: {result['error']}")
    
    elif args.mode == "batch":
        # 批量处理示例文件
        overview = pipeline.get_dataset_overview()
        sample_files = overview['sample_files'][:args.max_files]
        
        if not sample_files:
            print("错误: 未找到可处理的文件")
            return
            
        results = pipeline.process_batch(
            input_files=sample_files,
            output_dir=args.output_dir,
            generate_bvh=not args.no_bvh,
            generate_training_data=not args.no_training_data,
            perform_cleaning=not args.no_cleaning
        )
        
        # 显示统计结果
        success_count = sum(1 for r in results if r['success'])
        print(f"\n=== 批量处理结果 ===")
        print(f"总文件数: {len(results)}")
        print(f"成功处理: {success_count}")
        print(f"处理失败: {len(results) - success_count}")
        print(f"成功率: {success_count/len(results)*100:.1f}%")
    
    elif args.mode == "dataset":
        # 处理整个数据集
        results = pipeline.process_dataset(
            output_dir=args.output_dir,
            max_files=args.max_files,
            generate_bvh=not args.no_bvh,
            generate_training_data=not args.no_training_data,
            perform_cleaning=not args.no_cleaning
        )
        
        # 显示统计结果
        success_count = sum(1 for r in results if r['success'])
        print(f"\n=== 数据集处理结果 ===")
        print(f"总文件数: {len(results)}")
        print(f"成功处理: {success_count}")
        print(f"处理失败: {len(results) - success_count}")
        print(f"成功率: {success_count/len(results)*100:.1f}%")

if __name__ == "__main__":
    main()