"""
数据清洗与异常处理模块
根据《骨骼约束规范》和功能需求实现完整的数据质量检测与清洗功能

功能维度：
1. 数据有效性检查（缺失值、静态数据检测）
2. 生物力学合理性检查（关节角极限、肢体扭曲检测）
3. 运动学质量检查（过度抖动、足部滑步检测）
4. 质量评分与报告生成
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 默认阈值配置
DEFAULT_THRESHOLDS = {
    # 数据有效性检查阈值
    'nan_threshold': 1e-10,           # NaN/Inf检测阈值
    'static_variance_threshold': 1e-6, # 静态数据方差阈值
    'static_rotation_threshold': 1e-3, # 静态旋转变化阈值
    
    # 生物力学合理性检查阈值
    'knee_flexion_range': [0, 150],    # 膝关节屈伸范围(度)
    'elbow_flexion_range': [0, 150],   # 肘关节屈伸范围(度)
    'spine_rotation_threshold': 45,    # 脊柱旋转安全阈值(度)
    
    # 运动学质量检查阈值
    'angular_velocity_limit': 720,     # 角速度限制(度/秒)
    'foot_height_threshold': 0.05,     # 足部离地高度阈值(米)
    'foot_sliding_velocity': 0.1,      # 足部滑动速度阈值(米/秒)
}

@dataclass
class DataQualityReport:
    """数据质量报告数据类"""
    file_path: str
    file_name: str
    total_frames: int
    processing_time: str
    
    # 数据有效性检查结果
    validity_issues: Dict[str, Any]
    static_frame_count: int
    nan_frame_count: int
    
    # 生物力学合理性检查结果
    biomechanical_violations: Dict[str, Any]
    joint_violation_counts: Dict[str, int]
    
    # 运动学质量检查结果
    motion_quality_issues: Dict[str, Any]
    jitter_frame_count: int
    sliding_frame_count: int
    
    # 整体评分
    quality_score: float  # 0-100分
    recommendation: str   # 处理建议
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

class DataCleaner:
    """数据清洗器主类"""
    
    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        """
        初始化数据清洗器
        
        Args:
            thresholds: 自定义阈值配置
        """
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
            
        logger.info("DataCleaner初始化完成")
        logger.info(f"使用阈值配置: {self.thresholds}")

    def check_data_validity(self, poses: np.ndarray, trans: np.ndarray) -> Dict[str, Any]:
        """
        执行数据有效性检查
        
        Args:
            poses: 姿态数据
            trans: 位移数据
            
        Returns:
            有效性检查报告
        """
        logger.info("执行数据有效性检查...")
        
        validity_report = {
            'nan_check': self._check_nan_values(poses, trans),
            'static_check': self._check_static_data(poses, trans),
            'overall_valid': True,
            'issues_found': 0
        }
        
        # 统计问题数量
        if validity_report['nan_check']['has_nan']:
            validity_report['issues_found'] += 1
            validity_report['overall_valid'] = False
            
        if validity_report['static_check']['is_static']:
            validity_report['issues_found'] += 1
            validity_report['overall_valid'] = False
            
        logger.info(f"数据有效性检查完成: {'有效' if validity_report['overall_valid'] else '无效'}")
        return validity_report

    def _check_nan_values(self, poses: np.ndarray, trans: np.ndarray) -> Dict[str, Any]:
        """检测缺失值/NaN值"""
        issues = {
            'has_nan': False,
            'nan_frames': [],
            'nan_positions': [],
            'details': {}
        }
        
        # 检查姿态数据
        nan_mask = np.isnan(poses) | np.isinf(poses)
        if np.any(nan_mask):
            issues['has_nan'] = True
            nan_frames = np.where(np.any(nan_mask, axis=1))[0]
            issues['nan_frames'] = nan_frames.tolist()
            issues['nan_positions'] = np.where(nan_mask)
            issues['details']['pose_nan_count'] = np.sum(nan_mask)
            
        # 检查位移数据
        trans_nan_mask = np.isnan(trans) | np.isinf(trans)
        if np.any(trans_nan_mask):
            issues['has_nan'] = True
            nan_trans_frames = np.where(np.any(trans_nan_mask, axis=1))[0]
            issues['details']['trans_nan_frames'] = nan_trans_frames.tolist()
            issues['details']['trans_nan_count'] = np.sum(trans_nan_mask)
            
        return issues

    def _check_static_data(self, poses: np.ndarray, trans: np.ndarray) -> Dict[str, Any]:
        """检测静态数据"""
        issues = {
            'is_static': False,
            'static_frames': [],
            'root_variance': 0.0,
            'rotation_variance': 0.0,
            'details': {}
        }
        
        # 计算根节点位移方差
        root_variance = np.var(trans, axis=0)
        issues['root_variance'] = float(np.mean(root_variance))
        
        # 检查是否为静态（根节点位移方差过小）
        if issues['root_variance'] < self.thresholds['static_variance_threshold']:
            issues['is_static'] = True
            issues['static_frames'] = list(range(len(poses)))
            issues['details']['reason'] = '根节点位移方差过小'
            
        # 检查旋转变化（计算相邻帧间旋转差异）
        if len(poses) > 1:
            diff_poses = np.diff(poses, axis=0)
            rotation_variance = np.var(diff_poses, axis=0)
            issues['rotation_variance'] = float(np.mean(rotation_variance))
            
            # 如果旋转变化也很小，进一步确认为静态数据
            if (issues['rotation_variance'] < self.thresholds['static_rotation_threshold'] 
                and not issues['is_static']):
                issues['is_static'] = True
                issues['static_frames'] = list(range(len(poses)))
                issues['details']['reason'] = '关节旋转变化过小'
                
        return issues

    def check_biomechanical_reasonableness(self, poses: np.ndarray) -> Dict[str, Any]:
        """
        执行生物力学合理性检查
        
        Args:
            poses: 姿态数据
            
        Returns:
            生物力学检查报告
        """
        logger.info("执行生物力学合理性检查...")
        
        # 提取关节角度
        joint_angles = self._extract_joint_angles(poses)
        
        # 执行各项检查
        joint_limit_violations = self._check_joint_limits(joint_angles)
        
        biomechanical_report = {
            'joint_limit_checks': joint_limit_violations,
            'overall_reasonable': True,
            'violation_ratio': 0.0,
            'issues_found': 0
        }
        
        # 计算违规比例
        total_frames = poses.shape[0]
        all_violations = set()
        all_violations.update(joint_limit_violations['knee_violations'])
        all_violations.update(joint_limit_violations['elbow_violations'])
        all_violations.update(joint_limit_violations['spine_violations'])
        
        violation_ratio = len(all_violations) / total_frames if total_frames > 0 else 0
        biomechanical_report['violation_ratio'] = violation_ratio
        
        # 判断是否合理（如果违规帧超过一定比例则认为不合理）
        if violation_ratio > 0.3:  # 超过30%帧违规
            biomechanical_report['overall_reasonable'] = False
            biomechanical_report['issues_found'] += 1
            
        logger.info(f"生物力学检查完成: 违规比例 {violation_ratio:.2%}")
        return biomechanical_report

    def _extract_joint_angles(self, poses: np.ndarray) -> np.ndarray:
        """从姿态数据中提取各关节角度"""
        frames = poses.shape[0]
        joint_angles = np.zeros((frames, 24, 3))
        
        # 将轴角转换为欧拉角
        for frame in range(frames):
            for joint in range(24):
                start_idx = joint * 3
                end_idx = start_idx + 3
                axis_angle = poses[frame, start_idx:end_idx]
                
                if np.linalg.norm(axis_angle) < 1e-10:
                    rotation_matrix = np.eye(3)
                else:
                    rotation = Rotation.from_rotvec(axis_angle)
                    rotation_matrix = rotation.as_matrix()
                
                joint_angles[frame, joint] = self._rotation_matrix_to_euler_angles(rotation_matrix)
                
        return joint_angles

    def _rotation_matrix_to_euler_angles(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """旋转矩阵转欧拉角（XYZ顺序，度）"""
        # 提取矩阵元素
        r00, r01, r02 = rotation_matrix[0]
        r10, r11, r12 = rotation_matrix[1]
        r20, r21, r22 = rotation_matrix[2]
        
        # 处理万向锁情况
        if abs(r20) >= 1:
            y = np.sign(r20) * np.pi / 2
            x = 0
            z = np.arctan2(-r01, r11)
        else:
            y = -np.arcsin(r20)
            x = np.arctan2(r21, r22)
            z = np.arctan2(r10, r00)
            
        return np.degrees([x, y, z])

    def _check_joint_limits(self, joint_angles: np.ndarray) -> Dict[str, Any]:
        """检查关节角度是否超出生理范围"""
        violations = {
            'knee_violations': [],      # 膝关节违规
            'elbow_violations': [],     # 肘关节违规
            'spine_violations': [],     # 脊柱违规
            'violation_counts': {},
            'details': {}
        }
        
        # 膝关节检查（索引4和5）
        knee_ranges = self.thresholds['knee_flexion_range']
        for knee_idx in [4, 5]:  # left_knee, right_knee
            knee_flexion = joint_angles[:, knee_idx, 0]  # X轴旋转（屈伸）
            violation_mask = (knee_flexion < knee_ranges[0]) | (knee_flexion > knee_ranges[1])
            if np.any(violation_mask):
                violation_frames = np.where(violation_mask)[0].tolist()
                violations['knee_violations'].extend(violation_frames)
                violations['violation_counts'][f'knee_{knee_idx}'] = len(violation_frames)
        
        # 肘关节检查（索引18和19）
        elbow_ranges = self.thresholds['elbow_flexion_range']
        for elbow_idx in [18, 19]:  # left_elbow, right_elbow
            elbow_flexion = joint_angles[:, elbow_idx, 0]  # X轴旋转（屈伸）
            violation_mask = (elbow_flexion < elbow_ranges[0]) | (elbow_flexion > elbow_ranges[1])
            if np.any(violation_mask):
                violation_frames = np.where(violation_mask)[0].tolist()
                violations['elbow_violations'].extend(violation_frames)
                violations['violation_counts'][f'elbow_{elbow_idx}'] = len(violation_frames)
        
        # 脊柱旋转检查（索引3, 6, 9对应spine1, spine2, spine3）
        spine_threshold = self.thresholds['spine_rotation_threshold']
        for spine_idx in [3, 6, 9]:
            # 检查Y轴旋转（侧屈）和Z轴旋转（旋转）
            spine_y_rot = np.abs(joint_angles[:, spine_idx, 1])  # 侧屈
            spine_z_rot = np.abs(joint_angles[:, spine_idx, 2])  # 旋转
            
            y_violation = spine_y_rot > spine_threshold
            z_violation = spine_z_rot > spine_threshold
            
            if np.any(y_violation | z_violation):
                violation_frames = np.where(y_violation | z_violation)[0].tolist()
                violations['spine_violations'].extend(violation_frames)
                violations['violation_counts'][f'spine_{spine_idx}'] = len(violation_frames)
        
        # 去重并统计
        violations['knee_violations'] = list(set(violations['knee_violations']))
        violations['elbow_violations'] = list(set(violations['elbow_violations']))
        violations['spine_violations'] = list(set(violations['spine_violations']))
        
        violations['details']['total_knee_violations'] = len(violations['knee_violations'])
        violations['details']['total_elbow_violations'] = len(violations['elbow_violations'])
        violations['details']['total_spine_violations'] = len(violations['spine_violations'])
        
        return violations

    def check_motion_quality(self, poses: np.ndarray, trans: np.ndarray, 
                           framerate: float = 120.0) -> Dict[str, Any]:
        """
        执行运动学质量检查
        
        Args:
            poses: 姿态数据
            trans: 位移数据
            framerate: 帧率
            
        Returns:
            运动学质量检查报告
        """
        logger.info("执行运动学质量检查...")
        
        motion_report = {
            'jitter_detection': self._detect_excessive_jitter(poses, framerate),
            'foot_sliding_detection': self._detect_foot_sliding(poses, trans, framerate),
            'overall_good_quality': True,
            'issues_found': 0
        }
        
        # 统计问题
        if motion_report['jitter_detection']['has_jitter']:
            motion_report['issues_found'] += 1
            
        if motion_report['foot_sliding_detection']['has_sliding']:
            motion_report['issues_found'] += 1
            
        if motion_report['issues_found'] > 0:
            motion_report['overall_good_quality'] = False
            
        logger.info(f"运动学质量检查完成: 发现{motion_report['issues_found']}个问题")
        return motion_report

    def _detect_excessive_jitter(self, poses: np.ndarray, framerate: float) -> Dict[str, Any]:
        """检测过度抖动"""
        issues = {
            'has_jitter': False,
            'jitter_frames': [],
            'max_angular_velocity': 0.0,
            'details': {}
        }
        
        if len(poses) < 2:
            return issues
            
        # 计算帧间角度变化
        diff_poses = np.diff(poses, axis=0)  # (frames-1, 72)
        angular_changes = np.linalg.norm(diff_poses, axis=1)  # (frames-1,)
        
        # 转换为角速度（度/秒）
        time_delta = 1.0 / framerate
        angular_velocities = np.degrees(angular_changes) / time_delta
        
        issues['max_angular_velocity'] = float(np.max(angular_velocities))
        
        # 检测超过阈值的帧
        jitter_threshold = self.thresholds['angular_velocity_limit']
        jitter_mask = angular_velocities > jitter_threshold
        if np.any(jitter_mask):
            issues['has_jitter'] = True
            issues['jitter_frames'] = np.where(jitter_mask)[0].tolist()
            issues['details']['excessive_velocity_frames'] = len(issues['jitter_frames'])
            issues['details']['threshold_used'] = jitter_threshold
            
        return issues

    def _detect_foot_sliding(self, poses: np.ndarray, trans: np.ndarray, 
                           framerate: float) -> Dict[str, Any]:
        """检测足部滑步"""
        issues = {
            'has_sliding': False,
            'sliding_frames': [],
            'details': {}
        }
        
        # 简化的足部滑动检测
        if len(trans) < 2:
            return issues
            
        # 计算水平速度（忽略Y轴，即垂直方向）
        horizontal_velocity = np.linalg.norm(np.diff(trans[:, [0, 2]], axis=0), axis=1)
        
        # 简单检测：如果整体动作幅度较小，可能存在滑动
        avg_horizontal_speed = np.mean(horizontal_velocity)
        if avg_horizontal_speed < 0.01:  # 很小的水平运动
            issues['has_sliding'] = True
            issues['sliding_frames'] = list(range(len(poses)))
            issues['details']['avg_speed'] = float(avg_horizontal_speed)
            
        return issues

    def generate_quality_score(self, validity_report: Dict[str, Any], 
                             biomechanical_report: Dict[str, Any],
                             motion_report: Dict[str, Any]) -> float:
        """生成综合质量评分"""
        score = 100.0
        
        # 数据有效性扣分
        if not validity_report['overall_valid']:
            score -= 30
            
        # 生物力学合理性扣分
        violation_ratio = biomechanical_report.get('violation_ratio', 0)
        score -= violation_ratio * 40  # 最多扣40分
        
        # 运动学质量扣分
        if not motion_report['overall_good_quality']:
            score -= 20
            
        return max(0, min(100, score))  # 限制在0-100范围内

    def generate_recommendation(self, quality_score: float, 
                              issues_found: int) -> str:
        """生成处理建议"""
        if quality_score >= 90:
            return "数据质量优秀，可直接用于训练"
        elif quality_score >= 70:
            return "数据质量良好，建议轻微后处理"
        elif quality_score >= 50:
            return "数据质量一般，需要进行数据清洗和修复"
        elif quality_score >= 30:
            return "数据质量较差，建议丢弃或大幅修正"
        else:
            return "数据质量很差，强烈建议丢弃"

    def clean_single_file(self, file_data: Dict[str, Any], 
                         output_dir: Optional[str] = None) -> DataQualityReport:
        """
        清洗单个文件
        
        Args:
            file_data: 包含姿态和位移数据的字典
            output_dir: 输出目录（可选）
            
        Returns:
            数据质量报告
        """
        start_time = datetime.now()
        
        try:
            # 提取必要字段
            poses = file_data['poses'][:, :72]  # 取前72维身体姿态
            trans = file_data['trans']
            framerate = file_data.get('mocap_framerate', 120.0)
            file_path = file_data.get('_metadata', {}).get('file_path', 'unknown')
            file_name = file_data.get('_metadata', {}).get('file_name', 'unknown')
            
            # 执行各项检查
            validity_report = self.check_data_validity(poses, trans)
            biomechanical_report = self.check_biomechanical_reasonableness(poses)
            motion_report = self.check_motion_quality(poses, trans, framerate)
            
            # 生成综合评分
            quality_score = self.generate_quality_score(
                validity_report, biomechanical_report, motion_report
            )
            
            total_issues = (validity_report['issues_found'] + 
                          biomechanical_report['issues_found'] + 
                          motion_report['issues_found'])
                          
            recommendation = self.generate_recommendation(quality_score, total_issues)
            
            # 创建报告
            processing_time = str(datetime.now() - start_time)
            
            report = DataQualityReport(
                file_path=file_path,
                file_name=file_name,
                total_frames=len(poses),
                processing_time=processing_time,
                
                validity_issues=validity_report,
                static_frame_count=len(validity_report['static_check']['static_frames']),
                nan_frame_count=len(validity_report['nan_check']['nan_frames']),
                
                biomechanical_violations=biomechanical_report,
                joint_violation_counts=biomechanical_report['joint_limit_checks']['violation_counts'],
                
                motion_quality_issues=motion_report,
                jitter_frame_count=len(motion_report['jitter_detection']['jitter_frames']),
                sliding_frame_count=len(motion_report['foot_sliding_detection']['sliding_frames']),
                
                quality_score=quality_score,
                recommendation=recommendation
            )
            
            # 保存报告（如果指定了输出目录）
            if output_dir:
                self.save_report(report, output_dir)
                
            logger.info(f"文件处理完成: {file_name}")
            logger.info(f"质量评分: {quality_score:.1f}/100")
            logger.info(f"处理建议: {recommendation}")
            
            return report
            
        except Exception as e:
            logger.error(f"处理文件失败: {file_data.get('file_path', 'unknown')}")
            logger.error(f"错误详情: {str(e)}")
            raise

    def save_report(self, report: DataQualityReport, output_dir: str):
        """保存数据质量报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON报告
        json_path = output_path / f"{report.file_name}_quality_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            
        logger.info(f"质量报告已保存: {json_path}")

def main():
    """主函数 - 演示使用方法"""
    # 创建数据清洗器
    cleaner = DataCleaner()
    
    # 创建测试数据
    test_data = {
        'poses': np.random.randn(100, 156),  # 100帧，156维
        'trans': np.random.randn(100, 3),    # 100帧，3维位移
        'mocap_framerate': 120.0,
        '_metadata': {
            'file_path': './test.npz',
            'file_name': 'test.npz'
        }
    }
    
    try:
        # 清洗单个文件
        report = cleaner.clean_single_file(test_data, "./test_output")
        
        # 打印摘要信息
        print("\n=== 数据质量报告摘要 ===")
        print(f"文件: {report.file_name}")
        print(f"总帧数: {report.total_frames}")
        print(f"处理时间: {report.processing_time}")
        print(f"质量评分: {report.quality_score:.1f}/100")
        print(f"处理建议: {report.recommendation}")
        print(f"发现的问题数: {report.validity_issues['issues_found'] + report.biomechanical_violations['issues_found'] + report.motion_quality_issues['issues_found']}")
        
    except Exception as e:
        logger.error(f"处理失败: {e}")

if __name__ == "__main__":
    main()