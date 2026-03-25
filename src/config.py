"""
配置模块
Configuration Module

集中管理所有配置参数
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PathConfig:
    """路径配置"""
    # 项目根目录
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 数据目录
    data_dir: str = field(default='data')
    template_file: str = field(default='data/single.mat')

    # 训练数据
    good_signal_file: str = field(default='data/for_model/good_single_cycle.mat')
    bad_signal_file: str = field(default='data/for_model/bad.mat')

    # 可行性数据
    kexing_good_file: str = field(default='data/kexing/good.mat')
    kexing_bad_file: str = field(default='data/kexing/bad.mat')

    # 测试数据
    kexing_test_good_file: str = field(default='data/kexing_test/good.mat')
    kexing_test_bad_file: str = field(default='data/kexing_test/bad.mat')
    bad_test_file: str = field(default='data/for_model/bad_test.mat')

    # 输出目录
    results_dir: str = field(default='results')
    models_dir: str = field(default='results/models')
    features_dir: str = field(default='data/features')
    figures_dir: str = field(default='figures')

    def __post_init__(self):
        """确保所有路径都是绝对路径"""
        for attr in ['data_dir', 'template_file', 'good_signal_file', 'bad_signal_file',
                     'kexing_good_file', 'kexing_bad_file', 'kexing_test_good_file',
                     'kexing_test_bad_file', 'bad_test_file', 'results_dir',
                     'models_dir', 'features_dir', 'figures_dir']:
            path = getattr(self, attr)
            if not os.path.isabs(path):
                setattr(self, attr, os.path.join(self.root_dir, path))


@dataclass
class DTWConfig:
    """DTW匹配配置"""
    # 搜索半径（路径约束）
    radius: int = 1

    # 自适应选模板数量
    top_n: int = 200

    # 是否截断到N波
    truncate_to_n_wave: bool = True

    # 斜率约束范围
    slope_constraint: Tuple[float, float] = (0.5, 2.0)


@dataclass
class ClassifierConfig:
    """分类器配置"""
    # 分类器权重 [LR, SVM, RF]
    weights: List[float] = field(default_factory=lambda: [1/3, 1/3, 1/3])

    # LR参数
    lr_C: float = 1.0
    lr_max_iter: int = 1000

    # SVM参数
    svm_C: float = 1.0
    svm_kernel: str = 'rbf'
    svm_gamma: str = 'scale'

    # RF参数
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_random_state: int = 42

    # 训练参数
    test_size: float = 0.3
    random_state: int = 42
    cv_folds: int = 5


@dataclass
class SignalConfig:
    """信号处理配置"""
    # 采样率
    original_sample_rate: int = 1024
    target_sample_rate: int = 100

    # 小波变换参数
    wavelet: str = 'sym4'
    wavelet_level: int = 5
    wavelet_detail_levels: Tuple[int, ...] = (3, 4, 5)  # 使用第3-5层

    # 能量包络参数
    envelope_window_size: int = 20
    envelope_smooth_iterations: int = 3


@dataclass
class Config:
    """总配置类"""
    paths: PathConfig = field(default_factory=PathConfig)
    dtw: DTWConfig = field(default_factory=DTWConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)

    def ensure_dirs(self):
        """确保所有输出目录存在"""
        os.makedirs(self.paths.results_dir, exist_ok=True)
        os.makedirs(self.paths.models_dir, exist_ok=True)
        os.makedirs(self.paths.features_dir, exist_ok=True)
        os.makedirs(self.paths.figures_dir, exist_ok=True)


# 默认配置实例
default_config = Config()


def get_config() -> Config:
    """获取默认配置"""
    return default_config
