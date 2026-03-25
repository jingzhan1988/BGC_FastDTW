"""
Signal Quality Assessment System
信号质量评估系统

模块化的信号质量评估框架，基于DTW模板匹配和集成分类器
"""

from .config import Config, get_config, PathConfig, DTWConfig, ClassifierConfig, SignalConfig
from .data_loader import DataLoader, load_templates, load_training_data
from .signal_processor import SignalProcessor, preprocess_signal, find_n_wave, truncate_signal
from .feature_extractor import FeatureExtractor
from .dtw_matcher import DTWMatcher
from .ensemble_classifier import EnsembleClassifier, LRClassifier, SVMClassifier, RFClassifier

__all__ = [
    # 配置
    'Config',
    'get_config',
    'PathConfig',
    'DTWConfig',
    'ClassifierConfig',
    'SignalConfig',

    # 数据加载
    'DataLoader',
    'load_templates',
    'load_training_data',

    # 信号处理
    'SignalProcessor',
    'preprocess_signal',
    'find_n_wave',
    'truncate_signal',

    # 特征提取
    'FeatureExtractor',

    # DTW匹配
    'DTWMatcher',

    # 分类器
    'EnsembleClassifier',
    'LRClassifier',
    'SVMClassifier',
    'RFClassifier',
]

__version__ = '1.0.0'
__author__ = 'Signal Quality Assessment Team'
