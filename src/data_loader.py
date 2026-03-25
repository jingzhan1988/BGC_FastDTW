"""
数据加载模块
Data Loader Module

统一的数据加载接口
"""

import numpy as np
import scipy.io as scio
from typing import List, Tuple, Dict, Optional
import os

from .config import Config, get_config


class DataLoader:
    """数据加载器"""

    def __init__(self, config: Config = None):
        """
        初始化数据加载器

        Args:
            config: 配置对象
        """
        self.config = config or get_config()

    def load_mat(self, file_path: str) -> Dict:
        """
        加载MAT文件

        Args:
            file_path: MAT文件路径

        Returns:
            MAT文件数据字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MAT file not found: {file_path}")
        return scio.loadmat(file_path)

    def load_templates(self, file_path: str = None) -> List[np.ndarray]:
        """
        加载模板库

        Args:
            file_path: 模板文件路径，默认使用配置中的路径

        Returns:
            模板列表
        """
        file_path = file_path or self.config.paths.template_file

        data = self.load_mat(file_path)
        templates = [dat[-1][0] for dat in data['single_cycle']]
        templates = [np.asarray(temp).flatten() for temp in templates]

        # 去重和排序
        templates = self._preprocess_templates(templates)

        print(f"Loaded {len(templates)} templates from {file_path}")
        return templates

    def _preprocess_templates(self, templates: List[np.ndarray]) -> List[np.ndarray]:
        """
        预处理模板：去重和按长度排序

        Args:
            templates: 原始模板列表

        Returns:
            处理后的模板列表
        """
        # 转换为tuple去重
        unique_templates = list(set(map(tuple, templates)))
        # 转回numpy数组
        unique_templates = [np.array(t) for t in unique_templates]
        # 按长度排序
        sorted_templates = sorted(unique_templates, key=len)
        return sorted_templates

    def load_single_cycle_signals(self, file_path: str, key: str = 'single_cycle') -> List[np.ndarray]:
        """
        加载单心动周期信号

        Args:
            file_path: MAT文件路径
            key: 数据键名

        Returns:
            信号列表
        """
        data = self.load_mat(file_path)
        signals = [dat[-1][0] for dat in data[key]]
        signals = [np.asarray(s).flatten() for s in signals if len(s) > 0]

        print(f"Loaded {len(signals)} single cycle signals from {file_path}")
        return signals

    def load_multi_cycle_signals(self, file_path: str, key: str, start_row: int = 3) -> List[np.ndarray]:
        """
        加载多周期信号（如bad_signal格式）

        Args:
            file_path: MAT文件路径
            key: 数据键名
            start_row: 起始行（跳过前几行）

        Returns:
            信号列表
        """
        data = self.load_mat(file_path)
        signals = []

        raw_data = data[key][start_row:]

        for row in raw_data:
            for item in row:
                if hasattr(item, 'shape'):
                    if len(item.shape) > 1 and item.shape[1] > 1:
                        signals.append(np.asarray(item).flatten())
                    elif len(item) > 0:
                        item_flat = np.asarray(item).flatten()
                        if len(item_flat) > 0:
                            signals.append(item_flat)
                elif len(item) > 0:
                    item_flat = np.asarray(item).flatten()
                    if len(item_flat) > 0:
                        signals.append(item_flat)

        print(f"Loaded {len(signals)} multi-cycle signals from {file_path}")
        return signals

    def load_kexing_signals(self, file_path: str, key: str, start_row: int = 3) -> List[List[np.ndarray]]:
        """
        加载可行性分析数据（每个样本包含多个周期）

        Args:
            file_path: MAT文件路径
            key: 数据键名 ('good_signal_cycle' 或 'bad_signal_cycle')
            start_row: 起始行

        Returns:
            样本列表，每个样本是周期信号的列表
        """
        data = self.load_mat(file_path)
        raw_data = data[key][start_row:]

        samples = []
        for idx in range(raw_data.shape[1]):
            cycles = raw_data[:, idx]
            sample_signals = []
            for cycle in cycles:
                if hasattr(cycle, 'shape') and cycle.shape[1] > 1:
                    sample_signals.append(np.asarray(cycle).flatten())
            if sample_signals:
                samples.append(sample_signals)

        print(f"Loaded {len(samples)} samples from {file_path}")
        return samples

    def load_good_signals(self) -> List[np.ndarray]:
        """加载好信号数据"""
        return self.load_single_cycle_signals(
            self.config.paths.good_signal_file,
            key='single_cycle'
        )

    def load_bad_signals(self) -> List[np.ndarray]:
        """加载坏信号数据"""
        return self.load_multi_cycle_signals(
            self.config.paths.bad_signal_file,
            key='bad_signal',
            start_row=3
        )

    def load_bad_test_signals(self) -> List[np.ndarray]:
        """加载坏信号测试数据"""
        return self.load_multi_cycle_signals(
            self.config.paths.bad_test_file,
            key='bad_test',
            start_row=3
        )

    def load_training_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        加载训练数据

        Returns:
            (good_signals, bad_signals)
        """
        good_signals = self.load_good_signals()
        bad_signals = self.load_bad_signals()
        return good_signals, bad_signals

    def load_sqi_from_txt(self, file_path: str) -> np.ndarray:
        """
        从TXT文件加载SQI值

        Args:
            file_path: TXT文件路径

        Returns:
            SQI数组
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SQI file not found: {file_path}")
        return np.loadtxt(file_path)

    def save_sqi_to_txt(self, sqi_values: np.ndarray, file_path: str):
        """
        保存SQI值到TXT文件

        Args:
            sqi_values: SQI数组
            file_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savetxt(file_path, sqi_values)
        print(f"Saved SQI values to {file_path}")

    def save_features(self, features: np.ndarray, file_path: str):
        """
        保存特征矩阵

        Args:
            features: 特征矩阵
            file_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, features)
        print(f"Saved features to {file_path}, shape: {features.shape}")

    def load_features(self, file_path: str) -> np.ndarray:
        """
        加载特征矩阵

        Args:
            file_path: 特征文件路径

        Returns:
            特征矩阵
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        return np.load(file_path)


# 便捷函数
def load_templates(config: Config = None) -> List[np.ndarray]:
    """加载模板库的便捷函数"""
    loader = DataLoader(config)
    return loader.load_templates()


def load_training_data(config: Config = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """加载训练数据的便捷函数"""
    loader = DataLoader(config)
    return loader.load_training_data()
