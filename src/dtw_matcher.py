"""
DTW匹配模块
DTW Matcher Module

实现带路径约束的Fast-SDTW匹配，支持N波截断
"""

import numpy as np
from scipy.signal import find_peaks
from fastdtw import fastdtw
from typing import List, Tuple, Dict, Optional

from .config import Config, get_config
from .feature_extractor import FeatureExtractor
from .data_loader import DataLoader
from .signal_processor import SignalProcessor


class DTWMatcher:
    """DTW模板匹配器"""

    def __init__(self, config: Config = None):
        """
        初始化DTW匹配器

        Args:
            config: 配置对象
        """
        self.config = config or get_config()
        self.dtw_radius = self.config.dtw.radius
        self.top_n = self.config.dtw.top_n
        self.truncate_enabled = self.config.dtw.truncate_to_n_wave

        self.templates = None
        self.data_loader = DataLoader(self.config)
        self.signal_processor = SignalProcessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)

    def load_templates(self) -> List[np.ndarray]:
        """
        加载模板库

        Returns:
            模板列表
        """
        self.templates = self.data_loader.load_templates()
        return self.templates

    def _preprocess_templates(self, templates: List[np.ndarray]) -> List[np.ndarray]:
        """
        预处理模板：去重和按长度排序

        Args:
            templates: 原始模板列表

        Returns:
            处理后的模板列表
        """
        unique_templates = list(set(map(tuple, templates)))
        unique_templates = [np.array(t) for t in unique_templates]
        sorted_templates = sorted(unique_templates, key=len)
        return sorted_templates

    def select_templates(self, query: np.ndarray, top_n: int = None) -> List[np.ndarray]:
        """
        自适应选择模板（按长度选择最接近的top_n个）

        Args:
            query: 查询信号
            top_n: 选择数量

        Returns:
            选中的模板列表
        """
        if self.templates is None:
            self.load_templates()

        if top_n is None:
            top_n = self.top_n

        query_len = len(query)

        # 获取与query长度相同的序列
        same_length_templates = [t for t in self.templates if len(t) == query_len]

        if len(same_length_templates) >= top_n:
            return same_length_templates[:top_n]

        # 否则按长度差值排序
        length_diffs = [(t, abs(len(t) - query_len)) for t in self.templates]
        sorted_by_diff = sorted(length_diffs, key=lambda x: x[1])

        result = same_length_templates.copy()
        for t, _ in sorted_by_diff:
            if len(result) >= top_n:
                break
            # 避免重复添加
            if not any(np.array_equal(t, r) for r in result):
                result.append(t)

        return result[:top_n]

    def find_n_wave(self, signal: np.ndarray) -> Optional[int]:
        """
        检测N波位置

        Args:
            signal: 输入信号

        Returns:
            N波位置索引
        """
        return self.signal_processor.find_n_wave_position(signal)

    def truncate_to_n_wave(self, signal: np.ndarray, n_wave_idx: int = None) -> np.ndarray:
        """
        截断信号到N波位置

        Args:
            signal: 输入信号
            n_wave_idx: N波位置索引

        Returns:
            截断后的信号
        """
        if n_wave_idx is not None:
            signal = np.asarray(signal).flatten()
            if n_wave_idx < len(signal):
                return signal[:n_wave_idx + 1]
            return signal
        return self.signal_processor.truncate_to_n_wave(signal)

    def match_single(self, query: np.ndarray, truncate: bool = True) -> Dict:
        """
        匹配单个信号

        Args:
            query: 查询信号
            truncate: 是否截断到N波

        Returns:
            匹配结果字典: {
                'features': 特征字典,
                'best_template': 最佳匹配模板,
                'dtw_distance': DTW距离,
                'pearson': 皮尔逊系数,
                'path_smoothness': 路径光滑度
            }
        """
        query = np.asarray(query).flatten()

        # 可选：截断到N波
        if truncate:
            query_truncated = self.truncate_to_n_wave(query)
        else:
            query_truncated = query

        # 自适应选择模板
        selected_templates = self.select_templates(query_truncated)

        # 如果截断了query，也需要截断模板
        if truncate:
            truncated_templates = []
            for t in selected_templates:
                t_truncated = self.truncate_to_n_wave(t)
                truncated_templates.append(t_truncated)
            selected_templates = truncated_templates

        # 提取特征
        features, best_template = self.feature_extractor.extract_features_batch(
            query_truncated, selected_templates
        )

        if features is None:
            return {
                'features': None,
                'best_template': None,
                'dtw_distance': float('inf'),
                'pearson': 0.0,
                'path_smoothness': float('inf')
            }

        return {
            'features': features,
            'best_template': best_template,
            'dtw_distance': features['dtw_distance'],
            'pearson': features['pearson'],
            'path_smoothness': features['path_smoothness']
        }

    def match_batch(self, queries: List[np.ndarray], truncate: bool = True,
                    verbose: bool = True) -> List[Dict]:
        """
        批量匹配信号

        Args:
            queries: 查询信号列表
            truncate: 是否截断到N波
            verbose: 是否显示进度

        Returns:
            匹配结果列表
        """
        results = []

        if verbose:
            from tqdm import tqdm
            iterator = tqdm(queries, desc="Matching")
        else:
            iterator = queries

        for query in iterator:
            result = self.match_single(query, truncate=truncate)
            results.append(result)

        return results

    def get_feature_matrix(self, queries: List[np.ndarray], truncate: bool = True) -> np.ndarray:
        """
        获取特征矩阵

        Args:
            queries: 查询信号列表
            truncate: 是否截断到N波

        Returns:
            特征矩阵 shape: (n_samples, 3)
            列顺序: [pearson, dtw_distance, path_smoothness]
        """
        results = self.match_batch(queries, truncate=truncate)

        feature_matrix = []
        for r in results:
            if r['features'] is not None:
                feature_matrix.append([
                    r['pearson'],
                    r['dtw_distance'],
                    r['path_smoothness']
                ])
            else:
                feature_matrix.append([0.0, float('inf'), float('inf')])

        return np.array(feature_matrix)


def test_dtw_matcher():
    """测试DTW匹配器"""
    print("DTW Matcher Test:")

    # 创建测试数据
    t = np.linspace(0, 2*np.pi, 100)
    query = np.sin(t)

    # 模拟模板库
    templates = [
        np.sin(np.linspace(0, 2*np.pi, 95)),
        np.sin(np.linspace(0, 2*np.pi, 100)),
        np.sin(np.linspace(0, 2*np.pi, 105)),
    ]

    # 初始化匹配器（使用模拟模板）
    matcher = DTWMatcher()
    matcher.templates = templates

    # 测试N波检测
    n_wave = matcher.find_n_wave(query)
    print(f"  N-wave index: {n_wave}")

    # 测试特征提取
    result = matcher.match_single(query, truncate=False)
    print(f"  DTW Distance: {result['dtw_distance']:.4f}")
    print(f"  Pearson: {result['pearson']:.4f}")
    print(f"  Path Smoothness: {result['path_smoothness']:.4f}")

    print("Test completed!")


if __name__ == '__main__':
    test_dtw_matcher()
