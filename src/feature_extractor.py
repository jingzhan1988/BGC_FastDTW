"""
特征提取模块
Feature Extraction Module

提取三个信号质量特征:
1. 皮尔逊相关系数 (Pearson Correlation)
2. DTW距离 (Dynamic Time Warping Distance)
3. 路径光滑度 (Path Smoothness)
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import resample
from fastdtw import fastdtw
from typing import Tuple, List, Dict, Optional

from .config import Config, get_config


class FeatureExtractor:
    """特征提取类"""

    def __init__(self, config: Config = None, dtw_radius: int = None):
        """
        初始化特征提取器

        Args:
            config: 配置对象
            dtw_radius: DTW搜索半径，控制路径约束严格程度
        """
        self.config = config or get_config()
        self.dtw_radius = dtw_radius or self.config.dtw.radius

    def compute_pearson(self, query: np.ndarray, template: np.ndarray) -> float:
        """
        计算皮尔逊相关系数

        由于两个信号长度可能不同，需要先重采样到相同长度

        Args:
            query: 查询信号
            template: 模板信号

        Returns:
            皮尔逊相关系数 [-1, 1]
        """
        query = np.asarray(query).flatten()
        template = np.asarray(template).flatten()

        # 重采样到相同长度（取较长的长度）
        target_len = max(len(query), len(template))

        if len(query) != target_len:
            query_resampled = resample(query, target_len)
        else:
            query_resampled = query

        if len(template) != target_len:
            template_resampled = resample(template, target_len)
        else:
            template_resampled = template

        # 计算皮尔逊相关系数
        corr, _ = pearsonr(query_resampled, template_resampled)

        # 处理NaN情况
        if np.isnan(corr):
            corr = 0.0

        return corr

    def compute_dtw_with_path(self, query: np.ndarray, template: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """
        计算DTW距离并返回warping path

        Args:
            query: 查询信号
            template: 模板信号

        Returns:
            (distance, path): DTW距离和路径
        """
        query = np.asarray(query).flatten()
        template = np.asarray(template).flatten()

        # 使用fastdtw计算，radius参数控制搜索窗口（路径约束）
        distance, path = fastdtw(query, template, radius=self.dtw_radius)

        return distance, path

    def compute_path_smoothness(self, path: List[Tuple[int, int]]) -> float:
        """
        计算路径光滑度 (Path Smoothness, PS)

        光滑路径的特点: 斜率变化小，接近对角线
        非光滑路径: 存在大量水平/垂直移动（表示信号形变大）

        计算方法:
        1. 计算相邻点之间的移动方向
        2. 累加方向变化量
        3. 归一化

        Args:
            path: DTW路径 [(i1, j1), (i2, j2), ...]

        Returns:
            路径光滑度值，值越小表示越光滑
        """
        if len(path) < 3:
            return 0.0

        # 计算每步的移动方向 (dx, dy)
        directions = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            directions.append((dx, dy))

        # 计算方向变化的累积
        # 理想对角线移动: (1, 1)
        # 水平移动: (1, 0) 或 垂直移动: (0, 1) 表示形变
        smoothness = 0.0
        diagonal_count = 0

        for dx, dy in directions:
            if dx == 1 and dy == 1:
                # 对角线移动，最光滑
                diagonal_count += 1
            elif dx == 0 or dy == 0:
                # 水平或垂直移动，增加不光滑度
                smoothness += 1.0
            else:
                # 其他情况
                smoothness += abs(dx - dy) * 0.5

        # 归一化：除以路径长度
        normalized_smoothness = smoothness / len(path)

        return normalized_smoothness

    def extract_features(self, query: np.ndarray, template: np.ndarray) -> Dict[str, float]:
        """
        一次性提取所有特征

        Args:
            query: 查询信号
            template: 模板信号

        Returns:
            特征字典: {
                'pearson': 皮尔逊相关系数,
                'dtw_distance': DTW距离,
                'path_smoothness': 路径光滑度
            }
        """
        # 计算DTW距离和路径
        dtw_distance, path = self.compute_dtw_with_path(query, template)

        # 计算皮尔逊相关系数
        pearson_corr = self.compute_pearson(query, template)

        # 计算路径光滑度
        path_smoothness = self.compute_path_smoothness(path)

        return {
            'pearson': pearson_corr,
            'dtw_distance': dtw_distance,
            'path_smoothness': path_smoothness
        }

    def extract_features_batch(self, query: np.ndarray, templates: List[np.ndarray]) -> Tuple[Dict[str, float], np.ndarray]:
        """
        批量匹配模板并提取最佳匹配的特征

        Args:
            query: 查询信号
            templates: 模板列表

        Returns:
            (features, best_template): 最佳匹配的特征和对应模板
        """
        best_distance = float('inf')
        best_features = None
        best_template = None
        best_path = None

        for template in templates:
            distance, path = self.compute_dtw_with_path(query, template)

            if distance < best_distance:
                best_distance = distance
                best_path = path
                best_template = template

        # 计算最佳匹配的所有特征
        if best_template is not None:
            pearson_corr = self.compute_pearson(query, best_template)
            path_smoothness = self.compute_path_smoothness(best_path)

            best_features = {
                'pearson': pearson_corr,
                'dtw_distance': best_distance,
                'path_smoothness': path_smoothness
            }

        return best_features, best_template


def test_feature_extractor():
    """测试特征提取器"""
    # 创建测试信号
    t = np.linspace(0, 2*np.pi, 100)
    query = np.sin(t)
    template = np.sin(t + 0.1)  # 稍微有相位差的信号

    # 初始化特征提取器
    extractor = FeatureExtractor()

    # 提取特征
    features = extractor.extract_features(query, template)

    print("Feature Extraction Test:")
    print(f"  Pearson Correlation: {features['pearson']:.4f}")
    print(f"  DTW Distance: {features['dtw_distance']:.4f}")
    print(f"  Path Smoothness: {features['path_smoothness']:.4f}")

    # 验证皮尔逊系数范围
    assert -1 <= features['pearson'] <= 1, "Pearson correlation out of range"
    # 验证路径光滑度非负
    assert features['path_smoothness'] >= 0, "Path smoothness should be non-negative"

    print("All tests passed!")


if __name__ == '__main__':
    test_feature_extractor()
