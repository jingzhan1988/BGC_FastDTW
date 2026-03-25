"""
信号质量评估系统 - 主入口
Signal Quality Assessment System - Main Entry

提供命令行接口和完整的处理流程
"""

import os
import sys
import argparse
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    Config, get_config,
    DataLoader,
    DTWMatcher,
    EnsembleClassifier,
    SignalProcessor
)


class SignalQualityAssessment:
    """信号质量评估系统"""

    def __init__(self, config: Config = None):
        """
        初始化评估系统

        Args:
            config: 配置对象
        """
        self.config = config or get_config()
        self.config.ensure_dirs()

        self.data_loader = DataLoader(self.config)
        self.dtw_matcher = DTWMatcher(self.config)
        self.signal_processor = SignalProcessor(self.config)
        self.classifier = None

    def load_templates(self):
        """加载模板库"""
        print("Loading templates...")
        self.dtw_matcher.load_templates()
        print(f"Loaded {len(self.dtw_matcher.templates)} templates")

    def extract_features(self, signals: list, truncate: bool = True) -> np.ndarray:
        """
        提取特征矩阵

        Args:
            signals: 信号列表
            truncate: 是否截断到N波

        Returns:
            特征矩阵 (n_samples, 3)
        """
        if self.dtw_matcher.templates is None:
            self.load_templates()

        print(f"Extracting features from {len(signals)} signals...")
        feature_matrix = self.dtw_matcher.get_feature_matrix(signals, truncate=truncate)
        print(f"Feature matrix shape: {feature_matrix.shape}")

        return feature_matrix

    def train_classifier(self, X: np.ndarray, y: np.ndarray):
        """
        训练分类器

        Args:
            X: 特征矩阵
            y: 标签
        """
        print("Training ensemble classifier...")
        self.classifier = EnsembleClassifier(weights=self.config.classifier.weights)
        self.classifier.fit(X, y)
        print("Training completed!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测信号质量

        Args:
            X: 特征矩阵

        Returns:
            预测概率
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not trained. Call train_classifier first.")
        return self.classifier.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        评估模型性能

        Args:
            X: 特征矩阵
            y: 真实标签

        Returns:
            评估结果
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not trained.")
        return self.classifier.evaluate(X, y)

    def save_models(self, path: str = None):
        """保存模型"""
        path = path or self.config.paths.models_dir
        if self.classifier:
            self.classifier.save(path)
            print(f"Models saved to {path}")

    def load_models(self, path: str = None):
        """加载模型"""
        path = path or self.config.paths.models_dir
        self.classifier = EnsembleClassifier()
        self.classifier.load(path)
        print(f"Models loaded from {path}")

    def run_full_pipeline(self):
        """运行完整的训练流程"""
        print("="*60)
        print("Signal Quality Assessment - Full Pipeline")
        print("="*60)

        # 1. 加载模板
        self.load_templates()

        # 2. 加载训练数据
        print("\nLoading training data...")
        good_signals = self.data_loader.load_good_signals()
        bad_signals = self.data_loader.load_bad_signals()

        # 3. 提取特征
        print("\nExtracting features for good signals...")
        good_features = self.extract_features(good_signals)

        print("\nExtracting features for bad signals...")
        bad_features = self.extract_features(bad_signals)

        # 4. 构建训练集
        X = np.vstack([good_features, bad_features])
        y = np.array([1] * len(good_features) + [0] * len(bad_features))
        print(f"\nTotal samples: {len(y)} (Good: {sum(y==1)}, Bad: {sum(y==0)})")

        # 5. 训练分类器
        print("\n" + "-"*40)
        self.train_classifier(X, y)

        # 6. 评估
        print("\n" + "-"*40)
        print("Evaluation Results:")
        results = self.evaluate(X, y)
        for clf_name, metrics in results.items():
            print(f"\n{clf_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        # 7. 保存模型
        print("\n" + "-"*40)
        self.save_models()

        # 8. 保存特征
        self.data_loader.save_features(
            good_features,
            os.path.join(self.config.paths.features_dir, 'good_features.npy')
        )
        self.data_loader.save_features(
            bad_features,
            os.path.join(self.config.paths.features_dir, 'bad_features.npy')
        )

        print("\n" + "="*60)
        print("Pipeline completed!")
        print("="*60)

        return X, y, results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='Signal Quality Assessment System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train              # 运行完整训练流程
  python main.py --extract-features   # 仅提取特征
  python main.py --evaluate           # 评估已训练的模型
        """
    )

    parser.add_argument('--train', action='store_true',
                        help='Run full training pipeline')
    parser.add_argument('--extract-features', action='store_true',
                        help='Extract features only')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained models')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study')

    args = parser.parse_args()

    # 初始化系统
    sqa = SignalQualityAssessment()

    if args.train:
        sqa.run_full_pipeline()

    elif args.extract_features:
        sqa.load_templates()
        good_signals = sqa.data_loader.load_good_signals()
        bad_signals = sqa.data_loader.load_bad_signals()

        good_features = sqa.extract_features(good_signals)
        bad_features = sqa.extract_features(bad_signals)

        sqa.data_loader.save_features(
            good_features,
            os.path.join(sqa.config.paths.features_dir, 'good_features.npy')
        )
        sqa.data_loader.save_features(
            bad_features,
            os.path.join(sqa.config.paths.features_dir, 'bad_features.npy')
        )

    elif args.evaluate:
        sqa.load_models()
        good_features = sqa.data_loader.load_features(
            os.path.join(sqa.config.paths.features_dir, 'good_features.npy')
        )
        bad_features = sqa.data_loader.load_features(
            os.path.join(sqa.config.paths.features_dir, 'bad_features.npy')
        )

        X = np.vstack([good_features, bad_features])
        y = np.array([1] * len(good_features) + [0] * len(bad_features))

        results = sqa.evaluate(X, y)
        for clf_name, metrics in results.items():
            print(f"\n{clf_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

    elif args.ablation:
        # 运行消融实验脚本
        import scripts.ablation_study as ablation
        ablation.main()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
