"""
模型训练脚本
Train Models Script

使用模块化代码训练分类器
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import Config, get_config, DataLoader, EnsembleClassifier


def remove_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """使用IQR方法识别异常值"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (data >= lower_bound) & (data <= upper_bound)


def main():
    """主函数"""
    # 获取配置
    config = get_config()
    config.ensure_dirs()

    # 切换到项目根目录
    os.chdir(config.paths.root_dir)

    print("="*60)
    print("Model Training")
    print("="*60)

    # 初始化数据加载器
    data_loader = DataLoader(config)

    # 加载特征
    good_features_path = os.path.join(config.paths.features_dir, 'good_features.npy')
    bad_features_path = os.path.join(config.paths.features_dir, 'bad_features.npy')

    if os.path.exists(good_features_path) and os.path.exists(bad_features_path):
        print("\nLoading features from NPY files...")
        good_features = data_loader.load_features(good_features_path)
        bad_features = data_loader.load_features(bad_features_path)
    else:
        # 如果没有特征文件，从SQI文件加载
        print("\nNPY features not found, loading from SQI files...")
        good_sqi = data_loader.load_sqi_from_txt('SQIs_Good.txt')
        bad_sqi = data_loader.load_sqi_from_txt('SQIs_Bad.txt')

        # 去除异常值
        good_mask = remove_outliers_iqr(good_sqi)
        bad_mask = remove_outliers_iqr(bad_sqi)

        good_features = good_sqi[good_mask].reshape(-1, 1)
        bad_features = bad_sqi[bad_mask].reshape(-1, 1)

    print(f"\nGood features shape: {good_features.shape}")
    print(f"Bad features shape: {bad_features.shape}")

    # 构建训练集
    X = np.vstack([good_features, bad_features])
    y = np.array([1] * len(good_features) + [0] * len(bad_features))

    print(f"\nTotal samples: {len(y)}")
    print(f"  Good (y=1): {sum(y==1)}")
    print(f"  Bad (y=0): {sum(y==0)}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.classifier.test_size,
        random_state=config.classifier.random_state,
        stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 训练分类器
    print("\n" + "-"*40)
    print("Training Ensemble Classifier")
    print("-"*40)

    ensemble = EnsembleClassifier(weights=config.classifier.weights)
    ensemble.fit(X_train, y_train)

    # 评估
    print("\n" + "-"*40)
    print("Evaluation Results")
    print("-"*40)

    train_results = ensemble.evaluate(X_train, y_train)
    test_results = ensemble.evaluate(X_test, y_test)

    print("\nTraining Set:")
    for clf_name, metrics in train_results.items():
        print(f"\n  {clf_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")

    print("\nTest Set:")
    for clf_name, metrics in test_results.items():
        print(f"\n  {clf_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")

    # 保存模型
    print("\n" + "-"*40)
    print("Saving Models")
    print("-"*40)

    ensemble.save(config.paths.models_dir)

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == '__main__':
    main()
