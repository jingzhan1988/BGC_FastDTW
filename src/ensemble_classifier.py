"""
集成分类器模块
Ensemble Classifier Module

实现LR、SVM、RF三分类器的加权集成
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class BaseClassifier:
    """分类器基类"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        """训练模型"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: str) -> None:
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)

    def load(self, path: str) -> 'BaseClassifier':
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_fitted = True
        return self


class LRClassifier(BaseClassifier):
    """逻辑回归分类器"""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        super().__init__()
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear')


class SVMClassifier(BaseClassifier):
    """SVM分类器"""

    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: str = 'scale'):
        super().__init__()
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)


class RFClassifier(BaseClassifier):
    """随机森林分类器"""

    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )


class EnsembleClassifier:
    """加权集成分类器"""

    def __init__(self, weights: List[float] = None):
        """
        初始化集成分类器

        Args:
            weights: 三个分类器的权重 [w_lr, w_svm, w_rf]
                    默认等权重 [1/3, 1/3, 1/3]
        """
        self.classifiers = {
            'LR': LRClassifier(),
            'SVM': SVMClassifier(),
            'RF': RFClassifier()
        }

        if weights is None:
            self.weights = [1/3, 1/3, 1/3]
        else:
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleClassifier':
        """
        训练所有分类器

        Args:
            X: 特征矩阵 shape: (n_samples, n_features)
            y: 标签数组 shape: (n_samples,)

        Returns:
            self
        """
        print("Training ensemble classifier...")

        for name, clf in self.classifiers.items():
            print(f"  Training {name}...")
            clf.fit(X, y)

        self.is_fitted = True
        print("Training completed!")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        加权平均预测概率

        Args:
            X: 特征矩阵

        Returns:
            概率矩阵 shape: (n_samples, 2)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted yet")

        probs = np.zeros((X.shape[0], 2))

        for (name, clf), w in zip(self.classifiers.items(), self.weights):
            probs += w * clf.predict_proba(X)

        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征矩阵
            threshold: 分类阈值

        Returns:
            预测类别数组
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        获取各个分类器的单独预测

        Args:
            X: 特征矩阵

        Returns:
            各分类器预测结果字典
        """
        predictions = {}
        for name, clf in self.classifiers.items():
            predictions[name] = {
                'proba': clf.predict_proba(X),
                'pred': clf.predict(X)
            }
        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        评估所有分类器性能

        Args:
            X: 特征矩阵
            y: 真实标签

        Returns:
            评估结果字典
        """
        results = {}

        # 评估各个分类器
        for name, clf in self.classifiers.items():
            y_pred = clf.predict(X)
            y_proba = clf.predict_proba(X)[:, 1]

            results[name] = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0),
                'auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
            }

        # 评估集成分类器
        y_pred_ensemble = self.predict(X)
        y_proba_ensemble = self.predict_proba(X)[:, 1]

        results['Ensemble'] = {
            'accuracy': accuracy_score(y, y_pred_ensemble),
            'precision': precision_score(y, y_pred_ensemble, zero_division=0),
            'recall': recall_score(y, y_pred_ensemble, zero_division=0),
            'f1': f1_score(y, y_pred_ensemble, zero_division=0),
            'auc': roc_auc_score(y, y_proba_ensemble) if len(np.unique(y)) > 1 else 0.0
        }

        return results

    def save(self, dir_path: str) -> None:
        """
        保存所有模型

        Args:
            dir_path: 保存目录
        """
        import os
        os.makedirs(dir_path, exist_ok=True)

        for name, clf in self.classifiers.items():
            clf.save(os.path.join(dir_path, f'{name}.pickle'))

        # 保存权重
        with open(os.path.join(dir_path, 'weights.pickle'), 'wb') as f:
            pickle.dump(self.weights, f)

        print(f"Models saved to {dir_path}")

    def load(self, dir_path: str) -> 'EnsembleClassifier':
        """
        加载所有模型

        Args:
            dir_path: 模型目录

        Returns:
            self
        """
        import os

        for name, clf in self.classifiers.items():
            clf.load(os.path.join(dir_path, f'{name}.pickle'))

        # 加载权重
        with open(os.path.join(dir_path, 'weights.pickle'), 'rb') as f:
            self.weights = pickle.load(f)

        self.is_fitted = True
        print(f"Models loaded from {dir_path}")

        return self


def test_ensemble_classifier():
    """测试集成分类器"""
    from sklearn.datasets import make_classification

    print("Ensemble Classifier Test:")

    # 生成测试数据
    X, y = make_classification(
        n_samples=200,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )

    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 初始化并训练
    ensemble = EnsembleClassifier(weights=[1/3, 1/3, 1/3])
    ensemble.fit(X_train, y_train)

    # 评估
    results = ensemble.evaluate(X_test, y_test)

    print("\nResults:")
    for clf_name, metrics in results.items():
        print(f"\n  {clf_name}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")

    # 验证权重和为1
    assert abs(sum(ensemble.weights) - 1.0) < 1e-6, "Weights should sum to 1"

    print("\nTest completed!")


if __name__ == '__main__':
    test_ensemble_classifier()
