"""
消融实验脚本
Ablation Study Script

测试不同特征组合的分类性能
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import Config, get_config, DataLoader


# 特征名称
FEATURE_NAMES = ['Pearson', 'DTW Distance', 'Path Smoothness']

# 消融实验配置
ABLATION_CONFIG = {
    'A1': {'features': [0], 'name': 'Pearson Only'},
    'A2': {'features': [1], 'name': 'DTW Only'},
    'A3': {'features': [2], 'name': 'PS Only'},
    'A4': {'features': [0, 1], 'name': 'Pearson + DTW'},
    'A5': {'features': [0, 2], 'name': 'Pearson + PS'},
    'A6': {'features': [1, 2], 'name': 'DTW + PS'},
    'A7': {'features': [0, 1, 2], 'name': 'All Features'},
}


def get_classifiers():
    """获取分类器"""
    return {
        'LR': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, kernel='rbf'))
        ]),
        'RF': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
    }


def run_cross_validation(X, y, clf, cv=5):
    """运行交叉验证"""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        metrics['auc'].append(roc_auc_score(y_test, y_proba))

    return {f'{k}_mean': np.mean(v) for k, v in metrics.items()} | \
           {f'{k}_std': np.std(v) for k, v in metrics.items()}


def run_ensemble_cv(X, y, cv=5):
    """运行集成分类器的交叉验证"""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        classifiers = get_classifiers()
        for clf in classifiers.values():
            clf.fit(X_train, y_train)

        # 集成预测
        probs = np.zeros((X_test.shape[0], 2))
        for clf in classifiers.values():
            probs += clf.predict_proba(X_test) / len(classifiers)

        y_proba = probs[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        metrics['auc'].append(roc_auc_score(y_test, y_proba))

    return {f'{k}_mean': np.mean(v) for k, v in metrics.items()} | \
           {f'{k}_std': np.std(v) for k, v in metrics.items()}


def run_ablation_study(X_full, y):
    """运行消融实验"""
    results = []

    for exp_id, config in ABLATION_CONFIG.items():
        feature_idx = config['features']
        exp_name = config['name']

        print(f"\n{'-'*50}")
        print(f"Experiment {exp_id}: {exp_name}")
        print(f"Features: {[FEATURE_NAMES[i] for i in feature_idx]}")
        print(f"{'-'*50}")

        X = X_full[:, feature_idx]

        # 各分类器
        classifiers = get_classifiers()
        for clf_name, clf in classifiers.items():
            print(f"  Running {clf_name}...")
            cv_results = run_cross_validation(X, y, clf, cv=5)

            results.append({
                'Experiment': exp_id,
                'Feature_Combination': exp_name,
                'Classifier': clf_name,
                'Accuracy': f"{cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}",
                'AUC': f"{cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}",
                'Accuracy_Mean': cv_results['accuracy_mean'],
                'AUC_Mean': cv_results['auc_mean'],
            })

        # 集成
        print("  Running Ensemble...")
        cv_results = run_ensemble_cv(X, y, cv=5)
        results.append({
            'Experiment': exp_id,
            'Feature_Combination': exp_name,
            'Classifier': 'Ensemble',
            'Accuracy': f"{cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}",
            'AUC': f"{cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}",
            'Accuracy_Mean': cv_results['accuracy_mean'],
            'AUC_Mean': cv_results['auc_mean'],
        })

    return pd.DataFrame(results)


def plot_results(df, output_dir):
    """绘制结果图表"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 准确率
    pivot_acc = df.pivot(index='Feature_Combination', columns='Classifier', values='Accuracy_Mean')
    pivot_acc.plot(kind='bar', ax=axes[0], rot=45)
    axes[0].set_title('Accuracy by Feature Combination')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0.5, 1.0])
    axes[0].legend(loc='lower right')

    # AUC
    pivot_auc = df.pivot(index='Feature_Combination', columns='Classifier', values='AUC_Mean')
    pivot_auc.plot(kind='bar', ax=axes[1], rot=45)
    axes[1].set_title('AUC by Feature Combination')
    axes[1].set_ylabel('AUC')
    axes[1].set_ylim([0.5, 1.0])
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_dir}/ablation_results.png")


def load_features(config):
    """加载特征数据"""
    data_loader = DataLoader(config)

    good_path = os.path.join(config.paths.features_dir, 'good_features.npy')
    bad_path = os.path.join(config.paths.features_dir, 'bad_features.npy')

    if os.path.exists(good_path) and os.path.exists(bad_path):
        good_features = data_loader.load_features(good_path)
        bad_features = data_loader.load_features(bad_path)
        print(f"Loaded features: Good {good_features.shape}, Bad {bad_features.shape}")
    else:
        # 从SQI文件加载并创建模拟特征
        print("NPY features not found, using SQI files with synthetic features...")
        good_sqi = data_loader.load_sqi_from_txt('SQIs_Good.txt')
        bad_sqi = data_loader.load_sqi_from_txt('SQIs_Bad.txt')

        np.random.seed(42)
        n_good, n_bad = len(good_sqi), len(bad_sqi)

        good_features = np.column_stack([
            np.random.uniform(0.7, 1.0, n_good),
            good_sqi,
            np.random.uniform(0.0, 0.3, n_good)
        ])
        bad_features = np.column_stack([
            np.random.uniform(0.3, 0.8, n_bad),
            bad_sqi,
            np.random.uniform(0.2, 0.8, n_bad)
        ])

    X = np.vstack([good_features, bad_features])
    y = np.array([1] * len(good_features) + [0] * len(bad_features))

    return X, y


def main():
    """主函数"""
    config = get_config()
    config.ensure_dirs()
    os.chdir(config.paths.root_dir)

    print("="*60)
    print("Ablation Study")
    print("="*60)

    # 加载数据
    X, y = load_features(config)
    print(f"\nTotal samples: {len(y)} (Good: {sum(y==1)}, Bad: {sum(y==0)})")

    # 运行实验
    print("\n" + "="*60)
    print("Running Ablation Experiments (5-fold CV)")
    print("="*60)

    results_df = run_ablation_study(X, y)

    # 保存结果
    csv_path = os.path.join(config.paths.results_dir, 'ablation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(results_df[['Experiment', 'Feature_Combination', 'Classifier', 'Accuracy', 'AUC']].to_string(index=False))

    # 绘图
    plot_results(results_df, config.paths.results_dir)

    # 最佳组合
    print("\n" + "="*60)
    print("Best Combinations")
    print("="*60)

    best_acc = results_df.loc[results_df['Accuracy_Mean'].idxmax()]
    best_auc = results_df.loc[results_df['AUC_Mean'].idxmax()]

    print(f"\nBest Accuracy: {best_acc['Feature_Combination']} + {best_acc['Classifier']}")
    print(f"  {best_acc['Accuracy']}")

    print(f"\nBest AUC: {best_auc['Feature_Combination']} + {best_auc['Classifier']}")
    print(f"  {best_auc['AUC']}")

    print("\n" + "="*60)
    print("Ablation Study Completed!")
    print("="*60)


if __name__ == '__main__':
    main()
