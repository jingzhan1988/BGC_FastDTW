"""
特征生成脚本
Generate Features Script

使用模块化代码批量生成特征
"""

import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import Config, get_config, DataLoader, DTWMatcher


def main():
    """主函数"""
    # 获取配置
    config = get_config()
    config.ensure_dirs()

    # 切换到项目根目录
    os.chdir(config.paths.root_dir)

    print("="*60)
    print("Feature Generation")
    print("="*60)

    # 初始化组件
    data_loader = DataLoader(config)
    dtw_matcher = DTWMatcher(config)

    # 加载模板
    print("\nLoading templates...")
    dtw_matcher.load_templates()

    # 处理好信号
    print("\n" + "-"*40)
    print("Processing good signals...")
    print("-"*40)

    try:
        good_signals = data_loader.load_good_signals()
        good_features = dtw_matcher.get_feature_matrix(good_signals, truncate=True)

        output_path = os.path.join(config.paths.features_dir, 'good_features.npy')
        data_loader.save_features(good_features, output_path)
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    # 处理坏信号
    print("\n" + "-"*40)
    print("Processing bad signals...")
    print("-"*40)

    try:
        bad_signals = data_loader.load_bad_signals()
        bad_features = dtw_matcher.get_feature_matrix(bad_signals, truncate=True)

        output_path = os.path.join(config.paths.features_dir, 'bad_features.npy')
        data_loader.save_features(bad_features, output_path)
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    print("\n" + "="*60)
    print("Feature generation completed!")
    print("="*60)


if __name__ == '__main__':
    main()
