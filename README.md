# Signal Quality Assessment System

A DTW-based template matching system for physiological signal (BCG) quality assessment.

## Features

- **Fast-SDTW Matching**: Dynamic Time Warping with path constraints
- **N-wave Truncation**: Signal matching from start to N-wave for improved accuracy
- **Three-Feature Extraction**:
  - Pearson Correlation Coefficient
  - DTW Distance
  - Path Smoothness (PS)
- **Ensemble Classifier**: Weighted fusion of LR + SVM + RF
- **Ablation Study**: Feature importance analysis support

## Project Structure

```
workstation/
├── main.py                       # Main entry (CLI interface)
├── src/                          # Core modules
│   ├── __init__.py               # Module exports
│   ├── config.py                 # Configuration management
│   ├── data_loader.py            # Data loading utilities
│   ├── signal_processor.py       # Signal preprocessing
│   ├── feature_extractor.py      # Feature extraction
│   ├── dtw_matcher.py            # DTW matching
│   └── ensemble_classifier.py    # Ensemble classifier
├── scripts/                      # Scripts
│   ├── generate_features.py      # Feature generation
│   ├── train_models.py           # Model training
│   └── ablation_study.py         # Ablation study
├── data/                         # Data directory
│   ├── single.mat                # Template library
│   ├── for_model/                # Training data
│   ├── kexing/                   # Feasibility training data
│   ├── kexing_test/              # Feasibility test data
│   └── features/                 # Extracted features
├── results/                      # Results directory
│   ├── models/                   # Trained models
│   └── ablation_results.csv      # Ablation study results
├── figures/                      # Visualization plots
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

## Module Description

| Module | Description |
|--------|-------------|
| `config.py` | Centralized configuration for paths, DTW params, classifier params |
| `data_loader.py` | Load MAT files, templates, SQI data |
| `signal_processor.py` | Wavelet filtering, energy envelope, N-wave detection, cardiac cycle segmentation |
| `feature_extractor.py` | Extract Pearson correlation, DTW distance, path smoothness |
| `dtw_matcher.py` | Adaptive template selection, Fast-SDTW matching, N-wave truncation |
| `ensemble_classifier.py` | LR/SVM/RF classifiers and weighted ensemble |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Command Line Interface

```bash
# Run full training pipeline
python main.py --train

# Extract features only
python main.py --extract-features

# Evaluate trained models
python main.py --evaluate

# Run ablation study
python main.py --ablation
```

### Option 2: Modular API

```python
from src import DTWMatcher, EnsembleClassifier, DataLoader, get_config

# Get configuration
config = get_config()

# Load data
loader = DataLoader(config)
signals = loader.load_good_signals()

# DTW matching and feature extraction
matcher = DTWMatcher(config)
matcher.load_templates()
features = matcher.get_feature_matrix(signals)

# Train classifier
classifier = EnsembleClassifier()
classifier.fit(X_train, y_train)
probs = classifier.predict_proba(X_test)
```

### Option 3: Run Scripts

```bash
# Generate features
python scripts/generate_features.py

# Train models
python scripts/train_models.py

# Ablation study
python scripts/ablation_study.py
```

## Algorithm Pipeline

```
┌─────────┐    ┌──────────┐    ┌───────────────┐    ┌───────────┐    ┌──────────┐
│  Input  │ → │ Adaptive │ → │  Fast-SDTW    │ → │  Feature  │ → │ Ensemble │
│ Signal  │    │ Template │    │  Matching     │    │Extraction │    │Classifier│
│         │    │Selection │    │(N-wave Trunc.)│    │           │    │          │
└─────────┘    └──────────┘    └───────────────┘    └───────────┘    └──────────┘
                                                          │               │
                                                    ┌─────┴─────┐   ┌─────┴─────┐
                                                    │ Pearson   │   │ LR        │
                                                    │ DTW Dist. │   │ SVM       │
                                                    │ Path Smooth│   │ RF        │
                                                    └───────────┘   └─────┬─────┘
                                                                          │
                                                                   Weighted Average
                                                                    Probability
```

## Configuration

All parameters are managed centrally in `src/config.py`:

### DTW Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius` | 1 | DTW search radius (path constraint) |
| `top_n` | 200 | Number of templates for adaptive selection |
| `truncate_to_n_wave` | True | Whether to truncate signal at N-wave |

### Classifier Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `weights` | [1/3, 1/3, 1/3] | Weights for LR/SVM/RF |
| `test_size` | 0.3 | Test set ratio |
| `cv_folds` | 5 | Cross-validation folds |

### Signal Processing Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `wavelet` | 'sym4' | Wavelet basis function |
| `wavelet_level` | 5 | Wavelet decomposition levels |
| `target_sample_rate` | 100 | Target sampling rate (Hz) |

## Ablation Study

Testing classification performance with different feature combinations:

| Exp. | Features | Description |
|------|----------|-------------|
| A1 | Pearson | Single feature baseline |
| A2 | DTW | Single feature baseline (current method) |
| A3 | PS | Single feature baseline |
| A4 | Pearson + DTW | Two-feature combination |
| A5 | Pearson + PS | Two-feature combination |
| A6 | DTW + PS | Two-feature combination |
| A7 | All | All features (expected best) |

Results are saved to `results/ablation_results.csv`

## Data Format

### Template Library (single.mat)
```python
data['single_cycle']  # Single cardiac cycle template array
```

### Training Data
- Good signals: `data/for_model/good_single_cycle.mat`
- Bad signals: `data/for_model/bad.mat`

### Feature Files (.npy)
```python
# Shape: (n_samples, 3)
# Columns: [pearson, dtw_distance, path_smoothness]
features = np.load('data/features/good_features.npy')
```

## License

MIT License
