# An Efficient Approach for Lie Detection using EEG Signals and Ensemble Learning

## Overview

This project implements a comprehensive machine learning pipeline for lie detection using EEG (Electroencephalography) signals. The system employs multiple classification algorithms combined with ensemble methods to achieve robust and accurate predictions.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Methodology](#methodology)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features

- **Multiple ML Algorithms**: Implementation of 6 different classification models
- **Hyperparameter Optimization**: Automated hyperparameter tuning for each model
- **Ensemble Methods**: Three voting strategies for improved accuracy
- **10-Fold Cross-Validation**: Robust evaluation across multiple data splits
- **Comprehensive Visualization**: Bar plots, confusion matrices, and comparative analysis
- **Performance Metrics**: Accuracy, F1-score, precision, recall, and confusion matrices

## Dataset

The dataset consists of EEG signal recordings from 10 subjects (folds) with binary classification:
- **Class 0**: Truth-telling
- **Class 1**: Lying

Each fold contains:
- Training dataset: `Fold_X_Train_Dataset.csv`
- Testing dataset: `Fold_X_Test_Dataset.csv`

## Models Implemented

1. **Support Vector Machine (SVM)** with RBF kernel
2. **Logistic Regression** with L1/L2 regularization
3. **K-Nearest Neighbors (KNN)**
4. **Decision Tree Classifier**
5. **Random Forest Classifier**
6. **Gradient Boosting (XGBoost)**

## Methodology

### 1. Data Preprocessing
- Binary label conversion (1 → 0, 2 → 1)
- Train-validation-test split strategy
- Feature and target separation

### 2. Hyperparameter Tuning
Each model undergoes extensive hyperparameter optimization:
- **SVM**: C and gamma parameters
- **Logistic Regression**: Penalty type, C value, solver
- **KNN**: n_neighbors, weights, algorithm, leaf_size, p-value
- **Decision Tree**: Criterion, max_depth, min_samples_split, etc.
- **Random Forest**: n_estimators, max_features, max_depth, etc.
- **XGBoost**: n_estimators, learning_rate, gamma, subsample, etc.

### 3. Ensemble Methods

Three voting strategies are implemented:

#### Weighted Voting
- Combines predictions using validation accuracy as weights
- Aggregates probability scores from each model
- Threshold at 0.5 for final classification

#### Majority Voting
- Each model votes for a class
- Final prediction based on majority consensus
- Threshold: n//2 models

#### Unanimous Voting
- Conservative approach
- Predicts positive class only if all models agree
- Uses minimum prediction across all models

### 4. Evaluation
- 10-fold cross-validation
- Metrics: Accuracy, F1-score, Confusion Matrix
- Comparative analysis across subjects and ensemble methods

## Requirements

```
pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd eeg-lie-detection

# Install required packages
pip install pandas numpy scikit-learn xgboost seaborn matplotlib

# For Google Colab (if using Colab)
# The notebook automatically handles Google Drive mounting
```

## Results

### Average Accuracy Across 10 Folds

The project evaluates three ensemble configurations (3, 4, and 5 models) with three voting strategies:

- **Weighted Voting**: Typically achieves highest accuracy
- **Majority Voting**: Balanced approach with robust performance
- **Unanimous Voting**: Conservative predictions with high precision

### Visualization Outputs

The project generates multiple visualizations:
- Individual model performance per subject
- Validation vs. Test accuracy comparisons
- F-measure comparisons
- Ensemble method comparisons
- Subject-wise accuracy analysis

## Key Functions

- `userModel()`: Train and evaluate a single model
- `userModelTuning()`: Hyperparameter validation across folds
- `WeightedVotingClassification()`: Ensemble with weighted voting
- `MajorityVotingClassification()`: Ensemble with majority voting
- `UnanimousVotingClassification()`: Ensemble with unanimous voting
- `confusionMatrix()`: Visualize confusion matrix
- `barPlot()`: Generate performance bar plots
- `masterBarPlot()`: Compare ensemble methods across subjects

## Performance Optimization

- **Randomized Search**: For faster hyperparameter tuning
- **Validation Strategy**: Leave-one-out cross-validation for hyperparameter selection
- **Ensemble Diversity**: Combines models with different learning paradigms

## Future Enhancements

- [ ] Deep learning models (LSTM, CNN)
- [ ] Feature extraction and selection
- [ ] Real-time prediction capability
- [ ] Additional ensemble strategies (Stacking, Blending)
- [ ] Interpretability analysis (SHAP, LIME)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project is for research and educational purposes only. Lie detection systems should be used responsibly and ethically, considering privacy and consent regulations.
