# LELEC2870 Machine Learning Project

## Heart Failure Prediction in Smurf Society

A machine learning project to predict the risk of heart failure in Smurfs using clinical data and heart scan images.

### Dataset

The dataset includes:
- **Tabular data**: Clinical measurements (age, blood pressure, cholesterol, lifestyle habits, etc.)
- **Image data**: 48×48 pixel heart scan images from Smurf-sized MRI
- **Target variable**: Risk of developing heart failure within 10 years

Data files:
- `X_train.csv` / `X_test.csv`: Training and test features
- `y_train.csv` / `y_test.csv`: Training and test labels
- `Img_train/` / `Img_test/`: Heart scan images
- `X.csv`: Unlabeled data for final predictions

### Project Structure

**Part 1 - Linear Model**: Data preprocessing, feature selection, and baseline linear regression model

**Part 2 - Nonlinear Models**: Comparison of various nonlinear models (tree-based, neural networks, etc.) with hyperparameter tuning

**Part 3 - Image Integration**: Extract features from heart scans using deep learning and combine with tabular data

**Part 4 - Data Analysis**: Exploratory analysis to identify risk factors and at-risk groups

### Requirements

- Python 3.x
- Common ML libraries: scikit-learn, pandas, numpy
- Deep learning: pytorch (or tensorflow/keras)
- Visualization: matplotlib, seaborn

### Usage

1. Place the dataset files in the project directory
2. Run preprocessing and model training scripts for each part
3. Generate predictions on unlabeled data using the best model

### Evaluation Metric

Root Mean Square Error (RMSE) on unlabeled test set

---

*Academic Year 2025-2026*

