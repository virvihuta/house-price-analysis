# House Prices Regression Analysis

End-to-end machine learning workflow for the Ames Housing / House Prices regression problem, implemented in a single Jupyter notebook (main.ipynb).  
The project covers preprocessing, feature engineering, model training, hyperparameter tuning and ensembling.

---

## Getting Started

1. Clone the repository  
2. Place the dataset files in the following structure:

data/train.csv  
data/test.csv  

3. Open and run `main.ipynb` from top to bottom

---

## Environment & Dependencies

Python version: 3.9+

Libraries used:
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- catboost
- jupyter

---

## Installation

Create and activate a virtual environment:

`python -m venv .venv  `
`source .venv/bin/activate   (macOS / Linux)  `
`.venv\Scripts\activate      (Windows)`

Install dependencies:

`pip install -U pip`
`pip install pandas numpy scipy matplotlib seaborn scikit-learn xgboost lightgbm catboost jupyter  `

Run Jupyter:

jupyter notebook  

---

## Methodology

### Exploratory Data Analysis & Outliers
- Scatter plots of features vs. SalePrice
- Z-score inspection for extreme values
- Manual removal of strong outliers using selected Ids

### Missing Values
- Manual filling for domain-based missingness (e.g. no basement, no alley)
- Pipeline-based imputers for remaining missing values

### Feature Engineering
- Creation of derived features (e.g. total number of bathrooms)
- Removal of redundant or low-information columns

### Target Transformation
The target variable is transformed as:

SalePrice = log1p(SalePrice)

Predictions are converted back using:

SalePrice = expm1(prediction)

### Preprocessing Pipeline
Implemented using ColumnTransformer:
- Numeric features: mean imputation + standard scaling
- Ordinal features: most-frequent imputation + ordinal encoding
- Nominal features: most-frequent imputation + one-hot encoding

---

## Models Trained

Baseline and tuned models:
- Linear Regression
- Ridge Regression (GridSearchCV)
- RandomForestRegressor (GridSearchCV)
- GradientBoostingRegressor (GridSearchCV)
- XGBoost Regressor (GridSearchCV)
- LightGBM Regressor (GridSearchCV)
- CatBoost Regressor (GridSearchCV)

Ensemble models:
- VotingRegressor (weighted ensemble)
- StackingRegressor (meta-model ensemble)

---

## Results

Evaluation is performed on a hold-out validation split using the log-transformed target.

Hold-out split performance:

Model | MSE (log space) | RMSE (log space)
----- | --------------- | ----------------
VotingRegressor | 0.01427 | 0.11948
Ridge (best CV) | 0.01500 | 0.12248
Linear Regression | 0.01557 | 0.12479
CatBoost (best CV) | 0.01568 | 0.12521
Gradient Boosting (best CV) | 0.01569 | 0.12525
StackingRegressor | 0.01571 | 0.12535
XGBoost (best CV) | 0.01646 | 0.12829
LightGBM (best CV) | 0.01912 | 0.13828
Random Forest (best CV) | 0.02197 | 0.14821

Note: Since the target is log1p(SalePrice), an RMSE of ~0.12 corresponds to roughly 12–13% average multiplicative error.

---

## Future Improvements

Planned improvements for this project include:

- Refactoring the preprocessing and models into a single end-to-end `Pipeline` to fully eliminate data leakage during evaluation
- Improving validation strategy using cross-validation on the full pipeline
- Adding automated Kaggle submission generation
- Exploring **deep learning approaches with PyTorch**, including:
  - Fully connected neural networks for tabular data
  - Regularization techniques (dropout, batch normalization)
  - Custom loss monitoring and learning-rate scheduling
- Comparing classical ensemble methods with neural network performance
- Improving reproducibility and experiment tracking

These improvements will be implemented iteratively in future versions of the project.

Install
