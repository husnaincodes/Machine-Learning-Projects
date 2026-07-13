# Machine-Learning-Projects
# 🐉 Dragon Real Estate - Price Predictor

A machine learning project that predicts median house prices using the classic **Boston Housing dataset**. The pipeline covers data exploration, stratified train/test splitting, feature engineering, model training with Random Forest Regression, and evaluation via cross-validation.

## 📋 Overview

This project walks through a complete end-to-end ML workflow:

1. **Data Exploration** — inspect structure, distributions, and summary statistics
2. **Train/Test Split** — stratified split on the `CHAS` feature to preserve its distribution
3. **Correlation Analysis** — identify which features most strongly predict `MEDV` (median home value)
4. **Feature Engineering** — derive a new `TAXRM` (tax-to-rooms ratio) feature
5. **Preprocessing Pipeline** — standardize features using `StandardScaler`
6. **Model Training** — fit a `RandomForestRegressor`
7. **Evaluation** — measure performance with RMSE and 10-fold cross-validation
8. **Model Persistence** — save the trained model with `joblib`
9. **Final Testing** — evaluate on the held-out test set

## 📊 Dataset

The project uses the Boston Housing dataset (`housing_data.csv`), which contains 506 records of housing data across Boston suburbs, with features including:

| Feature | Description |
|---------|-------------|
| `CRIM` | Per capita crime rate by town |
| `ZN` | Proportion of residential land zoned for large lots |
| `INDUS` | Proportion of non-retail business acres per town |
| `CHAS` | Charles River dummy variable (1 if tract bounds river, else 0) |
| `NOX` | Nitric oxide concentration |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built before 1940 |
| `DIS` | Weighted distance to employment centers |
| `RAD` | Index of accessibility to radial highways |
| `TAX` | Property tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | Proportion of Black residents by town |
| `LSTAT` | % lower status of the population |
| `MEDV` | Median value of owner-occupied homes (target variable) |

> **Note:** The CSV's second row contains column descriptions rather than data, so it's skipped on load (`skiprows=[1]`).

## 🛠️ Tech Stack

- **Python 3**
- **pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **scikit-learn** — preprocessing, modeling, and evaluation
- **joblib** — model serialization

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/husnaincodes/dragon-real-estate.git
cd dragon-real-estate
pip install pandas numpy matplotlib scikit-learn joblib
```

## 🚀 Usage

Make sure `housing_data.csv` is in the project directory, then run:

```bash
python dragon_real_estate.py
```

This will:
- Print data summaries and correlation statistics to the console
- Display histogram and scatter plots
- Train the model and print RMSE / cross-validation scores
- Save the trained model as `housing_model.joblib`

### Loading the saved model

```python
from joblib import load
model = load('housing_model.joblib')
predictions = model.predict(prepared_data)
```

## 🔍 Methodology

**Stratified Splitting:** Rather than a purely random train/test split, `StratifiedShuffleSplit` is used on the `CHAS` feature to ensure both the training and test sets have a representative proportion of river-adjacent tracts.

**Feature Engineering:** A `TAXRM` feature (tax rate divided by average room count) is engineered to capture the relationship between property tax burden and property size, which showed correlation with `MEDV`.

**Model Choice:** `RandomForestRegressor` was selected after comparing against `DecisionTreeRegressor` and `LinearRegression` (see commented-out code in the script), offering better generalization through ensembling.

**Evaluation:** Beyond a simple train-set RMSE, 10-fold cross-validation is used to get a more reliable estimate of model performance and reduce the risk of overfitting to a single split.

## 📈 Results

The model outputs:
- Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) on the training set
- Cross-validation RMSE scores (mean and standard deviation) across 10 folds
- Final RMSE on the held-out test set

*(Run the script to see current metrics — results depend on the specific train/test split and random forest configuration.)*

## 📁 Project Structure

```
Machine Learning Project/
├── dragon_real_estate.py    # Main pipeline script
├── housing_data.csv         # Boston Housing dataset
├── housing_model.joblib     # Saved trained model (generated after running)
└── README.md                 # Project documentation
```

## 🔮 Future Improvements

- Hyperparameter tuning via `GridSearchCV` or `RandomizedSearchCV`
- Handling of missing values with `SimpleImputer` in the pipeline
- Additional feature engineering (e.g., polynomial features)
- Comparison against other regressors (Gradient Boosting, XGBoost)
- A simple web interface or API for making predictions on new data

## 👤 Author

**Husnain** — [@husnaincodes](https://github.com/husnaincodes)
