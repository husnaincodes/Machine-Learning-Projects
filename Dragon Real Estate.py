
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load



def main():
   
    # Load and explore data
   
    # Skip the second line, which contains column descriptions not data.
    housing_data = pd.read_csv('housing_data.csv', skiprows=[1])

    print(housing_data.head())
    housing_data.info()
    print(housing_data['CHAS'].value_counts())
    print(housing_data.describe())

    housing_data.hist(bins=50, figsize=(20, 15))
    plt.show()

 
    # Training and Testing Data Split
    
    # Simple random split (kept for reference)
    train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)
    print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

    # Stratified split on CHAS to preserve its distribution in train/test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_indices, test_indices in splitter.split(housing_data, housing_data['CHAS']):
        train_set = housing_data.iloc[train_indices]
        test_set = housing_data.iloc[test_indices]

    test_set.info()
    print(test_set.describe())
    print(test_set['CHAS'].value_counts())

    housing_data = train_set.copy()


    # Correlation Matrix

    corr_matrix = housing_data.corr()
    print(corr_matrix['MEDV'].sort_values(ascending=False))

    attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
    scatter_matrix(housing_data[attributes], figsize=(12, 8))
    plt.show()

   
    # Trying Out Attribute Combinations
    
    housing_data['TAXRM'] = housing_data['TAX'] / housing_data['RM']
    print(housing_data.head())

    corr_matrix = housing_data.corr()
    print(corr_matrix['MEDV'].sort_values(ascending=False))

    housing_data.plot(kind='scatter', x='TAXRM', y='MEDV', alpha=0.8)
    plt.show()
    housing_data.plot(kind='scatter', x='RM', y='MEDV', alpha=0.8)
    plt.show()
    print(housing_data.describe())

   
    # Pipeline
    
    my_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    housing_data_num_tr = my_pipeline.fit_transform(housing_data.drop('MEDV', axis=1))
    housing_labels = housing_data['MEDV'].copy()
    print(housing_data_num_tr.shape)

   
    # Selecting and training the model
   
    # Other candidates tried: DecisionTreeRegressor, LinearRegression
    model = RandomForestRegressor()
    model.fit(housing_data_num_tr, housing_labels)

    some_data = housing_data.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = my_pipeline.transform(some_data.drop('MEDV', axis=1))
    print("Predictions:", model.predict(some_data_prepared))
    print("Labels:", list(some_labels))

  
    # Evaluating the model (training set)
  
    housing_predictions = model.predict(housing_data_num_tr)
    mse = mean_squared_error(housing_labels, housing_predictions)
    rmse = np.sqrt(mse)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)

    # Better evaluation - cross validation
   
    scores = cross_val_score(
        model, housing_data_num_tr, housing_labels,
        scoring='neg_mean_squared_error', cv=10
    )
    rmse_scores = np.sqrt(-scores)
    print("Scores:", rmse_scores)
    print("Mean:", rmse_scores.mean())
    print("Standard deviation:", rmse_scores.std())

  
    # Save the model

    dump(model, 'housing_model.joblib')


    # Final evaluation on the test set
   
    X_test = test_set.drop("MEDV", axis=1)
    Y_test = test_set["MEDV"].copy()
    X_test_prepared = my_pipeline.transform(X_test)
    final_predictions = model.predict(X_test_prepared)
    final_mse = mean_squared_error(Y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("Final Mean Squared Error:", final_mse)
    print("Final Root Mean Squared Error:", final_rmse)


if __name__ == "__main__":
    main()