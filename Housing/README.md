Housing Price Prediction 
This project is based on examples from the book:

**[Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow (2nd Edition)](https://github.com/ageron/handson-ml2)**  
Author: **Aurélien Géron**

This implementation follows the structure and exercises from Chapter 2 (End-to-End ML Project) of the book.

## Dataset

- **Source**: California housing data from 1990 census
- **Target**: `median_house_value`
- **Features** include:
  - Median income
  - Housing age
  - Population
  - Rooms per household
  - Ocean proximity (categorical), and others..
## **Model Training**
- Models used:
  - `LinearRegression`
  - `DecisionTreeRegressor`
  - `RandomForestRegressor`
  - `SVR` (Support Vector Regressor)
- Cross-validation with `cross_val_score`
- Grid search (`GridSearchCV`) for hyperparameter tuning
