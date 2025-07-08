### Housing Price Prediction

This project is based on examples from the book:

**[Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow (2nd Edition)](https://github.com/ageron/handson-ml2)**  
Author: **Aurélien Géron**

This implementation follows the structure and exercises from Chapter 2 (End-to-End ML Project) of the book.

### Dataset
- **Source**: California housing data from 1990 census
- **Target**: `median_house_value`
- **Features** include:
  - Median income
  - Housing age
  - Population
  - Rooms per household
  - Ocean proximity (categorical), and others..
    
### **Model Training**
Models used
  - `LinearRegression`
  - `DecisionTreeRegressor`
  - `RandomForestRegressor`
  - `SVR` (Support Vector Regressor)

### Techniques Applied

- **Feature engineering** using a custom transformer (`CombinedAttributesAdder`) to create new features
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`
- **Missing value handling** using `SimpleImputer` with the `"median"` strategy
- **Feature scaling** with `StandardScaler` for normalized numerical inputs
- **Categorical encoding** via `OneHotEncoder` for the `ocean_proximity` feature
- **Data pipelines** built using `Pipeline` and `ColumnTransformer` to streamline preprocessing
- **Model training** with multiple regressors:
  - `LinearRegression`
  - `DecisionTreeRegressor`
  - `RandomForestRegressor`
  - `SVR` (Support Vector Regressor)
- **Hyperparameter tuning** using `GridSearchCV` for model selection and optimization
- **Cross-validation** with `cross_val_score` to estimate generalization error robustly

### Evaluation Metrics

Since this is a **regression problem**, the following metrics were used

- **RMSE (Root Mean Squared Error)**: Measures the standard deviation of prediction errors
- **MAE (Mean Absolute Error)**: Captures average absolute errors between predictions and actual values
- **R² Score (Coefficient of Determination)**: Reflects how much variance in the target is explained by the model
- **95% Confidence Interval for RMSE**: Estimated using `scipy.stats.t.interval`, `stats.sem`, and z/t-score methods
