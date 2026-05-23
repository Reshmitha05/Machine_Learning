# Regression & Forecasting Under Distribution Shift

**ELL 409: Introduction to Machine Learning — Course Project 1**

---

## What I Did

A supervised regression competition split into four structurally different parts (A–D), each requiring a different modeling strategy. I built **four separate pipelines** using scikit-learn, all within a single notebook. Evaluated by weighted Mean Absolute Error (lower is better).

Final score = `0.20×MAE_A + 0.25×MAE_B + 0.35×MAE_C + 0.20×MAE_D`

---

## The Problem

| Part | Challenge | Data Type |
|------|-----------|-----------|
| A | Linear signal with correlated covariates | Static regression |
| B | Nonlinear interaction active only in a restricted feature region | Static regression |
| C | Panel forecasting with temporal lag features | Time-series (panel) |
| D | Panel forecasting under distribution shift + outlier contamination | Time-series (panel) |

---

## My Approach

### The Key Trick — Combine Train + Validation Before Final Fit

After model selection on `valid.csv`, I concatenated `train.csv` + `valid.csv` and retrained on the full combined set before generating test predictions. This is explicitly allowed by the competition rules and gives the model more temporal coverage, which matters especially for Parts C and D.

```python
combined_df = pd.concat([train_df, valid_df], ignore_index=True)
```

### Shared Preprocessing

All parts share a common preprocessing structure:
- **Numeric features** (`x1–x20`, `z1–z8`, `m1–m6`): median imputation (handles NaNs in `m` columns) → scaling
- **Categorical features** (`dept`, `building_type`, `sensor_class`): constant imputation → one-hot encoding with `handle_unknown='ignore'`
- Built as scikit-learn `Pipeline` + `ColumnTransformer` so there's zero leakage between fit and transform

### Part-Specific Pipelines

**Part A — Ridge Regression**
- Features: all numeric + categorical
- `StandardScaler` + `Ridge(alpha=10.0)`
- The signal is largely linear with correlated covariates, so Ridge's L2 penalty handles multicollinearity cleanly

**Part B — Polynomial + Ridge**
- Features: all numeric + categorical
- `StandardScaler` → `PolynomialFeatures(degree=2)` → `Ridge(alpha=10.0)`
- Part B has nonlinear interactions active in a localized feature region — degree-2 polynomial features let the linear model capture those without switching to a tree-based model
- Ridge regularization prevents overfitting from the feature explosion

**Part C — Linear Forecasting with Lags**
- Features: numeric + time features (`dow`, `month_phase`, `holiday_flag`) + lag features (`y_lag1`, `y_lag7`) + categorical
- `StandardScaler` + `Ridge(alpha=1.0)`
- Lag features carry most of the temporal signal here; lower alpha since the panel data is less noisy

**Part D — Robust Linear Forecasting**
- Same features as Part C
- **`RobustScaler` instead of `StandardScaler`** — key difference for this part
- `RobustScaler` uses median and IQR instead of mean and std, so it's not thrown off by the outliers and distribution shift that characterize Part D
- `Ridge(alpha=10.0)` for stronger regularization

---

## Why These Choices

**Why Ridge over plain linear regression?**
Parts A and B have 20+ continuous features that are correlated. OLS would overfit or be numerically unstable; Ridge shrinks coefficients stably.

**Why not a tree model (XGBoost, RandomForest)?**
The competition structure hints at using methods "appropriate to each part" — polynomial features for Part B, robust scaling for Part D. These are also simpler, faster, and easier to reason about for a course project.

**Why RobustScaler specifically for Part D?**
StandardScaler computes mean and std — both are sensitive to outliers. RobustScaler uses the median and interquartile range, which are resistant to extreme values. When the data has distribution shift and outlier contamination, this makes a measurable difference.

**Why combine train + valid?**
For Parts C and D (panel time series), Part D's test timestamps (120–179) are later in time than Part C's (0–119). Training on both train and valid gives the model exposure to more of the temporal distribution before predicting the test window.

---

## Libraries Used

```
scikit-learn
pandas
numpy
```

---

## How to Run

The notebook was developed on Kaggle. To run locally:

```python
# Replace Kaggle paths with local paths
train_df = pd.read_csv('train.csv')
valid_df  = pd.read_csv('valid.csv')
test_df   = pd.read_csv('test.csv')

# Replace output path
final_submission.to_csv('submission.csv', index=False)
```

Everything else runs as-is.
