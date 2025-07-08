### Transaction Fraud Detection

This project applies **supervised machine learning** techniques to detect fraudulent online transactions in a **highly imbalanced dataset**.

The goal is to build models that can accurately detect fraud with minimal false positives, despite class imbalance.

###  Dataset

- **Source**: [Kaggle – Online Payments Fraud Detection](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-datase)
- **Type**: Binary classification
- **Fraud Rate**: Extremely low (<1%), making this a **class imbalance** challenge

### Models Used

- `LogisticRegression`
- `RandomForestClassifier`
- `XGBoostClassifier`  
  ↳ with **early stopping** and **GPU acceleration**

### Techniques Applied

- **Data balancing** using undersampling of the majority (non-fraud) class
- **Feature normalization/standardization** to improve model performance and convergence
- **Hyperparameter tuning** via `RandomizedSearchCV`
- **Early stopping** in XGBoost to avoid overfitting during training
- **Cross-validation** to evaluate model stability and performance on unseen data

###  Evaluation Metrics
Because of extreme class imbalance, accuracy alone is misleading. So, I emphasized

-  **Precision**: How many predicted frauds were actually fraud
-  **Recall**: How many actual frauds were caught
-  **F1 Score**: Balance between precision and recall
-  **AUROC**: Area Under ROC Curve
-  **AUPRC**: Area Under Precision-Recall Curve
