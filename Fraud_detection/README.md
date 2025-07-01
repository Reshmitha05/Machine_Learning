Transaction Fraud Detection
This applies supervised learning techniques to detect fraudulent online transactions in a highly imbalanced dataset. Dataset:[Kaggle: Online Payments Fraud Detection](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-datase

Models Used: Logistic Regression, Random Forest, XGBoost (with early stopping and GPU acceleration)

Techniques Applied: Data balancing (undersampling), Feature normalization, Hyperparameter tuning (RandomizedSearchCV), Early stopping

Evaluation Metrics: Accuracy, Precision, Recall, F1-score (with special focus on the fraud class), ROC Curve (AUROC: 0.92 for Logistic Regression), Precision-Recall Curve (AUPRC: 0.90+)
