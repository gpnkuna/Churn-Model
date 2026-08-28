# Customer Churn Prediction

A logistic regression model that predicts whether a telecom customer will churn, built on the IBM Telco Customer Churn dataset.

## Overview

This notebook walks through a full classification pipeline: loading raw customer data, cleaning it, engineering features, training a logistic regression model, and evaluating its performance on held-out data.

## Tech Stack

- **Python 3**
- **pandas** = data loading, cleaning, transformation
- **scikit-learn** = `StandardScaler`, `OneHotEncoder`, `train_test_split`, `LogisticRegression`, evaluation metrics
- **Jupyter Notebook** = development environment

## Dataset

The dataset (`../data/dataset.xlsx`) contains 7,043 customer records with demographic info, account details, and service subscriptions. The target variable is `Churn Value` (1 = customer churned, 0 = customer stayed).

**Columns removed during cleaning:**
- `CustomerID`, `Count`, `Lat Long`, `Latitude`, `Longitude`, identifiers or redundant geographic detail
- `Churn Reason` , over 50% missing
- `Churn Score`, `Churn Label`, `CLTV` , dropped to prevent data leakage (these are derived from or directly encode the target, so including them would let the model "see" the answer)

**Feature groups:**
- Numeric: `Zip Code`, `Tenure Months`, `Monthly Charges`, `Total Charges`
- Categorical: remaining 19 columns (Contract, Payment Method, Internet Service, etc.), one-hot encoded

## Pipeline

1. **Load & clean** = drop irrelevant/leaky columns, convert `Total Charges` to numeric, fill missing categorical values
2. **Split first** = `train_test_split` (80/20) happens *before* any scaling or encoding, to avoid the test set influencing preprocessing
3. **Scale** = `StandardScaler` fit on training data only, applied to both train and test
4. **Encode** = `OneHotEncoder` fit on training data only, applied to both train and test
5. **Train** = `LogisticRegression` (scikit-learn defaults)
6. **Evaluate** = accuracy, confusion matrix, precision/recall/F1

## Skills Learned

- **Identifying data leakage** = recognizing when a feature (like a pre-computed churn score, or a categorical restatement of the target) gives a model unfair access to the answer
- **Correct train/test split discipline** = fitting preprocessing steps (scaling, encoding) on training data only, then applying them to test data, to avoid test-set information leaking into training
- **Debugging silent failures** = tracing why one-hot encoded columns disappeared from a feature matrix due to how a sparse matrix was wrapped in a DataFrame
- **Data type correction** = diagnosing why a numeric column (`Total Charges`) was being read as text, and fixing it with `pd.to_numeric`
- **Index alignment** = understanding why `pd.concat` requires matching indices, and how to preserve them after transformations that return raw arrays
- **Evaluation beyond accuracy** = using confusion matrices and classification reports to properly assess a model on an imbalanced dataset
