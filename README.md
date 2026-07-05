# Insurance Claim Risk Prediction

A machine learning model that predicts whether an auto insurance policyholder will file 
a claim, built using the Porto Seguro Safe Driver dataset from Kaggle.

## Background
Accurately predicting claim risk is the foundation of insurance pricing. If a company 
charges everyone the same premium, high-risk customers become unprofitable while 
low-risk customers leave for cheaper competitors. This project builds a model that ranks 
policyholders by risk, enabling more accurate premium pricing — a core actuarial task.

## Dataset
The dataset was released by Porto Seguro, a Brazilian auto insurance company, as part of 
a Kaggle competition. It contains 595,000 anonymized policyholders with 57 features 
covering individual characteristics, regional data, and vehicle information. Only 3.6% 
of policyholders filed a claim, creating a significant class imbalance.

Dataset: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

## Methodology
1. **Data Cleaning** — Missing values encoded as -1 were imputed using mode for 
categorical columns and median for continuous columns. Two columns with over 65% 
missingness were dropped entirely.
2. **Feature Encoding** — Categorical columns were one-hot encoded to prevent 
misinterpretation of category labels as ordinal numbers.
3. **Modeling** — Three models were trained in order of complexity: a dummy classifier, 
logistic regression, and XGBoost with hyperparameter tuning via grid search.
4. **Evaluation** — Models were evaluated using AUC-ROC, which measures ranking ability 
rather than raw accuracy — the standard metric for imbalanced insurance datasets.

## Results
| Model | AUC Score |
|---|---|
| Dummy Classifier | 0.500 |
| Logistic Regression | 0.628 |
| XGBoost (tuned) | 0.631 |

The XGBoost model achieved a Gini coefficient of 0.262, close to top results from the 
original Kaggle competition. Feature importance analysis identified regional variables 
and car characteristics as the strongest predictors of claim risk.

## How to Run
1. Clone this repository
2. Install dependencies: `pip install pandas scikit-learn xgboost matplotlib`
3. Download train.csv from the Kaggle link above and place it in the project folder
4. Open `insurance_claim_prediction.ipynb` in Jupyter Notebook and run all cells

## Future Work
- Apply SMOTE to address class imbalance during training
- Test LightGBM as an alternative to XGBoost
- Explore interaction features between regional and vehicle variables
- Implement cross-validated evaluation for more robust scoring
