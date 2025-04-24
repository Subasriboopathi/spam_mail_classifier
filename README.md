# spam_mail_classifier

A Python-based machine-learning pipeline to detect spam emails with robust preprocessing, feature selection, over-sampling, and hyperparameter tuning.

## 🚀 Features

- **Data Imputation**: Automatically fills in missing numeric values with column means.  
- **Scaling & Selection**: Scales features into [0,1] and selects the top *k* predictors via chi-squared tests.  
- **Class Imbalance Handling**: Uses SMOTE to synthesize minority-class samples and balance the training set.  
- **Model Comparison & Tuning**: Runs a stratified `GridSearchCV` over Logistic Regression and Random Forest hyperparameters.  
- **Metrics & Reporting**: Outputs CV accuracy and test accuracy (as percentages), confusion matrix, classification report, and ROC-AUC.

## 📂 Repository Contents

- `spam_email_data.csv` — Original dataset with 7 engineered numeric features + `SpamLabel`.    
- `README.md` — This file.  

## 🔧 Installation

```bash
git clone https://github.com/Subasriboopathi/spam_mail_classifier.git
cd spam_mail_classifier
pip install -r requirements.txt
