# spam_mail_classifier

An end-to-end Python pipeline for detecting spam emails using feature selection, SMOTE balancing, and hyperparameter-tuned classifiers.  
Outputs cross-validation & test accuracies (in %), confusion matrix, classification report, and ROC-AUC.

## Overview

`spam_mail_classifier` loads a CSV of engineered numeric email features, imputes missing values, scales them into [0,1], and selects the most informative predictors via a chi-squared test. It balances the training set with SMOTE, then compares and tunes two models—Logistic Regression and Random Forest—using stratified GridSearchCV. Finally, it reports both CV and hold-out test metrics, including accuracy (%), confusion matrix, classification report, and ROC-AUC.

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
