import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load & impute
df = pd.read_csv("spam_email_data.csv")
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])

# 2. Features/target split
X = df.drop(columns=["SpamLabel"])
y = df["SpamLabel"]

# 3. Train/test stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# number of features actually present:
num_features = X_train.shape[1]

# 4. Pipeline builder: MinMax scales into [0,1] → chi2 → SMOTE → classifier
def make_pipeline(clf):
    return ImbPipeline([
        ("scaler", MinMaxScaler()),
        ("select", SelectKBest(score_func=chi2)),
        ("smote",  SMOTE(random_state=42)),
        ("clf",    clf)
    ])

# 5. Hyperparameter grid – only pick k ≤ your feature count
param_grid = [
    {
        "select__k": list(range(2, num_features + 1)),
        "clf": [LogisticRegression(solver="liblinear")],
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l1", "l2"],
    },
    {
        "select__k": list(range(2, num_features + 1)),
        "clf": [RandomForestClassifier(random_state=42)],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
    }
]

# 6. GridSearch with stratified 5-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    make_pipeline(LogisticRegression()),
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# 7. Fit & pick best
grid.fit(X_train, y_train)
best = grid.best_estimator_

# Print CV accuracy as percentage
print("Best params:", grid.best_params_)
print(f"CV Accuracy: {grid.best_score_ * 100:.2f}%")

# 8. Evaluate on hold-out
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

# Test accuracy as percentage
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy : {test_acc * 100:.2f}%")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"Test ROC AUC  : {roc_auc:.4f}")

