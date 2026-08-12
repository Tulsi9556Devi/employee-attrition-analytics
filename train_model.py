import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


# =========================
# 2. Select Relevant Features
# =========================

features = [
    "Age",
    "MonthlyIncome",
    "OverTime",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "WorkLifeBalance",
    "StockOptionLevel",
    "JobRole",
    "MaritalStatus",
    "BusinessTravel"
]

X = df[features]

y = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# =========================
# 3. Identify Data Types
# =========================

categorical_features = [
    "OverTime",
    "JobRole",
    "MaritalStatus",
    "BusinessTravel"
]

numerical_features = [
    "Age",
    "MonthlyIncome",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "WorkLifeBalance",
    "StockOptionLevel"
]


# =========================
# 4. Preprocessing
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# =========================
# 5. ML Pipeline
# =========================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# =========================
# 6. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 7. Train
# =========================

pipeline.fit(
    X_train,
    y_train
)


# =========================
# 8. Evaluate
# =========================

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


print("\n============================")
print("MODEL EVALUATION")
print("============================")

print(
    "\nAccuracy:",
    round(
        accuracy_score(
            y_test,
            predictions
        ),
        3
    )
)

print(
    "\nROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            probabilities
        ),
        3
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# =========================
# 9. Save Complete Pipeline
# =========================

joblib.dump(
    pipeline,
    "attrition_model.pkl"
)

print("\nModel saved as attrition_model.pkl")