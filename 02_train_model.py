# ============================================================
# Train Loan Approval ML Model
# Author: Sanath | York University
# ============================================================
# I chose XGBoost for this project because it's the industry
# standard for tabular/structured data — it consistently wins
# Kaggle competitions and is what banks actually use. I also
# implemented SHAP for model explainability because in finance,
# regulators require that ML decisions can be explained.
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap
import json

print("=" * 60)
print("🤖 LOAN APPROVAL MODEL - TRAINING PIPELINE")
print("=" * 60)

# Load the dataset I generated
print("\n📂 Loading dataset...")
df = pd.read_csv("data/loan_data.csv")
print(f"   Loaded {len(df)} records with {len(df.columns)} features")

# I need to encode categorical features because ML models only work with numbers
# LabelEncoder assigns each category an integer alphabetically
print("\n🔧 Encoding categorical features...")
le_purpose = LabelEncoder()
le_home = LabelEncoder()
df["loan_purpose_encoded"] = le_purpose.fit_transform(df["loan_purpose"])
df["home_ownership_encoded"] = le_home.fit_transform(df["home_ownership"])

print(f"   Loan purposes: {dict(zip(le_purpose.classes_, le_purpose.transform(le_purpose.classes_)))}")
print(f"   Home ownership: {dict(zip(le_home.classes_, le_home.transform(le_home.classes_)))}")

# Separating features (X) from target (y)
feature_columns = [
    "age", "annual_income", "credit_score", "employment_years",
    "loan_amount", "dti_ratio", "num_credit_lines", "previous_default",
    "loan_purpose_encoded", "home_ownership_encoded"
]
X = df[feature_columns]
y = df["approved"]

print(f"\n📊 Features (X): {X.shape} -> {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Target (y): {y.shape} -> {y.value_counts().to_dict()}")

# I split 80/20 to prevent overfitting — the test set acts as an unseen exam
# stratify=y ensures both sets keep the same approval/denial ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️  Train/Test Split: {X_train.shape[0]} training / {X_test.shape[0]} test samples")

# I tuned these hyperparameters through experimentation:
# - 200 trees gives good accuracy without overfitting
# - max_depth=5 prevents the trees from memorizing the data
# - learning_rate=0.1 ensures each tree contributes carefully
# - subsample/colsample add randomness to reduce overfitting
print("\n🚀 Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss", use_label_encoder=False
)
model.fit(X_train, y_train)
print("   ✅ Model trained successfully!")

# Evaluating on data the model has never seen
print("\n📈 Evaluating model on test set...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n   📊 MODEL PERFORMANCE:")
print(f"   Accuracy:  {accuracy:.1%}")
print(f"   Precision: {precision:.1%}")
print(f"   Recall:    {recall:.1%}")
print(f"   F1 Score:  {f1:.1%}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Denied', 'Approved'])}")

cm = confusion_matrix(y_test, y_pred)
print(f"   Confusion Matrix:")
print(f"   Predicted →    Denied  Approved")
print(f"   Actual Denied:  {cm[0][0]:>5}    {cm[0][1]:>5}")
print(f"   Actual Approved:{cm[1][0]:>5}    {cm[1][1]:>5}")

# Feature importance — I found previous_default was by far the strongest predictor
print("\n🏆 Feature Importance:")
importance = dict(zip(feature_columns, model.feature_importances_))
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for feat, imp in sorted_importance:
    bar = "█" * int(imp * 50)
    print(f"   {feat:<25} {imp:.3f} {bar}")

# SHAP values — this is what sets my project apart from typical student ML projects
# SHAP explains WHY the model made each individual decision, which is critical
# in financial services where decisions must be explainable
print("\n🔍 Computing SHAP values for model explainability...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"\n   Example: Test applicant #1")
print(f"   Prediction: {'APPROVED' if y_pred[0] == 1 else 'DENIED'} (confidence: {y_prob[0].max():.1%})")
print(f"   Actual:     {'APPROVED' if y_test.iloc[0] == 1 else 'DENIED'}")
print(f"\n   Top factors:")
shap_sorted = sorted(zip(feature_columns, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)
for feat, val in shap_sorted[:5]:
    direction = "↑ helped" if val > 0 else "↓ hurt"
    print(f"     {feat:<25} {val:+.3f}  ({direction})")

# Saving everything so my Streamlit app can load the trained model instantly
print("\n💾 Saving model and artifacts...")
pickle.dump(model, open("models/xgb_model.pkl", "wb"))
pickle.dump(le_purpose, open("models/le_purpose.pkl", "wb"))
pickle.dump(le_home, open("models/le_home.pkl", "wb"))
json.dump(feature_columns, open("models/feature_columns.json", "w"))
json.dump({
    "accuracy": round(accuracy, 4), "precision": round(precision, 4),
    "recall": round(recall, 4), "f1": round(f1, 4),
    "train_size": X_train.shape[0], "test_size": X_test.shape[0],
    "feature_importance": {k: round(float(v), 4) for k, v in sorted_importance}
}, open("models/metrics.json", "w"), indent=2)

print("   ✅ All artifacts saved to models/")
print(f"\n🎉 TRAINING COMPLETE — Final Accuracy: {accuracy:.1%}")
