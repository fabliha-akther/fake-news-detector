import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import shap

# Load data and prepare features
def load_data(filepath="data/features.csv"):
    data = pd.read_csv(filepath)

    features = [
        "hour", "dayofweek", "is_weekend",
        "content_length", "num_exclamations", "num_questions",
        "num_uppercase_words", "num_links", "num_hashtags", "keyword_hits", "sentiment_score"  # Include sentiment score
    ]

    X = data[features]
    y = data["label"]

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Logistic Regression Model
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Logistic Regression")
    print(classification_report(y_test, predictions))
    print("AUC:", roc_auc_score(y_test, probabilities))

# Train Random Forest Model
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Random Forest")
    print(classification_report(y_test, predictions))
    print("AUC:", roc_auc_score(y_test, probabilities))

# Train XGBoost Model
def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, scale_pos_weight=1.1)  # Adjust scale_pos_weight as needed
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("XGBoost")
    print(classification_report(y_test, predictions))
    print("AUC:", roc_auc_score(y_test, probabilities))

    model.save_model("model.xgb")
    print("Model saved as model.xgb")

    return model


# SHAP
def explain_with_shap(model, X_sample):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)
    return shap_values


def run_all_models():
    X_train, X_test, y_train, y_test = load_data()

    train_logistic_regression(X_train, y_train, X_test, y_test)

    train_random_forest(X_train, y_train, X_test, y_test)

    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

    return xgb_model, X_test

if __name__ == "__main__":
    model, X = run_all_models()
    sample = X.sample(n=100, random_state=42)
    shap_values = explain_with_shap(model, sample)
