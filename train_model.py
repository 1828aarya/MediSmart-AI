from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

DATA_PATH = Path(__file__).parent / "data" / "medicine_dataset.csv"
MODEL_PATH = Path(__file__).parent / "medicine_model.joblib"

df = pd.read_csv(DATA_PATH).fillna("")
required = {"medicine_name", "active_ingredient", "dosage_form", "strength", "category"}
if not required.issubset(df.columns):
    raise ValueError(f"Dataset must contain columns: {sorted(required)}")

# Combine medicine text fields into one input for text classification.
X = (
    df["medicine_name"].astype(str) + " " +
    df["active_ingredient"].astype(str) + " " +
    df["dosage_form"].astype(str) + " " +
    df["strength"].astype(str)
)
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),
    "Decision Tree": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("model", DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"))
    ]),
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
    ])
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1-score (macro)": f1_score(y_test, pred, average="macro", zero_division=0)
    })
    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

results_df = pd.DataFrame(results).sort_values("F1-score (macro)", ascending=False)
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Algorithm"]
best_model = models[best_name]
joblib.dump({"model": best_model, "algorithm": best_name}, MODEL_PATH)
print(f"\nSaved best educational model: {best_name} -> {MODEL_PATH}")
print("\nImportant: this is a small synthetic teaching dataset. Scores do not indicate clinical accuracy.")
