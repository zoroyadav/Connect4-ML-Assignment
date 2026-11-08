import os
import json
import time
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

def load_data():
    """Load Connect-4 dataset via ucimlrepo or fallback to local files."""
    print("Loading Connect-4 dataset...")
    try:
        from ucimlrepo import fetch_ucirepo
        c4 = fetch_ucirepo(id=26)
        X = c4.data.features.astype(str)
        y = c4.data.targets.iloc[:, 0].astype(str)
        df = pd.concat([X, y.rename("class")], axis=1)
    except Exception as e:
        print(f"ucimlrepo failed ({e}), loading from local or raw source...")
        data_path = os.path.join("connect+4", "connect-4.data")
        cols = [
            f"{c}{r}"
            for c in ["a", "b", "c", "d", "e", "f", "g"]
            for r in range(1, 7)
        ] + ["class"]
        df = pd.read_csv(data_path, header=None, names=cols)
        X = df.iloc[:, :-1].astype(str)
        y = df.iloc[:, -1].astype(str)

    print(f"Total instances: {len(df)}, Features: {X.shape[1]}")
    print(f"Class distribution:\n{y.value_counts()}")
    return df, X, y

def main():
    os.makedirs("model", exist_ok=True)
    df, X, y = load_data()

    # We sample a stratified representative subset of 15,000 instances
    # to balance high statistical power, fast training, and cloud deployment memory constraints.
    print("\nCreating stratified sample of 15,000 instances...")
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=15000, stratify=y, random_state=42
    )
    
    # 80-20 train-test split: 12,000 training instances, 3,000 test instances
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=42
    )
    
    # Save test_data.csv in project root as required by the assignment specification
    test_df = pd.concat([X_test, y_test.rename("class")], axis=1)
    test_df.to_csv("test_data.csv", index=False)
    print(f"Saved test_data.csv with {len(test_df)} samples to project root.")

    # Define standard encoder for categorical board positions ('x', 'o', 'b')
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Define the 5 ML classification models required by the assignment
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=14, min_samples_split=10, random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1
        ),
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Support Vector Machine": SVC(
            kernel="rbf", C=2.0, probability=True, random_state=42
        ),
    }

    results = []
    trained_pipelines = {}

    print("\n" + "="*80)
    print("TRAINING AND EVALUATING THE 5 CLASSIFICATION MODELS")
    print("="*80)

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        pipeline = Pipeline([("encoder", encoder), ("classifier", clf)])
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Predictions & Probabilities
        t_eval = time.time()
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        eval_time = time.time() - t_eval

        # Compute all 6 required evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="weighted"
        )
        prec = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        rec = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        mcc = matthews_corrcoef(y_test, y_pred)

        print(f"Completed in {train_time:.2f}s (Inference: {eval_time:.3f}s)")
        print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"AUC:       {auc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"MCC:       {mcc:.4f}")

        results.append({
            "ML Model Name": name,
            "Accuracy": round(float(acc), 4),
            "AUC": round(float(auc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
            "MCC": round(float(mcc), 4),
            "Train_Time_sec": round(float(train_time), 2),
            "Classes": list(pipeline.classes_)
        })

        # Save model pipeline into model/ directory
        filename_map = {
            "Logistic Regression": "logistic_regression.joblib",
            "Decision Tree": "decision_tree.joblib",
            "KNN": "knn.joblib",
            "Naive Bayes": "naive_bayes.joblib",
            "Support Vector Machine": "svm.joblib"
        }
        model_path = os.path.join("model", filename_map[name])
        joblib.dump(pipeline, model_path, compress=3, protocol=4)
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Saved {name} to {model_path} ({file_size_mb:.2f} MB)")

    # Print summary table
    results_df = pd.DataFrame(results)[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)
    print(results_df.to_string(index=False))

    # Save metrics to json and csv for easy loading in README and app
    with open(os.path.join("model", "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    results_df.to_csv(os.path.join("model", "comparison_table.csv"), index=False)
    print("\nMetrics saved to model/metrics.json and model/comparison_table.csv")

if __name__ == "__main__":
    main()
