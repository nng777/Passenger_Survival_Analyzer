from __future__ import annotations

import time
from typing import Dict, Any

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from titanic_predictor import TitanicPredictor


def run_models() -> Dict[str, Dict[str, Any]]:
    """Train four models and return their accuracy and training time."""
    predictor = TitanicPredictor()
    predictor.load_data()
    predictor.engineer_features()
    predictor.prepare_features()

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(probability=True, random_state=42),
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(predictor.X_train, predictor.y_train)
        train_time = time.perf_counter() - start
        y_pred = model.predict(predictor.X_test)
        acc = accuracy_score(predictor.y_test, y_pred)
        results[name] = {"accuracy": acc, "train_time": train_time}
    return results


def plot_accuracies(results: Dict[str, Dict[str, Any]]) -> None:
    """Display a bar chart comparing model accuracies."""
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]

    plt.bar(names, accuracies, color="skyblue")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    try:
        plt.show()
    except Exception as exc:
        print(f"Plotting failed: {exc}")


def main() -> None:
    """Run the model experiment."""
    print("\n🚢 TASK 5: THE MODEL EXPERIMENTER")
    results = run_models()

    best = max(results, key=lambda n: results[n]["accuracy"])
    fastest = min(results, key=lambda n: results[n]["train_time"])

    print("\nAccuracy and training time:")
    for name, info in results.items():
        print(f"{name:>20}: {info['accuracy']:.3f}  ({info['train_time']:.2f}s)")

    print(f"\nBest performing model : {best}")
    print(f"Fastest to train      : {fastest}")

    plot_accuracies(results)


if __name__ == "__main__":
    main()
