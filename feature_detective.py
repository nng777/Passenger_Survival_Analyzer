import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from titanic_predictor import TitanicPredictor


def run_experiment(drop_feature=None):
    """Train Logistic Regression without a specific feature."""
    predictor = TitanicPredictor()
    predictor.load_data()
    predictor.engineer_features()
    predictor.prepare_features()

    if drop_feature:
        predictor.X_train = predictor.X_train.drop(columns=[drop_feature], errors="ignore")
        predictor.X_test = predictor.X_test.drop(columns=[drop_feature], errors="ignore")

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(predictor.X_train, predictor.y_train)
    y_pred = model.predict(predictor.X_test)
    accuracy = accuracy_score(predictor.y_test, y_pred)
    return accuracy


def main():
    features_to_drop = [None, 'Sex_numeric', 'Pclass', 'Age', 'FamilySize']
    results = {}

    for feature in features_to_drop:
        acc = run_experiment(feature)
        key = feature if feature else 'All Features'
        results[key] = acc

    print("Feature Removal Experiment")
    print("==========================")
    for feature, acc in results.items():
        print(f"{feature:>12}: {acc:.3f}")


if __name__ == "__main__":
    main()