from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from titanic_predictor import TitanicPredictor


def engineer_features_fare(predictor: TitanicPredictor) -> pd.DataFrame:
    """Create engineered features including FarePerPerson and fare categories."""
    df = predictor.data.copy()

    # Handle missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Family related features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age groups (same as baseline)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Adult", "Middle", "Senior"],
    )

    # New fare related features
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["FareCategory"] = pd.cut(
        df["Fare"],
        bins=[-0.1, 0.0, 10, 50, float("inf")],
        labels=["Free", "Cheap", "Moderate", "Expensive"],
    )

    # Encode categorical variables
    le = LabelEncoder()
    df["Sex_numeric"] = le.fit_transform(df["Sex"])

    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
    df = pd.concat([df, embarked_dummies], axis=1)

    age_group_dummies = pd.get_dummies(df["AgeGroup"], prefix="AgeGroup")
    df = pd.concat([df, age_group_dummies], axis=1)

    fare_cat_dummies = pd.get_dummies(df["FareCategory"], prefix="FareCategory")
    df = pd.concat([df, fare_cat_dummies], axis=1)

    predictor.data_engineered = df
    return df


def prepare_features_fare(predictor: TitanicPredictor) -> None:
    """Prepare features using the fare investigator columns."""
    feature_columns = [
        "Pclass",
        "Sex_numeric",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FarePerPerson",
        "FamilySize",
        "IsAlone",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S",
        "AgeGroup_Child",
        "AgeGroup_Teen",
        "AgeGroup_Adult",
        "AgeGroup_Middle",
        "AgeGroup_Senior",
        "FareCategory_Free",
        "FareCategory_Cheap",
        "FareCategory_Moderate",
        "FareCategory_Expensive",
    ]

    available = [c for c in feature_columns if c in predictor.data_engineered.columns]

    X = predictor.data_engineered[available]
    y = predictor.data_engineered["Survived"]

    predictor.X_train, predictor.X_test, predictor.y_train, predictor.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    predictor.feature_names = available


def run_logistic_regression(predictor: TitanicPredictor) -> float:
    """Train Logistic Regression and return accuracy."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(predictor.X_train, predictor.y_train)
    y_pred = model.predict(predictor.X_test)
    return accuracy_score(predictor.y_test, y_pred)


def baseline_accuracy() -> float:
    """Accuracy using the original TitanicPredictor features."""
    predictor = TitanicPredictor()
    predictor.load_data()
    predictor.engineer_features()
    predictor.prepare_features()
    return run_logistic_regression(predictor)


def fare_feature_accuracy() -> tuple[float, pd.Series]:
    """Accuracy and survival rates using custom fare features."""
    predictor = TitanicPredictor()
    predictor.load_data()
    engineer_features_fare(predictor)
    prepare_features_fare(predictor)
    acc = run_logistic_regression(predictor)
    rates = predictor.data_engineered.groupby("FareCategory")["Survived"].mean().sort_index()
    return acc, rates


def main() -> None:
    print("\n🚢 TASK 3: THE FARE INVESTIGATOR")
    base_acc = baseline_accuracy()
    new_acc, rates = fare_feature_accuracy()

    print(f"\nBaseline accuracy : {base_acc:.3f}")
    print(f"Fare features accuracy: {new_acc:.3f}")

    diff = new_acc - base_acc
    if diff > 0:
        print("✅ Fare features improved accuracy!")
    elif diff < 0:
        print("❌ Fare features decreased accuracy.")
    else:
        print("⚖️  Accuracy unchanged.")

    print("\nSurvival rate by fare category:")
    for cat, rate in rates.items():
        print(f"   {cat}: {rate:.1%}")

    try:
        rates.plot(kind="bar", color="orange", title="Survival Rate by Fare Category")
        plt.xlabel("Fare Category")
        plt.ylabel("Survival Rate")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()