from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from titanic_predictor import TitanicPredictor


def engineer_features_with_title(predictor: TitanicPredictor) -> pd.DataFrame:
    """Create engineered features including passenger titles."""

    df = predictor.data.copy()

    # Handle missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Family related features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age and fare groups (same bins as the baseline)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Adult", "Middle", "Senior"],
    )
    df["FareGroup"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "VeryHigh"])

    # --------------------------------------------------
    # Title extraction
    # --------------------------------------------------
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\\.", expand=False)
    common_titles = ["Mr", "Mrs", "Miss", "Master"]
    df["Title"] = df["Title"].apply(lambda x: x if x in common_titles else "Other")

    # Encode categorical variables
    le = LabelEncoder()
    df["Sex_numeric"] = le.fit_transform(df["Sex"])

    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
    df = pd.concat([df, embarked_dummies], axis=1)

    age_group_dummies = pd.get_dummies(df["AgeGroup"], prefix="AgeGroup")
    df = pd.concat([df, age_group_dummies], axis=1)

    fare_group_dummies = pd.get_dummies(df["FareGroup"], prefix="FareGroup")
    df = pd.concat([df, fare_group_dummies], axis=1)

    title_dummies = pd.get_dummies(df["Title"], prefix="Title")
    df = pd.concat([df, title_dummies], axis=1)

    predictor.data_engineered = df
    return df


def prepare_features_title(predictor: TitanicPredictor) -> None:
    """Prepare model features including title dummy columns."""

    feature_columns = [
        "Pclass",
        "Sex_numeric",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
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
        "FareGroup_Low",
        "FareGroup_Medium",
        "FareGroup_High",
        "FareGroup_VeryHigh",
        "Title_Mr",
        "Title_Mrs",
        "Title_Miss",
        "Title_Master",
        "Title_Other",
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


def title_feature_accuracy() -> tuple[float, pd.Series]:
    """Accuracy and survival rates using title features."""

    predictor = TitanicPredictor()
    predictor.load_data()
    engineer_features_with_title(predictor)
    prepare_features_title(predictor)
    acc = run_logistic_regression(predictor)
    rates = predictor.data_engineered.groupby("Title")["Survived"].mean().sort_index()
    return acc, rates


def main() -> None:
    print("\n🚢 TASK 4: THE TITLE EXTRACTOR")
    base_acc = baseline_accuracy()
    new_acc, rates = title_feature_accuracy()

    print(f"\nBaseline accuracy : {base_acc:.3f}")
    print(f"Title feature accuracy: {new_acc:.3f}")

    diff = new_acc - base_acc
    if diff > 0:
        print("✅ Title features improved accuracy!")
    elif diff < 0:
        print("❌ Title features decreased accuracy.")
    else:
        print("⚖️  Accuracy unchanged.")

    print("\nSurvival rate by title:")
    for title, rate in rates.items():
        print(f"   {title}: {rate:.1%}")

    try:
        rates.plot(kind="bar", color="purple", title="Survival Rate by Title")
        plt.xlabel("Title")
        plt.ylabel("Survival Rate")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()