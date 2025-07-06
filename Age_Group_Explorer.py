import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from titanic_predictor import TitanicPredictor


def engineer_features_custom_ages(predictor: TitanicPredictor) -> pd.DataFrame:
    """Create engineered features using custom age groups."""
    df = predictor.data.copy()

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Family related features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # New age groups
    age_labels = ['Baby', 'Child', 'Teen', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[-0.1, 2, 12, 19, 59, 100],
        labels=age_labels,
    )

    # Fare groups
    df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])

    # Encode categorical variables
    le = LabelEncoder()
    df['Sex_numeric'] = le.fit_transform(df['Sex'])

    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)

    age_group_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
    for label in age_labels:
        col = f'AgeGroup_{label}'
        if col not in age_group_dummies:
            age_group_dummies[col] = 0
    df = pd.concat([df, age_group_dummies], axis=1)

    fare_group_dummies = pd.get_dummies(df['FareGroup'], prefix='FareGroup')
    df = pd.concat([df, fare_group_dummies], axis=1)

    predictor.data_engineered = df
    return df


def prepare_features_custom(predictor: TitanicPredictor):
    """Prepare features using the custom age group columns."""
    feature_columns = [
        'Pclass', 'Sex_numeric', 'Age', 'SibSp', 'Parch', 'Fare',
        'FamilySize', 'IsAlone',
        'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'AgeGroup_Baby', 'AgeGroup_Child', 'AgeGroup_Teen', 'AgeGroup_Adult', 'AgeGroup_Senior',
        'FareGroup_Low', 'FareGroup_Medium', 'FareGroup_High', 'FareGroup_VeryHigh'
    ]

    available_features = [c for c in feature_columns if c in predictor.data_engineered.columns]

    X = predictor.data_engineered[available_features]
    y = predictor.data_engineered['Survived']

    predictor.X_train, predictor.X_test, predictor.y_train, predictor.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    predictor.feature_names = available_features


def run_logistic_regression(predictor: TitanicPredictor) -> float:
    """Train Logistic Regression and return accuracy."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(predictor.X_train, predictor.y_train)
    y_pred = model.predict(predictor.X_test)
    return accuracy_score(predictor.y_test, y_pred)


def baseline_accuracy() -> float:
    """Accuracy using the original age groups from TitanicPredictor."""
    predictor = TitanicPredictor()
    predictor.load_data()
    predictor.engineer_features()
    predictor.prepare_features()
    return run_logistic_regression(predictor)


def custom_age_accuracy() -> tuple[float, pd.Series]:
    """Accuracy and survival rates using custom age groups."""
    predictor = TitanicPredictor()
    predictor.load_data()
    engineer_features_custom_ages(predictor)
    prepare_features_custom(predictor)
    acc = run_logistic_regression(predictor)
    rates = predictor.data_engineered.groupby('AgeGroup')['Survived'].mean().sort_index()
    return acc, rates


def main():
    print("\n TASK 2: AGE GROUP EXPLORER")
    base_acc = baseline_accuracy()
    new_acc, rates = custom_age_accuracy()

    print(f"\nBaseline accuracy : {base_acc:.3f}")
    print(f"New age groups accuracy: {new_acc:.3f}")

    diff = new_acc - base_acc
    if diff > 0:
        print("✅ New age groups improved accuracy!")
    elif diff < 0:
        print("❌ New age groups decreased accuracy.")
    else:
        print("⚖️  Accuracy unchanged.")

    print("\nSurvival rate by new age group:")
    for group, rate in rates.items():
        print(f"   {group}: {rate:.1%}")

    try:
        rates.plot(kind='bar', color='skyblue', title='Survival Rate by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Survival Rate')
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == '__main__':
    main()