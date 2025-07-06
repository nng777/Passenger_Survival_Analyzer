#!/usr/bin/env python3
"""
🚢 Titanic Survival Predictor
A beginner-friendly machine learning project demonstrating:
- Feature engineering and selection
- Model building with scikit-learn
- Performance evaluation and comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TitanicPredictor:
    """
    A comprehensive Titanic survival prediction system that demonstrates
    key machine learning concepts for beginners.
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.feature_names = []
        
    def load_data(self, file_path=None):
        """Load the Titanic dataset"""
        if file_path:
            self.data = pd.read_csv(file_path)
        else:
            # Create sample data if no file provided
            print("📊 Creating sample Titanic dataset...")
            self.data = self._create_sample_data()
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📈 Dataset shape: {self.data.shape}")
        return self.data
    
    def _create_sample_data(self):
        """Create a sample Titanic dataset for demonstration"""
        np.random.seed(42)
        n_samples = 891
        
        # Generate realistic sample data
        data = {
            'PassengerId': range(1, n_samples + 1),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(29, 14, n_samples),
            'SibSp': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.01]),
            'Parch': np.random.choice([0, 1, 2, 3], n_samples, p=[0.76, 0.13, 0.08, 0.03]),
            'Fare': np.random.exponential(32, n_samples),
            'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
        }
        
        df = pd.DataFrame(data)


        # Generate a simple Name column for tasks that need titles

        male_first = ['John', 'William', 'James', 'George', 'Charles']
        female_first = ['Mary', 'Anna', 'Emma', 'Elizabeth', 'Minnie']
        surnames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Wilson', 'Taylor', 'Anderson']

        titles_male = ['Mr', 'Master', 'Dr', 'Rev', 'Col']
        titles_female = ['Mrs', 'Miss', 'Dr']

        names = []
        for sex in df['Sex']:
            if sex == 'male':
                title = np.random.choice(titles_male, p=[0.7, 0.1, 0.1, 0.05, 0.05])
                first = np.random.choice(male_first)
            else:
                title = np.random.choice(titles_female, p=[0.55, 0.4, 0.05])
                first = np.random.choice(female_first)
            surname = np.random.choice(surnames)
            names.append(f"{surname}, {title}. {first}")

        df['Name'] = names


        # Add some missing values to simulate real data
        missing_age_idx = np.random.choice(df.index, size=int(0.2 * len(df)), replace=False)
        df.loc[missing_age_idx, 'Age'] = np.nan
        
        # Generate survival based on realistic patterns
        survival_prob = 0.3  # Base survival rate
        
        # Adjust probability based on features
        prob_adjustments = np.zeros(len(df))
        prob_adjustments += (df['Sex'] == 'female') * 0.4  # Women more likely to survive
        prob_adjustments += (df['Pclass'] == 1) * 0.2  # First class more likely
        prob_adjustments += (df['Pclass'] == 2) * 0.1  # Second class somewhat more likely
        prob_adjustments += (df['Age'] < 16) * 0.3  # Children more likely (where age is not null)
        prob_adjustments += (df['Age'] > 60) * -0.1  # Elderly less likely
        
        final_probs = np.clip(survival_prob + prob_adjustments, 0, 1)
        df['Survived'] = np.random.binomial(1, final_probs)
        
        return df
    
    def explore_data(self):
        """Explore and visualize the dataset"""
        print("\n🔍 DATASET EXPLORATION")
        print("=" * 50)
        
        # Basic info
        print(f"📊 Dataset Info:")
        print(f"   • Total passengers: {len(self.data)}")
        print(f"   • Features: {len(self.data.columns)}")
        print(f"   • Survivors: {self.data['Survived'].sum()} ({self.data['Survived'].mean():.1%})")
        
        # Missing values
        print(f"\n🔍 Missing Values:")
        missing = self.data.isnull().sum()
        for col, count in missing[missing > 0].items():
            print(f"   • {col}: {count} ({count/len(self.data):.1%})")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🚢 Titanic Dataset Exploration', fontsize=16, fontweight='bold')
        
        # Survival by gender
        survival_by_sex = self.data.groupby(['Sex', 'Survived']).size().unstack()
        survival_by_sex.plot(kind='bar', ax=axes[0,0], color=['#ff6b6b', '#4ecdc4'])
        axes[0,0].set_title('Survival by Gender')
        axes[0,0].set_xlabel('Gender')
        axes[0,0].set_ylabel('Count')
        axes[0,0].legend(['Did not survive', 'Survived'])
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # Survival by class
        survival_by_class = self.data.groupby(['Pclass', 'Survived']).size().unstack()
        survival_by_class.plot(kind='bar', ax=axes[0,1], color=['#ff6b6b', '#4ecdc4'])
        axes[0,1].set_title('Survival by Passenger Class')
        axes[0,1].set_xlabel('Passenger Class')
        axes[0,1].set_ylabel('Count')
        axes[0,1].legend(['Did not survive', 'Survived'])
        axes[0,1].tick_params(axis='x', rotation=0)
        
        # Age distribution
        self.data['Age'].hist(bins=30, ax=axes[1,0], alpha=0.7, color='skyblue')
        axes[1,0].set_title('Age Distribution')
        axes[1,0].set_xlabel('Age')
        axes[1,0].set_ylabel('Frequency')
        
        # Fare distribution
        self.data['Fare'].hist(bins=30, ax=axes[1,1], alpha=0.7, color='lightgreen')
        axes[1,1].set_title('Fare Distribution')
        axes[1,1].set_xlabel('Fare')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        return self.data.describe()
    
    def engineer_features(self):
        """Perform feature engineering on the dataset"""
        print("\n🔧 FEATURE ENGINEERING")
        print("=" * 50)
        
        # Make a copy to avoid modifying original data
        df = self.data.copy()
        
        print("1. 🩹 Handling Missing Values...")
        # Fill missing Age with median
        age_median = df['Age'].median()
        df['Age'].fillna(age_median, inplace=True)
        print(f"   • Filled {self.data['Age'].isnull().sum()} missing ages with median: {age_median:.1f}")
        
        # Fill missing Embarked with mode
        embarked_mode = df['Embarked'].mode()[0]
        df['Embarked'].fillna(embarked_mode, inplace=True)
        print(f"   • Filled missing embarkation with mode: {embarked_mode}")
        
        print("\n2. 🏗️ Creating New Features...")
        # Family size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        print(f"   • Created FamilySize feature (range: {df['FamilySize'].min()}-{df['FamilySize'].max()})")
        
        # Is alone
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        print(f"   • Created IsAlone feature ({df['IsAlone'].sum()} passengers traveled alone)")
        
        # Age groups
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                               labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        print(f"   • Created AgeGroup feature with 5 categories")
        
        # Fare groups
        df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
        print(f"   • Created FareGroup feature with 4 categories")
        
        print("\n3. 🔄 Converting Categorical Variables...")
        # Convert categorical variables to numerical
        le = LabelEncoder()
        
        # Binary encoding for Sex
        df['Sex_numeric'] = le.fit_transform(df['Sex'])
        print(f"   • Encoded Sex: male=0, female=1")
        
        # One-hot encoding for Embarked
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df = pd.concat([df, embarked_dummies], axis=1)
        print(f"   • One-hot encoded Embarked: {list(embarked_dummies.columns)}")
        
        # One-hot encoding for AgeGroup
        age_group_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
        df = pd.concat([df, age_group_dummies], axis=1)
        print(f"   • One-hot encoded AgeGroup: {list(age_group_dummies.columns)}")
        
        # One-hot encoding for FareGroup
        fare_group_dummies = pd.get_dummies(df['FareGroup'], prefix='FareGroup')
        df = pd.concat([df, fare_group_dummies], axis=1)
        print(f"   • One-hot encoded FareGroup: {list(fare_group_dummies.columns)}")
        
        self.data_engineered = df
        print(f"\n✅ Feature engineering complete! Dataset now has {len(df.columns)} columns.")
        return df
    
    def prepare_features(self):
        """Select and prepare features for modeling"""
        print("\n🎯 FEATURE SELECTION")
        print("=" * 50)
        
        # Select features for modeling
        feature_columns = [
            'Pclass', 'Sex_numeric', 'Age', 'SibSp', 'Parch', 'Fare',
            'FamilySize', 'IsAlone',
            'Embarked_C', 'Embarked_Q', 'Embarked_S',
            'AgeGroup_Child', 'AgeGroup_Teen', 'AgeGroup_Adult', 'AgeGroup_Middle', 'AgeGroup_Senior',
            'FareGroup_Low', 'FareGroup_Medium', 'FareGroup_High', 'FareGroup_VeryHigh'
        ]
        
        # Keep only features that exist in the dataset
        available_features = [col for col in feature_columns if col in self.data_engineered.columns]
        
        X = self.data_engineered[available_features]
        y = self.data_engineered['Survived']
        
        print(f"📊 Selected {len(available_features)} features for modeling:")
        for i, feature in enumerate(available_features, 1):
            print(f"   {i:2d}. {feature}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.feature_names = available_features
        
        print(f"\n📈 Data split:")
        print(f"   • Training set: {len(self.X_train)} samples")
        print(f"   • Test set: {len(self.X_test)} samples")
        print(f"   • Training survival rate: {self.y_train.mean():.1%}")
        print(f"   • Test survival rate: {self.y_test.mean():.1%}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_models(self):
        """Build and train multiple machine learning models"""
        print("\n🤖 MODEL BUILDING")
        print("=" * 50)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            # Store model and results
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"   ✅ Accuracy: {accuracy:.3f}")
            print(f"   📊 CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        return self.models
    
    def evaluate_models(self):
        """Evaluate and compare model performance"""
        print("\n📊 MODEL EVALUATION")
        print("=" * 50)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎯 Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Model accuracy comparison
        model_names = list(self.models.keys())
        accuracies = [self.models[name]['accuracy'] for name in model_names]
        cv_means = [self.models[name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0,0].bar(x - width/2, accuracies, width, label='Test Accuracy', color='skyblue')
        axes[0,0].bar(x + width/2, cv_means, width, label='CV Mean', color='lightgreen')
        axes[0,0].set_title('Model Accuracy Comparison')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(model_names, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Feature importance (Random Forest)
        rf_model = self.models['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Plot top 10 features
        top_features = feature_importance.tail(10)
        axes[0,1].barh(top_features['feature'], top_features['importance'], color='coral')
        axes[0,1].set_title('Top 10 Feature Importance (Random Forest)')
        axes[0,1].set_xlabel('Importance')
        
        # Confusion matrix for best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['accuracy'])
        best_predictions = self.models[best_model_name]['predictions']
        
        cm = confusion_matrix(self.y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        
        # Survival rate by prediction confidence (Logistic Regression)
        lr_model = self.models['Logistic Regression']['model']
        prediction_probs = lr_model.predict_proba(self.X_test)[:, 1]
        
        # Create probability bins
        prob_bins = np.linspace(0, 1, 11)
        bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
        actual_rates = []
        
        for i in range(len(prob_bins) - 1):
            mask = (prediction_probs >= prob_bins[i]) & (prediction_probs < prob_bins[i + 1])
            if mask.sum() > 0:
                actual_rates.append(self.y_test[mask].mean())
            else:
                actual_rates.append(0)
        
        axes[1,1].plot(bin_centers, actual_rates, 'o-', color='red', label='Actual')
        axes[1,1].plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
        axes[1,1].set_title('Prediction Calibration')
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Actual Survival Rate')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print(f"\n🏆 RESULTS SUMMARY")
        print("-" * 30)
        for name, results in self.models.items():
            print(f"\n{name}:")
            print(f"   • Test Accuracy: {results['accuracy']:.3f}")
            print(f"   • CV Score: {results['cv_mean']:.3f} (±{results['cv_std']:.3f})")
        
        print(f"\n🥇 Best Model: {best_model_name} (Accuracy: {self.models[best_model_name]['accuracy']:.3f})")
        
        # Classification report for best model
        print(f"\n📋 Detailed Classification Report - {best_model_name}:")
        print(classification_report(self.y_test, best_predictions, 
                                  target_names=['Did not survive', 'Survived']))
        
        return self.models
    
    def analyze_feature_importance(self):
        """Analyze which features are most important for prediction"""
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Get feature importance from Random Forest
        rf_model = self.models['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("📊 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:<20} {row['importance']:.3f}")
        
        # Interpretation
        print(f"\n🧠 Key Insights:")
        top_feature = feature_importance.iloc[0]['feature']
        print(f"   • Most important feature: {top_feature}")
        
        if 'Sex_numeric' in feature_importance['feature'].values:
            sex_importance = feature_importance[feature_importance['feature'] == 'Sex_numeric']['importance'].iloc[0]
            print(f"   • Gender importance: {sex_importance:.3f} (women had higher survival rates)")
        
        if 'Pclass' in feature_importance['feature'].values:
            class_importance = feature_importance[feature_importance['feature'] == 'Pclass']['importance'].iloc[0]
            print(f"   • Passenger class importance: {class_importance:.3f} (higher class = better survival)")
        
        if 'Age' in feature_importance['feature'].values:
            age_importance = feature_importance[feature_importance['feature'] == 'Age']['importance'].iloc[0]
            print(f"   • Age importance: {age_importance:.3f} (children had higher survival rates)")
        
        return feature_importance
    
    def predict_survival(self, passenger_data):
        """Predict survival for a new passenger"""
        print("\n🔮 MAKING PREDICTIONS")
        print("=" * 50)
        
        # Use the best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['accuracy'])
        best_model = self.models[best_model_name]['model']
        
        # Make prediction
        prediction = best_model.predict([passenger_data])[0]
        probability = best_model.predict_proba([passenger_data])[0]
        
        print(f"🤖 Using {best_model_name} for prediction...")
        print(f"📊 Prediction: {'SURVIVED' if prediction == 1 else 'DID NOT SURVIVE'}")
        print(f"🎯 Confidence: {max(probability):.1%}")
        print(f"📈 Survival probability: {probability[1]:.1%}")
        
        return prediction, probability
    
    def run_complete_analysis(self):
        """Run the complete machine learning pipeline"""
        print("🚢" * 20)
        print("🚢 TITANIC SURVIVAL PREDICTION - COMPLETE ANALYSIS")
        print("🚢" * 20)
        
        # Step 1: Load and explore data
        if self.data is None:
            self.load_data()
        
        # Step 2: Explore the data
        self.explore_data()
        
        # Step 3: Feature engineering
        self.engineer_features()
        
        # Step 4: Prepare features
        self.prepare_features()
        
        # Step 5: Build models
        self.build_models()
        
        # Step 6: Evaluate models
        self.evaluate_models()
        
        # Step 7: Analyze feature importance
        self.analyze_feature_importance()
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 50)
        print("🎓 What you learned:")
        print("   • How to handle missing data")
        print("   • Feature engineering techniques")
        print("   • Building multiple ML models")
        print("   • Evaluating model performance")
        print("   • Understanding feature importance")
        print("\n🚀 Next steps:")
        print("   • Try different feature engineering approaches")
        print("   • Experiment with hyperparameter tuning")
        print("   • Apply these techniques to other datasets")


def main():
    """Main function to run the Titanic predictor"""
    print("🚢 Welcome to the Titanic Survival Predictor!")
    print("This is your introduction to machine learning with scikit-learn.\n")
    
    # Create predictor instance
    predictor = TitanicPredictor()
    
    # Run complete analysis
    predictor.run_complete_analysis()
    
    # Example prediction
    print("\n" + "="*50)
    print("🎯 EXAMPLE PREDICTION")
    print("="*50)
    
    # Create example passenger data (matching the feature order)
    example_passenger = [
        1,    # Pclass (3rd class)
        1,    # Sex_numeric (male)
        25,   # Age
        0,    # SibSp
        0,    # Parch
        7.25, # Fare
        2,    # FamilySize
        0,    # IsAlone
        0, 0, 1,  # Embarked (Southampton)
        0, 0, 1, 0, 0,  # AgeGroup (Adult)
        1, 0, 0, 0      # FareGroup (Low)
    ]
    
    prediction, probability = predictor.predict_survival(example_passenger)
    
    print(f"\n💡 This passenger would likely have {'SURVIVED' if prediction == 1 else 'NOT SURVIVED'}")
    print(f"   Based on historical patterns in the data.")


if __name__ == "__main__":
    main() 