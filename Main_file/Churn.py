"""
Churn Prediction Pipeline
-------------------------
A modular, production-grade pipeline for binary classification.
Compares XGBoost, Random Forest, and Logistic Regression with:
- Automatic Class Imbalance Handling
- Hyperparameter Tuning (GridSearch)
- Comprehensive Visualization (SHAP, ROC, PR Curves)


"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # This forces it to use the stable, built-in Windows UI engine
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb

from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
)
from sklearn.datasets import make_classification

# Configuration to suppress warnings for cleaner output
warnings.filterwarnings('ignore')
sns.set(style="whitegrid", palette="muted")


class ChurnPipeline:
    """
    End-to-end pipeline for training, evaluating, and visualizing
    churn prediction models.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_estimators = {}
        self.results = pd.DataFrame()
        self.X_test = None
        self.y_test = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads real data and uses ALL columns.
        """
        print("Loading real data...")
        
        # 1. Read your file
        df = pd.read_csv("Customer Churn.csv") 

        # 2. Define Target
        target_col = 'Churn' 
        
        # 3. Define X (Features) and y (Target)
        # We ONLY drop the target column. We keep everything else (IDs, etc.)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 4. Handle Categorical Data
        # Convert all text columns (including IDs if they are text) to numbers
        X = pd.get_dummies(X, drop_first=True)
        
        return X, y

    def preprocess_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Any, Any, Any]:
        """Splits data into training and testing sets."""
        print("Splitting data...")
        return train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

    def get_hyperparameters(self, scale_pos_weight: float) -> Dict[str, Any]:
        """
        Defines model pipelines and hyperparameter grids.
        """
        return {
            "LogisticRegression": {
                "model": Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(class_weight="balanced", solver="liblinear"))
                ]),
                "params": {
                    "clf__C": [0.01, 0.1, 1, 10],
                    "clf__penalty": ["l1", "l2"]
                }
            },
            "RandomForest": {
                "model": RandomForestClassifier(class_weight="balanced", random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_leaf": [2, 5]
                }
            },
            "XGBoost": {
                "model": xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=self.random_state
                ),
                "params": {
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7],
                    "n_estimators": [100, 200]
                }
            }
        }

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Performs Grid Search CV for each model.
        """
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        config = self.get_hyperparameters(scale_pos_weight)

        print(f"\nStarting training with {len(config)} models...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for name, settings in config.items():
            print(f"Training {name}...")
            grid = GridSearchCV(
                settings["model"],
                settings["params"],
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)
            self.best_estimators[name] = grid.best_estimator_
            print(f"  Best Params: {grid.best_params_}")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Calculates metrics for all trained models.
        """
        self.X_test = X_test
        self.y_test = y_test
        metrics_list = []

        print("\nEvaluating models...")
        for name, model in self.best_estimators.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics_list.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob)
            })

        self.results = pd.DataFrame(metrics_list).set_index("Model")
        print("\n--- Performance Comparison ---")
        print(self.results)
        return self.results

    def plot_confusion_matrices(self):
        """Plots side-by-side confusion matrices."""
        num_models = len(self.best_estimators)
        fig, axes = plt.subplots(1, num_models, figsize=(18, 5))
        
        for ax, (name, model) in zip(axes, self.best_estimators.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title(f"{name}\nConfusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        
        plt.tight_layout()
        plt.show()

    def plot_roc_pr_curves(self):
        """Plots Overlaid ROC and Precision-Recall Curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for name, model in self.best_estimators.items():
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')

        # Formatting ROC
        ax1.plot([0, 1], [0, 1], 'k--', lw=1)
        ax1.set_title("ROC Curves Comparison")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend()

        # Formatting PR
        ax2.set_title("Precision-Recall Curves Comparison")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def save_models(self, models_dir: str = "saved_models"):
        """Saves all trained models and the test set to disk."""
        os.makedirs(models_dir, exist_ok=True)
        for name, model in self.best_estimators.items():
            path = os.path.join(models_dir, f"{name}.pkl")
            joblib.dump(model, path)
            print(f"  Saved {name} -> {path}")
        joblib.dump((self.X_test, self.y_test), os.path.join(models_dir, "test_data.pkl"))
        print("  Saved test data.")

    def load_models(self, models_dir: str = "saved_models"):
        """Loads all saved models and test set from disk."""
        model_names = ["LogisticRegression", "RandomForest", "XGBoost"]
        for name in model_names:
            path = os.path.join(models_dir, f"{name}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}. Run the notebook first to train and save models.")
            self.best_estimators[name] = joblib.load(path)
            print(f"  Loaded {name} from {path}")
        self.X_test, self.y_test = joblib.load(os.path.join(models_dir, "test_data.pkl"))
        print("  Loaded test data.")

    def explain_best_model(self):
        """
        Uses SHAP to explain the best performing model (Selection based on F1).
        Note: SHAP works best natively with Tree models (XGBoost/RF).
        """
        best_model_name = self.results["F1-Score"].idxmax()
        model = self.best_estimators[best_model_name]
        
        print(f"\nExplaining best model: {best_model_name} using SHAP...")

        # Handling Pipeline vs Native objects for SHAP
        if "Pipeline" in str(type(model)):
            # SHAP explainer for linear models
            explainer = shap.LinearExplainer(
                model.named_steps['clf'], 
                model.named_steps['scaler'].transform(self.X_test)
            )
            shap_values = explainer.shap_values(
                model.named_steps['scaler'].transform(self.X_test)
            )
        else:
            # TreeExplainer for XGBoost/RandomForest
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_test)

        # 1. Summary Plot (Beeswarm)
        plt.figure(figsize=(10, 6))
        plt.title(f"SHAP Summary Plot ({best_model_name})")
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.show()

        # 2. Native Feature Importance (Gain/Gini)
        if hasattr(model, "feature_importances_"):
            plt.figure(figsize=(10, 6))
            importances = pd.Series(
                model.feature_importances_, index=self.X_test.columns
            ).sort_values(ascending=False).head(10)
            
            sns.barplot(x=importances.values, y=importances.index, palette="viridis")
            plt.title(f"Native Feature Importance ({best_model_name})")
            plt.xlabel("Importance Score")
            plt.show()


if __name__ == "__main__":
    pipeline = ChurnPipeline()

    # Load pre-trained models saved by Churn_Modelling.ipynb
    print("Loading saved models...")
    pipeline.load_models("saved_models")

    # Evaluate
    pipeline.evaluate(pipeline.X_test, pipeline.y_test)

    # Visualize
    print("\nGenerating visualizations...")
    pipeline.plot_confusion_matrices()
    pipeline.plot_roc_pr_curves()

    # Explain
    pipeline.explain_best_model()