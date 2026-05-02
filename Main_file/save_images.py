import os, joblib, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

BASE      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, '..', 'saved_models')
OUT       = os.path.join(BASE, '..', 'images')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, '..', 'Customer Churn.csv'))
models = {n: joblib.load(os.path.join(MODELS_DIR, f'{n}.pkl'))
          for n in ['LogisticRegression', 'RandomForest', 'XGBoost']}
X_test, y_test = joblib.load(os.path.join(MODELS_DIR, 'test_data.pkl'))

# 1. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, model) in zip(axes, models.items()):
    sns.heatmap(confusion_matrix(y_test, model.predict(X_test)),
                annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'confusion_matrices.png'), dpi=150)
plt.close()
print('Saved: confusion_matrices.png')

# 2. ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title('ROC Curves')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'roc_curves.png'), dpi=150)
plt.close()
print('Saved: roc_curves.png')

# 3. Precision-Recall Curves
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(recall, precision, label=f'{name} (AUC={auc(recall, precision):.2f})')
ax.set_title('Precision-Recall Curves')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'precision_recall_curves.png'), dpi=150)
plt.close()
print('Saved: precision_recall_curves.png')

# 4. Feature Importance (Random Forest)
model = models['RandomForest']
imp = pd.Series(model.feature_importances_, index=X_test.columns)\
        .sort_values(ascending=True).tail(12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(imp.index, imp.values, color='steelblue', edgecolor='none')
ax.set_title('Feature Importance - Random Forest')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'feature_importance.png'), dpi=150)
plt.close()
print('Saved: feature_importance.png')

# 5. SHAP Values (XGBoost)
explainer  = shap.TreeExplainer(models['XGBoost'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'shap_values.png'), dpi=150)
plt.close()
print('Saved: shap_values.png')

print(f'\nAll images saved to: {os.path.abspath(OUT)}')
