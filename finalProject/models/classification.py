import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_predict
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

#looaaaad it up
df = pd.read_csv('../data/all_data_yesorno.csv')
X = df.drop(columns=['Profit Yes or No'])
y = df['Profit Yes or No']

#scale everything
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#balance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

#built in random forest model from sklearn
model = RandomForestClassifier(n_estimators=100, random_state=42)

#n-fold cross validation
cv_results = cross_validate(
    model, 
    X_resampled, 
    y_resampled, 
    cv=13, 
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True,
    return_estimator=True
)

#----------------------------------#
#print results
print("\n=== Cross-Validation Results ===")
print(f"Mean Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Mean Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Mean Recall: {cv_results['test_recall'].mean():.4f}")
print(f"Mean F1-Score: {cv_results['test_f1'].mean():.4f}")
print(f"Mean ROC AUC: {cv_results['test_roc_auc'].mean():.4f}")

#----------------------------------#
#generate some predictions to create a confusion matrix
y_pred_cv = cross_val_predict(model, X_resampled, y_resampled, cv=5)
conf_matrix_cv = confusion_matrix(y_resampled, y_pred_cv)

#print confusion matrices
print("\n=== Confusion Matrix (Raw Counts) ===")
print(conf_matrix_cv)
conf_matrix_norm = conf_matrix_cv.astype('float') / conf_matrix_cv.sum(axis=1)[:, np.newaxis]
print("\n=== Confusion Matrix (Normalized) ===")
print(np.round(conf_matrix_norm, 2))

#----------------------------------#
#assess feature importance and correlations
feature_importances = np.mean([
    est.feature_importances_ for est in cv_results['estimator']
], axis=0)
feature_names = df.drop(columns=['Profit Yes or No']).columns


feature_corr = df.corr()['Profit Yes or No'].drop('Profit Yes or No')

#show both in one graph
feature_analysis = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances,
    'Correlation': [feature_corr[f] for f in feature_names]
}).sort_values('Importance', ascending=False)

#----------------------------------#
#plotting
plt.figure(figsize=(12, 8))
bars = plt.barh(
    feature_analysis['Feature'],
    feature_analysis['Importance'],
    color=np.where(feature_analysis['Correlation'] > 0, 'green', 'red'),
    alpha=0.6
)
for i, (imp, corr) in enumerate(zip(feature_analysis['Importance'], 
                                   feature_analysis['Correlation'])):
    plt.text(imp/2, i, f'Corr: {corr:.2f}', va='center', color='white', fontweight='bold')

plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance and Profitability Correlation\n(Green = Positive, Red = Negative)')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.legend(
    handles=[
        plt.Rectangle((0,0),1,1,fc='green',alpha=0.6),
        plt.Rectangle((0,0),1,1,fc='red',alpha=0.6)
    ],
    labels=['Positive Correlation', 'Negative Correlation'],
    loc='lower right'
)
plt.tight_layout()
plt.show()
#----------------------------------#