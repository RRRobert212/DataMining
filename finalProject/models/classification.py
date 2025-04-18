import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('../data/all_data_yesorno.csv')

# Replace 'ProfitYesNo' with your actual column name for the target variable (0 or 1)
X = df.drop(columns=['Profit Yes or No'])  # Feature matrix (exclude the target variable)
y = df['Profit Yes or No']  # Target variable (Profit Yes/No)

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Feature Importance
feature_importances = model.feature_importances_

# Plot Feature Importances
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from Random Forest')
plt.show()

# Confusion Matrix Plot
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['Not Profitable', 'Profitable'], yticklabels=['Not Profitable', 'Profitable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Cross-validation (for additional performance evaluation)
cross_val_acc = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='accuracy')
print(f"Cross-validated Accuracy: {cross_val_acc.mean():.4f}")

# Additional Evaluation
print(f"Random Forest Classifier Accuracy (Test Set): {accuracy:.4f}")
print("Classification Report (Test Set):")
print(classification_rep)
print("Confusion Matrix (Test Set):")
print(conf_matrix)

# Correlation between features and target (ProfitYesNo)
df_corr = df.copy()
df_corr['Profit Yes or No'] = y
correlation_matrix = df_corr.corr()

# Plot the correlation of features with the target (ProfitYesNo)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix[['Profit Yes or No']].sort_values(by='Profit Yes or No', ascending=False), annot=True, cmap='coolwarm')
plt.title('Feature Correlation with Profit (Yes/No)')
plt.show()

# Visualize the relationship between a key feature (e.g., 'AggressionFactor') and the target variable
# Replace 'AggressionFactor' with the actual feature name that you want to analyze
plt.figure(figsize=(8, 6))
sns.boxplot(x='Profit Yes or No', y='AggressionFactor', data=df)
plt.title('Aggression Factor vs Profitability')
plt.xlabel('Profit (0 = No, 1 = Yes)')
plt.ylabel('Aggression Factor')
plt.show()

# You can add similar plots for other features as needed (e.g., scatter plots or boxplots for 'AggressionFactor', etc.)
