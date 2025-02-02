import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# Loading the Data
data = pd.read_csv("D:/Misc. Work/PIP summer prog/feature_statistics.csv")

# Drop 'person_id' column if it exists
if 'person_id' in data.columns:
    data = data.drop(columns=['person_id'])

# Histograms
data.iloc[:, :-1].hist(figsize=(15, 10))
plt.suptitle('Histograms of Acceleration Features')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Box plots
plt.figure(figsize=(15, 8))
sns.boxplot(data=data.iloc[:, :-1])
plt.title('Box Plots of Acceleration Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(15, 8))
correlation = data.iloc[:, :-1].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Acceleration Features')
plt.show()

# Separating features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]  # Taking the last column as 'label'

# Training-Testing Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training using Decision Tree
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
clf.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = clf.predict(X_test_scaled)

# Compute and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualizing the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Checking for Overfitting
train_pred = clf.predict(X_train_scaled)
train_acc = accuracy_score(y_train, train_pred)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
if train_acc - accuracy_score(y_test, y_pred) > 0.05:  # Threshold of 0.05, can be adjusted
    print("Warning: The model might be overfitting!")
else:
    print("The model seems fine.")

# Visualization of the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Fallen', 'Fallen'], rounded=True)
plt.show()
