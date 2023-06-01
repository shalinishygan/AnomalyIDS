import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the NSL-KDD dataset
data = pd.read_csv('path/to/nsl-kdd-dataset.csv')

# Preprocess the dataset
dataset = dataset.dropna()

# Split the dataset into features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy and display confusion
matrix accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
print('Confusion Matrix:')
print(confusion_mat)

# Plot the confusion matrix
labels = ['normal', 'anomaly']
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = [0, 1]

plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()