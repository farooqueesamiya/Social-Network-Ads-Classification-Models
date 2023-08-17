import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Load the dataset from a CSV file
dataset = pd.read_csv('Social_Network_Ads.csv')

# Extract features (X) and target (y) values from the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Standardize features using StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# List of classifiers
classifiers = [
    DecisionTreeClassifier(criterion='entropy', random_state=0),
    SVC(kernel='rbf', random_state=0),
    GaussianNB(),
    RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0),
    SVC(kernel='linear', random_state=0),
    KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
    LogisticRegression(random_state=0)
]

# Loop through classifiers
for idx, classifier in enumerate(classifiers):
    classifier.fit(X_train, y_train)
    
    # Predict target values on the test set
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy score and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print classifier info, accuracy score, and confusion matrix
    print(f"Classifier {idx + 1} - {classifier.__class__.__name__}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Visualize the decision boundary
    plt.figure(figsize=(8, 6))
    X_set, y_set = sc.inverse_transform(X_test), y_test
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('salmon', 'dodgerblue'))(i), label=j)
    plt.title(f'{classifier.__class__.__name__} - Decision Boundary (Test set)\nAccuracy: {accuracy:.2f}')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
