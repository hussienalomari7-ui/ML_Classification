# ============================================================
# K-Nearest Neighbors (KNN) for Tumor Classification
# Educational / Academic Version
# ============================================================

# ------------------------------------------------------------
# 1) Import the required libraries
# ------------------------------------------------------------

# pandas:
# Used to read and manipulate tabular data (DataFrame)
import pandas as pd

# numpy:
# Used for numerical operations and arrays
import numpy as np

# KNeighborsClassifier:
# This is the KNN classification model from scikit-learn
from sklearn.neighbors import KNeighborsClassifier

# train_test_split:
# Used to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Metrics for evaluating classification performance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

# matplotlib:
# Used for plotting graphs
import matplotlib.pyplot as plt

# seaborn:
# Used for better data visualization
import seaborn as sns

# warnings:
# Used to suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ------------------------------------------------------------
# 2) Set a random seed for reproducibility
# ------------------------------------------------------------

# random_state helps us get the same train-test split every time
# we run the code, so the experiment becomes reproducible
rs = 123


# ------------------------------------------------------------
# 3) Load the dataset
# ------------------------------------------------------------

# This is the dataset URL used in the lab
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/tumor.csv"

# Read the CSV file into a pandas DataFrame
tumor_df = pd.read_csv(dataset_url)

# Show the first 5 rows to inspect the data
print("First 5 rows of the dataset:")
print(tumor_df.head())

# Show column names
print("\nColumn names:")
print(tumor_df.columns.tolist())


# ------------------------------------------------------------
# 4) Separate features (X) and target (y)
# ------------------------------------------------------------

# X contains all columns except the last one
# These are the input features used by the model
X = tumor_df.iloc[:, :-1]

# y contains only the last column
# This is the target variable (the label we want to predict)
y = tumor_df.iloc[:, -1:]

print("\nShape of X (features):", X.shape)
print("Shape of y (target):", y.shape)


# ------------------------------------------------------------
# 5) Explore the dataset
# ------------------------------------------------------------

# describe() gives summary statistics for numerical columns:
# count, mean, std, min, max, quartiles, etc.
print("\nStatistical summary of the features:")
print(X.describe())

# value_counts(normalize=True) shows class distribution as percentages
print("\nTarget class distribution (normalized):")
print(y.value_counts(normalize=True))

# Plot the target distribution
plt.figure(figsize=(6, 4))
y.value_counts().plot.bar(color=["green", "red"])
plt.title("Target Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# ------------------------------------------------------------
# 6) Split the data into training and testing sets
# ------------------------------------------------------------

# test_size=0.2 means:
# 80% of the data will be used for training
# 20% of the data will be used for testing

# stratify=y means:
# preserve the original class distribution in both train and test sets

# random_state=rs means:
# use the fixed seed defined earlier for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=rs
)

print("\nTraining set shape:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting set shape:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ------------------------------------------------------------
# 7) Build the first KNN model
# ------------------------------------------------------------

# Create a KNN classifier with k = 5
# This means the model will look at the 5 nearest neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
# y_train.values.ravel() converts y from shape (n_samples, 1)
# into shape (n_samples,), which is the preferred shape for sklearn
knn_model.fit(X_train, y_train.values.ravel())


# ------------------------------------------------------------
# 8) Make predictions on the test set
# ------------------------------------------------------------

# predict() returns the predicted class labels for X_test
preds = knn_model.predict(X_test)

print("\nFirst 10 predictions:")
print(preds[:10])


# ------------------------------------------------------------
# 9) Define an evaluation function
# ------------------------------------------------------------

def evaluate_metrics(y_true, y_pred):
    """
    This function calculates important classification metrics:
    - accuracy
    - precision
    - recall
    - f1-score
    """

    results = {}

    # Accuracy = correct predictions / total predictions
    results["accuracy"] = accuracy_score(y_true, y_pred)

    # precision_recall_fscore_support computes precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary"
    )

    results["precision"] = precision
    results["recall"] = recall
    results["f1score"] = f1

    return results


# ------------------------------------------------------------
# 10) Evaluate the model
# ------------------------------------------------------------

metrics_result = evaluate_metrics(y_test, preds)

print("\nEvaluation metrics for K=5:")
print(metrics_result)

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, preds))


# ------------------------------------------------------------
# 11) Search for the best K from 1 to 50
# ------------------------------------------------------------

# We will store the results for each K value
results_all_k = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1score": []
}

# Try every K from 1 to 50
for k in range(1, 51):
    # Create a new model with the current k
    model = KNeighborsClassifier(n_neighbors=k)

    # Train the model on training data
    model.fit(X_train, y_train.values.ravel())

    # Predict on test data
    predictions = model.predict(X_test)

    # Evaluate
    result = evaluate_metrics(y_test, predictions)

    # Store results
    results_all_k["accuracy"].append(result["accuracy"])
    results_all_k["precision"].append(result["precision"])
    results_all_k["recall"].append(result["recall"])
    results_all_k["f1score"].append(result["f1score"])


# ------------------------------------------------------------
# 12) Convert results to DataFrame for easier analysis
# ------------------------------------------------------------

result_df = pd.DataFrame(results_all_k)

print("\nResults table (first 10 rows):")
print(result_df.head(10))


# ------------------------------------------------------------
# 13) Find the best K based on F1-score
# ------------------------------------------------------------

# idxmax() returns the index of the maximum value
# Since index starts at 0 but K starts at 1, we add 1
best_k = result_df["f1score"].idxmax() + 1

print("\nBest K based on F1-score:", best_k)


# ------------------------------------------------------------
# 14) Retrain the final model using the best K
# ------------------------------------------------------------

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train.values.ravel())

final_preds = final_model.predict(X_test)

final_metrics = evaluate_metrics(y_test, final_preds)

print("\nFinal model evaluation using best K:")
print(final_metrics)


# ------------------------------------------------------------
# 15) Plot F1-score versus K
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), result_df["f1score"], marker="o")
plt.xlabel("K")
plt.ylabel("F1 Score")
plt.title("F1 Score for Different Values of K")
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# 16) Example of real model usage on a new patient
# ------------------------------------------------------------

# IMPORTANT:
# The order of values must match the exact order of X.columns
print("\nFeature order:")
print(X.columns.tolist())

# Example patient
# Replace these values with a real patient's values if needed
new_patient = pd.DataFrame(
    [[5, 1, 1, 1, 2, 1, 3, 1, 1]],
    columns=X.columns
)

# Predict the class of the new patient
new_prediction = final_model.predict(new_patient)

# Predict class probabilities
new_probabilities = final_model.predict_proba(new_patient)

print("\nPrediction for the new patient:")
print("Predicted class:", new_prediction[0])
print("Predicted probabilities:", new_probabilities[0])

# Human-readable interpretation
# NOTE:
# In many versions of this lab:
# 0 may represent Benign and 1 may represent Malignant
# Verify this mapping in your dataset before using it in a report
if new_prediction[0] == 0:
    print("Predicted result: Benign tumor")
else:
    print("Predicted result: Malignant tumor")