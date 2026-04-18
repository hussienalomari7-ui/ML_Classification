# ============================================================
# KNN Tumor Classifier - Practical Interactive Version
# ============================================================

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------------------------------------------
# 1) Load and prepare the dataset
# ------------------------------------------------------------

rs = 123

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/tumor.csv"
tumor_df = pd.read_csv(dataset_url)

X = tumor_df.iloc[:, :-1]
y = tumor_df.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=rs
)

# ------------------------------------------------------------
# 2) Train a final model
# ------------------------------------------------------------

# You can change this value manually,
# or replace it with the best_k you found in your experiment
best_k = 5

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train.values.ravel())

# ------------------------------------------------------------
# 3) Optional: evaluation function
# ------------------------------------------------------------

def evaluate_metrics(y_true, y_pred):
    results = {}
    results["accuracy"] = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary"
    )

    results["precision"] = precision
    results["recall"] = recall
    results["f1score"] = f1
    return results

# Quick test evaluation
test_preds = knn_model.predict(X_test)
print("Model test performance:")
print(evaluate_metrics(y_test, test_preds))

# ------------------------------------------------------------
# 4) Function to read patient values from user input
# ------------------------------------------------------------

def get_patient_input(feature_names):
    """
    Ask the user to enter each feature value one by one.
    Returns a list of numeric values in the correct feature order.
    """
    patient_values = []

    print("\nPlease enter the patient's values exactly as requested.")
    print("Use numbers only.\n")

    for feature in feature_names:
        while True:
            try:
                value = float(input(f"Enter value for '{feature}': "))
                patient_values.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value only.")

    return patient_values

# ------------------------------------------------------------
# 5) Function to predict tumor class for a new patient
# ------------------------------------------------------------

def predict_tumor_from_values(model, feature_names, patient_values):
    """
    Convert the input values into a DataFrame,
    then predict both class and probabilities.
    """
    new_patient = pd.DataFrame([patient_values], columns=feature_names)

    prediction = model.predict(new_patient)[0]
    probabilities = model.predict_proba(new_patient)[0]

    # NOTE:
    # This mapping assumes:
    # 0 = Benign
    # 1 = Malignant
    # Verify the mapping with your dataset if needed
    if prediction == 0:
        label = "Benign"
    else:
        label = "Malignant"

    result = {
        "prediction_numeric": prediction,
        "prediction_label": label,
        "probability_class_0": probabilities[0],
        "probability_class_1": probabilities[1]
    }

    return result

# ------------------------------------------------------------
# 6) Main interactive program
# ------------------------------------------------------------

print("\nFeature order expected by the model:")
for i, col in enumerate(X.columns, start=1):
    print(f"{i}. {col}")

# Read patient values from the user
patient_values = get_patient_input(X.columns)

# Predict
result = predict_tumor_from_values(knn_model, X.columns, patient_values)

# Display results
print("\n================ Prediction Result ================")
print("Predicted numeric class:", result["prediction_numeric"])
print("Predicted label:", result["prediction_label"])
print("Probability of class 0:", result["probability_class_0"])
print("Probability of class 1:", result["probability_class_1"])
print("===================================================")