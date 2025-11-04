import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

script_dir = "C:/Users/ollie/OneDrive - Bath Spa University/AI Ass/AI-Assessment-S1"
print("Script directory:", script_dir)

images_dir = os.path.join(script_dir, "images")
results_dir = os.path.join(script_dir, "results")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
file_path = os.path.join(script_dir, "car.data")

if not os.path.exists(file_path): #Error handling
    print("ERROR: File not found at", file_path)
    print("Files in directory:", os.listdir(script_dir))
    raise FileNotFoundError("Could not find car.data")

print("Loading data from:", file_path)
data = pd.read_csv(file_path, names=columns)
print("Data loaded:", data.shape[0], "rows")

data_str = data.copy()

encoders = {col: LabelEncoder() for col in columns}
for col in columns:
    data[col] = encoders[col].fit_transform(data[col])

X = data.drop("class", axis=1)
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test_idx = X_test.index

clf = DecisionTreeClassifier(criterion="entropy", random_state=42) #Decision Tree Model - Assistance from LLM for implementation 
clf.fit(X_train, y_train)

model_path = os.path.join(script_dir, "decision_tree_model.joblib")
joblib.dump(clf, model_path)
print("Model saved")

def manual_rule_instance(row): ##Rule-Based System Design logic - Assistance from LLM for debugging & implementation
    buying = row["buying"]
    maint = row["maint"]
    doors = row["doors"]
    persons = row["persons"]
    lug_boot = row["lug_boot"]
    safety = row["safety"]
    if safety == "high" and buying in ("low", "med"):
        return "vgood"
    if safety == "high" and buying == "high" and maint in ("low", "med"):
        return "good"
    if safety == "med" and persons == "more" and lug_boot == "big":
        return "good"
    if safety == "med" and buying in ("low", "med"):
        return "acc"
    if persons == "2" and doors == "2" and buying == "vhigh":
        return "unacc"
    if maint == "vhigh" and buying == "vhigh":
        return "unacc"
    return "unacc"

test_original = data_str.loc[X_test_idx].reset_index(drop=True)
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

y_pred_tree = clf.predict(X_test_reset)
y_pred_tree_str = encoders["class"].inverse_transform(y_pred_tree)
y_test_str = encoders["class"].inverse_transform(y_test_reset)

y_pred_manual = test_original.apply(manual_rule_instance, axis=1).values
y_pred_manual_enc = encoders["class"].transform(y_pred_manual)

print("") #UI For formatting output to user
print("=" * 60)
print("RESULTS")
print("=" * 60)
print("Decision Tree Accuracy:", accuracy_score(y_test_reset, y_pred_tree)) #Decision Tree results
print("Manual Rules Accuracy:", accuracy_score(y_test_str, y_pred_manual)) #RB Algo results
print("")
print("-" * 60)
print("Decision Tree Classification Report:")
print("-" * 60)
print(classification_report(y_test_reset, y_pred_tree, target_names=encoders["class"].classes_))
print("")
print("-" * 60)
print("Manual Rules Classification Report:")
print("-" * 60)
print(classification_report(y_test_str, y_pred_manual, target_names=encoders["class"].classes_))