# Rule-Based AI vs Decision Tree Classification – Car Evaluation

This repository contains the source code and dataset used for a university assignment investigating how effectively a manually written rule-based system can replicate human-like decision-making when evaluating car acceptability, and comparing this to a data-driven decision tree model.

## 📌 Objective

To evaluate whether a hand-crafted rule-based AI system can match the performance of a machine-learned model in classifying cars based on common consumer attributes such as safety, capacity, and cost.

The experiment uses the **Car Evaluation Dataset** from the UCI Machine Learning Repository.

---

## 🧠 Methods

Two AI approaches were implemented:

| Model | Description | Nature |
|-------|------------|--------|
| **Rule-Based System** | Manually engineered IF-THEN rules based on intuitive human logic | Symbolic, deterministic |
| **Decision Tree Classifier** | Learned model using `scikit-learn` (`DecisionTreeClassifier`, entropy criterion) | Statistical, data-driven |

Both models were evaluated using accuracy, precision, recall, F1 score and support on the same 80/20 train-test split.

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `RB_AI.py` | Full Python script containing both the manual rule system and decision tree implementation |
| `car.data` | Raw dataset from UCI repository |
| `car.names` *(optional)* | Attribute information file |
| `decision_tree_model.joblib` *(optional)* | Saved model output for reproducibility |
| *(Images/Results folder if generated)* | Confusion matrix / metric outputs |

---

## 📊 Dataset

**Car Evaluation Dataset**  
Source: UCI Machine Learning Repository  
Link: https://archive.ics.uci.edu/dataset/19/car+evaluation  

| Attribute | Values |
|----------|--------|
| Buying price | vhigh, high, med, low |
| Maintenance cost | vhigh, high, med, low |
| Doors | 2, 3, 4, 5+ |
| Persons | 2, 4, more |
| Luggage boot size | small, med, big |
| Safety | low, med, high |
| Class (label) | unacc, acc, good, vgood |

---

## 🏗️ Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib
