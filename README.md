# Student Placement Prediction Using CGPA

A complete, beginner-friendly machine learning mini-project that predicts whether a college student is likely to be placed during campus recruitments based on their **CGPA** (Cumulative Grade Point Average).

Built purely with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 1. Project Overview

Predicting student placement outcomes helps students and academic counselors understand the relationship between academic consistency and campus placement opportunities. In this project:
- **Input Feature:** `CGPA` (Numeric value on a 0.0 to 10.0 scale)
- **Target Variable:** `Placement Status` (Binary: `0 = Not Placed`, `1 = Placed`)
- **Machine Learning Model:** Logistic Regression
- **User Interface:** Streamlit (single-page interactive dashboard)

> **Disclaimer:** The dataset (`dataset.csv`) is synthetic and generated strictly for academic and demonstration purposes. It does not reflect real-world university placement records.

---

## 2. Machine Learning Algorithm: Logistic Regression

**Logistic Regression** is a fundamental supervised learning classification algorithm used to predict the probability of a categorical dependent variable (here, binary placement status: $y \in \{0, 1\}$).

### How it Works:
1. **Linear Combination:**
   The algorithm first computes a linear combination of the input feature ($x = \text{CGPA}$):
   $$z = w \cdot x + b$$
   Where $w$ is the learned weight (coefficient) and $b$ is the intercept (bias).

2. **Sigmoid Activation Function:**
   To convert this continuous value $z$ into a valid probability between $0$ and $1$, the logistic (sigmoid) function is applied:
   $$P(\text{Placed} = 1 \mid \text{CGPA}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

3. **Classification Decision:**
   - If $P \ge 0.50$, the student is predicted as **"Likely Placed"** ($1$).
   - If $P < 0.50$, the student is predicted as **"Likely Not Placed"** ($0$).

---

## 3. Project Structure

```text
placement-prediction/
├── app.py                # Streamlit web application & user interface
├── train_model.py        # Model training, evaluation, and saving script
├── dataset.csv           # Synthetic dataset (200 records of CGPA vs Placed)
├── model.pkl             # Serialized trained Logistic Regression model
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation and guide
```

---

## 4. Setup & Installation

Follow these steps to run the project on your machine:

### Step 1: Open Terminal & Navigate to the Project Folder
```bash
cd "/Users/aditya/Projects/ML mnp/placement-prediction"
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Execution Commands

### Step 4: Train the Machine Learning Model
Train the Logistic Regression model and generate `model.pkl`:
```bash
python train_model.py
```

**Expected Output:**
```text
============================================================
🎓 Student Placement Prediction - Model Training
============================================================
[1/5] Loading dataset... Total records: 200
[2/5] Preparing features (CGPA) and target (placed)...
[3/5] Splitting data into training (80%) and testing (20%) sets...
[4/5] Training Logistic Regression model...
      Learned coefficient (weight): 2.6802
      Learned intercept (bias):     -19.0483
      50% Probability CGPA Boundary: 7.11
[5/5] Evaluating model performance on test set...
----------------------------------------
Accuracy:  87.50%
Precision: 95.00%
Recall:    82.61%
F1-Score:  88.37%
----------------------------------------
✅ Model successfully trained and saved to: model.pkl
============================================================
```

### Step 5: Launch the Streamlit Web Application
Run the Streamlit application in your terminal:
```bash
streamlit run app.py
```
Streamlit will automatically open your default web browser at `http://localhost:8501`.

---

## 6. User Interface Features

1. **Placement Predictor:**
   - Numeric input field for CGPA (0.00 to 10.00).
   - Interactive **Predict Placement** button.
   - Outcome badge showing **"Likely Placed"** or **"Likely Not Placed"**.
   - Estimated placement probability gauge (e.g., `85.40%`).

2. **Data & Sigmoid Visualization:**
   - Scatter points of historical student placement data.
   - Smooth S-curve (Sigmoid function) showing how placement probability rises with CGPA.
   - Dynamic red marker and dotted projection lines indicating your current CGPA input.

3. **Model Evaluation Metrics:**
   - Four clean metric displays: **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
   - **Confusion Matrix** visual heatmap with breakdown of True Negatives, False Positives, False Negatives, and True Positives.

---

## 7. Edge Cases & Testing Guide

| Test Scenario | CGPA Input | Expected Result | Placement Probability |
| :--- | :--- | :--- | :--- |
| Low Academic Score | `5.50` | Likely Not Placed | ~3.5% |
| Average Score | `6.80` | Likely Not Placed | ~30.5% |
| Decision Boundary | `7.11` | Boundary Point | ~50.0% |
| Above Average Score | `7.80` | Likely Placed | ~86.5% |
| High Academic Score | `9.20` | Likely Placed | ~99.6% |
| Boundary Minimum | `0.00` | Likely Not Placed | ~0.0% |
| Boundary Maximum | `10.00` | Likely Placed | ~99.9% |

---

## 8. Technology Stack

- **Language:** Python 3.9+
- **Web App:** Streamlit
- **Data Handling:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib
- **Model Serialization:** Joblib
