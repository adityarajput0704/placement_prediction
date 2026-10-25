"""
Student Placement Prediction - Model Training Script

This script trains a Logistic Regression model to predict whether a student
is likely to be placed based on their CGPA. It evaluates the model on a test
dataset and saves the trained model to 'model.pkl'.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

def main():
    print("=" * 60)
    print("🎓 Student Placement Prediction - Model Training")
    print("=" * 60)

    # 1. Resolve file paths relative to this script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset.csv")
    model_path = os.path.join(script_dir, "model.pkl")

    # 2. Load the dataset
    print(f"\n[1/5] Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"      Total records loaded: {len(df)}")
    print(f"      Dataset head:\n{df.head()}")

    # 3. Prepare features (X) and target (y)
    print("\n[2/5] Preparing features and target...")
    X = df[['cgpa']]        # Input feature: CGPA (2D DataFrame for scikit-learn)
    y = df['placed']       # Target label: 0 (Not Placed) or 1 (Placed)

    # 4. Split dataset into training set (80%) and testing set (20%)
    print("\n[3/5] Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training samples: {len(X_train)}")
    print(f"      Testing samples:  {len(X_test)}")

    # 5. Initialize and train the Logistic Regression model
    print("\n[4/5] Training Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Display learned model parameters
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    decision_threshold_cgpa = -intercept / slope
    print(f"      Learned coefficient (weight): {slope:.4f}")
    print(f"      Learned intercept (bias):     {intercept:.4f}")
    print(f"      50% Probability CGPA Boundary: {decision_threshold_cgpa:.2f}")

    # 6. Evaluate model on test data
    print("\n[5/5] Evaluating model performance on test set...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("-" * 40)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("-" * 40)
    print("\nConfusion Matrix:")
    print("                   Predicted 0    Predicted 1")
    print(f"Actual 0 (Not Placed):   {cm[0][0]:<14} {cm[0][1]}")
    print(f"Actual 1 (Placed):       {cm[1][0]:<14} {cm[1][1]}")
    print("-" * 40)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Placed (0)", "Placed (1)"]))

    # 7. Save trained model to disk
    joblib.dump(model, model_path)
    print(f"✅ Model successfully trained and saved to: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
