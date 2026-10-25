"""
Student Placement Prediction - Streamlit Application

A clean, single-page machine learning mini-project interface
for predicting student placement outcomes based on CGPA using Logistic Regression.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import streamlit as st

# Configure the Streamlit page layout and title
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# Helper Functions: Load Data and Model
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

@st.cache_data
def load_dataset():
    """Load the synthetic student dataset from CSV."""
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None

@st.cache_resource
def load_trained_model():
    """Load the trained Logistic Regression model from joblib pickle."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Load dataset and model
df = load_dataset()
model = load_trained_model()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.title("Student Placement Prediction")
st.subheader("Predict student placement outcomes using CGPA.")
st.caption(
    "This beginner-friendly mini-project uses a Logistic Regression model trained on student CGPA data "
    "to calculate the probability of placement and predict whether a student is likely to be placed."
)

st.divider()

# Ensure model and dataset are loaded properly
if model is None or df is None:
    st.error(
        "⚠️ Model file (`model.pkl`) or dataset (`dataset.csv`) not found! "
        "Please run `python train_model.py` first to generate the dataset and train the model."
    )
    st.stop()

# ---------------------------------------------------------
# Section 1: User Input & Prediction
# ---------------------------------------------------------
st.header("1. Placement Prediction")

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    # Numeric CGPA input field ranging from 0 to 10
    cgpa_input = st.number_input(
        label="Enter Student CGPA (Scale 0.0 to 10.0):",
        min_value=0.0,
        max_value=10.0,
        value=7.50,
        step=0.10,
        format="%.2f",
        help="Input the cumulative grade point average between 0.0 and 10.0"
    )

    # Predict Placement button
    predict_clicked = st.button("Predict Placement", type="primary", use_container_width=True)

# Keep track of prediction in session state for chart highlighting
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.last_cgpa = 7.50
    st.session_state.last_prob = 0.0
    st.session_state.last_label = ""

if predict_clicked:
    # Validate CGPA input range
    if cgpa_input < 0.0 or cgpa_input > 10.0:
        st.error("Invalid CGPA! Please provide a value between 0.0 and 10.0.")
    else:
        # Create input DataFrame with appropriate feature name
        input_data = pd.DataFrame({"cgpa": [cgpa_input]})

        # Predict outcome and placement probability using the trained model
        pred_class = model.predict(input_data)[0]
        prob_placed = model.predict_proba(input_data)[0][1]

        # Update session state
        st.session_state.prediction_made = True
        st.session_state.last_cgpa = cgpa_input
        st.session_state.last_prob = prob_placed
        st.session_state.last_label = "Likely Placed" if pred_class == 1 else "Likely Not Placed"

with col_result:
    if st.session_state.prediction_made:
        st.subheader("Prediction Result")

        label = st.session_state.last_label
        probability = st.session_state.last_prob

        # Display outcome status
        if label == "Likely Placed":
            st.success(f"Outcome: **{label}**")
        else:
            st.warning(f"Outcome: **{label}**")

        # Display estimated placement probability
        st.metric(
            label="Estimated Placement Probability",
            value=f"{probability * 100:.2f}%"
        )

        st.caption(
            f"Based on the learned logistic curve, a CGPA of {st.session_state.last_cgpa:.2f} "
            f"gives an estimated placement likelihood of {probability * 100:.2f}%."
        )
    else:
        st.info("Enter a CGPA value on the left and click **Predict Placement** to view the outcome and probability.")

st.divider()

# ---------------------------------------------------------
# Section 2: Data & Model Visualization Chart
# ---------------------------------------------------------
st.header("2. CGPA vs Placement Outcome Visualization")
st.write(
    "The plot below shows the distribution of students in the dataset along with the "
    "Logistic Regression sigmoid curve representing the probability of placement for any given CGPA."
)

# Generate Matplotlib chart with clean white background and blue accents
fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="#FFFFFF")
ax.set_facecolor("#F8FAFC")

# Scatter plot of training/sample data points
not_placed = df[df["placed"] == 0]
placed = df[df["placed"] == 1]

ax.scatter(
    not_placed["cgpa"],
    not_placed["placed"],
    color="#94A3B8",
    alpha=0.65,
    s=35,
    edgecolors="none",
    label="Sample: Not Placed (0)"
)

ax.scatter(
    placed["cgpa"],
    placed["placed"],
    color="#2563EB",
    alpha=0.75,
    s=35,
    edgecolors="none",
    label="Sample: Placed (1)"
)

# Plot the fitted Logistic Regression sigmoid probability curve
x_vals = np.linspace(0, 10, 300)
x_curve_df = pd.DataFrame({"cgpa": x_vals})
y_probs = model.predict_proba(x_curve_df)[:, 1]

ax.plot(
    x_vals,
    y_probs,
    color="#1D4ED8",
    linewidth=2.5,
    label="Logistic Regression Probability Curve"
)

# Decision threshold at 0.5 probability
ax.axhline(
    y=0.5,
    color="#CBD5E1",
    linestyle="--",
    linewidth=1.2,
    label="Decision Boundary (0.50 Probability)"
)

# Highlight candidate position if prediction has been performed
if st.session_state.prediction_made:
    candidate_cgpa = st.session_state.last_cgpa
    candidate_prob = st.session_state.last_prob

    # Point on sigmoid curve
    ax.scatter(
        [candidate_cgpa],
        [candidate_prob],
        color="#DC2626",
        s=120,
        zorder=5,
        edgecolors="#FFFFFF",
        linewidth=2,
        label=f"Current Input (CGPA: {candidate_cgpa:.2f})"
    )

    # Vertical reference line
    ax.vlines(
        x=candidate_cgpa,
        ymin=0,
        ymax=candidate_prob,
        color="#DC2626",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8
    )

    # Horizontal reference line
    ax.hlines(
        y=candidate_prob,
        xmin=0,
        xmax=candidate_cgpa,
        color="#DC2626",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8
    )

# Formatting axes and labels
ax.set_xlim(0, 10)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Student CGPA (0 to 10)", fontsize=11, color="#1E293B")
ax.set_ylabel("Placement Probability / Outcome", fontsize=11, color="#1E293B")
ax.grid(True, linestyle=":", alpha=0.5, color="#CBD5E1")
ax.legend(loc="center left", frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=9)

# Clean spine styling
for spine in ax.spines.values():
    spine.set_color("#E2E8F0")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

# ---------------------------------------------------------
# Section 3: Model Evaluation Metrics
# ---------------------------------------------------------
st.header("3. Model Evaluation Metrics")
st.write(
    "These performance metrics were evaluated on the held-out test set (20% of dataset records) "
    "using standard classification metrics and a confusion matrix."
)

# Compute metrics dynamically using the same test split strategy as train_model.py
X = df[["cgpa"]]
y = df["placed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display 4 metrics in columns
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(label="Accuracy", value=f"{accuracy * 100:.1f}%")
    st.caption("Proportion of total predictions that were correct.")

with metric_col2:
    st.metric(label="Precision", value=f"{precision * 100:.1f}%")
    st.caption("Accuracy of positive (Placed) predictions.")

with metric_col3:
    st.metric(label="Recall", value=f"{recall * 100:.1f}%")
    st.caption("Proportion of actual placed students identified.")

with metric_col4:
    st.metric(label="F1-Score", value=f"{f1 * 100:.1f}%")
    st.caption("Harmonic mean of Precision and Recall.")

st.write("")

# Confusion Matrix Visual Display
st.subheader("Confusion Matrix")

cm_col1, cm_col2 = st.columns([1, 1], gap="large")

with cm_col1:
    fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5), facecolor="#FFFFFF")
    im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    # Clean axes and labels
    ax_cm.set_title("Test Set Confusion Matrix", fontsize=11, color="#1E293B", pad=12)
    classes = ["Not Placed (0)", "Placed (1)"]
    tick_marks = np.arange(len(classes))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(classes, fontsize=9, color="#1E293B")
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(classes, fontsize=9, color="#1E293B")
    ax_cm.set_ylabel("Actual Label", fontsize=10, color="#1E293B")
    ax_cm.set_xlabel("Predicted Label", fontsize=10, color="#1E293B")

    # Annotate counts inside the confusion matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "#1E293B",
                fontsize=12, fontweight="bold"
            )

    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with cm_col2:
    st.write("**Confusion Matrix Breakdown:**")
    st.write(f"- **True Negatives (TN):** {cm[0][0]} — Correctly predicted Not Placed.")
    st.write(f"- **False Positives (FP):** {cm[0][1]} — Incorrectly predicted Placed.")
    st.write(f"- **False Negatives (FN):** {cm[1][0]} — Incorrectly predicted Not Placed.")
    st.write(f"- **True Positives (TP):** {cm[1][1]} — Correctly predicted Placed.")
    st.write(f"- **Total Test Records Evaluated:** {len(y_test)} students.")

st.divider()

# ---------------------------------------------------------
# Synthetic Dataset Notice / Disclaimer
# ---------------------------------------------------------
st.info(
    "📌 **Dataset Notice:** The dataset used in this project (`dataset.csv`) is synthetic and generated "
    "strictly for academic and educational demonstration purposes. Its performance does not represent "
    "actual university placement statistics or real-world campus recruitment outcomes."
)
