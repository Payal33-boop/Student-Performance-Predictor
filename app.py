
import streamlit as st
import pandas as pd
import joblib

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="centered"
)

# Load trained model
model = joblib.load("student_model.pkl")

# Application heading
st.title("📚 Student Exam Score Predictor")

st.write(
    "Predict a student's final examination performance "
    "using Machine Learning."
)

st.divider()

st.subheader("Enter Student Details")

# Student input form
with st.form("student_form"):

    age = st.number_input(
        "Student Age",
        min_value=15,
        max_value=22,
        value=18
    )

    studytime = st.selectbox(
        "Weekly Study Time",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Less than 2 hours",
            2: "2 to 5 hours",
            3: "5 to 10 hours",
            4: "More than 10 hours"
        }[x]
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=5
    )

    failures = st.number_input(
        "Previous Class Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    g1 = st.number_input(
        "First Period Grade (0-20)",
        min_value=0,
        max_value=20,
        value=12
    )

    g2 = st.number_input(
        "Second Period Grade (0-20)",
        min_value=0,
        max_value=20,
        value=13
    )

    schoolsup = st.selectbox(
        "Extra Educational Support",
        ["yes", "no"]
    )

    famsup = st.selectbox(
        "Family Educational Support",
        ["yes", "no"]
    )

    internet = st.selectbox(
        "Internet Access",
        ["yes", "no"]
    )

    activities = st.selectbox(
        "Extracurricular Activities",
        ["yes", "no"]
    )

    higher = st.selectbox(
        "Interested in Higher Education",
        ["yes", "no"]
    )

    submit = st.form_submit_button(
        "Predict Exam Score",
        use_container_width=True
    )

# Prediction
if submit:

    input_data = pd.DataFrame([{
        "studytime": studytime,
        "absences": absences,
        "failures": failures,
        "G1": g1,
        "G2": g2,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "internet": internet,
        "activities": activities,
        "higher": higher,
        "age": age
    }])

    prediction = model.predict(input_data)[0]

    # Convert marks from 20 to 100
    predicted_score = max(
        0,
        min(100, prediction * 5)
    )

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Final Examination Score",
        f"{predicted_score:.1f}/100"
    )

    st.progress(predicted_score / 100)

    if predicted_score >= 80:
       performance = "Excellent"
       st.success("Excellent Performance!")

    elif predicted_score >= 60:
        performance = "Good"
        st.success("Good Performance!")

    elif predicted_score >= 40:
        performance = "Average"
        st.warning("Average Performance")

    else:
        performance = "Needs Improvement"
        st.error("Needs Improvement")

 # Prediction and performance code
    report_data = input_data.copy()

    report_data["Predicted Score (/100)"] = round(predicted_score, 2)
    report_data["Performance"] = performance

    csv_report = report_data.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv_report,
        file_name="student_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.info( 
        "This prediction is an estimate generated "
        "by a Machine Learning model. It is not a "
        "guaranteed examination result."
    )

st.divider()

st.caption(
    "Student Exam Score & Performance Predictor | "
    "Powered by Random Forest Regression"
)

# ----------------------------------
# STUDENT PERFORMANCE ANALYSIS
# ----------------------------------

st.divider()

st.title("📊 Student Performance Dashboard")

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Dataset overview
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Students", len(df))

with col2:
    st.metric("Average Final Grade", round(df["G3"].mean(), 2))

with col3:
    st.metric("Total Features", len(df.columns))

# Display dataset
with st.expander("View Student Dataset"):
    st.dataframe(df, use_container_width=True)

# Grade distribution
st.subheader("📈 Final Grade Distribution")

grade_counts = (
    df["G3"]
    .value_counts()
    .sort_index()
)

st.bar_chart(grade_counts)

# Study time vs final grade
st.subheader("📚 Study Time vs Average Final Grade")

study_analysis = (
    df.groupby("studytime")["G3"]
    .mean()
    .round(2)
)

study_analysis.index = [
    "Less than 2 hours",
    "2 to 5 hours",
    "5 to 10 hours",
    "More than 10 hours"
]

st.bar_chart(study_analysis)

# Absences vs average grade
st.subheader("📉 Absences vs Average Final Grade")

absence_analysis = (
    df.groupby("absences")["G3"]
    .mean()
    .sort_index()
)

st.line_chart(absence_analysis)

st.caption(
    "Dashboard based on the UCI Student Performance Mathematics dataset."
)

# ----------------------------------
# MODEL PERFORMANCE VISUALIZATION
# ----------------------------------

st.divider()

st.title("🤖 Machine Learning Model Performance")

st.subheader("Actual vs Predicted Student Marks")

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Select the same features used during model training
features = [
    "studytime", "absences", "failures", "G1", "G2",
    "schoolsup", "famsup", "internet", "activities",
    "higher", "age"
]

X = df[features]
y = df["G3"]

# Recreate the same test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Predict using the saved model
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Display model metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", round(mae, 2))

with col2:
    st.metric("RMSE", round(rmse, 2))

with col3:
    st.metric("R² Score", round(r2, 2))

# Create comparison graph
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

ax.plot(
    [0, 20],
    [0, 20],
    linestyle="--"
)

ax.set_xlabel("Actual Final Marks")
ax.set_ylabel("Predicted Final Marks")
ax.set_title("Actual vs Predicted Marks")

ax.set_xlim(0, 20)
ax.set_ylim(0, 20)

ax.grid(True)

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Evaluation is performed on the held-out test dataset."
)

