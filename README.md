# 📚 Student Exam Score & Performance Predictor

**A Machine Learning-based web application for predicting student examination scores and analyzing academic performance.**

## 📌 1. Project Overview

The Student Exam Score & Performance Predictor is a Machine Learning project developed using Python and Streamlit.

The application predicts a student's expected final examination score based on academic information such as study time, previous examination grades, absences, and educational support.

It also provides an interactive dashboard to analyze student performance through data visualization.

## 🎯 2. Project Objectives

- Predict students' final examination scores using Machine Learning.
- Analyze academic performance using historical student data.
- Understand the relationship between study time and examination results.
- Visualize student grades and academic patterns.
- Evaluate the accuracy of the trained ML model.
- Generate downloadable student prediction reports.

## 🛠️ 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | Web application interface |
| Pandas | Data processing |
| NumPy | Numerical calculations |
| Scikit-learn | Machine Learning |
| Matplotlib | Data visualization |
| Joblib | Saving and loading ML models |

## 📊 4. Dataset Information

The project uses the **UCI Student Performance Dataset**, specifically the Mathematics dataset (`student-mat.csv`).

- Total student records: 395
- Total dataset attributes: 33
- Target variable: G3 (final examination grade)
- Original grading scale: 0–20

The application converts predicted marks to a 0–100 scale for easier understanding.

**Dataset source:**  
https://archive.ics.uci.edu/dataset/320/student+performance

## 🤖 5. Machine Learning Algorithm

The project uses **Random Forest Regression** to predict final examination grades.

### Input Features

- Study time
- Number of absences
- Previous class failures
- First period grade (G1)
- Second period grade (G2)
- Age
- Extra educational support
- Family educational support
- Internet access
- Extracurricular activities
- Interest in higher education

The model uses an 80:20 train-test split.

Since G1 and G2 are included as inputs, the application is designed to estimate the final grade after earlier examination grades are available.

## ✨ 6. Main Features

1. Student academic information input form.
2. Final examination score prediction.
3. Student performance classification.
4. Dataset overview and statistics.
5. Final grade distribution chart.
6. Study time vs average final grade analysis.
7. Absences vs average final grade analysis.
8. Actual vs predicted marks visualization.
9. Model evaluation using MAE, RMSE and R².
10. Downloadable prediction reports in CSV format.

## 📈 7. Model Evaluation

The trained Random Forest model was evaluated using:

| Evaluation Metric | Recorded Result |
|---|---:|
| Mean Absolute Error (MAE) | 1.05 |
| Root Mean Squared Error (RMSE) | 1.73 |
| R² Score | 0.85 |

These are the recorded results from the initial training run.

## 📁 8. Project Structure

```text
Student-Performance-Predictor/
│
├── data/
│   └── student-mat.csv
│
├── app.py
├── train_model.py
├── student_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ 9. Installation and Execution

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the project directory

```bash
cd Student-Performance-Predictor
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Train the Machine Learning model

```bash
python train_model.py
```

### Step 7: Run the application

```bash
streamlit run app.py
```

The application will open locally in your browser.

## 🔮 10. Future Enhancements

- Compare different Machine Learning algorithms.
- Improve model performance through feature engineering.
- Add student login and prediction history.
- Integrate additional student datasets.
- Deploy the application online.
- Develop personalized academic recommendations.

## ⚠️ 11. Limitations

- The dataset represents students from two Portuguese schools.
- Predictions are estimates, not guaranteed examination results.
- Performance may differ when applied to students from other educational backgrounds.
- The model depends on the quality and availability of input information.

## 🏁 12. Conclusion

This project demonstrates the application of Machine Learning in educational performance prediction.

It combines data analysis, regression modeling, and interactive visualization to provide an easy-to-use student performance prediction system.

**Developed as a B.Tech CSE (AI/ML) Minor Project.**