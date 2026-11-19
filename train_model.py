
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load the student dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select input features
features = [
    "studytime", "absences", "failures", "G1", "G2",
    "schoolsup", "famsup", "internet", "activities",
    "higher", "age"
]

X = df[features]
y = df["G3"]

# 3. Identify categorical columns
categorical_features = [
    "schoolsup", "famsup", "internet",
    "activities", "higher"
]

# 4. Convert categorical data into numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# 5. Create the Machine Learning model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2
    ))
])

# 6. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 7. Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# 8. Make predictions
predictions = model.predict(X_test)

# 9. Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# 10. Save the trained model
joblib.dump(model, "student_model.pkl")

print("\nModel trained successfully!")
print("Model saved as student_model.pkl")
