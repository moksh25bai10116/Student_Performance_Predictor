import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "student_performance.csv")

df = pd.read_csv(data_path)
df = df.dropna()

print(df.head())

# Split features
X = df.drop("final_score", axis=1)
y = df["final_score"]

numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(exclude=np.number).columns

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

dt_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(max_depth=5, random_state=42))
])

# Train
lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

# Evaluation
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 2))

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_dt, "Decision Tree")

# Risk classification
def risk(score):
    if score >= 75:
        return "Low"
    elif score >= 50:
        return "Medium"
    else:
        return "High"

# Sample prediction
sample = pd.DataFrame([{
    "study_hours": 5,
    "attendance": 70,
    "sleep_hours": 6,
    "previous_score": 60,
    "assignments_score": 65,
    "participation": 5,
    "extracurricular": "No",
    "internet_access": "Yes",
    "parent_education": "Bachelor"
}])

pred = lr_model.predict(sample)[0]
print("\nPredicted Score:", round(pred, 2))
print("Risk Level:", risk(pred))

# Plot

# Create folder safely
output_dir = os.path.join(base_dir, "outputs", "plots")
os.makedirs(output_dir, exist_ok=True)

# Actual vs Predicted scatter
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.7)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted")
plot_path = os.path.join(output_dir, "prediction_plot.png")
plt.savefig(plot_path)
print(f"Plot saved at: {plot_path}")
plt.show()

# Distribution of final scores
plt.figure(figsize=(8, 6))
df["final_score"].hist(bins=20)
plt.title("Distribution of Final Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
hist_path = os.path.join(output_dir, "score_distribution.png")
plt.savefig(hist_path)
print(f"Histogram saved at: {hist_path}")
plt.show()