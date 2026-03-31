import os
import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

# Generate realistic distributions for student data
study_hours = np.random.normal(6, 2, n).clip(1, 12)
attendance = np.random.normal(85, 10, n).clip(50, 100)
sleep_hours = np.random.normal(7, 1, n).clip(4, 10)
previous_score = np.random.normal(70, 12, n).clip(35, 95)
assignments_score = np.random.normal(75, 10, n).clip(40, 100)
participation = np.random.randint(1, 11, n)

extracurricular = np.random.choice(["Yes", "No"], n, p=[0.4, 0.6])
internet_access = np.random.choice(["Yes", "No"], n, p=[0.85, 0.15])
parent_education = np.random.choice(["HighSchool", "Bachelor", "Master"], n, p=[0.3, 0.45, 0.25])

data = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "sleep_hours": sleep_hours,
    "previous_score": previous_score,
    "assignments_score": assignments_score,
    "participation": participation,
    "extracurricular": extracurricular,
    "internet_access": internet_access,
    "parent_education": parent_education
})

# Score formula with controlled weights + random noise
noise = np.random.normal(0, 4, n)
raw_score = (
    study_hours * 2.5 +
    attendance * 0.25 +
    sleep_hours * 1.2 +
    previous_score * 0.35 +
    assignments_score * 0.25 +
    participation * 1.0 +
    noise
)

# Convert to roughly 0-100 and clip to avoid outliers at extremes
final_score = 100 * (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
final_score = np.clip(final_score, 10, 98)

# Add slight spread by applying small gaussian scaling to avoid all 0/100
final_score = np.clip(final_score + np.random.normal(0, 2, n), 5, 100)

data["final_score"] = final_score

# Save data to CSV for main.py
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(base_dir, "data", "student_performance.csv")

data.to_csv(output_path, index=False)
print(f"Generated {n} records and saved to {output_path}")