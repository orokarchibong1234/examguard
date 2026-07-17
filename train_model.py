import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("Generating synthetic exam behaviour training data...")

np.random.seed(42)
n_normal = 1000
n_anomalous = 100

# ----------------------------------------
# NORMAL STUDENT BEHAVIOUR
# ----------------------------------------
normal_data = {
    "score": np.random.normal(65, 15, n_normal).clip(0, 100),
    "total_clicks": np.random.normal(80, 20, n_normal).clip(10, 200),
    "time_on_exam": np.random.normal(1200, 300, n_normal).clip(300, 1800),
    "tab_switches": np.random.normal(1, 1, n_normal).clip(0, 5),
    "avg_time_per_question": np.random.normal(240, 60, n_normal).clip(30, 600),
    "activity_rate": np.random.normal(0.07, 0.02, n_normal).clip(0.01, 0.2),
}

# ----------------------------------------
# ANOMALOUS STUDENT BEHAVIOUR
# ----------------------------------------
anomalous_data = {
    # Low score + high tab switches = cheating
    "score": np.random.normal(85, 5, n_anomalous).clip(0, 100),
    "total_clicks": np.random.normal(200, 50, n_anomalous).clip(50, 500),
    "time_on_exam": np.random.normal(300, 100, n_anomalous).clip(60, 600),
    "tab_switches": np.random.normal(10, 3, n_anomalous).clip(5, 30),
    "avg_time_per_question": np.random.normal(60, 20, n_anomalous).clip(10, 120),
    "activity_rate": np.random.normal(0.5, 0.1, n_anomalous).clip(0.2, 1.0),
}

# ----------------------------------------
# COMBINE DATA
# ----------------------------------------
normal_df = pd.DataFrame(normal_data)
anomalous_df = pd.DataFrame(anomalous_data)
df = pd.concat([normal_df, anomalous_df], ignore_index=True)

print(f"Training data shape: {df.shape}")
print(f"Normal samples: {n_normal}")
print(f"Anomalous samples: {n_anomalous}")

# ----------------------------------------
# SCALE FEATURES
# ----------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ----------------------------------------
# TRAIN ISOLATION FOREST
# ----------------------------------------
print("\nTraining Isolation Forest model...")

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)
model.fit(X_scaled)

print("Model training completed!")

# ----------------------------------------
# SAVE MODEL AND SCALER
# ----------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/isolation_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model saved to models/isolation_forest.pkl")
print("Scaler saved to models/scaler.pkl")

# ----------------------------------------
# TEST PREDICTION
# ----------------------------------------
print("\nTesting model...")

test_normal = scaler.transform([[65, 80, 1200, 1, 240, 0.07]])
test_anomalous = scaler.transform([[95, 300, 200, 15, 40, 0.8]])

print(f"Normal student prediction: {'Normal' if model.predict(test_normal)[0] == 1 else 'Anomalous'}")
print(f"Anomalous student prediction: {'Normal' if model.predict(test_anomalous)[0] == 1 else 'Anomalous'}")
print("\nDone!")