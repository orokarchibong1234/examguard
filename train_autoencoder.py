import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("Setting up Autoencoder training...")

# Try importing tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not found. Installing...")
    import subprocess
    subprocess.run(["pip", "install", "tensorflow"])
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense

# ----------------------------------------
# GENERATE TRAINING DATA
# ----------------------------------------
print("\nGenerating training data...")

np.random.seed(42)
n_normal = 1000
n_anomalous = 100

# Normal behaviour
normal_data = {
    "score": np.random.normal(65, 15, n_normal).clip(0, 100),
    "total_clicks": np.random.normal(80, 20, n_normal).clip(10, 200),
    "time_on_exam": np.random.normal(1200, 300, n_normal).clip(300, 1800),
    "tab_switches": np.random.normal(1, 1, n_normal).clip(0, 5),
    "avg_time_per_question": np.random.normal(240, 60, n_normal).clip(30, 600),
    "activity_rate": np.random.normal(0.07, 0.02, n_normal).clip(0.01, 0.2),
}

# Anomalous behaviour
anomalous_data = {
    "score": np.random.normal(85, 5, n_anomalous).clip(0, 100),
    "total_clicks": np.random.normal(200, 50, n_anomalous).clip(50, 500),
    "time_on_exam": np.random.normal(300, 100, n_anomalous).clip(60, 600),
    "tab_switches": np.random.normal(10, 3, n_anomalous).clip(5, 30),
    "avg_time_per_question": np.random.normal(60, 20, n_anomalous).clip(10, 120),
    "activity_rate": np.random.normal(0.5, 0.1, n_anomalous).clip(0.2, 1.0),
}

normal_df = pd.DataFrame(normal_data)
anomalous_df = pd.DataFrame(anomalous_data)

print(f"Normal samples: {len(normal_df)}")
print(f"Anomalous samples: {len(anomalous_df)}")

# ----------------------------------------
# SCALE DATA
# ----------------------------------------
scaler = joblib.load("models/scaler.pkl")
X_normal = scaler.transform(normal_df)
X_all = scaler.transform(pd.concat([normal_df, anomalous_df], ignore_index=True))

# ----------------------------------------
# BUILD AUTOENCODER
# ----------------------------------------
print("\nBuilding Autoencoder...")

input_dim = X_normal.shape[1]  # 6 features

inputs = Input(shape=(input_dim,))

# Encoder
encoded = Dense(8, activation='relu')(inputs)
encoded = Dense(4, activation='relu')(encoded)
bottleneck = Dense(2, activation='relu')(encoded)

# Decoder
decoded = Dense(4, activation='relu')(bottleneck)
decoded = Dense(8, activation='relu')(decoded)
outputs = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

# ----------------------------------------
# TRAIN ON NORMAL DATA ONLY
# ----------------------------------------
print("\nTraining Autoencoder on normal data only...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = autoencoder.fit(
    X_normal, X_normal,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining complete!")

# ----------------------------------------
# CALCULATE THRESHOLD
# ----------------------------------------
print("\nCalculating anomaly threshold...")

reconstructions = autoencoder.predict(X_normal)
mse_errors = np.mean(np.power(X_normal - reconstructions, 2), axis=1)

threshold = np.percentile(mse_errors, 95)
print(f"Reconstruction error threshold: {threshold:.4f}")

# ----------------------------------------
# TEST MODEL
# ----------------------------------------
print("\nTesting model...")

test_normal = scaler.transform([[65, 80, 1200, 1, 240, 0.07]])
test_anomalous = scaler.transform([[95, 300, 200, 15, 40, 0.8]])

recon_normal = autoencoder.predict(test_normal)
recon_anomalous = autoencoder.predict(test_anomalous)

error_normal = np.mean(np.power(test_normal - recon_normal, 2))
error_anomalous = np.mean(np.power(test_anomalous - recon_anomalous, 2))

print(f"Normal student error: {error_normal:.4f} → {'Anomalous' if error_normal > threshold else 'Normal'}")
print(f"Anomalous student error: {error_anomalous:.4f} → {'Anomalous' if error_anomalous > threshold else 'Normal'}")

# ----------------------------------------
# SAVE MODEL AND THRESHOLD
# ----------------------------------------
os.makedirs("models", exist_ok=True)
autoencoder.save("models/autoencoder.keras")
joblib.dump(threshold, "models/autoencoder_threshold.pkl")

print("\nAutoencoder saved to models/autoencoder.keras")
print(f"Threshold saved to models/autoencoder_threshold.pkl")
print("\nDone!")