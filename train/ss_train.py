import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("features_shutter_only.csv")
df = df.dropna(subset=["Shutter", "feat_brightness"])
df = df[df["Shutter"] > 0]

# Features and Target (use log(1/Shutter))
feature_cols = [
    "feat_laplacian", "feat_tenengrad", "feat_pbm", "feat_edge_density",
    "feat_brightness", "feat_hist_mean", "feat_hist_var"
]
X = df[feature_cols].values.astype(np.float32)
y_raw = df["Shutter"].values.astype(np.float32)
y = np.log(1.0 / y_raw)  # More stable regression target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# Define MLP model
def build_model(input_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(X_train.shape[1])

# Train model
early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
lr_scheduler = callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=8,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# Save model & scaler
model.save("stable_shutter_model.h5")
joblib.dump(scaler, "stable_shutter_scaler.pkl")

# Predict and inverse transform: predicted shutter = 1 / exp(log_value)
y_val_pred_log = model.predict(X_val).flatten()
y_val_pred = 1.0 / np.exp(y_val_pred_log)
y_val_true = 1.0 / np.exp(y_val)

# Clip negatives
y_val_pred = np.clip(y_val_pred, 0, None)




# Plot Actual vs. Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_val_true, y_val_pred, c='blue', label='Predictions')
plt.plot([min(y_val_true), max(y_val_true)], [min(y_val_true), max(y_val_true)], 'r--', label='Ideal')
plt.xlabel("Actual Shutter Speed")
plt.ylabel("Predicted Shutter Speed")
plt.title("Actual vs. Predicted Shutter Speed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("final_shutter_prediction.png")
plt.show()

# Residual Plot
plt.figure(figsize=(8, 6))
residuals = y_val_true - y_val_pred
plt.scatter(y_val_pred, residuals, c='blue', label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.xlabel("Predicted Shutter Speed")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Shutter Speed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("residuals_plot.png")
plt.show()

