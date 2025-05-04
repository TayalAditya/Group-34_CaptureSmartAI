"""
ISO Model Training Script  ─ TRUE‑LABEL VERSION
================================================
• Trains a categorical ISO classifier on hand‑crafted image features.
• *No* bias/random relabeling is injected during training – the model
  learns genuine feature→ISO relationships.
• Exposure compensation (ISO uplift based on shutter speed) will be
  handled **after inference**, ensuring data integrity and model
  generalisation.
Outputs
-------
1. iso_classifier_model.h5     – Keras H5 model
2. iso_feature_scaler.pkl      – StandardScaler for features
3. iso_class_to_label.pkl      – mapping {class idx: ISO value}
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ------------------------------------------------------------------
# 1. Load and clean dataset
# ------------------------------------------------------------------

data = pd.read_csv("features_iso_only.csv")
# keep rows that have ISO and at least one feature
feature_cols = [col for col in data.columns if col.startswith("feat_")]
needed_cols = feature_cols + ["ISO"]
data = data.dropna(subset=needed_cols)

# ------------------------------------------------------------------
# 2. Optional: round ISO to nearest standard stop so we have discrete
#    classes with enough samples (helps soft‑max classifier)
# ------------------------------------------------------------------
std_isos = np.array([50, 64, 80, 100, 125, 160, 200, 250, 320,
                     400, 500, 640, 800, 1000, 1250, 1600,
                     2000, 2500, 3200, 4000, 5000, 6400])

def round_to_nearest_iso(x):
    return int(std_isos[np.argmin(np.abs(std_isos - x))])

data["ISO"] = data["ISO"].apply(round_to_nearest_iso)

# keep ISO classes that appear at least twice (filter noise)
iso_counts = data["ISO"].value_counts()
valid_isos = iso_counts[iso_counts >= 2].index.tolist()
data = data[data["ISO"].isin(valid_isos)]

# ------------------------------------------------------------------
# 3. Encode ISO↔class
# ------------------------------------------------------------------
iso_values = sorted(data["ISO"].unique())
iso_to_class = {iso: idx for idx, iso in enumerate(iso_values)}
class_to_iso = {v: k for k, v in iso_to_class.items()}
data["ISO_class"] = data["ISO"].map(iso_to_class)

# ------------------------------------------------------------------
# 4. Feature matrix & labels
# ------------------------------------------------------------------
X = data[feature_cols].values.astype("float32")
Y = tf.keras.utils.to_categorical(data["ISO_class"].values,
                                  num_classes=len(iso_values))

# ------------------------------------------------------------------
# 5. Train‑validation split & scaling
# ------------------------------------------------------------------
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# ------------------------------------------------------------------
# 6. Build MLP classifier
# ------------------------------------------------------------------

def build_iso_classifier(input_dim, n_classes):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_iso_classifier(X_train.shape[1], Y_train.shape[1])
model.summary()

# ------------------------------------------------------------------
# 7. Class weights: down‑weight rare, very high ISOs
# ------------------------------------------------------------------
class_weights = {}
for cls_idx, iso_val in class_to_iso.items():
    if iso_val > 2500:
        class_weights[cls_idx] = 0.4
    elif iso_val > 1500:
        class_weights[cls_idx] = 0.6
    elif iso_val > 1000:
        class_weights[cls_idx] = 0.8
    else:
        class_weights[cls_idx] = 1.0

# ------------------------------------------------------------------
# 8. Fit model
# ------------------------------------------------------------------
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10,
                            restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=5, min_lr=1e-6)
]

model.fit(X_train, Y_train,
          validation_data=(X_val, Y_val),
          epochs=100,
          batch_size=16,
          class_weight=class_weights,
          callbacks=callbacks_list,
          verbose=1)

# ------------------------------------------------------------------
# 9. Save artefacts
# ------------------------------------------------------------------
model.save("iso_classifier_model.h5")
joblib.dump(scaler, "iso_feature_scaler.pkl")
joblib.dump(class_to_iso, "iso_class_to_label.pkl")

print("✅ ISO classifier trained & artefacts saved.")

# ------------------------------------------------------------------
# 10. Exposure‑ratio helper (use at inference, not during training)
# ------------------------------------------------------------------

def adjust_iso_for_fast_shutter(pred_iso, pred_shutter_fast,
                                baseline_shutter=1/30, iso_cap=6400):
    """Scale ISO upward to compensate for a faster shutter.

    Parameters
    ----------
    pred_iso : int or float
        ISO predicted by the classifier (before scaling).
    pred_shutter_fast : float
        Shutter time predicted by SS model (seconds).
    baseline_shutter : float
        Reference shutter assumed to give correct exposure on the same scene.
    iso_cap : int
        Maximum ISO allowed after scaling.
    """
    # If shutter is faster than baseline, compute scale factor
    k = baseline_shutter / pred_shutter_fast
    iso_scaled = pred_iso * k

    # Round to nearest standard stop
    std_stops = std_isos  # reuse same array
    iso_final = int(std_stops[np.argmin(np.abs(std_stops - iso_scaled))])

    return int(np.clip(iso_final, 50, iso_cap))
