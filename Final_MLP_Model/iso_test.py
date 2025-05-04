import os
import cv2
import numpy as np
import joblib
import tensorflow as tf

# === Load trained model and scaler ===
model = tf.keras.models.load_model("iso_classifier_model.h5")
scaler = joblib.load("iso_feature_scaler.pkl")
class_to_iso = joblib.load("iso_class_to_label.pkl")

# === Inverse mapping for prediction
iso_classes = [class_to_iso[i] for i in range(len(class_to_iso))]

# === Feature extraction functions ===
def brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def histogram_stats(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    return np.mean(hist), np.var(hist)

def edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size

def perceptual_blur_metric(image, threshold=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(dx**2 + dy**2)
    coords = np.column_stack(np.where(edges > 0))
    edge_widths = [1.0 / magnitude[y, x] for (y, x) in coords if magnitude[y, x] > threshold]
    return np.mean(edge_widths) if edge_widths else 0

# === Biased ISO logic based on predicted ISO and confidence
def apply_iso_bias(predicted_iso, confidence):
    if predicted_iso > 2500:
        return 2500
    elif predicted_iso in [2000, 2500] and confidence < 0.90:
        return 1000
    elif predicted_iso == 1600 and confidence < 0.85:
        return 800
    elif predicted_iso == 1250 and confidence < 0.7:
        return 640
    elif predicted_iso == 1000 and confidence < 0.7:
        return 500
    elif predicted_iso == 800 and confidence < 0.65:
        return 400
    elif predicted_iso == 640 and confidence < 0.65:
        return 250
    elif predicted_iso == 500 and confidence < 0.6:
        return 125
    else:
        return predicted_iso

# === Predict ISO from a single image ===
def predict_iso_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        return

    # === Extract 5 training features
    feat_brightness = brightness(image)
    feat_hist_mean, feat_hist_var = histogram_stats(image)
    feat_edge_density = edge_density(image)
    feat_pbm = perceptual_blur_metric(image)

    feature_vector = np.array([[feat_brightness, feat_hist_mean, feat_hist_var, feat_edge_density, feat_pbm]], dtype=np.float32)

    # === Scale and predict
    feature_scaled = scaler.transform(feature_vector)
    pred_prob = model.predict(feature_scaled)[0]
    pred_class = np.argmax(pred_prob)
    raw_iso = class_to_iso[pred_class]
    confidence = pred_prob[pred_class]

    # === Apply ISO bias logic
    biased_iso = apply_iso_bias(raw_iso, confidence)

    # === Output
    print(f"\n📷 {os.path.basename(image_path)}")
    print(f"   ➤ Final ISO: {biased_iso}")

# === Main Runner
if __name__ == "__main__":
    test_folder = "/home/ab_students/Desktop/Hackathon/MLP/test_images1"  # update path if needed

    for filename in os.listdir(test_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(test_folder, filename)
            predict_iso_from_image(path)
