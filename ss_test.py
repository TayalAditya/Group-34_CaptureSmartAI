import os
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# === Feature Extraction Functions ===
def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tenengrad_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return np.mean(gx**2 + gy**2)

def perceptual_blur_metric(image, threshold=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(dx**2 + dy**2)
    coords = np.column_stack(np.where(edges > 0))
    edge_widths = [1.0 / magnitude[y, x] for (y, x) in coords if magnitude[y, x] > threshold]
    return np.mean(edge_widths) if edge_widths else 0

def edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size

def brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def histogram_stats(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    return np.mean(hist), np.var(hist)

# === Prediction Function ===
def predict_shutter_speed(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read: {image_path}")
        return

    # Extract features
    lap = laplacian_variance(image)
    ten = tenengrad_score(image)
    pbm = perceptual_blur_metric(image)
    edge = edge_density(image)
    bright = brightness(image)
    hist_mean, hist_var = histogram_stats(image)

    feature_vector = np.array([[lap, ten, pbm, edge, bright, hist_mean, hist_var]], dtype=np.float32)

    try:
        scaler = joblib.load("stable_shutter_scaler.pkl")
        model = load_model("stable_shutter_model.h5", compile=False)
    except Exception as e:
        print("❌ Failed to load model or scaler:", e)
        return

    # Scale and Predict
    scaled_input = scaler.transform(feature_vector)
    log_shutter_inv = model.predict(scaled_input)[0][0]
    shutter = 1.0 / np.exp(log_shutter_inv)  # reverse log(1/shutter)

    # Clamp result for sanity
    shutter = max(1e-6, min(shutter, 60.0))

    print(f"📷 {os.path.basename(image_path)}")
    print(f"   ➤ Predicted Shutter Speed: {shutter:.6f} sec")

# === Run on all images in a folder ===
if __name__ == "__main__":
    test_folder = "/home/ab_students/Desktop/Hackathon/MLP/test_images4"  # update as needed

    for filename in os.listdir(test_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(test_folder, filename)
            predict_shutter_speed(path)

