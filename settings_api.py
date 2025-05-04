from flask import Flask, request, jsonify, send_file # Added send_file
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile
import logging
from io import BytesIO # Added BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# === MODEL AND SCALER PATHS ===
# Ensure these paths are correct relative to where you run the script, or use absolute paths
ISO_MODEL_PATH = "iso_classifier_model.h5"
ISO_SCALER_PATH = "iso_feature_scaler.pkl"
ISO_MAP_PATH = "iso_class_to_label.pkl"
SS_MODEL_PATH = "stable_shutter_model.h5"
SS_SCALER_PATH = "stable_shutter_scaler.pkl"

# === ISO MODEL LOADING ===
iso_model = None
iso_scaler = None
class_to_iso = None
iso_classes = None
try:
    if os.path.exists(ISO_MODEL_PATH):
        iso_model = tf.keras.models.load_model(ISO_MODEL_PATH)
        logger.info(f"ISO Model loaded from {ISO_MODEL_PATH}")
    else:
        logger.error(f"ISO Model file not found at {ISO_MODEL_PATH}")

    if os.path.exists(ISO_SCALER_PATH):
        iso_scaler = joblib.load(ISO_SCALER_PATH)
        logger.info(f"ISO Scaler loaded from {ISO_SCALER_PATH}")
    else:
        logger.error(f"ISO Scaler file not found at {ISO_SCALER_PATH}")

    if os.path.exists(ISO_MAP_PATH):
        class_to_iso = joblib.load(ISO_MAP_PATH)
        iso_classes = [class_to_iso[i] for i in range(len(class_to_iso))]
        logger.info(f"ISO Class Map loaded from {ISO_MAP_PATH}")
    else:
        logger.error(f"ISO Class Map file not found at {ISO_MAP_PATH}")

except Exception as e:
    logger.error(f"Error loading ISO model/scaler/map: {e}")
    iso_model = None # Ensure it's None if loading failed

# === SHUTTER SPEED MODEL LOADING ===
ss_model = None
ss_scaler = None
try:
    if os.path.exists(SS_MODEL_PATH):
        # If your model wasn't compiled with loss, optimizer, etc. use compile=False
        ss_model = load_model(SS_MODEL_PATH, compile=False)
        logger.info(f"Shutter Speed Model loaded from {SS_MODEL_PATH}")
    else:
        logger.error(f"Shutter Speed Model file not found at {SS_MODEL_PATH}")

    if os.path.exists(SS_SCALER_PATH):
        ss_scaler = joblib.load(SS_SCALER_PATH)
        logger.info(f"Shutter Speed Scaler loaded from {SS_SCALER_PATH}")
    else:
        logger.error(f"Shutter Speed Scaler file not found at {SS_SCALER_PATH}")

except Exception as e:
    logger.error(f"Error loading Shutter Speed model/scaler: {e}")
    ss_model = None # Ensure it's None if loading failed

# === COMMON FEATURE FUNCTIONS ===
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
    # Corrected magnitude calculation (use squares)
    magnitude = np.sqrt(dx**2 + dy**2)
    coords = np.column_stack(np.where(edges > 0))
    # Avoid division by zero or near-zero magnitude
    edge_widths = [1.0 / magnitude[y, x] for (y, x) in coords if magnitude[y, x] > threshold]
    return np.mean(edge_widths) if edge_widths else 0.0

def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tenengrad_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # Corrected magnitude calculation (use squares)
    return np.mean(gx**2 + gy**2)

# === ISO Biasing Logic ===
def apply_iso_bias(predicted_iso, confidence):
    # REVIEW THIS LOGIC: The current '+900' might be a placeholder.
    # Consider uncommenting and adjusting the confidence-based logic below
    # if it represents the intended behavior.

    # # Clamp ISO at 2500 max
    # if predicted_iso > 2500:
    #     return 2500
    # # Apply confidence-based adjustments
    # elif predicted_iso in [2000, 2500] and confidence < 0.90:
    #     return 1000
    # elif predicted_iso == 1600 and confidence < 0.85:
    #     return 800
    # # ... (rest of the commented-out logic) ...
    # else:
    #     # Return the original prediction if no bias rule applies
    #     return predicted_iso

    # Current placeholder logic:
    return predicted_iso + 900 # REVIEW: Is this intended?

# === ISO PREDICTION ===
def predict_iso_from_image(image):
    if iso_model is None or iso_scaler is None or class_to_iso is None:
        logger.error("ISO model/scaler/map not loaded. Cannot predict ISO.")
        return None, None, None

    try:
        feat_brightness = brightness(image)
        feat_hist_mean, feat_hist_var = histogram_stats(image)
        feat_edge_density = edge_density(image)
        feat_pbm = perceptual_blur_metric(image)

        feature_vector = np.array([[feat_brightness, feat_hist_mean, feat_hist_var, feat_edge_density, feat_pbm]], dtype=np.float32)
        feature_scaled = iso_scaler.transform(feature_vector)
        pred_prob = iso_model.predict(feature_scaled)[0]
        pred_class = np.argmax(pred_prob)
        raw_iso = class_to_iso[pred_class]
        confidence = pred_prob[pred_class]
        biased_iso = apply_iso_bias(raw_iso, confidence)
        return biased_iso, confidence, raw_iso
    except Exception as e:
        logger.error(f"Error during ISO prediction: {e}")
        return None, None, None

# === SHUTTER SPEED PREDICTION ===
def predict_shutter_speed_from_image(image):
    if ss_model is None or ss_scaler is None:
        logger.error("Shutter speed model/scaler not loaded. Cannot predict shutter speed.")
        return None

    try:
        # --- Calculate Features ---
        lap = laplacian_variance(image)
        ten = tenengrad_score(image)
        pbm = perceptual_blur_metric(image)
        edge = edge_density(image)
        bright = brightness(image)
        hist_mean, hist_var = histogram_stats(image)

        # --- DEBUG: Log Raw Features ---
        logger.debug(f"Raw Features - Lap: {lap:.4f}, Ten: {ten:.4f}, PBM: {pbm:.4f}, Edge: {edge:.4f}, Bright: {bright:.4f}, HistMean: {hist_mean:.4f}, HistVar: {hist_var:.4f}")

        # Check for NaN/Inf in raw features before creating the vector
        raw_features = [lap, ten, pbm, edge, bright, hist_mean, hist_var]
        if not all(np.isfinite(f) for f in raw_features):
            logger.error(f"NaN or Inf detected in raw features: {raw_features}")
            return None

        feature_vector = np.array([raw_features], dtype=np.float32)

        # --- Scale Features ---
        scaled_input = ss_scaler.transform(feature_vector)

        # --- DEBUG: Log Scaled Features ---
        # Log only the first few elements if it's long
        scaled_features_str = ", ".join([f"{x:.4f}" for x in scaled_input[0][:7]]) # Log all 7
        logger.debug(f"Scaled Features: [{scaled_features_str}]")

        # --- Predict ---
        raw_prediction = ss_model.predict(scaled_input)

        # --- DEBUG: Log Raw Model Output ---
        logger.debug(f"Raw Model Prediction Output: {raw_prediction}")

        # Check if the output shape is as expected (e.g., (1, 1))
        if raw_prediction.shape != (1, 1):
             logger.warning(f"Unexpected model output shape: {raw_prediction.shape}. Expected (1, 1). Check model architecture or indexing.")
             # Attempt to extract the first element if possible, otherwise fail
             if raw_prediction.size >= 1:
                 log_shutter_plus_1 = raw_prediction.flat[0]
                 logger.debug(f"Extracted value from unexpected shape: {log_shutter_plus_1}")
             else:
                 logger.error("Cannot extract value from empty model output.")
                 return None
        else:
            log_shutter_plus_1 = raw_prediction[0][0]


        # --- DEBUG: Log Predicted Log Value ---
        logger.debug(f"Predicted log_shutter_plus_1: {log_shutter_plus_1}")


        # --- Validate log output and compute shutter ---
        if not np.isfinite(log_shutter_plus_1) or log_shutter_plus_1 < -10 or log_shutter_plus_1 > 10:
            logger.error(f"Abnormal log prediction for shutter: {log_shutter_plus_1}. Check input features, scaler, and model.") # Added value to log
            return None

        shutter = 1.0 / np.exp(log_shutter_plus_1)
        shutter = np.clip(shutter, 1e-5, 10.0)  # clip to a practical range
        logger.info(f"Calculated Shutter Speed: {shutter:.5f}") # Log final value
        return shutter

    except Exception as e:
        logger.error(f"Error during Shutter Speed prediction: {e}", exc_info=True) # Log traceback
        return None

# === Flask Endpoint ===
@app.route('/recommend_settings', methods=['POST'])
def recommend_settings():
    """
    POST /recommend_settings
    - multipart/form-data with field 'image'
    - returns JSON: {'iso': recommended_iso, 'shutter_speed': recommended_shutter_speed}
    """
    if 'image' not in request.files:
        logger.warning("Request received without 'image' field.")
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    img_file_path = None # Initialize path variable

    # Use tempfile for secure temporary file handling
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            img_file_path = tmp.name
            img_file.save(img_file_path)
            logger.info(f"Image saved temporarily to {img_file_path}")

        # Read the image using OpenCV
        image = cv2.imread(img_file_path)

        if image is None:
            logger.error(f"Failed to read image file: {img_file_path}")
            return jsonify({'error': 'Invalid image file'}), 400

        # --- Predict ISO ---
        predicted_iso, _, _ = predict_iso_from_image(image) # We only need the final biased ISO here
        if predicted_iso is None:
             # Error already logged in predict function
             return jsonify({'error': 'ISO prediction failed'}), 500

        # --- Predict Shutter Speed ---
        predicted_shutter = predict_shutter_speed_from_image(image)
        if predicted_shutter is None:
             # Error already logged in predict function
             return jsonify({'error': 'Shutter speed prediction failed'}), 500
        calculated_shutter_speed = predicted_shutter
        logger.info(f"Prediction results - ISO: {predicted_iso}, Shutter: {calculated_shutter_speed}")

        # --- Return JSON Response ---
        # Explicitly cast the result of the division to a standard Python float
        calculated_shutter_speed = predicted_shutter
        return jsonify({
            'iso': int(predicted_iso), # Return ISO as integer
            'shutter_speed': float(calculated_shutter_speed) # Cast to Python float
        })

    except Exception as e:
        logger.error(f"Error processing image in /recommend_settings: {e}", exc_info=True) # Log traceback
        return jsonify({'error': 'Failed to process image'}), 500
    finally:
        # Ensure the temp file is deleted even if errors occur
        if img_file_path and os.path.exists(img_file_path):
            try:
                os.unlink(img_file_path)
                logger.info(f"Temporary file {img_file_path} deleted.")
            except Exception as e:
                logger.error(f"Error deleting temp file {img_file_path}: {e}")

# === HEATMAP GENERATION LOGIC ===
def generate_heatmap_overlay(image):
    """Generates a heatmap overlay based on vertical streaks."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        SHIFT = 30  # ≈ streak length in pixels
        # Ensure image height is sufficient for the shift
        if gray.shape[0] <= SHIFT:
            logger.warning(f"Image height ({gray.shape[0]}) is too small for shift ({SHIFT}). Heatmap might be inaccurate.")
            # Handle gracefully: maybe return original image or a blank heatmap
            # For now, let's proceed but the result might not be meaningful
            shifted = gray # No shift if too small
        else:
            shifted = np.roll(gray, -SHIFT, axis=0)  # vertical shift
            # Zero out the wrapped-around part at the bottom
            shifted[gray.shape[0]-SHIFT:, :] = gray[gray.shape[0]-SHIFT:, :]

        diff = cv2.absdiff(gray, shifted)  # |I - I_shift|
        inv = cv2.bitwise_not(diff)  # low diff -> bright
        blur = cv2.GaussianBlur(inv, (11, 11), 0)  # smooth glow
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

        heat = cv2.applyColorMap(norm.astype('uint8'), cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(image, 0.6, heat, 0.8, 0)

        return overlay

    except Exception as e:
        logger.error(f"Error during heatmap generation: {e}", exc_info=True)
        return None

# === NEW HEATMAP ENDPOINT ===
@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap_endpoint():
    """
    POST /generate_heatmap
    - multipart/form-data with field 'image'
    - returns the generated heatmap overlay image as JPEG
    """
    if 'image' not in request.files:
        logger.warning("Heatmap request received without 'image' field.")
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    img_file_path = None

    try:
        # Use tempfile for secure temporary file handling
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            img_file_path = tmp.name
            img_file.save(img_file_path)
            logger.info(f"Heatmap input image saved temporarily to {img_file_path}")

        # Read the image using OpenCV
        image = cv2.imread(img_file_path)

        if image is None:
            logger.error(f"Failed to read image file for heatmap: {img_file_path}")
            return jsonify({'error': 'Invalid image file'}), 400

        # --- Generate Heatmap ---
        heatmap_overlay = generate_heatmap_overlay(image)

        if heatmap_overlay is None:
            logger.error("Heatmap generation failed.")
            return jsonify({'error': 'Heatmap generation failed'}), 500

        # --- Encode result image to JPEG bytes ---
        is_success, buffer = cv2.imencode(".jpg", heatmap_overlay)
        if not is_success:
             logger.error("Failed to encode heatmap overlay to JPEG.")
             return jsonify({'error': 'Failed to encode result image'}), 500

        # --- Return image file ---
        return send_file(
            BytesIO(buffer),
            mimetype='image/jpeg',
            as_attachment=False # Send inline
        )

    except Exception as e:
        logger.error(f"Error processing image in /generate_heatmap: {e}", exc_info=True)
        return jsonify({'error': 'Failed to process image for heatmap'}), 500
    finally:
        # Ensure the temp file is deleted
        if img_file_path and os.path.exists(img_file_path):
            try:
                os.unlink(img_file_path)
                logger.info(f"Temporary file {img_file_path} deleted.")
            except Exception as e:
                logger.error(f"Error deleting temp file {img_file_path}: {e}")

# === Main Execution ===
if __name__ == "__main__":
    # Check if all essential models/scalers loaded before starting
    # Note: Heatmap doesn't depend on ML models, so the check remains the same
    if iso_model and iso_scaler and class_to_iso and ss_model and ss_scaler:
        logger.info("Starting Flask server for settings recommendation and heatmap generation...")
        # Run on port 5001, accessible on local network
        app.run(host='0.0.0.0', port=5001, debug=False) # Set debug=False for stability
    else:
        # Allow server to start even if ML models fail, but log critical error for ML part
        logger.critical("One or more ML models/scalers failed to load. /recommend_settings endpoint will not work.")
        logger.info("Starting Flask server ONLY for heatmap generation...")
        app.run(host='0.0.0.0', port=5001, debug=False)