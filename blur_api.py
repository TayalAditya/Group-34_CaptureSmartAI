from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import tempfile
import os
import io # For sending image data from memory
import logging
import tensorflow as tf # Import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---- Configuration ----
MODEL_PATH = 'blur_detection_model_v2.h5' # Make sure this path is correct
IMAGE_SIZE = (224, 224) # Input size expected by your MobileNetV2 model
# --- ---

# ---- Load Model ----
model = None
if os.path.exists(MODEL_PATH):
    try:
        # Explicitly tell load_model about 'mse' using the class
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError()} # Use the class instance
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model from {MODEL_PATH}: {e}")
else:
    logger.error(f"Model file not found at {MODEL_PATH}")
# --- ---


# ---- Traditional CV-based blur metrics ----

def detect_blur_laplacian(image):
    # ...existing code...
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_blur_tenengrad(image, ksize=3):
    # ...existing code...
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # Corrected Tenengrad calculation (use gx**2 + gy**2)
    fm = np.sqrt(gx**2 + gy**2)
    return np.mean(fm) # Often mean is used, but mean of squares is also common

def perceptual_blur_metric(image, threshold=0.1):
    # ...existing code...
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(dx**2 + dy**2)

    edge_widths = []
    coords = np.column_stack(np.where(edges > 0))
    for (y, x) in coords:
        # Add a check for zero magnitude to avoid division by zero
        mag_val = magnitude[y, x]
        if mag_val > threshold:
            edge_widths.append(1.0 / mag_val)
    return float(np.mean(edge_widths)) if edge_widths else 0.0

# ---- Model Prediction ----
def predict_blur_with_model(image, target_size=IMAGE_SIZE):
    """Preprocesses image and predicts blur score using the loaded Keras model."""
    if model is None:
        logger.error("Model not loaded, cannot predict.")
        return None # Or raise an exception

    try:
        # Resize and preprocess for MobileNetV2
        # Ensure image is in BGR format if read by cv2.imread
        image_resized = cv2.resize(image, target_size)
        image_array = img_to_array(image_resized) # Converts to float32, scales to [0, 255]
        image_array = preprocess_input(image_array) # MobileNetV2 specific preprocessing
        image_array = np.expand_dims(image_array, axis=0) # Add batch dimension

        # Predict blur severity (value between 0 and 100, based on your model training)
        predicted_blur = model.predict(image_array)[0][0]
        return float(predicted_blur)
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        return None


# --- Placeholder for your actual unblurring function ---
def perform_unblurring(input_image_path):
    # ...existing code...
    # Load image
    img = cv2.imread(input_image_path)
    if img is None:
        return None

    # >>> THIS IS WHERE YOUR COMPLEX UNBLURRING LOGIC GOES <<<
    # Example: Apply a simple sharpening filter (NOT real unblurring)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    unblurred_img = cv2.filter2D(img, -1, kernel)

    # Encode the processed image to JPEG format in memory
    is_success, buffer = cv2.imencode(".jpg", unblurred_img)
    if not is_success:
        return None

    return io.BytesIO(buffer) # Return image data as a file-like object
# ---------------------------------------------------------

# ---- Flask endpoint ----

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    POST /analyze
    - multipart/form-data with field 'image'
    - returns JSON: laplacian_variance, tenengrad_score,
      perceptual_blur_metric, predicted_blur_score (capped at 100)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']

    # Create temp file but get its path immediately
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img_file_path = tmp.name # Store the path

    try:
        # Save the uploaded file to the temp path
        img_file.save(img_file_path)
        tmp.close() # Close the file handle so cv2 can read it

        # Read the image using the path
        image = cv2.imread(img_file_path)

        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # --- Calculate Traditional Metrics ---
        lap = detect_blur_laplacian(image)
        ten = detect_blur_tenengrad(image)
        pbm = perceptual_blur_metric(image)

        # --- Predict Blur with Model ---
        predicted_score = predict_blur_with_model(image)
        if predicted_score is None:
            # Handle case where model prediction failed
             return jsonify({'error': 'Model prediction failed'}), 500

        # --- Calculate final score and cap at 100 ---
        final_predicted_score = 5 * predicted_score
        # Use min() to cap the score at 100
        capped_score = min(final_predicted_score, 100.0)
        logger.info(f"Raw predicted score: {predicted_score:.4f}, Scaled score: {final_predicted_score:.4f}, Capped score: {capped_score:.4f}")


        return jsonify({
            'laplacian_variance': lap,
            'tenengrad_score': ten,
            'perceptual_blur_metric': pbm,
            'predicted_blur_score': capped_score # Use the capped score
        })

    except Exception as e:
        logger.error(f"Error processing image in /analyze: {e}", exc_info=True) # Log traceback
        return jsonify({'error': 'Failed to process image'}), 500
    finally:
        # Ensure the temp file is deleted
        if os.path.exists(img_file_path):
            try:
                os.unlink(img_file_path)
            except Exception as e:
                logger.error(f"Error deleting temp file {img_file_path}: {e}")

@app.route('/unblur', methods=['POST'])
def unblur_image():
    # ... existing code ...
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img_file_path = tmp.name

    try:
        img_file.save(img_file_path)
        tmp.close()

        # Perform the unblurring
        unblurred_image_data = perform_unblurring(img_file_path)

        if unblurred_image_data is None:
             return jsonify({'error': 'Failed to process or unblur image'}), 500

        # Send the image data back
        unblurred_image_data.seek(0) # Reset stream position
        return send_file(
            unblurred_image_data,
            mimetype='image/jpeg',
            as_attachment=False # Send inline
        )

    except Exception as e:
        app.logger.error(f"Error in /unblur: {e}")
        return jsonify({'error': 'Server error during unblurring'}), 500
    finally:
        if os.path.exists(img_file_path):
            try:
                os.unlink(img_file_path)
            except Exception as e:
                app.logger.error(f"Error deleting temp file {img_file_path}: {e}")


if __name__ == '__main__':
    # Make sure the model path is correct relative to where the script is run
    # Or use an absolute path
    if model is not None:
        # Run the app (consider using a production server like Gunicorn/Waitress for deployment)
        app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for production
    else:
        logger.critical("Application cannot start because the model failed to load.")