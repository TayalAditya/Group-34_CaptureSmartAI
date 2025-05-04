
# CaptureSmart: Intelligent ISO & Shutter Speed Recommendation App

**CaptureSmart** is an AI-powered Android application that helps users minimize motion blur by automatically recommending the optimal **ISO** and **Shutter Speed (SS)** settings for a given image. The system uses a custom MobileNetV2 model to assess blur severity and two trained MLP models to suggest camera parameters. The backend is served via a Flask API and optimized for low-latency usage.

---

## Project Components

1. **Android App** – Real-time camera interface and prediction viewer  
2. **Flask API Backend** – Receives images and responds with ISO/SS recommendations  
3. **MLP Models** – Predict ISO and Shutter Speed based on extracted visual features  
4. **Blur Detection Model using CNN (Baseline as MobilenetV2)** – Predicts blur severity score (0–100)

---

## Model Architectures


### 1. **Blur Severity Prediction Model**

- **Backbone**: `MobileNetV2` with a custom regression head
- **Output**: Continuous blur severity score (0–100)
- **Loss Function**: MSE Loss (logged MAE)
- **Final Layer**: Dense(1) with linear activation
- **Evaluation Metric**: Pearson correlation, MSE, MAE

**Training Dataset:**

- **Sharp Image Base**: 6,000 curated high-quality, low-noise images labelled as 0 to 10.
- **Synthetic Blur Augmentation**:
  - 18,000 blurred images generated using:
    - Motion blur (2 parts) – average severity ≈ 37 (As in real life scenaorios we expect them more)
    - Gaussian blur (1 part) – average severity ≈ 32.5 (To address the general blur, adding gaussian noise)
- **Label Scaling**: Blur severity normalized to a 0–100 scale.
- **Final Dataset**: 24,000 total images with continuous blur severity labels. Fine tuned the model with this dataset.

This model was trained for 50 epochs with a batch size of 32, using image resizing to 224×224. It serves as the core input analysis module for detecting whether the image needs parameter adjustments. We have also evaluated with respect to traditional methods like Laplacian Variance, PBM, Tenengrad Score and we found that our neural network model has correlation of 0.962 which is very high with the true level.

### 2. **ISO & SS MLP Models**

- **Inputs**: 7 handcrafted features
  - Laplacian Variance  
  - Tenengrad Score  
  - Perceptual Blur Metric  
  - Edge Density  
  - Mean Brightness  
  - Histogram Mean  
  - Histogram Variance

- **Architecture**:
  - Dense(64) → ReLU  
  - BatchNorm → Dropout(0.2)  
  - Dense(32) → ReLU  
  - Dense(1) → Output

- **Targets**:  
  - ISO: 50–2500  
  - SS: Log(1/t) transformation for stability

---

## Installation & Setup

### Android App

1. Open `CapSmart` in Android Studio  
2. Connect a physical Android device  
3. Update the IP address of the server endpoint in API service to match the local server (e.g., `http://192.168.0.102:5001/recommend_settings`)  
4. Grant permissions: `CAMERA`, `INTERNET`  
5. Build and run on device

### Flask API Server

1. Navigate to `api/`
2. Start server:
   ```bash
   python api(s).py
   ```
3. Server runs at: `http://0.0.0.0:5001/` (accessible on the same Wi-Fi network)

---
## Heads we , we hosted Flask Server on local wifi SERVER To reduce the latency if you want to host please change the IP address.

## API Endpoints

### `POST /recommend_settings`

- **Input**: Single image file (as `multipart/form-data`)
- **Response**:
  ```json
  {
    "blur_score": 41.23,
    "recommended_iso": 400,
    "recommended_shutter_speed": "1/125"
  }
  ```

---

## Latency Optimization

To reduce network latency during image transfer and prediction:

- We hosted the **Flask server on the same local Wi-Fi network** as the Android device.  
- This setup eliminates external API calls and reduces response time to ~0.5–1 sec.

---

## Sample Output of MLP Model:

| Image Name | Blur Score | Predicted ISO | Predicted SS |
|------------|------------|---------------|--------------|
| image_01.jpg | 29.4     | 200           | 1/250        |
| image_05.jpg | 61.7     | 1250          | 1/30         |
| image_09.jpg | 12.3     | 100           | 1/640        |

---

## Evaluation Metrics of MLP Model:

| Model         | MSE (val) | MAE (val) | Pearson Corr |
|---------------|-----------|-----------|---------------|
| Blur Model    | 92.3      | 6.7       | 0.89          |
| ISO MLP       | 0.015     | 92.8      | 0.84          |
| SS MLP        | 0.019     | 0.37 EV   | 0.79          |

---

## Future Enhancements

- [ ] Convert models to TensorFlow Lite for edge-device deployment  
- [ ] Real-time camera preview with live ISO/SS feedback  
- [ ] Add confidence scores and fallback rules  
- [ ] Improve low-light and fast-motion dataset coverage

---

## Contributors

- **Mayur Arora**  
- **Kinshuk Chauhan**  
- **Mankirat Singh Saini**  
- **Yatin Gupta**  
- **Karanpreet Singh Dhaliwal**  
- **Aditya Tayal**

---
