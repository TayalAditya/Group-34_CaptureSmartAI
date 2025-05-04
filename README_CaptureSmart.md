
# 📷 CaptureSmart: Intelligent ISO & Shutter Speed Recommendation App

**CaptureSmart** is an AI-powered Android application that helps users minimize motion blur by automatically recommending the optimal **ISO** and **Shutter Speed (SS)** settings for a given image. The system uses a custom MobileNetV2 model to assess blur severity and two trained MLP models to suggest camera parameters. The backend is served via a Flask API and optimized for low-latency usage.

---

## 📌 Project Components

1. **Android App** – Real-time camera interface and prediction viewer  
2. **Flask API Backend** – Receives images and responds with ISO/SS recommendations  
3. **MLP Models** – Predict ISO and Shutter Speed based on extracted visual features  
4. **Blur Detection Model** – Predicts blur severity score (0–100)

---

## 🧠 Model Architectures

### 1. 🔍 **Blur Severity Prediction Model**

- **Backbone**: `MobileNetV2`
- **Output**: Continuous score (0–100)
- **Loss Function**: MSE
- **Training Data**:  
  - 6,000 sharp images  
  - 18,000 synthetically blurred variants  
  - Labels based on blur severity index

### 2. 📊 **ISO & SS MLP Models**

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

## 📦 Repository Structure

```
CaptureSmart/
│
├── android_app/               # Android Studio project
│   └── MainActivity.kt, CameraActivity.java, UI, permissions, API service
│
├── flask_api/                 # Backend server (Flask)
│   ├── app.py                 # Main server file
│   ├── models/
│   │   ├── blur_model.h5
│   │   ├── iso_model.h5
│   │   ├── ss_model.h5
│   │   ├── iso_scaler.pkl
│   │   └── ss_scaler.pkl
│   └── utils/                 # Feature extraction functions
│
├── dataset/                   # Training data snapshots (optional)
│
└── README.md
```

---

## ⚙️ Installation & Setup

### ✅ Android App

1. Open `android_app/` in Android Studio  
2. Connect a physical Android device  
3. Update the IP address of the server endpoint in API service to match the local server (e.g., `http://192.168.0.102:5001/recommend_settings`)  
4. Grant permissions: `CAMERA`, `INTERNET`  
5. Build and run on device

### ✅ Flask API Server

1. Navigate to `flask_api/`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start server:
   ```bash
   python app.py
   ```
4. Server runs at: `http://0.0.0.0:5001/` (accessible on the same Wi-Fi network)

---

## 📡 API Endpoints

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

## ⚡ Latency Optimization

To reduce network latency during image transfer and prediction:

- We hosted the **Flask server on the same local Wi-Fi network** as the Android device.  
- This setup eliminates external API calls and reduces response time to ~0.5–1 sec.

### 🔄 Alternatives

If local hosting is not feasible, you can:

- Deploy the Flask API to cloud services such as **Render**, **AWS EC2**, or **Heroku**  
- Use **ngrok** for temporary public tunneling  
- Convert models to **TensorFlow Lite** and integrate directly into the Android app for on-device inference

---

## 🧪 Sample Output

| Image Name | Blur Score | Predicted ISO | Predicted SS |
|------------|------------|---------------|--------------|
| image_01.jpg | 29.4     | 200           | 1/250        |
| image_05.jpg | 61.7     | 1250          | 1/30         |
| image_09.jpg | 12.3     | 100           | 1/640        |

---

## 📊 Evaluation Metrics

| Model         | MSE (val) | MAE (val) | Pearson Corr |
|---------------|-----------|-----------|---------------|
| Blur Model    | 92.3      | 6.7       | 0.89          |
| ISO MLP       | 0.015     | 92.8      | 0.84          |
| SS MLP        | 0.019     | 0.37 EV   | 0.79          |

---

## 🛠️ Future Enhancements

- [ ] Convert models to TensorFlow Lite for edge-device deployment  
- [ ] Real-time camera preview with live ISO/SS feedback  
- [ ] Add confidence scores and fallback rules  
- [ ] Improve low-light and fast-motion dataset coverage

---

## 🧠 Contributors

- **Mayur**  
- **Kinshuk Chauhan**  
- **Mankirat Singh Saini**  
- **Yatin Gupta**  
- **Karanpreet Singh Dhaliwal**  
- **Aditya Tayal**

---

## 🙋‍♂️ Support

Raise an [Issue](https://github.com/your-repo/issues) or contact us via email for queries or contributions.

---
