# 📸 CaptureSmart: Intelligent Camera Settings Assistant

![Android](https://img.shields.io/badge/Platform-Android-green)
![Python](https://img.shields.io/badge/Language-Python-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey)
![Hackathon](https://img.shields.io/badge/Event-CS671_Hackathon-purple)

**CaptureSmart** is an AI-powered Android application developed for the **CS-671: Deep Learning & Its Applications Hackathon** at **IIT Mandi** (Group 34). The system helps photographers minimize motion blur by automatically recommending optimal **ISO** and **Shutter Speed** settings based on real-time image analysis.

## ✨ Key Features
- Real-time motion blur detection using custom MobileNetV2 model
- Intelligent ISO and shutter speed recommendations
- Low-latency Flask API backend
- Camera parameter prediction using trained MLP models
- Wi-Fi optimized for rapid image processing (~0.5-1s response)
- Comprehensive blur assessment with 7 visual feature extraction
- Synthetic blur augmentation pipeline for robust training

## 🧩 System Architecture
Android Camera → Capture Image → Send to Flask API → Preprocess Image → Blur Detection Model (MobileNetV2) → Extract Features (Laplacian, Tenengrad, etc) → ISO & Shutter Speed Prediction (MLP) → Return JSON to App → Display Recommendations

## 🛠️ Installation

### Backend Setup
Clone repository:
git clone https://github.com/yourusername/CaptureSmart.git  
cd CaptureSmart/API/

Create and activate virtual environment:
python -m venv venv  
source venv/bin/activate  (for Windows: venv\Scripts\activate)

Install dependencies:
pip install -r requirements.txt

Start server (runs on http://0.0.0.0:5001):
python api.py

### Android App Setup
- Open `Android_App` in Android Studio
- Update API endpoint in `app/src/main/java/com/example/capturesmart/network/NetworkService.kt`:
  private const val BASE_URL = "http://YOUR_LOCAL_IP:5001/"
- Build and run on physical Android device (API 26+)
- Grant camera and internet permissions when prompted

## 📂 Directory Structure

CaptureSmart/  
├── API/ (Flask server)  
│   ├── api.py  
│   ├── requirements.txt  
│   ├── blur_detection.py  
│   └── settings_predictor.py  
├── Android_App/  
│   ├── app/src/main/java/com/example/capturesmart/  
│   │   ├── camera/  
│   │   ├── network/  
│   │   └── ui/  
│   ├── res/  
│   └── build.gradle  
├── Models/  
│   ├── Blur_Detection/blur_detection_model_v2.h5  
│   └── Settings_Prediction/  
│       ├── iso_classifier_model.h5  
│       ├── iso_class_to_label.pkl  
│       ├── iso_feature_scaler.pkl  
│       ├── final_mlp_model_shutter.h5  
│       ├── final_scaler_shutter.pkl  
│       └── stable_shutter_model.h5  
├── Training/  
│   ├── Blur_Detection/  
│   │   ├── blur_detection_train.ipynb  
│   │   ├── blur_detection_test.ipynb  
│   │   └── dataset_creation.py  
│   └── Settings_Prediction/  
│       ├── iso_train.py  
│       ├── iso_test.py  
│       ├── ss_train.py  
│       ├── ss_test.py  
│       └── feature_extractor.py  
├── Dataset/  
│   ├── sharp_images/  
│   ├── motion_blur/  
│   └── gaussian_blur/  
├── ProjectReportGroup34.pdf  
├── LICENSE  
└── README.md

## 🌐 API Endpoint

POST /recommend_settings  
Request: image (JPEG, multipart/form-data)  
Response:  
{
  "blur_score": 41.23,
  "recommended_iso": 400,
  "recommended_shutter_speed": "1/125"
}

## 🔬 Technical Details

### Blur Detection Model
- Architecture: MobileNetV2 with custom regression head  
- Input: 224×224 RGB images  
- Output: Continuous blur score (0-100)  
- Training Data: 6,000 sharp images + 18,000 blurred (12k motion, 6k Gaussian)  
- Augmentations: Random crop, flip, rotate  
- Training: 50 epochs, batch size 32, Adam optimizer  
- MAE: 6.7 | Pearson Correlation: 0.89

### ISO & Shutter Speed Prediction

#### Features Used:
- Laplacian Variance  
- Tenengrad Score  
- Perceptual Blur Metric  
- Edge Density  
- Mean Brightness  
- Histogram Mean  
- Histogram Variance

#### Model:
- MLP: Dense(64) → ReLU → BatchNorm → Dropout → Dense(32) → ReLU → Output  
- ISO: Range 50–2500, MAE: 92.8  
- Shutter Speed: Log(1/t) transformation, MAE: 0.37 EV

## 📊 Performance Comparison

| Method               | MSE   | MAE  | Pearson |
|----------------------|-------|------|---------|
| Our MobileNetV2      | 92.3  | 6.7  | 0.89    |
| Laplacian Variance   | 142.6 | 9.8  | 0.71    |
| Tenengrad Score      | 138.2 | 9.5  | 0.73    |
| Perceptual Blur      | 125.9 | 8.2  | 0.79    |

## 📸 Sample Recommendations

| Scenario              | Blur Score | Recommended ISO | Recommended SS |
|-----------------------|------------|------------------|----------------|
| Bright daylight       | 12.3       | 100              | 1/640          |
| Indoor portrait       | 29.4       | 200              | 1/250          |
| Fast-moving subject   | 61.7       | 1250             | 1/30           |
| Low-light scene       | 84.2       | 2500             | 1/15           |

## 🚀 Future Roadmap (To not be continued for now)

### Feature Expansion
- Aperture recommendation  
- White balance auto-adjustment  
- Long exposure + night mode  
- Subject tracking  

### Performance Boost
- Quantization-aware training  
- Model pruning  
- Knowledge distillation  
- Hardware acceleration  

## 👥 Contributors (Group 34 - IIT Mandi)

| Name                       | 
|----------------------------|
| Aditya Tayal               | 
| Mayur Arora                |
| Kinshuk Chauhan            |
| Mankirat Singh Saini       |
| Yatin Gupta                |
| Karanpreet Singh Dhaliwal  |



## 🔗 Acknowledgements

- Course: CS-671: Deep Learning & Its Applications  
- Institution: Indian Institute of Technology, Mandi  
- Mentors: Dr. Aditya Nigam, Dr. Arnav Bhavsar  
- Dataset Sources: Manual Images clicked by our cameras
