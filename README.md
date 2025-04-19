# Facial Sentiment Analysis using Convolutional Neural Network (CNN)  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8.2-orange)  
![OpenCV](https://img.shields.io/badge/OpenCV-4.6.0-green)  
![License](https://img.shields.io/badge/License-MIT-lightgrey)  

## 📌 Table of Contents  
- [Project Background](#-project-background)  
- [Key Features](#-key-features)  
- [Technologies Used](#-technologies-used)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Detailed Results](#-detailed-results)  
- [Future Work](#-future-work)  
- [Acknowledgments](#-acknowledgments)  
- [License](#-license)  

---

## 🌟 Project Background  
**Year:** 2022-2023 (Engineering Internship)  
**Supervisor:** Mr. Bassem Hmissa (LEONI WIRING SYSTEMS)  
**Institution:** National School of Electronics and Telecommunications of Sfax, Tunisia  

### 🎯 Motivation  
Commercial APIs (e.g., Microsoft Face API) treat facial analysis as a "black box," limiting customization and interpretability. This project was developed to:  
- **Demystify AI models** by building face detection, emotion recognition, and facial identification systems from scratch.  
- **Analyze hyperparameter impact** (epochs, batch size, optimizers) on model accuracy.  
- **Provide a flexible SaaS solution** adaptable to client-specific datasets.  

---

## 🚀 Key Features  
### 1. **Face Detection**  
- **Haar Cascade**: Real-time detection using OpenCV.  
- **YOLOv4**: Higher accuracy (94.82%) but sensitive to face angles/lighting.  
  - *Limitations*: Struggles with rotated faces
     (![-2](https://github.com/user-attachments/assets/85f12ba4-72d1-4295-ac7a-141bc9b03362))
    and color resemblance
    (![-1](https://github.com/user-attachments/assets/e4bc905a-b6cc-4517-9340-198b075fbc97).  

### 2. **Emotion Detection**  
- **7-class CNN** (angry, disgust, fear, happy, neutral, sad, surprise).  
- **Dataset**: FER-2013 (35,887 images, 48x48px).  
- **Best Model**:  
  ```python
  Model.add(ZeroPadding2D())  
  Model.add(Conv2D(32, (3,3), activation='relu'))  
  Model.add(MaxPooling2D())  
  Model.add(Dense(7, activation='softmax'))  
  ```

### 3. **Facial Recognition**  
- **Dataset**: LFW (13,233 images) + custom-injected faces.  
- **Custom CNN**: Achieved 76.34% training accuracy.   

---

## 🛠️ Technologies Used  
| Category          | Tools/Libraries                                                                 |  
|-------------------|---------------------------------------------------------------------------------|  
| **Frameworks**    | TensorFlow, Keras, Darknet (YOLOv4)                                             |  
| **Computer Vision**| OpenCV, PIL                                                                     |  
| **Data Processing**| NumPy, Pandas, Matplotlib                                                       |  
| **Environment**   | Google Colab, Kaggle API                                                        |  

---

## 📦 Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/mariam-khediri/Facial-Sentiment-Analysis-using-CNN.git  
   cd facial-sentiment-analysis  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  # Sample requirements.txt included below  
   ```  
   **`requirements.txt`**:  
   ```  
   tensorflow==2.8.2  
   opencv-python==4.6.0  
   numpy>=1.21.0  
   matplotlib>=3.5.0  
   ```  
3. Download pretrained models:  
   - [YOLOv4 weights](https://github.com/AlexeyAB/darknet/releases)  
   - [Haar Cascade XML](https://github.com/opencv/opencv/tree/master/data/haarcascades)  

---

## 🖥️ Usage  
### Real-Time Emotion Detection  
```bash  
python emotion_detection.py --mode=camera  
```  
**Output**:  
- Processes webcam feed → detects faces → predicts emotions (![image](https://github.com/user-attachments/assets/703329cf-f829-416f-baf3-6bb7e54bc00d)
).  
- Supports image input: `--image=test.jpg`.  

### Facial Recognition  
```bash  
python facial_recognition.py --dataset=lfw --epochs=120  
```  
**Output**:  
- Recognizes faces from LFW dataset or custom-injected images ![image](https://github.com/user-attachments/assets/69a96fd9-c405-4681-9d7f-6e4da1f1035c)
 

---

## 📊 Detailed Results  
### Emotion Detection (CNN)  
| Hyperparameter      | Best Value  | Accuracy (Train/Test) |  
|---------------------|-------------|-----------------------|  
| Optimizer           | Adam (lr=0.001) | 94% / 51%            |  
| Epochs             | 30          | 94.75% (Train)       |  
| Batch Size          | 64          | 47.38% (Test)        |  

**Key Observations**:  
- Higher epochs → overfitting (test loss ↑ to 6.21).  
- SGD optimizer underperformed (max 25.83% test accuracy).  

### Facial Recognition (Custom CNN)  
| Batch Size | Epochs | Train Accuracy | Test Accuracy |  
|------------|--------|----------------|---------------|  
| 64         | 120    | 76.34%         | 14.32%        |  
| 32         | 25     | 32.37%         | 10.34%        |  

**Challenges**:  
- Low test accuracy due to dataset complexity (5,749 identities).  
- Dropout layers reduced overfitting (![image](https://github.com/user-attachments/assets/8902501a-12f7-4342-90ab-0ee5ea271418)
 vs ![image](https://github.com/user-attachments/assets/bc08f6b9-729a-4377-a89a-f8d4b9886c66)
).  

---

## 🔮 Future Work  
- **Data Augmentation**: Improve test accuracy with transformations (rotation, scaling).  
- **Transformer Models**: Experiment with Vision Transformers (ViT).  
- **API Deployment**: Flask/Django backend for SaaS integration.  

---

## 🙏 Acknowledgments  
- **Professional Supervisor**: Mr. Bassem Hmissa (LEONI).  
- **Dataset Providers**: Kaggle (FER-2013, LFW).  
- **Institutional Support**: Sfax University, Tunisia.  

---

## 📜 License  
MIT © [Mariem Khedhiri](https://github.com/mariam-khediri)  
