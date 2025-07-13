# Real-time-speech-emotion-detection-
# 🎤 Speech Emotion Detection (Real-time using Machine Learning)

This is a collaborative project that focuses on detecting emotions from **human speech in real time** using classical Machine Learning techniques. It includes everything from data preprocessing and model training to a live audio interface using Flask.

> 🔍 **Goal:** Given a voice recording, detect the speaker's emotion such as *happy, sad, angry, calm, etc.*

---

## 📌 What this project does

1. Takes voice input from a **microphone or uploaded audio file** (`.wav`)
2. Extracts audio features like:
   - **MFCCs**
   - **Chroma**
   - **Spectral Contrast**
   - **Mel Spectrogram**
   - **Tonnetz**
   - **RMS Energy**
   - **Zero Crossing Rate**
3. Uses **feature selection (SelectKBest)** and **data scaling**
4. Trains a **Multi-layer Perceptron (MLPClassifier)** on balanced data (using **SMOTE**)
5. Predicts the emotion using the trained model
6. Displays emotion on a real-time **Flask web interface**

---

## 🛠️ Technologies Used

| Task                      | Tools / Libraries                      |
|---------------------------|----------------------------------------|
| Audio processing          | `librosa`, `numpy`, `sounddevice`     |
| Feature selection         | `scikit-learn` (SelectKBest, f_classif) |
| Model training            | `MLPClassifier` from `sklearn.neural_network` |
| Data balancing            | `imblearn` (SMOTE)                     |
| Web app                   | `Flask`, `HTML`, `JavaScript`         |
| Model serialization       | `joblib`                               |
| Visualization (confusion matrix, etc.) | `matplotlib`, `seaborn`     |

---

## 🎯 Emotions Detected

This model supports the following 8 emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 👥 Contributors

- **Ankur Kumar** – Data preprocessing, model training, model evaluation
- **Palak Pandey** – Web integration, audio recording & UI


