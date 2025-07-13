from flask import Flask, render_template, request, jsonify
import joblib
import os
from Backend.record import record_audio
from Backend.extract import extract_features_live
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Load model & pre-processing tools
model = joblib.load("model\emotion_model.pkl")
scaler = joblib.load("model\emotion_scaler.pkl")
selector = joblib.load("model\emotion_selector.pkl")

emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    
    record_audio()  # saves as live_input.wav

    # Extract features
    features = extract_features_live("live_input.wav")
    features_selected = selector.transform(features)
    features_scaled = scaler.transform(features_selected)

    # Prediction
    predicted_class = model.predict(features_scaled)[0]
    emotion = emotion_labels[predicted_class]
    print(emotion)

    
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
