import librosa
import numpy as np

def extract_features_live(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = np.mean(chroma.T, axis=0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel = np.mean(mel.T, axis=0)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast.T, axis=0)
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    tonnetz = np.mean(tonnetz.T, axis=0)
    rms = librosa.feature.rms(y=y)
    rms = np.mean(rms.T, axis=0)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr = np.mean(zcr.T, axis=0)

    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz, rms, zcr])
    
    return features.reshape(1, -1)
