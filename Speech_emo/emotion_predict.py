import numpy as np
import tensorflow as tf
import librosa
import os 
import time 
from Speech_emo import model
# Function to preprocess audio file
def preprocess_audio(file_path, sr=16000, duration=2.5, offset=0.5, n_mfcc=128, n_fft=2048, hop_length=512):
    audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if mfccs.shape[1] < 157:
        pad_width = 157 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > 157:
        mfccs = mfccs[:, :157]
    return mfccs

# Function to predict emotion from audio file
def predict_emotion(model, file_path):
    mfccs = preprocess_audio(file_path)
    mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1])

    predictions = model.predict(mfccs)
    
    prediction_percentages = (predictions[0] * 100).tolist()

    return prediction_percentages

def predict_diarize_emo(diarize_segment):
    start = time.time()
    emo_label = []
    for link in diarize_segment['link']:
        prediction_percentages = predict_emotion(model = model, file_path=link)
        emo_label.append(prediction_percentages)
        
    diarize_segment['emotion'] = emo_label
    print(f"Finished predicting emotion in {time.time() - start}s")
    return diarize_segment


