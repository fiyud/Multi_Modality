import numpy as np
import tensorflow as tf
import librosa
import os 
import time 
import torch
from torch.nn import functional as F
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r"D:\NCKHSV.2024-2025\Services\aidemo")))
from Speech_emo_2 import model

# Define emotion labels
emotion_labels = ['Angry', 'Anxiety', 'Happy', 'Sad', 'Neutral']

def preprocess_audio(file_path, sr=16000, duration=5, n_mfcc=128, n_fft=2048, hop_length=512):
    wave_form, _ = librosa.load(file_path, sr=sr)
    
    # Pad or trim the audio to match expected duration
    if len(wave_form) < sr * duration:
        wave_form = np.pad(wave_form, (0, sr * duration - len(wave_form)), mode='symmetric')
    else:
        wave_form = wave_form[:sr * duration]
    
    # Compute MFCC features
    mfcc_features = librosa.feature.mfcc(y=wave_form, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    return np.array([mfcc_features])

def predict_emotion(model, file_path):
    # Preprocess audio file
    mfcc_features = preprocess_audio(file_path)
    mfcc_features = torch.tensor(mfcc_features).unsqueeze(0)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(mfcc_features)
    
    # Convert to percentage
    percentages = (F.softmax(output.squeeze(), dim=0) * 100).numpy().tolist()
    
    # Get the predicted label
    predicted_label = emotion_labels[np.argmax(percentages)]
    
    return {'percentages': percentages, 'predicted_label': predicted_label}

# def predict_diarize_emo(diarize_segment):
#     start = time.time()
#     emo_label = []
#     for link in diarize_segment['link']:
#         prediction = predict_emotion(model, file_path=link)
#         emo_label.append(prediction)
        
#     diarize_segment['emotion'] = emo_label
#     # print(f"Finished predicting text emotion in {time.time() - start}s")
#     return diarize_segment

def predict_diarize_emo(diarize_segment):
    start = time.time()
    emo_label = []
    for link in diarize_segment['link']:
        prediction = predict_emotion(model, file_path=link)
        emo_label.append(prediction['percentages'])  # <-- Extract percentages
    diarize_segment['emotion'] = emo_label
    return diarize_segment

# def run_inference(audio_folder, model):
#     """
#     Run inference on a folder of audio files and return classification results with percentages.

#     Parameters
#     ----------
#     audio_folder: str
#         Path to the folder containing audio files.

#     model: torch.nn.Module
#         The pre-trained emotion classification model.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame containing file paths, classification percentages, and predicted labels.
#     """
#     results = []

#     for file_name in os.listdir(audio_folder):
#         if file_name.endswith('.wav'):
#             file_path = os.path.join(audio_folder, file_name)
#             prediction = predict_emotion(model, file_path)
#             results.append({
#                 'file_path': file_path,
#                 'prediction_percentages': prediction['percentages'],
#                 'predicted_label': prediction['predicted_label']
#             })

#     df = pd.DataFrame(results)
#     df.to_csv("inference_results2.csv", index=False)
#     return df

# if __name__ == "__main__":
#     audio_folder = r"D:\NCKHSV.2024-2025\Services\emodata_v2\n"
#     results_df = run_inference(audio_folder, model)
#     print(results_df)