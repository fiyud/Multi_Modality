import os
from Audio2text.separate_audio import *
from Audio2text import diarize_model
from Denoiser.denoiser_run import denoise_audio
from S2T import *
from Speech_emo_2.emotion_predict import *
from Speech_emo_2 import *
from PhoBertCNN import *
from PhoBertCNN.phobert import CNN, encoder_generator, predict_sentence, load_phobert_cnn, predict_text_emo
from tqdm import tqdm
from scipy.stats import entropy
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

phoBertCNN, cnn = load_phobert_cnn('D:/NCKHSV.2024-2025/Services/phoberttask2a_2.pt', r'D:/NCKHSV.2024-2025/Services/cnntask2a_2.pt', 'cpu')
saving_folder = "D:/NCKHSV.2024-2025/Services/aidemo/Audio_saving_file/Combined/"
denoised_output_folder = "D:/NCKHSV.2024-2025/Services/aidemo/Audio_saving_file/Denoised"
segment_path = "D:/NCKHSV.2024-2025/Services/aidemo/Audio_saving_file/Segmented"

def main(audio_link):

    # step 1: denoise audio
    print("------Denosing audio------")
    audio_denoised_link = denoise_audio(audio_link=audio_link, output_folder=denoised_output_folder)
    # print('Done denoising audio')
    
    # step 2: diaries audio using speaker diarization
    print("------Diarizing audio------")
    audio = whisperx.load_audio(audio_denoised_link) # load and embed audio for diarization
    diarize_segments = get_DiarizeSegments(diarize_model = diarize_model,audio=audio, min_speakers = 2, max_speakers= 4)
    print('Done diarizing audio')
    print(diarize_segments)
    
    # step 2.5: filter out the audio segments that are less than the threshold
    print('------Filtering out audio segments less than threshold------')
    diarize_segments = audio_threshold(diarize_segments, threshold=600)
    print('Done filtering out audio segments')
    print(diarize_segments)
    
    print('-----Segmenting audio-----')
    diarize_segments = cut_and_save_segments(audio_link=audio_denoised_link, diarize_segments=diarize_segments, format_type = 'wav', save_path = segment_path)
    print('Done segmenting audio')
    print(diarize_segments)
   
    # step 3: speech emotion
    print("------Predicting emotion------")
    diarize_segments = predict_diarize_emo(diarize_segments)
    print('Done predicting emotion')
    # print(diarize_segments)
    
    # step 4: speech to text
    print("------Transcribing audio------")
    diarize_segments = text_diarize_save(diarize_segments)
    print("Done transcribing audio")
    
    # step 4.5: text emotion classification
    print("------Predicting emotion from text------")
    diarize_segments = predict_text_emo(diarize_segments, phoBertCNN, cnn, "cpu", encoder_generator=encoder_generator)

    # step 5: post processing
    print("Post processing")
    result = post_process1(diarize_segments)
    print("-------All Done--------")

    # print(result)
    return result

def audio_threshold(diarize_segments, threshold=600):
    """
    Filter out the audio segments that are less than the threshold
    """
    diarize_segments['duration'] = diarize_segments['end'] - diarize_segments['start']
    
    return diarize_segments[diarize_segments['duration'] > threshold]

def post_process(diarize_segments):

    text_emotion_labels = ['clean', 'offensive', 'hate']
    # emotion_labels = ['sad', 'happy', 'angry', 'neutral', 'fear']
    emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad']

    negative_emotions = {'sad', 'angry', 'fear'}
    negative_texts = {'offensive', 'hate'}
    
    for _, row in diarize_segments.iterrows():
        emotion_values = row['emotion']  
        text_emotion_values = row['emotion_text']  
        
        max_emotion = emotion_labels[np.argmax(emotion_values)]
        max_text_emotion = text_emotion_labels[np.argmax(text_emotion_values)]
        
        if max_emotion in negative_emotions or max_text_emotion in negative_texts:
            return "bad"

    return "good"

def calculate_entropy(probabilities):
    probabilities = np.array(probabilities)
    probabilities = np.clip(probabilities, 1e-10, 1.0)  # Avoid log(0)
    probabilities = probabilities / np.sum(probabilities)
    return entropy(probabilities)

def get_top_k_indices(probabilities, k=2):
    return np.argsort(probabilities)[-k:]

def post_process1(diarize_segments, 
                emotion_threshold=0.5,
                text_threshold=0.5,
                alpha=10.0, 
                beta=5.0,  
                risk_threshold=0.5,
                top_k=2):
    """
    Parameters:
    -----------
    diarize_segments : pandas.DataFrame
        DataFrame containing 'emotion' and 'emotion_text' columns
    emotion_threshold : float
        Threshold for emotion probabilities
    text_threshold : float
        Threshold for text emotion probabilities
    alpha : float
        Steepness parameter for risk sigmoid
    beta : float
        Steepness parameter for final decision boundary
    risk_threshold : float
        Threshold for final risk assessment
    top_k : int
        Number of top emotions to consider
    Returns:
    --------
    str
        'bad' if risky content detected, 'good' otherwise
    """

    text_emotion_labels = ['clean', 'offensive', 'hate']
    emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad']
    negative_emotions = {'angry', 'fear', 'sad'}
    negative_texts = {'offensive', 'hate'}
    
    max_emotion_entropy = np.log(len(emotion_labels))
    max_text_entropy = np.log(len(text_emotion_labels))
    
    total_segment_risk = 0
    
    for _, row in diarize_segments.iterrows():
        # distributions
        emotion_values = np.array(row['emotion'])
        text_emotion_values = np.array(row['emotion_text'])
        
        emotion_entropy = calculate_entropy(emotion_values)
        text_entropy = calculate_entropy(text_emotion_values)
        
        emotion_confidence = 1 - (emotion_entropy / max_emotion_entropy)
        text_confidence = 1 - (text_entropy / max_text_entropy)
        
        segment_risk = 0
        
        top_emotion_indices = get_top_k_indices(emotion_values, top_k)
        for idx in top_emotion_indices:
            if emotion_labels[idx] in negative_emotions:
                risk = sigmoid(emotion_values[idx] - emotion_threshold, alpha)
                segment_risk += emotion_confidence * risk
        
        top_text_indices = get_top_k_indices(text_emotion_values, top_k)
        for idx in top_text_indices:
            if text_emotion_labels[idx] in negative_texts:
                risk = sigmoid(text_emotion_values[idx] - text_threshold, alpha)
                segment_risk += text_confidence * risk
        
        total_segment_risk += segment_risk
    
    if len(diarize_segments) > 0:
        average_risk = total_segment_risk / len(diarize_segments)
    else:
        average_risk = 0
    
    posterior_bad = sigmoid(average_risk - risk_threshold, beta)
    
    return "bad" if posterior_bad > 0.5 else "good"

def sigmoid(x, alpha=10.0):
    return 1 / (1 + np.exp(-alpha * x))

def post_process_sigmoid(diarize_segments, emotion_threshold=0.5, text_emotion_threshold=0.5, top_n=2, alpha=10.0, risk_threshold=0.5):
    text_emotion_labels = ['clean', 'offensive', 'hate']
    emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad']

    negative_emotions = {'angry', 'fear', 'sad'}
    negative_texts = {'offensive', 'hate'}

    total_risk = 0

    for _, row in diarize_segments.iterrows():
        emotion_values = np.array(row['emotion'])
        text_emotion_values = np.array(row['emotion_text'])

        if emotion_values.ndim == 0 or text_emotion_values.ndim == 0:
            raise ValueError("Emotion values must be lists/arrays, not scalars.")

        top_emotion_indices = np.argsort(emotion_values)[-top_n:]
        top_text_indices = np.argsort(text_emotion_values)[-top_n:]

        for idx in top_emotion_indices:
            if emotion_labels[idx] in negative_emotions:
                risk = sigmoid(emotion_values[idx] - emotion_threshold, alpha)
                total_risk += risk

        for idx in top_text_indices:
            if text_emotion_labels[idx] in negative_texts:
                risk = sigmoid(text_emotion_values[idx] - text_emotion_threshold, alpha)
                total_risk += risk

    return "bad" if total_risk > risk_threshold else "good"

# def predict_and_evaluate(audio_folders):
#     correct_predictions = 0
#     total_predictions = 0

#     for true_label, folder_path in tqdm(audio_folders.items()):
#         for file_name in tqdm(os.listdir(folder_path)):
#             if file_name.endswith(".wav"): 
#                 file_path = os.path.join(folder_path, file_name)
#                 print(f"Processing: {file_path}")

#                 predicted_label = main(file_path)

#                 if predicted_label == true_label:
#                     correct_predictions += 1

#                 total_predictions += 1

#     accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
#     print(f"Accuracy: {accuracy * 100:.2f}%")
#     return accuracy

def predict_and_evaluate(audio_folders):
    results = []
    correct_predictions = 0
    total_predictions = 0

    for true_label, folder_path in tqdm(audio_folders.items()):
        for file_name in tqdm(os.listdir(folder_path)):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)

                predicted_label = main(file_path)

                if predicted_label == true_label:
                    correct_predictions += 1
                total_predictions += 1

    results_df = pd.DataFrame(results)

    y_true = results_df["ground_truth"]
    y_pred = results_df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="bad")
    recall = recall_score(y_true, y_pred, pos_label="bad")
    f1 = f1_score(y_true, y_pred, pos_label="bad")

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    audio_folders = {
    "bad": "D:/NCKHSV.2024-2025/Services/emodata_v2/b",
    "good": "D:/NCKHSV.2024-2025/Services/emodata_v2/n"
    }

    predict_and_evaluate(audio_folders)