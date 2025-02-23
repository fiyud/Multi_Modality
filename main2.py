import os
from Audio2text.separate_audio import *
from Audio2text import diarize_model
from Denoiser.denoiser_run import denoise_audio
from S2T import *
from Speech_emo_2.emotion_predict import *
from Speech_emo_2 import *
from PhoBertCNN import *
from PhoBertCNN.phobert import CNN,encoder_generator, predict_sentence, load_phobert_cnn, predict_text_emo

phoBertCNN, cnn = load_phobert_cnn('C:/Users/Admin/Desktop/Desktop/PhoBertCnn/savefolder/phoberttask2a_2.pt', r'C:/Users/Admin/Desktop/Desktop/PhoBertCnn/savefolder/cnntask2a_2.pt', 'cpu')
saving_folder = "C:/Users/Admin/Desktop/Desktop/aidemo/aidemo/Audio_saving_file/Combined/"
denoised_output_folder = "C:/Users/Admin/Desktop/Desktop/aidemo/aidemo/Audio_saving_file/Denoised"
segment_path = "C:/Users/Admin/Desktop/Desktop/aidemo/aidemo/Audio_saving_file/Segmented"

def main(audio_link):

    # step 1: denoise audio
    print("------Denosing audio------")
    audio_denoised_link = denoise_audio(audio_link=audio_link, output_folder=denoised_output_folder)
    print('Done denoising audio')
    
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
    print(diarize_segments)
    
    # step 4: speech to text
    print("------Transcribing audio------")
    diarize_segments = text_diarize_save(diarize_segments)
    print("Done transcribing audio")
    
    # step 4.5: text emotion classification
    print("------Predicting emotion from text------")
    diarize_segments = predict_text_emo(diarize_segments, phoBertCNN, cnn, "cpu", encoder_generator=encoder_generator)

    # step 5: post processing
    print("Post processing")
    result = post_process(diarize_segments)
    print("-------All Done--------")	
    # return 'A: xin chào (10% vui, 50% bình thường, 40% sợ)B: xin chào (10% vui, 50% bình thường, 40% sợ)', 200
    print(result)
    return result

def audio_threshold(diarize_segments, threshold=600):
    """
    Filter out the audio segments that are less than the threshold
    """
    diarize_segments['duration'] = diarize_segments['end'] - diarize_segments['start']
    
    return diarize_segments[diarize_segments['duration'] > threshold]

def post_process(diarize_segments):
    segment_labels = []
    # print("Dỉaizeeee", diarize_segments)
    # emotion_labels = ['happy', 'sad', 'neutral', 'angry', 'fear']
    emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad']
    for i in diarize_segments['emotion']:
        # Suppose segment['emotion'] is a dict like {'happy': 0.75, 'sad': 0.1, ...}
        emotion_dict = dict(zip(emotion_labels, i))
        label = process_segment_emotion(emotion_dict)
        segment_labels.append(label)
    
    final_label = aggregate_segment_results(segment_labels)
    
    return final_label


def classify_intensity(emotion, probability):
    if probability < 0.33:
        return f"weak {emotion}"
    elif probability < 0.70:
        return f"normal {emotion}"
    else:
        return f"strong {emotion}"

def process_segment_emotion(segment_emotion_probs):
    emotion, prob = max(segment_emotion_probs.items(), key=lambda item: item[1])
    intensity_label = classify_intensity(emotion, prob)
    return intensity_label

def aggregate_segment_results(segment_labels):
    intensity_weight = {
        "weak": 1,
        "normal": 2,
        "strong": 3
    }
    
    emotion_scores = {emotion: 0 for emotion in ['happy', 'sad', 'neutral', 'angry', 'fear']}
    
    for label in segment_labels:
        parts = label.split()  # e.g., ["weak", "happy"]
        if len(parts) == 2:
            intensity, emotion = parts
            emotion_scores[emotion] += intensity_weight.get(intensity, 0)
    
    final_emotion = max(emotion_scores, key=emotion_scores.get)
    return final_emotion

def calculate_emotion_avg(emotion_avg):
    
    bad_value = sum(emotion_avg[i] for i in [0, 1, 3])
    happy_value = emotion_avg[2]  
    normal_value = emotion_avg[4] 
    return bad_value, happy_value, normal_value

def calculate_emotion_text_avg(emotion_text_avg):
    clean_value = emotion_text_avg[0]
    bad_text_value = sum(emotion_text_avg[i] for i in [1, 2])
    return clean_value, bad_text_value

def get_highest_emotion(emotion_avg):
    return max(range(len(emotion_avg)), key=lambda i: emotion_avg[i])

def process_emotion_text(emotion_text_avg, highest_emotion_avg):
   
    clean = emotion_text_avg[0]
    bad_text = sum(emotion_text_avg[i] for i in [1, 2])
    if bad_text > clean:
        return "bad" 
    else:
        if highest_emotion_avg == 2:  # happy
            return "happy"
        elif highest_emotion_avg == 4:  # neutral
            return "neutral"
        else:
            return "bad" 

import os

# Đường dẫn tới các thư mục chứa audio
audio_folders = {
    "bad": "C://Users/Admin/Desktop/Desktop/aidemo/emodata/emodata/b",
    "happy": "C://Users/Admin/Desktop/Desktop/aidemo/emodata/emodata/h",
    "neutral": "C://Users/Admin/Desktop/Desktop/aidemo/emodata/emodata/n"
}

def predict_and_evaluate(audio_folders):
    correct_predictions = 0
    total_predictions = 0

    for true_label, folder_path in audio_folders.items():
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"): 
                file_path = os.path.join(folder_path, file_name)
                print(f"Processing: {file_path}")

                predicted_label = main(file_path)

                if predicted_label == true_label:
                    correct_predictions += 1

                total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    predict_and_evaluate(audio_folders)

