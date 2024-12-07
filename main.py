from flask import Flask, request, jsonify
import os
from Audio2text.separate_audio import *
from Audio2text import diarize_model
from Denoiser.denoiser_run import denoise_audio
from S2T import *
from Speech_emo.emotion_predict import *
from Speech_emo import *

saving_folder = "H://Learning Files/Project AI/Aidemo/aidemo/Audio_saving_file/Web_audio"
denoised_output_folder = "H://Learning Files/Project AI/Aidemo/aidemo/Audio_saving_file/Denoised/"
segment_path = "H://Learning Files/Project AI/Aidemo/aidemo/Audio_saving_file/Segmented/"

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello, world!")

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
        
    
    # saving file to the folder
    print("------Saving file------")
    file_saving_path = saving_folder + '/' + file.filename
    file.save(file_saving_path)

    print('Done saving file:', file_saving_path)
    
    # step 1: denoise audio
    print("------Denosing audio------")
    audio_denoised_link = denoise_audio(audio_link=file_saving_path, output_folder=denoised_output_folder)
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
    print(diarize_segments)
    
    # step 5: return to client
    print("Post processing")
    string = post_process(diarize_segments)
    print("-------All Done--------")	
    print(string)
    # return 'A: xin chào (10% vui, 50% bình thường, 40% sợ)B: xin chào (10% vui, 50% bình thường, 40% sợ)', 200

    return string, 200

def audio_threshold(diarize_segments, threshold=600):
    """
    Filter out the audio segments that are less than the threshold
    """
    diarize_segments['duration'] = diarize_segments['end'] - diarize_segments['start']
    
    return diarize_segments[diarize_segments['duration'] > threshold]

def post_process(diarize_segments):
    string_output = ''
    for i in range(len(diarize_segments)):
        id = diarize_segments.iloc[i]['speaker']
        text = diarize_segments.iloc[i]['text']['text']
        emotion = diarize_segments.iloc[i]['emotion']
        #  map to the emotion label
        
        temp = zip(emotion, emotion_labels)
        emotion = [f'{emo:.2f}% {label}' for emo, label in temp]
        
        string_output += id + ': ' + text + ' (' + ','.join(emotion) + ')'   
    return string_output

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
    
