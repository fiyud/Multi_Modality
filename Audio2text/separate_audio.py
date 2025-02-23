import whisperx
import pandas as pd
from pydub import AudioSegment
import torch
import numpy as np
import os 

def audio2text(model, audio_link: str, batch_size):
    """
    Transcribe the audio file to text using the whisper large model model
    """
    audio = whisperx.load_audio(audio_link)
    result = model.transcribe(audio, batch_size=batch_size)
    text = " ".join(i['text'] for i in result['segments'])
    
    return text

def get_DiarizeSegments(diarize_model, audio: np.ndarray, min_speakers=2, max_speakers=2):
    """
    Diarize the audio file to get the speaker segments (table format)
    """
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    diarize_segments[['start', 'end']] = diarize_segments[['start', 'end']].apply(lambda x: x.round(2)) * 1000

    return diarize_segments

def cut_and_save_segments(audio_link: str, diarize_segments, format_type = 'wav', save_path = "./audio_segments/"):
    """
    Cut the audio file into segments based on the diarize segments and save them.
    """
    if not audio_link.endswith(('.mp3', '.wav', '.ogg')):
        raise ValueError("Unsupported audio format. Please provide a .mp3, .wav, .ogg, or .flac file.")
    
    if audio_link.endswith('.mp3'):
        audio = AudioSegment.from_mp3(audio_link)
    elif audio_link.endswith('.wav'):
        audio = AudioSegment.from_wav(audio_link)
    else:
        audio = AudioSegment.from_ogg(audio_link)
        
    os.makedirs(save_path, exist_ok=True)

    file_link = []
   
    for i, (start, end) in enumerate(diarize_segments[['start', 'end']].to_numpy()):
        segment = audio[start:end]
        output_file = f"segment_{i+1}.{format_type}"
        
        save_link = save_path + output_file
        segment.export(save_link, format=format_type)
        # print(f"Saved {output_file}")
        file_link.append(save_link)
        
    diarize_segments['link'] = file_link
    
    return diarize_segments

def combine_audio(diarize_segments, save_path = "./audio_segments/", format_type = 'wav'):
    """
    Combine the audio segments of the same speaker.
    """
    
    info_df = pd.DataFrame(columns = ['id', 'link'])
    
    speaker_label = diarize_segments['speaker'].unique()
    
    for label in speaker_label:
        combine_audio = AudioSegment.empty()
        temp = diarize_segments[diarize_segments['speaker'] == label]
        audio_files = temp['link']
        for audio_file in audio_files:
            audio = AudioSegment.from_file(audio_file)
            combine_audio += audio
            
        save_combine_link = save_path + f'{label}.{format_type}'
        combine_audio.export(save_combine_link, format = format_type)
        
        data = {'id': label, 'link': save_combine_link}
        
        info_df = info_df._append(data, ignore_index = True)
    return info_df
