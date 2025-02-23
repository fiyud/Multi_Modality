from transformers import pipeline

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline("automatic-speech-recognition", model=r"D:\NCKHSV.2024-2025\Services\aidemo\S2T", device=device)

def speech_to_text(audio_link):
    text = pipe(audio_link)
    return text

def text_diarize_save(diarize_segment):
    text_label = []
    
    for link in diarize_segment['link']:
        text = speech_to_text(link)
        diarize_segment['text'] = text
        text_label.append(text)
        
    diarize_segment['text'] = text_label
    return diarize_segment