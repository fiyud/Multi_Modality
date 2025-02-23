import numpy as np

def post_process_weighted(diarize_segments, emotion_weights=None, text_weights=None):
    """
    Post-process with weighted scoring system for both audio and text emotions.
    
    Args:
        diarize_segments: DataFrame containing emotion predictions
        emotion_weights: Dictionary mapping emotion labels to their weights
        text_weights: Dictionary mapping text emotion labels to their weights
    """
    text_emotion_labels = ['clean', 'offensive', 'hate']
    # emotion_labels = ['sad', 'happy', 'angry', 'neutral', 'fear']
    emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad']

    if emotion_weights is None:
        emotion_weights = {
            'sad': 0.7,
            'angry': 0.8,
            'fear': 0.7,
            'happy': -0.5,
            'neutral': -0.3
        }
    
    if text_weights is None:
        text_weights = {
            'hate': 0.9,
            'offensive': 0.7,
            'clean': -0.8
        }
    
    total_score = 0
    segment_count = len(diarize_segments)
    
    for _, row in diarize_segments.iterrows():
        # Convert to numpy arrays
        emotion_probs = np.array(row['emotion'])
        text_emotion_probs = np.array(row['emotion_text'])
        
        if emotion_probs.ndim > 0:
            emotion_score = sum(
                prob * emotion_weights[emotion_labels[i]]
                for i, prob in enumerate(emotion_probs)
            )
        else:
            emotion_score = 0
        
        if text_emotion_probs.ndim > 0:
            text_score = sum(
                prob * text_weights[text_emotion_labels[i]]
                for i, prob in enumerate(text_emotion_probs)
            )
        else:
            text_score = 0
        
        # Combine scores (you can adjust the weights between emotion and text)
        segment_score = (emotion_score * 0.6) + (text_score * 0.4)
        total_score += segment_score
    
    # Normalize score by number of segments
    final_score = total_score / segment_count if segment_count > 0 else 0
    
    return "bad" if final_score > 0.15 else "good"

def post_process_temporal(diarize_segments, window_size=3):
    """
    Post-process considering temporal patterns in the conversation.
    
    Args:
        diarize_segments: DataFrame containing emotion predictions
        window_size: Number of segments to consider for temporal patterns
    """
    text_emotion_labels = ['clean', 'offensive', 'hate']
    emotion_labels = ['sad', 'happy', 'angry', 'neutral', 'fear']
    
    negative_emotions = {'sad', 'angry', 'fear'}
    negative_texts = {'offensive', 'hate'}
    
    # Sort segments by start time
    diarize_segments = diarize_segments.sort_values('start')
    
    negative_count = 0
    max_negative_window = 0
    
    # Sliding window analysis
    for i in range(len(diarize_segments) - window_size + 1):
        window = diarize_segments.iloc[i:i+window_size]
        window_negative_count = 0
        
        for _, row in window.iterrows():
            max_emotion = emotion_labels[np.argmax(row['emotion'])]
            max_text_emotion = text_emotion_labels[np.argmax(row['emotion_text'])]
            
            if max_emotion in negative_emotions or max_text_emotion in negative_texts:
                window_negative_count += 1
        
        max_negative_window = max(max_negative_window, window_negative_count)
        
    # Consider it "bad" if there's a concentration of negative segments
    return "bad" if max_negative_window >= (window_size * 0.6) else "good"

def post_process_ensemble(diarize_segments):
    weighted_pred = post_process_weighted(diarize_segments)
    temporal_pred = post_process_temporal(diarize_segments)
    
    # Simple majority voting (you can add more methods)
    predictions = [weighted_pred, temporal_pred]
    bad_count = sum(1 for pred in predictions if pred == "bad")
    
    return "bad" if bad_count >= len(predictions)/2 else "good"
