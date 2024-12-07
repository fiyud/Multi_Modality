import tensorflow as tf

model_path = "H://Learning Files/Project AI/Aidemo/aidemo/Speech_emo/models/7733.h5"
model = tf.keras.models.load_model(model_path, compile=False)
emotion_labels = ['Angry', 'Anxiety', 'Happy', 'Sad', 'Neutral']
