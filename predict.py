import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_emotion(model, tokenizer, label_encoder, input_text, max_length):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    prediction = model.predict(padded_input_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
    return predicted_label[0]
