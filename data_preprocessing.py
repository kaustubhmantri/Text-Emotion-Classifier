import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import keras

def load_data(file_path):
    data = pd.read_csv(file_path, sep=';')
    data.columns = ["Text", "Emotions"]
    texts = data["Text"].tolist()
    labels = data["Emotions"].tolist()
    return texts, labels

def preprocess_texts(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return tokenizer, padded_sequences, max_length

def encode_labels(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return label_encoder, keras.utils.to_categorical(labels)
