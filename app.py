import streamlit as st
from data_processing import load_data, preprocess_texts, encode_labels
from model import build_model, train_model
from predict import predict_emotion
import os

# Set the page configuration
st.set_page_config(
    page_title="Text Emotion Classifier",
    page_icon="🌟",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Add Theme Toggle (Light/Dark Mode)
theme_mode = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
if theme_mode == "Dark":
    st.markdown(
        """
        <style>body { background-color: #2e2e2e; color: white; }</style>
        """, unsafe_allow_html=True,
    )

# Custom CSS styles for improved layout and appearance
st.markdown(
    """
    <style>
        /* Page background image */
        body {
            background-image: url('bgimage.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }

        /* Title styling with animation */
        .title {
            font-family: 'Arial Black', sans-serif;
            color: #1f77b4;
            font-size: 3rem;
            text-align: center;
            margin-top: 50px;
            margin-bottom: 30px;
            animation: fadeIn 2s ease-in-out;
        }

        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }

        /* Text input styling */
        .stTextInput input {
            border: 2px solid #1f77b4;
            padding: 10px;
            border-radius: 10px;
            font-size: 1.2rem;
        }

        /* Prediction result */
        .result-box {
            background-color: #e6f2ff;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Button styling */
        button {
            background: linear-gradient(45deg, #1f77b4, #167fa6);
            color: white;
            font-size: 1.2rem;
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        button:hover {
            transform: scale(1.1);
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        }

        /* Emoji icons for emotions */
        .emotion-icons {
            text-align: center;
            margin-top: 20px;
        }
        .emotion-icons img {
            margin: 0 10px;
            width: 60px;
            height: 60px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title with custom CSS class
st.markdown('<div class="title">Text Emotion Classifier</div>', unsafe_allow_html=True)

# Display emotion icons
st.markdown(
    """
    <div class="emotion-icons">
        <img src="https://emojipedia-us.s3.amazonaws.com/source/skype/289/slightly-smiling-face_1f642.png" alt="happy" />
        <img src="https://emojipedia-us.s3.amazonaws.com/source/skype/289/face-with-tears-of-joy_1f602.png" alt="joy" />
        <img src="https://emojipedia-us.s3.amazonaws.com/source/skype/289/crying-face_1f622.png" alt="sad" />
        <img src="https://emojipedia-us.s3.amazonaws.com/source/skype/289/angry-face_1f620.png" alt="angry" />
        <img src="https://emojipedia-us.s3.amazonaws.com/source/skype/289/face-screaming-in-fear_1f631.png" alt="fear" />
        <img src="https://emojipedia-us.s3.amazonaws.com/source/skype/289/smiling-face-with-sunglasses_1f60e.png" alt="cool" />
    </div>
    """,
    unsafe_allow_html=True,
)

# Function to get corresponding emoji for predicted emotion
def get_emotion_icon(emotion):
    emotion_icons = {
        "joy": "😂",
        "sadness": "😢",
        "anger": "😠",
        "fear": "😱",
        "happy": "🙂",
        "surprise": "😲",
        "cool": "😎"
    }
    return emotion_icons.get(emotion.lower(), "🙂")  # Default emoji if not found

# Load and preprocess data
texts, labels = load_data('train.txt')
tokenizer, padded_sequences, max_length = preprocess_texts(texts)
label_encoder, one_hot_labels = encode_labels(labels)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences, one_hot_labels, test_size=0.2)

# Build and train the model
model = build_model(input_dim=len(tokenizer.word_index) + 1, max_length=max_length, num_classes=len(one_hot_labels[0]))
if os.path.exists('emotion_model.h5'):
    model.load_weights('emotion_model.h5')
else:
    train_model(model, xtrain, ytrain, xtest, ytest)
    model.save('emotion_model.h5')

# Input text for prediction
input_text = st.text_input("Enter text to predict emotion:")
if input_text:
    # Display spinner while predicting
    with st.spinner('Predicting emotion...'):
        predicted_emotion = predict_emotion(model, tokenizer, label_encoder, input_text, max_length)
        emotion_icon = get_emotion_icon(predicted_emotion)
        st.markdown(f'<div class="result-box">Predicted Emotion: {predicted_emotion} {emotion_icon}</div>', unsafe_allow_html=True)


