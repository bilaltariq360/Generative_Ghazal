import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load dataset and train LabelEncoder (same one used during training)
data = pd.read_csv("ghazals.csv").iloc[:500]
poetry_lines = data["Poetry"].dropna().tolist()

text = " ".join(poetry_lines)
words = text.split()

word_encoder = LabelEncoder()
word_encoder.fit(words)
word_to_index = {word: i for i, word in enumerate(word_encoder.classes_)}
index_to_word = {i: word for word, i in word_to_index.items()}

# Load the trained model
model = tf.keras.models.load_model("ghazal_model.h5")

st.set_page_config(page_title="Ghazal Generator", page_icon="📜")

# Streamlit UI
st.title("Ghazal Generator with GRU")

# User Inputs
start_text = st.text_input("Enter starting words:", "dil ke armaan")
words_per_line = st.slider("Words per Line:", 3, 15, 5)
total_lines = st.slider("Total Lines:", 2, 10, 5)

# Function to Generate Ghazal
def generate_ghazal(start_text, words_per_line, total_lines):
    generated_text = start_text.split()
    
    for _ in range(total_lines * words_per_line):
        # Convert input words to numbers
        encoded_input = [word_to_index.get(word, 0) for word in generated_text[-5:]]
        encoded_input = pad_sequences([encoded_input], maxlen=5, truncating="pre")
        
        # Predict next word
        predicted_index = np.argmax(model.predict(encoded_input), axis=-1)[0]
        next_word = index_to_word.get(predicted_index, "")

        generated_text.append(next_word)

        # Add newline every 'words_per_line' words
        if len(generated_text) % words_per_line == 0:
            generated_text.append("\n")

    return " ".join(generated_text)

# Generate and display Ghazal
if st.button("Generate Ghazal"):
    ghazal = generate_ghazal(start_text, words_per_line, total_lines)
    st.text_area("Generated Ghazal:", ghazal, height=200)
