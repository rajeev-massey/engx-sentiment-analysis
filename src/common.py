import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib

# Constants for model configuration
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 128
DROPOUT_RATE = 0.2
EPOCHS = 5
BATCH_SIZE = 64
LATENT_DIM = 100
LIMIT= 10000

# Function to load dataset from a file
def load_dataset(file_path):
    return pd.read_csv(file_path).limit(LIMIT)

# Function to preprocess the text data
def preprocess_text_data(reviews):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    word_index = tokenizer.word_index
    data_padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data_padded, word_index, tokenizer

# Function to encode labels
def encode_labels(labels):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(labels)

# Function to split the data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to save tokenizer to a file
def save_tokenizer(tokenizer, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(tokenizer, file_path)

# Function to load tokenizer from a file
def load_tokenizer(file_path):
    return joblib.load(file_path)

# Function to save model to a file
def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    model.save(file_path)

# Function to load model from a file
def load_model(file_path):
    from tensorflow.keras.models import load_model
    return load_model(file_path)