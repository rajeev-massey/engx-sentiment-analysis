import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from config import config

class RNNModel:
    def __init__(self):
        self.model_path = config.get('rnn_model')['model_path']
        self.epochs = config.get('rnn_model')['epochs']
        self.batch_size = config.get('rnn_model')['batch_size']
        self.tokenizer = Tokenizer()

    def load_data(self, data_path):
        data = pd.read_csv(data_path)
        return data

    def preprocess(self, data):
        data['review'] = data['review'].apply(lambda x: x.lower())
        self.tokenizer.fit_on_texts(data['review'].values)
        X = self.tokenizer.texts_to_sequences(data['review'].values)
        X = pad_sequences(X)
        Y = pd.get_dummies(data['sentiment']).values
        return train_test_split(X, Y, test_size=1 - config.get('common')['train_test_ratio'])

    def build_model(self, input_length):
        model = Sequential()
        model.add(Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=128, input_length=input_length))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_and_evaluate(self, data_path, limit=None):
        data = self.load_data(data_path)
        if limit:
            data = data.head(limit)
        X_train, X_test, Y_train, Y_test = self.preprocess(data)
        model = self.build_model(X_train.shape[1])

        model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_test, Y_test), verbose=2)

        # Ensure the model is saved with the correct extension
        model.save(f"{self.model_path}.h5")

        loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
        return accuracy


    def predict(self, text):
        model = tf.keras.models.load_model(f"{self.model_path}.h5")
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=model.input_shape[1])
        pred = model.predict(padded)
        return np.argmax(pred, axis=1)[0]

    def batch_predict(self, texts):
        model = tf.keras.models.load_model(f"{self.model_path}.h5")
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=model.input_shape[1])
        preds = model.predict(padded)
        return {text: np.argmax(pred, axis=1)[0] for text, pred in zip(texts, preds)}
