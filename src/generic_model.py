import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class GenericModel:
    def __init__(self, config):
        self.data_path = config.get('common')['data_path']
        self.train_test_ratio = config.get('common')['train_test_ratio']
        self.target_column = config.get('common')['target_column']
        self.text_column = config.get('common')['text_column']
        self.max_words = config.get('common')['max_words']
        self.max_len = config.get('common')['max_len']
        self.model_type = config.get('model')['type']
        self.model_path = config.get('model')['model_path']
        self.epochs = config.get('model')['epochs']
        self.batch_size = config.get('model')['batch_size']
        self.tokenizer = Tokenizer(num_words=self.max_words)

    def load_data(self):
        data = pd.read_csv(self.data_path)
        return data

    def preprocess_data(self, data):
        self.tokenizer.fit_on_texts(data[self.text_column])
        X = pad_sequences(self.tokenizer.texts_to_sequences(data[self.text_column]), maxlen=self.max_len)
        Y = data[self.target_column]
        return train_test_split(X, Y, test_size=1 - self.train_test_ratio, random_state=42)

    def build_model(self, input_length):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words, output_dim=128, input_length=input_length))
        if self.model_type == "LSTM":
            model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        else:
            model.add(SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_and_evaluate(self):
        data = self.load_data()
        X_train, X_test, Y_train, Y_test = self.preprocess_data(data)
        model = self.build_model(X_train.shape[1])
        model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)
        model.save(f"{self.model_path}.h5")
        loss, accuracy = model.evaluate(X_test, Y_test)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')
        return model

    def predict(self, text):
        model = tf.keras.models.load_model(f"{self.model_path}.h5")
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
        prediction = model.predict(padded_sequence)
        sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
        return sentiment