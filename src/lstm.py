from keras.layers import Bidirectional
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from common import EMBEDDING_DIM, LSTM_UNITS, DROPOUT_RATE, BATCH_SIZE, EPOCHS, save_model, load_model

# Function to build LSTM model
def build_lstm_model(vocab_size, input_length):
    model = Sequential()
    # Add embedding layer
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=input_length))
    # Add bidirectional LSTM layer
    model.add(Bidirectional(LSTM(LSTM_UNITS * 2, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE)))
    # Add output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to train LSTM model
def train_lstm_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    return history

# Function to evaluate LSTM model
def evaluate_lstm_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy