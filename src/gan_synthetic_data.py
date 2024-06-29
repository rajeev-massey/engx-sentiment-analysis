import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from config import config
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class GAN:
    def __init__(self, config):
        self.noise_dim = 100
        self.synthetic_data_rows = config.get('synthetic_data_rows', 1000)  # This should be read from the config, if available
        self.tokenizer = Tokenizer()
        self.model_path = config.get('rnn_model')['model_path']  # Model path from config
        self.epochs = config.get('rnn_model')['epochs']  # Epochs from config
        self.batch_size = config.get('rnn_model')['batch_size']  # Batch size from config
        self.input_dim = 1545  # Setting input dimension based on model requirements

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.noise_dim, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(self.input_dim, activation='tanh'))
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def compile_models(self, generator, discriminator):
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        discriminator.trainable = False

        gan_input = Input(shape=(self.noise_dim,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=Adam())
        return gan

    def preprocess_real_data(self, real_data):
        self.tokenizer.fit_on_texts(real_data[:, 0])
        sequences = self.tokenizer.texts_to_sequences(real_data[:, 0])
        padded_sequences = pad_sequences(sequences, maxlen=self.input_dim)
        return padded_sequences

    def train(self, real_data, epochs=1000, batch_size=32):
        half_batch = int(batch_size / 2)
        generator = self.build_generator()
        discriminator = self.build_discriminator()
        gan = self.compile_models(generator, discriminator)

        real_data = self.preprocess_real_data(real_data)

        for epoch in range(epochs):
            start_time = time.time()

            idx = np.random.randint(0, real_data.shape[0], half_batch)
            real_reviews = real_data[idx]

            noise = np.random.normal(0, 1, (half_batch, self.noise_dim))
            gen_reviews = generator.predict(noise)

            d_loss_real = discriminator.train_on_batch(real_reviews, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(gen_reviews, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            valid_y = np.array([1] * batch_size)
            g_loss = gan.train_on_batch(noise, valid_y)

            if epoch % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"{epoch} [D loss: {d_loss[0]:.4f} | D accuracy: {d_loss[1]:.4f}] [G loss: {g_loss:.4f}] Time: {elapsed_time:.2f}s")

        noise = np.random.normal(0, 1, (self.synthetic_data_rows, self.noise_dim))
        synthetic_data = generator.predict(noise)
        print(f"Synthetic data shape: {synthetic_data.shape}")  # Debugging stateme
        return synthetic_data

    def save_synthetic_data(self, synthetic_data):
        df = pd.DataFrame(synthetic_data)
        headers = [f"feature_{i}" for i in range(df.shape[1])]
        df.columns = headers
        df.to_csv('../data/synthetic_data.csv', index=False)

    def generate_synthetic_data(self, real_data_path, limit=None):
        real_data = pd.read_csv(real_data_path, on_bad_lines='skip').values
        if limit:
            real_data = real_data[:limit]
        synthetic_data = self.train(real_data)
        self.save_synthetic_data(synthetic_data)

    def retrain_with_synthetic(self, synthetic_data_path):
        synthetic_data = pd.read_csv(synthetic_data_path).values

        X_train = synthetic_data[:, :-1]  # Assuming the last column is the target
        Y_train = synthetic_data[:, -1]

        model = load_model(f"{self.model_path}.h5")

        model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

        model.save(f"{self.model_path}.h5")
        loss, accuracy = model.evaluate(X_train, Y_train, verbose=2)
        return accuracy

if __name__ == "__main__":
    config = {
        'synthetic_data_rows': 1000,
        'rnn_model': {
            'model_path': '../models/rnn/model',
            'epochs': 10,
            'batch_size': 32
        }
    }

    gan = GAN(config)
#    gan.generate_synthetic_data('../data/reviews_test.csv')
    accuracy = gan.retrain_with_synthetic('../data/synthetic_data.csv')
    print(f"Retrained model accuracy: {accuracy}")