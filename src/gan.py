import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model, Sequential

from common import LATENT_DIM


# Function to build generator model
def build_generator():
    model = Sequential()
    # Add dense layer
    model.add(Dense(512, input_dim=LATENT_DIM))
    # Add LeakyReLU activation
    model.add(LeakyReLU(alpha=0.2))
    # Add batch normalization
    model.add(BatchNormalization(momentum=0.8))
    # Add dropout for regularization
    model.add(Dropout(0.3))
    # Add another dense layer
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    # Output layer
    model.add(Dense(LATENT_DIM, activation='tanh'))
    model.add(Reshape((LATENT_DIM,)))
    return model

# Function to build discriminator model
def build_discriminator():
    model = Sequential()
    # Add dense layer
    model.add(Dense(512, input_dim=LATENT_DIM))
    model.add(LeakyReLU(alpha=0.2))
    # Add batch normalization
    model.add(BatchNormalization(momentum=0.8))
    # Add dropout for regularization
    model.add(Dropout(0.3))
    # Add another dense layer
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    # Output layer
    model.add(Dense(LATENT_DIM, activation='tanh'))
    model.add(Reshape((LATENT_DIM,)))
    return model

# Function to build GAN model by combining generator and discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(LATENT_DIM,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

# Function to generate synthetic data using the generator model
def generate_synthetic_data(generator, data_size):
    noise = np.random.normal(0, 1, (data_size, LATENT_DIM))
    synthetic_data = generator.predict(noise)
    return synthetic_data