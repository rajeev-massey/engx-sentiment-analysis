import numpy as np
from common import load_dataset, preprocess_text_data, encode_labels, split_data, save_model, load_model, \
    save_tokenizer, load_tokenizer
from gan import build_generator, build_discriminator, build_gan, generate_synthetic_data
from lstm import build_lstm_model, train_lstm_model, evaluate_lstm_model

# Paths to save and load models and tokenizers
TOKENIZER_PATH = '../models/tokenizer.joblib'
LSTM_MODEL_PATH = '../models/lstm_model.h5'
GENERATOR_MODEL_PATH = '../models/generator.h5'
DISCRIMINATOR_MODEL_PATH = '../models/discriminator.h5'
GAN_MODEL_PATH = '../models/gan.h5'


def main():
    # Load and preprocess data
    data = load_dataset('../data/reviews.csv')
    data_padded, word_index, tokenizer = preprocess_text_data(data['review'])
    labels = encode_labels(data['sentiment'])
    X_train, X_test, y_train, y_test = split_data(data_padded, labels)

    # Save tokenizer
    save_tokenizer(tokenizer, TOKENIZER_PATH)

    # Build and train LSTM(RNN) model
    vocab_size = len(word_index) + 1
    input_length = 100

    lstm_model = build_lstm_model(vocab_size, input_length)
    train_lstm_model(lstm_model, X_train, y_train, X_test, y_test)

    # Save LSTM model
    save_model(lstm_model, LSTM_MODEL_PATH)

    # Evaluate LSTM model
    original_accuracy = evaluate_lstm_model(lstm_model, X_test, y_test)
    print(f'Original LSTM Model Accuracy: {original_accuracy}')

    # Build GAN model
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)

    # Save GAN models
    save_model(generator, GENERATOR_MODEL_PATH)
    save_model(discriminator, DISCRIMINATOR_MODEL_PATH)
    save_model(gan, GAN_MODEL_PATH)

    # Generate synthetic data
    synthetic_reviews = generate_synthetic_data(generator, len(X_train))

    # Combine real and synthetic data
    X_train_augmented = np.vstack((X_train, synthetic_reviews))
    y_train_augmented = np.hstack((y_train, np.zeros(len(synthetic_reviews))))

    # Retrain LSTM model with augmented data
    train_lstm_model(lstm_model, X_train_augmented, y_train_augmented, X_test, y_test)

    # Save the retrained LSTM model
    save_model(lstm_model, LSTM_MODEL_PATH)

    # Evaluate the model again
    augmented_accuracy = evaluate_lstm_model(lstm_model, X_test, y_test)
    print(f'LSTM Model Accuracy after GAN augmentation: {augmented_accuracy}')
    print(f'Improvement: {augmented_accuracy - original_accuracy}')


if __name__ == "__main__":
    main()