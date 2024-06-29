# Sentiment Analysis Project

This project showcases two main approaches to sentiment analysis on textual data: using PySpark and deep learning models (LSTM and LSTM with GAN). The goal is to classify text data into positive or negative sentiment categories based on the content of the text.

## Part 1: Sentiment Analysis with PySpark

### Overview
This part demonstrates how to perform sentiment analysis using PySpark, a powerful tool for big data processing. It involves data processing, feature extraction, and machine learning steps implemented in Python using the PySpark framework.

### Project Structure
- `src/spark_ml.py`: Main Python script for sentiment analysis using PySpark.
- `data/reviews.csv`: Sample dataset for training and testing the sentiment analysis model.
- `models/`: Directory for saving trained models (created by the script).

### Requirements
- Python 3.6 or higher
- Apache Spark 2.4.5 or higher
- Java 8 or 11
- Hadoop 2.7 or higher (optional, for HDFS support)

### Setup and Running
1. Install Python dependencies: `pip install pyspark`
2. Configure Spark: Set `SPARK_HOME` and add Spark's `bin` directory to `PATH`.
3. Prepare the data in the `data/` directory.
4. Run the project: `python src/spark_ml.py`

### How It Works
- Data is loaded, preprocessed, and split into training and test sets.
- Logistic Regression and Naive Bayes models are trained and evaluated.

## Part 2: Sentiment Analysis with LSTM and LSTM with GAN

### Overview
This part explores deep learning approaches for sentiment analysis using LSTM (Long Short-Term Memory) networks and an advanced version with Generative Adversarial Networks (GAN).

### Project Structure
- `src/lstm.py`: Script for sentiment analysis using LSTM.
- `src/lstm_gan.py`: Script for sentiment analysis using LSTM with GAN.
- `data/reviews.csv`: Dataset for model training and testing.
- `models/`: Directory for trained models.

### Requirements
- Python 3.6 or higher
- TensorFlow 2.x
- Keras

### Setup and Running
1. Install Python dependencies: `pip install tensorflow keras`
2. Prepare the dataset in the `data/` directory.
3. Run LSTM model: `python src/lstm.py`
4. Run LSTM with GAN model: `python src/lstm_gan.py`

### How It Works
- Data is preprocessed and converted into a suitable format for training.
- LSTM and LSTM with GAN models are trained on the dataset and evaluated.

## Contributing
Contributions are welcome. Please fork the repository, make your changes, and submit a pull request.