from src.generic_model import GenericModel
from src.spark_ml import SparkMLSentiment
from src.config import Config

def main():
    config = Config('../resources/properties.yaml')

    # load configurations
    limit_records = config.get('common')['limit_records']
    data_path = config.get('common')['data_path']

    # SparkML Sentiment Analysis
    print(f'SparkML Sentiment Analysis with {limit_records} records: {data_path}')
    spark_ml = SparkMLSentiment(config)
    spark_ml_accuracy = spark_ml.train_and_evaluate(data_path, limit=limit_records)
    print(f'SparkML Accuracy: {spark_ml_accuracy}')


    # Single Text Prediction
    text = "This is a complete statement for sentiment analysis."
    print(f'SparkML Prediction: {spark_ml.predict(text)}')

    # Batch Text Prediction
    texts = [
        "This is an example text for sentiment analysis.",
        "Another example text.",
        "I really enjoyed the movie.",
        "The product quality is poor.",
        "I love this restaurant!",
        "The customer service was excellent.",
        "I'm not satisfied with the service.",
        "The book was captivating and I couldn't put it down.",
        "The plot of the movie was predictable and boring.",
        "The food was delicious and the ambiance was great.",
        "The hotel room was clean and comfortable.",
        "The flight was delayed and the airline didn't provide any updates.",
        "The concert was amazing!",
        "The game was thrilling till the end.",
        "The new software update is full of bugs.",
        "The car's performance is impressive.",
        "The phone's battery life is disappointing.",
        "The staff at the hospital were very caring and attentive.",
        "The course content is outdated.",
        "The park is clean and well-maintained.",
        "The app crashes frequently.",
        "The event was well-organized.",
        "The delivery was late and the package was damaged.",
        "The museum was informative and interesting.",
        "The beach was crowded and dirty."
    ]
    print(f'SparkML Batch Prediction: {spark_ml.batch_predict(texts)}')
    #print(f'RNN Batch Prediction: {rnn_model.batch_predict(texts)}')

if __name__ == "__main__":
    main()
