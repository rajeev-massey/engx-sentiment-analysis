from datetime import datetime

from pyspark.sql import SparkSession
#ML Libraries from py
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline


class SparkMLSentiment:
    def __init__(self, config):
        """
        Initializes the SparkMLSentiment class with configuration settings.

        :param config: A dictionary containing configuration settings for the Spark session and other parameters.
        """
        self.spark = self._init_spark_session(config)
        self.config = config
        self.text_column = "review"
        self.label_column = "sentiment"
        self.indexed_label_column = "indexedLabel"

    def _init_spark_session(self, config):
        """
       Initializes and returns a Spark session based on the provided configuration.

       :param config: Configuration dictionary for setting up the Spark session.
       :return: A SparkSession object.
       """
        spark = SparkSession.builder \
            .appName(config.get('spark')['config']['appName']) \
            .config("spark.executor.memory", config.get('spark')['config']['executor_memory']) \
            .config("spark.driver.memory", config.get('spark')['config']['driver_memory']) \
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
            .config("spark.hadoop.fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem") \
            .config("spark.memory.fraction", "0.8") \
            .getOrCreate()
        return spark

    def load_data(self, data_path):
        """
        Loads data from a specified path into a Spark DataFrame, applying basic preprocessing like dropping NA values.

        :param data_path: The path to the data file.
        :return: A Spark DataFrame containing the loaded data.
        """
        dataset = self.spark.read.csv(data_path, header=True, inferSchema=True)
        dataset = dataset.dropna().limit(self.config.get('common')['limit'])
        return dataset

    def preprocess(self, dataset):
        """
        Preprocesses the dataset by tokenizing text, removing stopwords, and extracting TF-IDF features.

        :param dataset: The Spark DataFrame to preprocess.
        :return: A DataFrame with the preprocessing pipeline applied.
        """
        indexer = StringIndexer(inputCol=self.label_column, outputCol=self.indexed_label_column)
        indexed = indexer.fit(dataset).transform(dataset)

        tokenizer = Tokenizer(inputCol=self.text_column, outputCol="tokens")
        stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
        hashingTF = HashingTF(inputCol="filtered", outputCol="tf_features", numFeatures=2000)
        idf = IDF(inputCol="tf_features", outputCol="idf_features")

        pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashingTF, idf])
        model_pipeline = pipeline.fit(indexed)
        return model_pipeline.transform(indexed)

    def train_models(self, dataset):
        """
       Trains Logistic Regression and Naive Bayes models on the dataset.

       :param dataset: The preprocessed dataset to train the models on.
       :return: The accuracy of the Logistic Regression and Naive Bayes models.
       """
        train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=12345)

        lr = LogisticRegression(labelCol=self.indexed_label_column, featuresCol="idf_features")
        nb = NaiveBayes(labelCol=self.indexed_label_column, featuresCol="idf_features")

        self.lr_model = lr.fit(train_data)
        self.nb_model = nb.fit(train_data)

        lr_predictions = self.lr_model.transform(test_data)
        nb_predictions = self.nb_model.transform(test_data)

        evaluator = MulticlassClassificationEvaluator(labelCol=self.indexed_label_column, predictionCol="prediction")

        self.lr_accuracy = evaluator.evaluate(lr_predictions, {evaluator.metricName: "accuracy"}) * 100
        self.nb_accuracy = evaluator.evaluate(nb_predictions, {evaluator.metricName: "accuracy"}) * 100

        return self.lr_accuracy, self.nb_accuracy

    def predict_lr(self, text):
        """
        Predicts the sentiment of the given text using the trained Logistic Regression model.

        :param text:
        :return:
        """
        dataset = self.spark.createDataFrame([(text,)], ["review"])
        processed_dataset = self.preprocess(dataset)
        prediction = self.lr_model.transform(processed_dataset)
        predicted_label = prediction.select("prediction").collect()[0]["prediction"]
        sentiment = "Positive" if predicted_label == 1.0 else "Negative"
        return sentiment

    def predict_nb(self, text):
        """
        Predicts the sentiment of the given text using the trained Naive Bayes model.

        :param text:
        :return:
        """
        dataset = self.spark.createDataFrame([(text,)], ["review"])
        processed_dataset = self.preprocess(dataset)
        prediction = self.nb_model.transform(processed_dataset)
        predicted_label = prediction.select("prediction").collect()[0]["prediction"]
        sentiment = "Positive" if predicted_label == 1.0 else "Negative"


if __name__ == "__main__":
    config = {
        'common': {
            'limit': 5000
        },
        'spark': {
            'config': {
                'appName': 'SentimentAnalysis',
                'executor_memory': '4g',
                'driver_memory': '4g'
            },
            'ml0lib': {
                'model_path': '../models'
            }
        }
    }

    spark_sentiment = SparkMLSentiment(config)
    dataset = spark_sentiment.load_data("../data/reviews.csv")

    start_time = datetime.now()

    processed_dataset = spark_sentiment.preprocess(dataset)
    lr_accuracy, nb_accuracy = spark_sentiment.train_models(processed_dataset)

    end_time = datetime.now()


    print(f"Logistic Regression Accuracy: {lr_accuracy}%")
    print(f"Naive Bayes Accuracy: {nb_accuracy}%")

    print(f"Process started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Process ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {end_time - start_time}")


# Stop the Spark session
    spark_sentiment.spark.stop()
