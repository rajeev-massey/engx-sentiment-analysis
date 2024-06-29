from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Start Spark session with optimized settings
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

# Load the IMDB dataset
dataset = spark.read.csv("reviews.csv", header=True, inferSchema=True)

# Select necessary columns and cache the dataset
dataset = dataset.select(col("review").alias("text"), col("sentiment").alias("label")).cache()

# Define the text and label column names
text_column_name = "text"
label_column_name = "label"

# Create a tokenizer
tokenizer = Tokenizer(inputCol=text_column_name, outputCol="tokens")

# Create a stop words remover
stopwords = StopWordsRemover(inputCol="tokens", outputCol="filtered")

# Create a HashingTF with reduced feature dimension
hashingTF = HashingTF(inputCol="filtered", outputCol="tf_features", numFeatures=1000)

# Create an IDF in the pipeline
idf = IDF(inputCol="tf_features", outputCol="idf_features")

# Create a logistic regression model
lr = LogisticRegression(labelCol=label_column_name, featuresCol="idf_features")

# Create a pipeline
pipeline = Pipeline(stages=[tokenizer, stopwords, hashingTF, idf, lr])

# Fit the pipeline to the dataset
model = pipeline.fit(dataset)

# Evaluate the model on the dataset
predictions = model.transform(dataset)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol=label_column_name, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Print the accuracy
print(f"Accuracy: {accuracy}")

# Additional evaluation metrics
evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_column_name, predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)

evaluator_precision = MulticlassClassificationEvaluator(labelCol=label_column_name, predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator_precision.evaluate(predictions)

evaluator_recall = MulticlassClassificationEvaluator(labelCol=label_column_name, predictionCol="prediction", metricName="weightedRecall")
recall = evaluator_recall.evaluate(predictions)

# Print additional evaluation metrics
print(f"F1 Score: {f1_score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Stop the Spark session
spark.stop()
