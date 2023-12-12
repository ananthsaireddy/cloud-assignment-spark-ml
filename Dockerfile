FROM jupyter/all-spark-notebook

WORKDIR /app

COPY . /app

CMD ["spark-submit", "--class", "com.example.testingSpark", "--master", "local[*]", "cloud-assignment-spark-ml-1.0.jar", "decision_tree_trained/",  "ValidationDataset.csv"]
