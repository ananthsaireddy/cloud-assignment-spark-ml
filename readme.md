# Cloud Assignment Spark ML

create the EMR cluster and give sufficient permissions to ssh into master node

run the training code using spark-submit
```
spark-submit --class com.example.TrainingSpark --master yarn s3://cloud-assignment-spark-ml/cloud-assignment-spark-ml-1.0.jar
```

run the below command to test using spark-submit
```
spark-submit --class com.example.TestingSpark --master yarn s3://cloud-assignment-spark-ml/cloud-assignment-spark-ml-1.0.jar
```

build docker image and push to docker hub using below commands
```
docker login
docker build -t <username>/cloud-assignment-spark-ml .
docker push  <username>/cloud-assignment-spark-ml
```

run the docker image for prediction on the ValidationDataset.csv
```
docker run <username>/cloud-assignment-spark-ml
```

