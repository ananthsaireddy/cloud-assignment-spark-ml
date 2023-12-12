package com.example;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import java.io.IOException;

public class TrainingSpark {
    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession
                .builder()
                .appName("Cloud Assignment Spark ML")
                .getOrCreate();

        String[] columnNames = {"fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol", "quality"};

        Dataset<Row> train = spark.read().format("csv")
                .option("delimiter", ";")
                .option("inferSchema", "true")
                .option("header", "true")
                .load("s3://cloud-assignment-spark-ml/TrainingDataset.csv");

        for (int i = 0; i < columnNames.length; i++) {
            train = train.withColumnRenamed(train.columns()[i], columnNames[i]);
        }

        VectorAssembler featureAssembler = new VectorAssembler()
                .setInputCols(java.util.Arrays.copyOfRange(columnNames, 0, columnNames.length - 1))
                .setOutputCol("features");

        train = featureAssembler.transform(train);

        StandardScaler featureScaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(true);

        StandardScalerModel featureScalerModel = featureScaler.fit(train);
        train = featureScalerModel.transform(train);

        Dataset<Row> validation = spark.read().format("csv")
                .option("delimiter", ";")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("s3://cloud-assignment-spark-ml/ValidationDataset.csv");

        for (int i = 0; i < columnNames.length; i++) {
            validation = validation.withColumnRenamed(validation.columns()[i], columnNames[i]);
        }

        validation = featureAssembler.transform(validation);
        StandardScalerModel validationFeatureScalerModel = featureScaler.fit(validation);
        Dataset<Row> val = validationFeatureScalerModel.transform(validation);

        LogisticRegression lr = new LogisticRegression()
                .setFeaturesCol("scaledFeatures")
                .setLabelCol("quality");
        LogisticRegressionModel lrModel = lr.fit(train);
        Dataset<Row> lrPredictions = lrModel.transform(val);

        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setFeaturesCol("scaledFeatures")
                .setLabelCol("quality");
        DecisionTreeClassificationModel dtModel = dt.fit(train);
        dtModel.save("s3://cloud-assignment-spark-ml/decision_tree_trained/");
        Dataset<Row> dtPredictions = dtModel.transform(val);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double lrF1 = evaluator.evaluate(lrPredictions);
        double dtF1 = evaluator.evaluate(dtPredictions);

        System.out.println("Logistic Regression F1 Score: " + lrF1);
        System.out.println("Decision Tree F1 Score: " + dtF1);

        spark.stop();
    }
}
