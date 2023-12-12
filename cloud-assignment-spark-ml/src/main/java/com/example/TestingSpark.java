package com.example;


import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;

public class TestingSpark {


    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("Cloud Assignment Spark ML")
                .getOrCreate();

        String modelPath = args.length > 0 ? args[0] : "s3://cloud-assignment-spark-ml/decision_tree_trained/";
        String inputFile = args.length > 1 ? args[1] : "s3://cloud-assignment-spark-ml/ValidationDataset.csv";

        String[] columnNames = {"fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol", "quality"};

        Dataset<Row> test = spark.read().format("csv")
                .option("delimiter", ";")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(inputFile);

        for (int i = 0; i < columnNames.length; i++) {
            test = test.withColumnRenamed(test.columns()[i], columnNames[i]);
        }

        VectorAssembler featureAssembler = new VectorAssembler()
                .setInputCols(java.util.Arrays.copyOfRange(columnNames, 0, columnNames.length - 1))
                .setOutputCol("features");

        test = featureAssembler.transform(test);

        StandardScaler featureScalar = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(true);

        StandardScalerModel featureScalerModel = featureScalar.fit(test);
        test = featureScalerModel.transform(test);

        DecisionTreeClassificationModel savedModel = DecisionTreeClassificationModel.load(modelPath);

        Dataset<Row> predictions = savedModel.transform(test);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1 Score= " + f1);

        spark.stop();
    }
}
