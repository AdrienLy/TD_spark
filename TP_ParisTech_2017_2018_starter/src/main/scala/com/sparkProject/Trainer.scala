package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.{SparkConf, SparkContext, sql}

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession.builder.config(conf).appName("TP_spark").getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")
    // Lecture du fichier parquet
    val df = spark.read.parquet("/Users/lyadrien/TP/Spark/TP_ParisTech_2017_2018_starter/prepared_trainingset")

    // Creation d'un token
    val tokenizer = new RegexTokenizer().setPattern("\\W+").setGaps(true).setInputCol("text").setOutputCol("tokens")

    // filtre sur les stop-word
    val remover = new StopWordsRemover().setInputCol("tokens").setOutputCol("tokens2")

    // comptage de la fréquence d'un terme
    val countV = new CountVectorizer().setInputCol("tokens2").setOutputCol("count")

    //Dernière étape du TF-IDF avec le nombre de document dans lequel le terme apparait
    val tfidf = new IDF().setInputCol("count").setOutputCol("tfidf")

    //Convertir la variable catégorielle “country2” en quantités numériques
    val country_indexed = new StringIndexer().setInputCol("country2").setOutputCol("country_indexed").setHandleInvalid("skip")

    //Convertir la variable catégorielle “currency2” en quantités numériques
    val currency_indexed = new StringIndexer().setInputCol("currency2").setOutputCol("currency_indexed").setHandleInvalid("skip")

    //One Hot Encoder des colonnes currency et country _indexed
    val currency_onehot = new OneHotEncoder().setInputCol("currency_indexed").setOutputCol("currency_onehot")
    val country_onehot = new OneHotEncoder().setInputCol("country_indexed").setOutputCol("country_onehot")

    //Assemblage des features
    val assembler = new VectorAssembler().setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot")).setOutputCol("features")

    //régression linéaire
    val lr = new LogisticRegression().setElasticNetParam(0.0).setFitIntercept(true).setFeaturesCol("features").setLabelCol("final_status").setStandardization(true).setPredictionCol("predictions").setRawPredictionCol("raw_predictions").setThresholds(Array(0.7, 0.3)).setTol(1.0e-6).setMaxIter(300)

    //Création du pipeline
    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, countV, tfidf, country_indexed, currency_indexed, country_onehot,currency_onehot, assembler, lr))
//    val model = pipeline.fit(df)
//    val new_df = model.transform(df)

    // k
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 12345)

    //l
    // Choix de la metric pour selectionné les hyper parametre
    val f1 = new MulticlassClassificationEvaluator().setMetricName("f1").setLabelCol("final_status").setPredictionCol("predictions")

    // Choix des hyper-parametre à tester
    val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4,10e-2)).addGrid(countV.minDF,Array(55.0,75.0,95.0)).build()

    // Mise en place du modèle
    val trainValidationSplit = new TrainValidationSplit().setEstimator(pipeline).setEvaluator(f1).setEstimatorParamMaps(paramGrid).setTrainRatio(0.7)

    val modelLR = trainValidationSplit.fit(training)

    //m
    val df_WithPredictions =modelLR.transform(test)

    // Evaluation du modèle sur le jeu de test
    val F1 = f1.evaluate(df_WithPredictions)
    println(s"F1 score = $F1")
    //n
    df_WithPredictions.groupBy("final_status", "predictions").count.show()







  }
}
