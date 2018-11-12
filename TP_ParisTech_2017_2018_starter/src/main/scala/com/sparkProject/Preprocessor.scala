package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP
    // on vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation de la SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc et donc aux mécanismes de distribution des calculs.)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    val df = spark.read.option("header" ,  "true").csv("train_clean.csv").cache()

    val featured_df = df.withColumn("deadline" ,(df("deadline").cast("Int")))
        .withColumn("state_changed_at", (df("state_changed_at").cast("Int")))
        .withColumn("created_at", (df("created_at").cast("Int")))
        .withColumn("launched_at", (df("launched_at").cast("Int")))
        .withColumn("final_status", (df("final_status").cast("Int")))
        .withColumn("backers_count", (df("backers_count").cast("Int")))
    featured_df.show(10)
    featured_df.printSchema()
    featured_df.describe().show()
    //featured_df.filter($"country"===”False”).groupBy("currency").count.orderBy($"count".desc).show(50)



    println("hello world ! from Preprocessor")


  }

}
