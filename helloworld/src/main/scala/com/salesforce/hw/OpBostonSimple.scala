/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.hw

import com.salesforce.hw.boston.OpBoston.{customRead, randomSeed}
import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.{CustomReader, DataReaders}
import com.salesforce.op.stages.impl.classification.{BinaryClassificationModelSelector, MultiClassificationModelSelector}
import com.salesforce.op.stages.impl.classification.ClassificationModelsToTry._
import com.salesforce.op.stages.impl.regression.RegressionModelSelector
import com.salesforce.op.stages.impl.tuning.{DataCutter, DataSplitter}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

case class BostonHouse
(
  rowId: Int,
  crim: Double,
  zn: Double,
  indus: Double,
  chas: String,
  nox: Double,
  rm: Double,
  age: Double,
  dis: Double,
  rad: Int,
  tax: Double,
  ptratio: Double,
  b: Double,
  lstat: Double,
  medv: Option[Double]
)


/**
  * A simplified Optimus Prime example classification app using the Titanic dataset
  */
object OpBostonSimple {

  /**
    * Run this from the command line with
    * ./gradlew sparkSubmit -Dmain=com.salesforce.hw.OpTitanicSimple -Dargs=/full/path/to/csv/file
    */
  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      println("You need to pass in the CSV file path as an argument")
      sys.exit(1)
    }
    val csvFilePath = args(0)
    println(s"Using user-supplied CSV file path: $csvFilePath")

    // hack for having train and test csv path
    val testCsvFilePath = args(1)

    // Set up a SparkSession as normal
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName.stripSuffix("$"))
    implicit val spark = SparkSession.builder.config(conf).getOrCreate()

    ////////////////////////////////////////////////////////////////////////////////
    // RAW FEATURE DEFINITIONS
    /////////////////////////////////////////////////////////////////////////////////

    // Define features using the OP types based on the data
    val rowId = FeatureBuilder.Integral[BostonHouse].extract(_.rowId.toIntegral).asPredictor

    val crim = FeatureBuilder.RealNN[BostonHouse].extract(_.crim.toRealNN).asPredictor

    val zn = FeatureBuilder.RealNN[BostonHouse].extract(_.zn.toRealNN).asPredictor

    val indus = FeatureBuilder.RealNN[BostonHouse].extract(_.indus.toRealNN).asPredictor

    val chas = FeatureBuilder.PickList[BostonHouse].extract(x => Option(x.chas).toPickList).asPredictor

    val nox = FeatureBuilder.RealNN[BostonHouse].extract(_.nox.toRealNN).asPredictor

    val rm = FeatureBuilder.RealNN[BostonHouse].extract(_.rm.toRealNN).asPredictor

    val age = FeatureBuilder.RealNN[BostonHouse].extract(_.age.toRealNN).asPredictor

    val dis = FeatureBuilder.RealNN[BostonHouse].extract(_.dis.toRealNN).asPredictor

    val rad = FeatureBuilder.Integral[BostonHouse].extract(_.rad.toIntegral).asPredictor

    val tax = FeatureBuilder.RealNN[BostonHouse].extract(_.tax.toRealNN).asPredictor

    val ptratio = FeatureBuilder.RealNN[BostonHouse].extract(_.ptratio.toRealNN).asPredictor

    val b = FeatureBuilder.RealNN[BostonHouse].extract(_.b.toRealNN).asPredictor

    val lstat = FeatureBuilder.RealNN[BostonHouse].extract(_.lstat.toRealNN).asPredictor

    val medv = FeatureBuilder.RealNN[BostonHouse].extract(_.medv.getOrElse(0.0).toRealNN).asResponse


    ////////////////////////////////////////////////////////////////////////////////
    // TRANSFORMED FEATURES
    /////////////////////////////////////////////////////////////////////////////////

    val bostonFeatures = Seq(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat).transmogrify()

    // Optionally check the features with a sanity checker
    val sanityCheck = true
    val finalFeatures = if (sanityCheck) medv.sanityCheck(bostonFeatures) else bostonFeatures

    // Define the model we want to use (here a simple logistic regression) and get the resulting output
    val prediction =
      RegressionModelSelector.withCrossValidation(dataSplitter = None)
//      RegressionModelSelector.withTrainValidationSplit(dataSplitter = Option(DataSplitter(reserveTestFraction = 0.3)))
        .setInput(medv, finalFeatures).getOutput()


    val evaluator = Evaluators.Regression()
      .setLabelCol(medv)
      .setPredictionCol(prediction)

    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW
    /////////////////////////////////////////////////////////////////////////////////

    def customRead(path: Option[String], spark: SparkSession): RDD[BostonHouse] = {
      require(path.isDefined, "The path is not set")
      val myFile = spark.sparkContext.textFile(path.get)

      myFile.filter(_.nonEmpty).zipWithIndex.map { case (x, number) =>
        val words = x.replaceAll("\\s+", " ").replaceAll(s"^\\s+(?m)", "").replaceAll(s"(?m)\\s+$$", "").split(" ")
        BostonHouse(number.toInt, words(0).toDouble, words(1).toDouble, words(2).toDouble, words(3), words(4).toDouble,
          words(5).toDouble, words(6).toDouble, words(7).toDouble, words(8).toInt, words(9).toDouble,
          words(10).toDouble, words(11).toDouble, words(12).toDouble, Option(words(13).toDouble))
      }
    }

//    import spark.implicits._ // Needed for Encoders for the Passenger case class
    // Define a way to read data into our Passenger class from our CSV file
    val trainDataReader = new CustomReader[BostonHouse](key = _.rowId.toString) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[BostonHouse], Dataset[BostonHouse]] = {
        val train = customRead(Some(csvFilePath), spark)
        Left(train)
      }
    }

    val workflow =
      new OpWorkflow()
        .setResultFeatures(medv, prediction)
        .setReader(trainDataReader)

    // Fit the workflow to the data
    val fittedWorkflow = workflow.train()
    println(s"Summary: ${fittedWorkflow.summary()}")


    import spark.implicits._
    // test set will do a reg reader
    val testDataReader = DataReaders.Simple.csvCase[BostonHouse](
      path = Option(testCsvFilePath),
      key = _.rowId.toString
    )

    println("Scoring test dataset")
    fittedWorkflow.setReader(testDataReader)
    var dataframeTest = fittedWorkflow.score()


    println("Transformed test dataframe columns:")
    dataframeTest.columns.foreach(println)

    dataframeTest = dataframeTest.drop(dataframeTest.col("medv"))

    println("Transformed test dataframe columns (after drops):")
    dataframeTest.columns.foreach(println)
    dataframeTest.write.format("com.databricks.spark.csv").save("bostonTestResults-9.csv")


  }
}
