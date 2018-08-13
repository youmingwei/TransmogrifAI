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

import com.salesforce.hw.iris.Iris
import com.salesforce.hw.iris.OpIris.randomSeed
import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataReaders
import com.salesforce.op.stages.impl.classification.{BinaryClassificationModelSelector, MultiClassificationModelSelector}
import com.salesforce.op.stages.impl.classification.ClassificationModelsToTry._
import com.salesforce.op.stages.impl.tuning.DataCutter
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

case class Iris
(
  sepalLength: Option[Double],
  sepalWidth: Option[Double],
  petalLength: Option[Double],
  petalWidth: Option[Double],
  irisClass: String
)


/**
  * A simplified Optimus Prime example classification app using the Titanic dataset
  */
object OpIrisSimple {

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

    // Set up a SparkSession as normal
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName.stripSuffix("$"))
    implicit val spark = SparkSession.builder.config(conf).getOrCreate()

    ////////////////////////////////////////////////////////////////////////////////
    // RAW FEATURE DEFINITIONS
    /////////////////////////////////////////////////////////////////////////////////

    // Define features using the OP types based on the data
//    val id = FeatureBuilder.Integral[Iris].extract(_.getID.toIntegral).asPredictor
    val sepalLength = FeatureBuilder.Real[Iris].extract(_.sepalLength.toReal).asPredictor
    val sepalWidth = FeatureBuilder.Real[Iris].extract(_.sepalWidth.toReal).asPredictor
    val petalLength = FeatureBuilder.Real[Iris].extract(_.petalLength.toReal).asPredictor
    val petalWidth = FeatureBuilder.Real[Iris].extract(_.petalWidth.toReal).asPredictor
    val irisClass = FeatureBuilder.PickList[Iris].extract(_.irisClass.toPickList).asResponse.indexed()

    ////////////////////////////////////////////////////////////////////////////////
    // TRANSFORMED FEATURES
    /////////////////////////////////////////////////////////////////////////////////

    val irisFeatures = Seq(sepalLength, sepalWidth, petalLength, petalWidth).transmogrify()

    // Optionally check the features with a sanity checker
    val sanityCheck = true
    val finalFeatures = if (sanityCheck) irisClass.sanityCheck(irisFeatures) else irisFeatures

    // Define the model we want to use (here a simple logistic regression) and get the resulting output
    val (prediction, rawPrediction, prob) =
      MultiClassificationModelSelector.withCrossValidation(splitter = Some(DataCutter(reserveTestFraction = 0.3)))
//    MultiClassificationModelSelector.withTrainValidationSplit(splitter = Some(DataCutter(reserveTestFraction = 0.3)))
        .setInput(irisClass, finalFeatures).getOutput()

    val evaluator = Evaluators.MultiClassification()
      .setLabelCol(irisClass)
      .setRawPredictionCol(rawPrediction)
      .setPredictionCol(prediction)
      .setProbabilityCol(prob)

    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW
    /////////////////////////////////////////////////////////////////////////////////

    import spark.implicits._ // Needed for Encoders for the Passenger case class
    // Define a way to read data into our Passenger class from our CSV file
    val trainDataReader = DataReaders.Simple.csvCase[Iris](
      path = Option(csvFilePath)
      //      key = _.id.toString
    )

    // Define a new workflow and attach our data reader
    val workflow =
      new OpWorkflow()
        .setResultFeatures(irisClass, rawPrediction, prob, prediction)
        .setReader(trainDataReader)

    // Fit the workflow to the data
    val fittedWorkflow = workflow.train()
    println(s"Summary: ${fittedWorkflow.summary()}")

  }
}
