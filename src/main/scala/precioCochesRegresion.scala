import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
object precioCochesRegresion {

  def main(args: Array[String]): Unit = {
    //Reducir el número de LOG
    Logger.getLogger("org").setLevel(Level.OFF)
    //Creando el contexto del Servidor
    val sc = new SparkContext("local","precioCochesRegresion", System.getenv("SPARK_HOME"))
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("precioCochesRegresion")
      .config("log4j.rootCategory", "ERROR, console")
      .getOrCreate()


    val data = spark.read.format("csv").option("header","true").option("inferschema","true").option("delimiter",",").load("resources/true_car_listings.csv").toDF()

    val labelIndexer = new StringIndexer().setInputCol("City").setOutputCol("cityIndexed").fit(data).transform(data)
    val labelIndexer2 = new StringIndexer().setInputCol("State").setOutputCol("stateIndexed").fit(labelIndexer).transform(labelIndexer)
    val labelIndexer3 = new StringIndexer().setInputCol("Vin").setOutputCol("vinIndexed").fit(labelIndexer2).transform(labelIndexer2)
    val labelIndexer4 = new StringIndexer().setInputCol("Make").setOutputCol("makeIndexed").fit(labelIndexer3).transform(labelIndexer3)
    val labelIndexer5 = new StringIndexer().setInputCol("Model").setOutputCol("modelIndexed").fit(labelIndexer4).transform(labelIndexer4)

    val data2 = labelIndexer5.drop("City","State","Vin","Make","Model")

    val featureCols = Array("Year","Mileage","cityIndexed","stateIndexed","vinIndexed","makeIndexed","modelIndexed")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(data2)

    df2.show(5)

    val lr = new LinearRegression()
      .setMaxIter(25)
      .setRegParam(0.1)
      .setElasticNetParam(0.7)
      .setLabelCol("Price")
      .setFeaturesCol("features")

    // Fit the model
    val lrModel = lr.fit(df2)
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

  }

}
