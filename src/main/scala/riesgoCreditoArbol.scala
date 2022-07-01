import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession

object riesgoCreditoArbol {
  def main(args: Array[String]): Unit = {
    //Reducir el número de LOG
    Logger.getLogger("org").setLevel(Level.OFF)
    //Creando el contexto del Servidor
    val sc = new SparkContext("local","riesgoCreditoArbol", System.getenv("SPARK_HOME"))
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("CargaJSON")
      .config("log4j.rootCategory", "ERROR, console")
      .getOrCreate()
    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .load("resources/clean_data.csv")

    df.printSchema()

    val inputColumns = Array("Gender", "Own_car", "Own_property", "Work_phone", "Phone", "Email", "Unemployed", "Num_children", "Num_family", "Account_length", "Total_income", "Age", "Years_employed")
    val assembler = new VectorAssembler().setInputCols(inputColumns).setOutputCol("features")

    val featureSet = assembler.transform(df)

    // split data random in trainingset (70%) and testset (30%)
    val seed = 5043
    val trainingAndTestSet = featureSet.randomSplit(Array[Double](0.7, 0.3), seed)
    val trainingSet = trainingAndTestSet(0)
    val testSet = trainingAndTestSet(1)

    // train the algorithm based on a Random Forest Classification Algorithm with default values// train the algorithm based on a Random Forest Classification Algorithm with default values

    val randomForestClassifier = new RandomForestClassifier().setSeed(seed)
    //randomForestClassifier.setMaxDepth(4)
    val model = randomForestClassifier.fit(trainingSet)

    val predictions = model.transform(testSet)

    // evaluate the model
    val evaluator = new MulticlassClassificationEvaluator()

    System.out.println("accuracy: " + evaluator.evaluate(predictions))
  }
}
