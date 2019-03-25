from pyspark.sql import SparkSession


class SparkJson:
    def __init__(self):
        """Initialize a basic SparkSession and setting to warning log4j level
        """
        self.spark = (SparkSession
                      .builder
                      .appName('isepAI')
                      .getOrCreate())
        self.spark.sparkContext.setLogLevel('WARN')

    def read_json(self, path: str):
        """Read multiLines json data
        """
        return (self.spark
                .read
                .option("multiLine", True)
                .json(path))
