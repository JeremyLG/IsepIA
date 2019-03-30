import os
import logging
from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)


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

    def write_json(self, df: DataFrame, usecase: str, path: str):
        logger.info("Writing dataframe to JSON")
        (df
         .coalesce(1)
         .write
         .mode("overwrite")
         .json(path))
        logger.info("Renaming Spark file to human readable name")
        for file in os.listdir(path):
            if file.startswith("part") & file.endswith(".json"):
                os.rename(path + "/" + file, path + "/" + usecase + ".json")
