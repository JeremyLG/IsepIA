from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit, input_file_name, col, date_format
from pyspark.sql.types import TimestampType


def transform(self, f):
    return f(self)


DataFrame.transform = transform
DATA_PATH = "/home/jeremy/Documents/isepAI/data"


class EsConf:
    def __init__(self):
        self.log = "spark/_doc"
        self.host = "localhost"
        self.port = 9200


class SparkJson:
    def __init__(self):
        self.spark = (SparkSession
                      .builder
                      .appName('isepAI')
                      .getOrCreate())
        self.spark.sparkContext.setLogLevel('WARN')

    def read(self, path: str):
        return (self.spark
                .read
                .option("multiLine", True)
                .json(path))

    def build_wildcard_directory(self, log_type: str):
        return DATA_PATH + "/*/" + log_type + "/*/*.json"

    def write_to_es(self, df, es_conf):
        (df.write
         .format("org.elasticsearch.spark.sql")
         .option("es.resource", es_conf.log)
         .option("es.nodes", es_conf.host)
         .option("es.port", es_conf.port)
         # .option("es.ingest.pipeline", "fix_date_1173")
         .option("es.mapping.date.rich", "false")
         .mode("overwrite")
         .save())


def filename_col():
    def inner(df):
        return df.withColumn("filename", input_file_name())
    return inner


def with_funny(word):
    def inner(df):
        return df.withColumn("funny", lit(word))
    return inner


def cast_timestamp():
    def inner(df):
        return (df
                .withColumn("time", col("time").cast(TimestampType()))
                .withColumn("timestamp", date_format(col("time"), "yyyy/MM/dd HH:mm:ss"))
                .drop("time"))
    return inner


if __name__ == '__main__':
    spark = SparkJson()
    es_conf = EsConf()
    read_path = spark.build_wildcard_directory("Previsions")
    # read_path = spark.build_wildcard_directory("Analyses")
    # read_path = spark.build_wildcard_directory("Observations")
    df = (spark
          .read(read_path)
          .transform(with_funny("yo"))
          .transform(filename_col())
          .transform(cast_timestamp()))
    df.printSchema()
    df.show()
    spark.write_to_es(df, es_conf)
