from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType
from pyspark.sql import SparkSession


def cast_string_timestamp():
    """Cast a string timestamp to a compatible format for indexing in elasticsearch
    """
    def inner(df):
        return (df
                .withColumn("time", F.col("time").cast(TimestampType()))
                .withColumn("timestamp", F.date_format(F.col("time"), "yyyy/MM/dd HH:mm:ss"))
                .drop("time"))
    return inner


def cast_date_hour_timestamp():
    """Cast a string date with hour to a compatible timestamp format for elasticsearch
    """
    def inner(df):
        return (df
                .withColumn("time", F.unix_timestamp(F.col("time"),
                                                     "yyyy-MM-dd-HH").cast(TimestampType()))
                .withColumn("timestamp", F.date_format(F.col("time"), "yyyy/MM/dd HH:mm:ss"))
                .drop("time"))
    return inner


def add_filepath_col():
    """Add a column to dataframe with the file path as content
    """
    def inner(df):
        return df.withColumn("filepath", F.input_file_name())
    return inner


def check_spark_version():
    spark = SparkSession.builder.getOrCreate()
    if spark.version < "2.4.0":
        raise EnvironmentError(f"""Spark doit être en version 2.4.0 pour appliquer la fonction sql
                               reverse, la version actuelle est : {spark.version}""")


def add_parse_filepath_col():
    """Parse the filepath column in order to create other columns station, api
    """
    check_spark_version()

    def inner(df):
        return (df
                .transform(add_filepath_col())
                .withColumn("reversed_split", F.reverse(F.split(F.col("filepath"), "/")))
                .withColumn("api", F.col("reversed_split").getItem(1))
                .withColumn("station", F.col("reversed_split").getItem(3)))
    return inner


def array_columns_to_rows():
    """Explode two array columns to rows with arrays_zip
    """
    check_spark_version()

    def inner(df):
        return (df
                .withColumn("tmp", F.arrays_zip("source", "temperature"))
                .withColumn("tmp", F.explode("tmp"))
                .select(F.col("tmp.source").alias("exploded_source"),
                        F.col("tmp.temperature").alias("exploded_temperature"),
                        *[c for c in df.columns])
                .drop("source", "temperature")
                )
    return inner


def array_observations_to_rows():
    """Explode the observation column to rows
    """
    def inner(df):
        return (df
                .withColumn("tmp", F.explode("observation"))
                .select(F.col("tmp.time").alias("time"),
                        F.col("tmp.temperature").alias("temperature"),
                        F.col("tmp.precipitation").alias("precipitation"),
                        F.col("tmp.humidity").alias("humidity"),
                        "station"))
    return inner


def write_to_es(df, es_conf):
    """Write a Spark dataframe to elastic index with es_conf as a dictionnary of elastic indexing
    options such as the host, port, ...
    """
    (df.write
     .format("org.elasticsearch.spark.sql")
     .options(**es_conf)
     .mode("overwrite")
     .save())


def transform(self, f):
    """Same as transform function from Scala Spark, will be native in Spark 3.0
    """
    return f(self)
