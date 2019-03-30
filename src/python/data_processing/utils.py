from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType
from pyspark.sql import SparkSession


def cast_string_timestamp(input_time_col="time", output_time_col="timestamp"):
    """Cast a string timestamp to a compatible format for indexing in elasticsearch
    """
    def inner(df):
        return (df
                .withColumn(input_time_col, F.col(input_time_col).cast(TimestampType()))
                .withColumn(output_time_col, F.date_format(F.col(input_time_col),
                                                           "yyyy/MM/dd HH:mm:ss"))
                .drop(input_time_col))
    return inner


def cast_date_hour_timestamp(input_time_col="time", output_time_col="timestamp"):
    """Cast a string date with hour to a compatible timestamp format for elasticsearch
    """
    def inner(df):
        return (df
                .withColumn(input_time_col,
                            F.unix_timestamp(F.col(input_time_col), "yyyy-MM-dd-HH")
                            .cast(TimestampType()))
                .withColumn(output_time_col,
                            F.date_format(F.col(input_time_col), "yyyy/MM/dd HH:mm:ss"))
                .drop(input_time_col))
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


def add_parse_filepath_col(usecase):
    """Parse the filepath column in order to create other columns station, api
    """
    check_spark_version()

    def inner(df):
        df_tmp = (df
                  .transform(add_filepath_col())
                  .withColumn("reversed_split", F.reverse(F.split(F.col("filepath"), "/"))))

        if usecase == "Previsions":
            return (df_tmp.withColumn("api", F.col("reversed_split").getItem(1))
                    .withColumn("station", F.col("reversed_split").getItem(3))
                    .withColumn("tmp_ts_file", F.col("reversed_split").getItem(0))
                    .withColumn("tmp_ts_file", F.split(F.col("tmp_ts_file"), "\\.").getItem(0)))
        else:
            return (df_tmp
                    .withColumn("station", F.col("reversed_split").getItem(2)))
    return inner


def array_analyses_to_rows():
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
                .drop("source", "temperature", "filepath")
                .withColumnRenamed("exploded_source", "source")
                .withColumnRenamed("exploded_temperature", "temperature")
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


def array_previsions_to_rows():
    """Explode the observation column to rows
    """
    def inner(df):
        return (df
                .withColumn("tmp", F.explode("forecast"))
                .select(F.col("tmp.cloud_cover").alias("cloud_cover"),
                        F.col("tmp.temperature").alias("temperature"),
                        F.col("tmp.precipitation").alias("precipitation"),
                        F.col("tmp.humidity").alias("humidity"),
                        F.col("tmp.wind").alias("wind"),
                        F.col("tmp.wind_dir").alias("wind_dir"),
                        F.col("tmp.pressure").alias("pressure"),
                        F.col("tmp.time").alias("time"),
                        "station", "api", "latitude_q", "latitude_r", "longitude_q", "longitude_r",
                        "source", "time_q", "time_r", "tmp_ts_file"))
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
