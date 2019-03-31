import logging

from data_processing.spark_session import SparkJson
from data_processing.factory import Factory
from data_processing.utils import write_to_es

logger = logging.getLogger(__name__)
DATA_PATH = "data/"

es_conf = {
    "es.resource": "spark/_doc",
    "es.nodes": "localhost",
    "es.port": 9200,
    "es.mapping.date.rich": False
}


def build_wildcard_directory(log_type: str):
    if log_type == "Previsions":
        return DATA_PATH + "/raw/*/" + log_type + "/*/2019*.json"
    elif log_type in ["Observations", "Analyses"]:
        return DATA_PATH + "/raw/*/" + log_type + "/*.json"
    else:
        raise ValueError(f"Le usecase {log_type} n'existe pas")


def process(process_usecases, write_es):
    spark_json = SparkJson()
    logger.info("Starting to process with Spark the 3 data sources")
    for usecase in process_usecases.split(","):
        logger.info(f"Starting usecase : {usecase}")
        read_path = build_wildcard_directory(usecase)
        factory = Factory(usecase, spark_json, read_path)
        df = factory.process()
        spark_json.write_json(df, usecase, DATA_PATH + "/output/" + usecase)
        if write_es:
            logger.info("Writing to ElasticSearch index")
            es_conf["es.resource"] = usecase.lower() + "/_doc"
            write_to_es(df, es_conf)


if __name__ == '__main__':
    process("Analyses,Observations,Previsions", True)
