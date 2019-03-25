from data_processing.spark_session import SparkJson
from data_processing.factory import Factory
from data_processing.utils import write_to_es


DATA_PATH = "/home/jeremy/Documents/isepAI/data"
es_conf = {
    "es.resource": "spark/_doc",
    "es.nodes": "localhost",
    "es.port": 9200,
    "es.mapping.date.rich": False
}
write_es = False


def build_wildcard_directory(log_type: str):
    if log_type == "Previsions":
        return DATA_PATH + "/*/" + log_type + "/*/*.json"
    elif log_type in ["Observations", "Analyses"]:
        return DATA_PATH + "/*/" + log_type + "/*.json"
    else:
        raise ValueError(f"Le usecase {log_type} n'existe pas")


if __name__ == '__main__':
    spark_json = SparkJson()
    usecase = "Observations"  # Previsions ou Analyses ou Observations
    read_path = build_wildcard_directory(usecase)
    factory = Factory(usecase, spark_json, read_path)
    df = factory.process()
    df.printSchema()
    df.show(50, False)
    if write_es:
        write_to_es(df, es_conf)
