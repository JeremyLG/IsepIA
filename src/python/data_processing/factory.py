from data_processing.utils import (cast_string_timestamp,
                                   transform,
                                   add_parse_filepath_col,
                                   cast_date_hour_timestamp,
                                   array_columns_to_rows,
                                   array_observations_to_rows)
from pyspark.sql import DataFrame
DataFrame.transform = transform


class Factory:
    def __init__(self, usecase, spark_json, read_path):
        self.usecase = usecase
        self.json_df = spark_json.read_json(read_path)
        self.json_df.show(50)

    def previsions(self):
        df = (self
              .json_df
              .transform(cast_string_timestamp())
              .transform(add_parse_filepath_col()))
        return df

    def analyses(self):
        df = (self
              .json_df
              .transform(cast_date_hour_timestamp())
              .transform(array_columns_to_rows()))
        return df

    def observations(self):
        df = (self
              .json_df
              .transform(array_observations_to_rows())
              .transform(cast_string_timestamp()))
        return df

    def process(self):
        if self.usecase == "Previsions":
            return self.previsions()
        elif self.usecase == "Analyses":
            return self.analyses()
        elif self.usecase == "Observations":
            return self.observations()
        else:
            raise ValueError(f"Le usecase {self.usecase} n'existe pas")
