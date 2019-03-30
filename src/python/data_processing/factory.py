from data_processing.utils import (cast_string_timestamp,
                                   transform,
                                   add_parse_filepath_col,
                                   cast_date_hour_timestamp,
                                   array_analyses_to_rows,
                                   array_observations_to_rows,
                                   array_previsions_to_rows)
from pyspark.sql import DataFrame
DataFrame.transform = transform


class Factory:
    def __init__(self, usecase, spark_json, read_path):
        self.usecase = usecase
        self.json_df = spark_json.read_json(read_path)

    def previsions(self):
        df = (self
              .json_df
              .transform(add_parse_filepath_col(self.usecase))
              .transform(array_previsions_to_rows())
              .transform(cast_string_timestamp())
              .transform(cast_string_timestamp(input_time_col="time_q",
                                               output_time_col="time_query"))
              .transform(cast_string_timestamp(input_time_col="time_r",
                                               output_time_col="time_received"))
              .transform(cast_date_hour_timestamp(input_time_col="tmp_ts_file",
                                                  output_time_col="ts_file")))
        return df

    def analyses(self):
        df = (self
              .json_df
              .transform(add_parse_filepath_col(self.usecase))
              .transform(cast_date_hour_timestamp())
              .transform(array_analyses_to_rows()))
        return df

    def observations(self):
        df = (self
              .json_df
              .transform(add_parse_filepath_col(self.usecase))
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
