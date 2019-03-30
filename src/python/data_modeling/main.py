import data_modeling.classification as C
import data_modeling.regression as R
import logging

logger = logging.getLogger(__name__)


def main(conf_dict):
    for ml_usecase in conf_dict["ml_usecases"].split(","):
        logger.info(f"Starting {ml_usecase} usecases")
        if ml_usecase == "classification":
            C.main(conf_dict["classification_usecase"])
        if ml_usecase == "regression":
            R.main(conf_dict["regression_usecase"])
