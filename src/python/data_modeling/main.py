import data_modeling.classification as C
import data_modeling.regression as R
import logging

logger = logging.getLogger(__name__)


def main(ml_usecases):
    for ml_usecase in ml_usecases.split(","):
        if ml_usecase == "classification":
            logger.info("Starting classification usecases")
            C.main()
        if ml_usecase == "regression":
            logger.info("Starting regression usecase")
            R.main()
