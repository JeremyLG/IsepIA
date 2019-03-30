import yaml
import logging
import logging.config
import logging.handlers

from data_modeling import main as M
from data_processing import main as P

CONF_DIR = "/home/jeremy/Documents/isepAI/conf/"


def main(conf_dict):
    logging.config.dictConfig(yaml.safe_load(open(CONF_DIR + 'logging.yml', 'r')))
    for mode in conf_dict["mode"].split(","):
        if mode == "data_processing":
            logging.info("Starting Spark Processing")
            P.process(conf_dict["process_usecases"], conf_dict["write_es"])
        if mode == "data_modeling":
            logging.info("Starting PyTorch Modeling")
            M.main(conf_dict["ml_usecases"])


if __name__ == '__main__':
    conf_dict = yaml.safe_load(open(CONF_DIR + 'prod.yml', 'r'))
    open(conf_dict["project_path"] + "logs/info.log", 'w').close()
    main(conf_dict)
