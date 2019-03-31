# IsepIA Deep Learning Hackathon

## Goal of the competition

The aim is to improve decision-making within a few hours, so the focus will be on immediate
forecasting. The most interesting parameters in agriculture are temperature, especially
for extreme cold events (frost at a sensitive stage of the crop which can destroy flowers
and therefore future fruits in arboriculture or vines), rainfall, which makes it possible to
control crop irrigation, wind which constrains cultural interventions and can also cause
crop lodging.

## Technologies used

- Python3
- PyTorch : deep learning framework
- Spark - Elastic : flatten and visualize json data

## Potential subjects to explore

1. Can we find signs of the arrival of a given event (e.g. low temperature, below 2°C, rain above 5 mm, other...) by analyzing forecasts and observations.
2. Can we determine with good certainty the most reliable forecast based on past forecasts and observations?
3. Can we integrate observation data into predictive models, but also forecast errors?
4. Propose AI models that can be applied to this data and that are of interest to farmers?

## What you need to do

```shell
git clone https://github.com/JeremyLG/isepIA
cd isepIA
python3 -m venv isep
source isep/bin/activate
pip3 install -r requirements.txt
python3 src/python/main.py
```

## Configuration

- **prod.yml** : Different options while executing the main
  - project_path : where the project is located on your OS
  - write_es : should the processing also save the data on Elasticsearch
  - mode : string separed by commas to defined what you should do. For example *data_processing,data_modeling*
  - process_usecases : string separed by commas which defines the data sources the script will process with Spark
  - ml_usecases : string separed by commas to execute regression and/or classification
  - classification_usecase and regression_usecase : target column for the training, should be either *temperature* or *humidity*
  - save_models : should PyTorch save the models in models directory
- **logging.yml** : Dictionnary config for logging the information while processing and training
