# Powering the Orkney Cloud
 _Modeling and Understanding Curtailment in Orkney_

This is the repo for my Msc Thesis project, revolving around modeling and understanding generator curtailment in the Active Network Management system of the Orkney power grid.

See http://curtailment.net for a graph tool to explore the data.

See http://forecast.curtailment.net for a curtailment forecast based on the models we have developed.

## Contents

- Datasets
  - The main dataset used for modeling is found in `datasets/Dataset-01-12-2018-to-01-03-2019.csv`
  - A clean version is found in `datasets/Dataset-01-12-2018-to-01-03-2019-cleaned.csv`
  - Eday data found in `datasets/eday-winter-2018-2019.csv`
  - The dataset used for evaluating predictive models is found in `datasets/predictive_dataset_april_2019.csv`
- Scripts
  - `scrape.py` contains the data collection and storing.
  - `prepros.py` contains data preprocessing, alogn with database querying and data cleaning.
  - `descriptive.py` contains the rule-based descriptive models and evaluation methods.
  - `model.py` contains methods for training, evaluating, saving and loading ANN-based models.
  - `plotter.py` contains methods for generating visualizations and plots.
  - `prepros-eday-data.py`, a simple script the parses the eday CSV-files and creates a dataframe.
  - `eday-power-curve.py` contains the code for constructing the power curves for the ERE wind turbine.
  - `metoffice.py` contains the methods for evaluating the predictive models.


## Config

To run the files, you need to make a `config.py` file with the following variables and place it in the main folder:
```
reader_user = "xxxx" # username for DB user with reading rights
reader_pw = "xxxxxx" # password for DB user with reading rights
writer_user = "xxxxx" # username for DB user with writing rights
writer_pw = "xxxxxx" # password for DB user with writing rights
API_KEY = "xxxxxxxx" # API Key for OpenWeatherMap
MET_API_KEY = "xxxxxxxxxxxxxxxxx" # API Key for UK Met office
SERVER = "xxx.xxx.xxx.xxx:xxxxx" # Server address and port for MongoDB
DATA_PATH = "./datasets/" # Or whereever you wanna put it.
```
