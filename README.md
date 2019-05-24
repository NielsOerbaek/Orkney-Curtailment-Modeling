# Powering the Orkney Cloud

This is the repo for a Msc Thesis project revolving around modeling and understanding curtailment in the Orkney Active Network Management system.

See http://curtailment.net for a graph tool to explore the data.

See http://forecast.curtailment.net for a curtailment forecast based on the models we have developed.

TODO: Add the scraping files to this folder. 

### Config

You should make a `config.py` file with the following variables at place it in the main folder:
```
reader_user = "xxxx"
reader_pw = "xxxxxx"
writer_user = "xxxxx"
writer_pw = "xxxxxx"
API_KEY = "xxxxxxxx" # API Key for OpenWeatherMap
MET_API_KEY = "xxxxxxxxxxxxxxxxx" # API Key for UK Met office
SERVER = "xxx.xxx.xxx.xxx:xxxxx"
DATA_PATH = "./datasets/" # Or whereever you wanna put it.
```
