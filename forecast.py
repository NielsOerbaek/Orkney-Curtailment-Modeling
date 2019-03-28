import pymongo
import requests as re
from datetime import datetime
import config

from pprint import pprint

# Connect to db
client = pymongo.MongoClient("mongodb://"+config.writer_user+":"+config.writer_pw+"@"+config.SERVER+"/sse-data")
db = client["sse-data"]

# Scraping the live demand and generation of the Orkney ANM
met_col = db["metforecast"]
met_scrape = re.get("http://datapoint.metoffice.gov.uk/public/data/val/wxfcs/all/json/354165?res=3hourly&key="+config.MET_API_KEY).json()
met_scrape["timestamp"] = datetime.utcnow().timestamp()

def renameKeys(object, old_name, new_name):
    if isinstance(object,dict):
        for key,val in object.items():
            if key == old_name: object[new_name] = object.pop(old_name)
            try:
                renameKeys(val, old_name, new_name)
            except: pass
    elif isinstance(object, list):
        for val in object:
            try:
                renameKeys(val, old_name, new_name)
            except: pass

renameKeys(met_scrape, "$", "dollar")

met_id = met_col.insert_one(met_scrape)
