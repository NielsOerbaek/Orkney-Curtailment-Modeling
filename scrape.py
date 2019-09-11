import pymongo
import requests as re
import time
from bs4 import BeautifulSoup
from datetime import datetime
import config

client = pymongo.MongoClient("mongodb://"+config.writer_user+":"+config.writer_pw+"@"+config.SERVER+"/sse-data")
db = client["sse-data"]


# Scraping the live demand and generation of the Orkney ANM
demand_col = db["demand"]
demand_scrape = re.get("https://www.ssen.co.uk/Sse_Components/Views/Controls/FormControls/Handlers/ActiveNetworkManagementHandler.ashx?action=graph&contentId=14973").json()
demand = dict()
demand["timestamp"] = time.time()
demand["data"] = demand_scrape["data"]["datasets"]
demand_id = demand_col.insert_one(demand)

print(demand_id)

# Scraping the live status of the Orkney ANM
status_col = db["ANM_status"]
page = re.get("https://www.ssen.co.uk/ANMGeneration/").text
soup = BeautifulSoup(page, 'html.parser')
table = soup.find('table', attrs={'class':'table'})
rows = table.find_all('tr')[2:]

def parse_symbol(td):
	classes = td.span["class"]
	if "glyphicon-ok-sign" in classes:
		return "GREEN"
	if "glyphicon-warning-sign" in classes:
		return "YELLOW"
	if "glyphicon-remove-sign" in classes:
		return "RED"

status = dict()
status["timestamp"] = time.time()
for row in rows:
	label = row.find('td', attrs={'class':'ZoneData-ZoneLabel'}).contents[0].strip()
	status[label] = dict()
	status[label]["label"] = label
	symbols = row.find_all('td', attrs={'class':'ZoneData-NoText'})
	status[label]["ANM_Operation"] = parse_symbol(symbols[0])
	status[label]["SHEPD_Equipment"] = parse_symbol(symbols[1])
	status[label]["Generator_Site_Issues"] = parse_symbol(symbols[2])

status_id = status_col.insert_one(status)

print(status_id)

# Scraping the live demand and generation of the Orkney ANM
weather_col = db["weather"]
weather_scrape = re.get("http://api.openweathermap.org/data/2.5/weather?lat=59.1885692&lon=-2.8229873&APPID="+config.API_KEY).json()
weather_id = weather_col.insert_one(weather_scrape)
print(weather_id)

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

print(met_id)
