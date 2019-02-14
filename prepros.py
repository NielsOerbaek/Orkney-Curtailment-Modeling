import pymongo
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import config

zone_names = ["Core Zone", "Zone 1", "Zone 1A", "Zone 2", "Zone 2A", "Zone 2B", "Zone 3", "Zone 4", "Zone 4A"]
timesteps = 6*3
batch_size = 32

client = pymongo.MongoClient("mongodb://"+config.reader_user+":"+config.reader_pw+"@167.99.129.50:27017/sse-data")
db = client["sse-data"]
status = db["ANM_status"]

def getDate(s): return datetime.fromtimestamp(int(s["timestamp"]))
def isCurtailed(z): return z["ANM_Operation"] == "YELLOW"
def isStopped(z): return z["ANM_Operation"] == "RED"

def getDataFrame(start=0, end=0):

    if end == 0: end = start+86400

    data_dict = dict()

    for s in status.find({"timestamp": {"$gt": start-1, "$lt": end}}):
        d = getDate(s).replace(microsecond=0,second=0)
        zs = dict()
        for z in zone_names:
            ss = int(isStopped(s[z]))
            cs = int(isCurtailed(s[z]) or ss)
            zs[z] = cs
        data_dict[d] = zs

    demand = db["demand"]

    for s in demand.find({"timestamp": {"$gt": start-1, "$lt": end}}):
        d = getDate(s).replace(microsecond=0,second=0)
        if d in data_dict.keys() and len(data_dict[d]) < 20:
            data_dict[d]["Demand"] = s["data"][0]["data"][0]
            data_dict[d]["Generation"] = s["data"][2]["data"][1]+s["data"][3]["data"][1]

    df = pd.DataFrame.from_dict(data_dict, orient="index")

    return df


def getWeatherDataFrame(start=0, end=0, hours=0):

    if end == 0: end = start+86400

    data = dict()

    weather = db["weather"]

    for w in weather.find({"dt": {"$gt": start-1+(hours*3600), "$lt": end+(hours*3600)}}):
        # Make datetime object at round to nearest 10 minutes.
        time = datetime.fromtimestamp(w["dt"]).replace(microsecond=0,second=0) - timedelta(hours=hours)
        data[time] = w["wind"]
        data[time]["pressure"] = w["main"]["pressure"]
        data[time]["temp"] = w["main"]["temp"]-273.15 # Conversion from Kelvin to Celcius

    df = pd.DataFrame.from_dict(data, orient="index")

    return df

def makeDataset(start, stop):
    start = datetime.strptime(start, '%Y-%m-%d').timestamp()
    stop = datetime.strptime(stop, '%Y-%m-%d').timestamp()

    sse_df = getDataFrame(start, stop)
    w_df = getWeatherDataFrame(start, stop, 3)

    # Split out target values
    y_df = sse_df[["Zone 2"]]

    # Join SSE-data, with weather- and time-data.
    w_cols = w_df.columns
    x_df = sse_df[["Demand","Generation"]].join(w_df).ffill().bfill()
    # Use some rolling average and shifting to make transitions between the hourly readings.
    x_df[w_cols] = x_df[w_cols].rolling(6, min_periods=1).mean().shift(-5).ffill()
    # Add relevand time and date information
    x_df["hour"] = [d.hour+1 for d in x_df.index]
    x_df["day"] = [d.day+1 for d in x_df.index]
    x_df["month"] = [d.month+1 for d in x_df.index]
    x_df["weekday"] = [d.weekday()+1 for d in x_df.index]

    # Go from dataframes to numpy arrays
    x, y = x_df.values, y_df.values
    return x, y


def splitData(x,y): return train_test_split(x, y, test_size=0.1, shuffle=False)
def makeTimeseries(x,y): return TimeseriesGenerator(x,y,timesteps,batch_size=batch_size)
def normalizeData(x): return normalize(x, axis=0, norm="max", return_norm=True) #TODO: Find out if l2 is the better norm here
