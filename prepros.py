import pymongo
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import config

zone_names = ["Core Zone", "Zone 1", "Zone 1A", "Zone 2", "Zone 2A", "Zone 2B", "Zone 3", "Zone 4", "Zone 4A"]
hours_forecast = 3
timesteps = 6*hours_forecast
batch_size = 1

client = pymongo.MongoClient("mongodb://"+config.reader_user+":"+config.reader_pw+"@167.99.129.50:27017/sse-data")
db = client["sse-data"]

def getDate(s): return datetime.fromtimestamp(int(s))
def isCurtailed(z): return z["ANM_Operation"] == "YELLOW"
def isStopped(z): return z["ANM_Operation"] == "RED"

def getDemandGen(start=0, end=0):
    if end == 0: end = start+86400
    demand = db["demand"]
    data = dict()

    for s in demand.find({"timestamp": {"$gt": start-1, "$lt": end}}):
        d = getDate(s["timestamp"]).replace(microsecond=0,second=0)
        data[d] = dict()
        data[d]["Demand"] = s["data"][0]["data"][0]
        data[d]["Generation"] = s["data"][2]["data"][1]+s["data"][3]["data"][1]

    df = pd.DataFrame.from_dict(data, orient="index")
    return df

def getANMStatus(start=0, end=0, hours=0):
    if end == 0: end = start+86400
    status = db["ANM_status"]
    data = dict()

    for s in status.find({"timestamp": {"$gt": start-1+(hours*3600), "$lt": end+(hours*3600)}}):
        d = getDate(s["timestamp"]).replace(microsecond=0,second=0) - timedelta(hours=hours)
        zs = dict()
        for z in zone_names:
            ss = int(isStopped(s[z]))
            cs = int(isCurtailed(s[z]) or ss)
            zs[z] = cs
        data[d] = zs

    df = pd.DataFrame.from_dict(data, orient="index")
    return df


def getWeather(start=0, end=0, hours=0):
    if end == 0: end = start+86400
    weather = db["weather"]
    data = dict()

    for w in weather.find({"dt": {"$gt": start-1+(hours*3600), "$lt": end+(hours*3600)}}):
        # Make datetime object at round to nearest 10 minutes.
        time = getDate(w["dt"]).replace(microsecond=0,second=0) - timedelta(hours=hours)
        data[time] = dict()
        data[time]["speed"] = w["wind"]["speed"]
        if "deg" in w["wind"].keys(): data[time]["deg"] = w["wind"]["deg"]
        else: data[time]["deg"] = -1
        data[time]["pressure"] = w["main"]["pressure"]
        data[time]["temp"] = w["main"]["temp"]-273.15 # Conversion from Kelvin to Celcius

    df = pd.DataFrame.from_dict(data, orient="index")
    return df

def makeDataset(start, stop, hours_forecast=3):
    start = datetime.strptime(start, '%Y-%m-%d').timestamp()
    stop = datetime.strptime(stop, '%Y-%m-%d').timestamp()

    demgen_df = getDemandGen(start, stop)
    w_df = getWeather(start, stop, hours_forecast)
    y_df = getANMStatus(start, stop, hours_forecast)

    # Join DemGen-data, with weather- and time-data
    x_df = demgen_df.join(w_df).ffill().bfill()
    # Use some rolling average and shifting to make transitions between the hourly readings
    w_cols = w_df.columns
    x_df[w_cols] = x_df[w_cols].rolling(6, min_periods=1).mean().shift(-5).ffill()
    # Add relevand time and date information
    x_df["hour"] = [d.hour+1 for d in x_df.index]
    x_df["day"] = [d.day+1 for d in x_df.index]
    x_df["month"] = [d.month+1 for d in x_df.index]
    x_df["weekday"] = [d.weekday()+1 for d in x_df.index]

    # Align x and y by dropping times that are in one but not the other
    dump_x = x_df[~x_df.index.isin(y_df.index)].index
    dump_y = y_df[~y_df.index.isin(x_df.index)].index
    x_df.drop(dump_x, inplace=True)
    y_df.drop(dump_y, inplace=True)

    # Go from dataframes to numpy arrays
    x, y = x_df.values, y_df.values
    return x, y


def splitData(x,y): return train_test_split(x, y, test_size=0.1, shuffle=False)
def makeTimeseries(x,y): return TimeseriesGenerator(x,y,timesteps,batch_size=batch_size)
def normalizeData(x): return normalize(x, axis=0, norm="l2", return_norm=True)
def reduceZones(y): return np.maximum.reduce(y, 1, initial=0)[:,np.newaxis]
def rebuildDate(x): return datetime(datetime.now().year, int(x[-2])-1, int(x[-3])-1, int(x[-4])-1) + timedelta(hours=hours_forecast)

def getPredictionData(hours=hours_forecast):
    print("hello!")
    # NOTE: You could also add previous ANM zone status' to the train input. Maybe that could make sense.
