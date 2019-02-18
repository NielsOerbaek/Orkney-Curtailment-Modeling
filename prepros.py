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
timesteps = 24
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
        d = getDate(s["timestamp"])
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
        d = getDate(s["timestamp"]) - timedelta(hours=hours)
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
        time = getDate(w["dt"]) - timedelta(hours=hours)
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

    # Get dataframes and round them to nearest hour.
    demgen_df = getDemandGen(start, stop).resample("10min").mean()
    w_df = getWeather(start, stop, hours_forecast).resample("10min").mean()
    y_df = getANMStatus(start, stop, hours_forecast).resample("10min").max()

    # TODO: Make a separate weather dataset for the forecast data.

    # Join DemGen-data, with weather- and time-data and average over 1 hour of readings
    x_df = pd.DataFrame(index=pd.date_range(start=datetime.fromtimestamp(start), end=datetime.fromtimestamp(stop), freq='10min', closed="left"))
    x_df = x_df.join(demgen_df).join(w_df).ffill().bfill()
    # Add relevant time and date information
    x_df["hour"] = [d.hour+1 for d in x_df.index]
    x_df["day"] = [d.day+1 for d in x_df.index]
    x_df["month"] = [d.month+1 for d in x_df.index]
    x_df["weekday"] = [d.weekday()+1 for d in x_df.index]

    # Align x and y by dropping times that are in one but not the other
    dump_x = x_df[~x_df.index.isin(y_df.index)].index
    dump_y = y_df[~y_df.index.isin(x_df.index)].index
    x_df.drop(dump_x, inplace=True)
    y_df.drop(dump_y, inplace=True)

    print(x_df.shape)

    # Go from dataframes to numpy arrays
    x, y = x_df.values, y_df.values
    return x, y


def makeTimeseries(x,y):
    samples = len(x)-timesteps+1
    ts = np.zeros(shape=(samples,timesteps,x.shape[1]))
    for i in range(samples):
        ts[i] = x[i:i+timesteps]
    return ts, y[timesteps-1:]

def splitData(x,y): return train_test_split(x, y, test_size=0.1, shuffle=False)
def normalizeData(x): return normalize(x, axis=0, norm="l2", return_norm=True)
def reduceZones(y): return np.maximum.reduce(y, 1, initial=0)[:,np.newaxis]
def rebuildDate(x): return datetime(datetime.now().year, int(x[-2])-1, int(x[-3])-1, int(x[-4])-1) + timedelta(hours=hours_forecast)

def getPredictionData(hours=hours_forecast):
    print("hello!")
    # NOTE: You could also add previous ANM zone status' to the train input. Maybe that could make sense.
