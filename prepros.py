import pymongo
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
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

def makeDataset(start, stop, hours_forecast=3, norms=None):
    start = datetime.strptime(start, '%Y-%m-%d').timestamp()
    stop = datetime.strptime(stop, '%Y-%m-%d').timestamp()

    # Get dataframes and round them to nearest hour.
    demgen_df = getDemandGen(start, stop).resample("10min").mean() # Demand Generation data
    hw_df = getWeather(start, stop, 0).resample("10min").mean() # Historical Weather data
    f_df = getWeather(start, stop, hours_forecast).resample("10min").mean() # Weather Forecast data

    y_df = getANMStatus(start, stop, hours_forecast).resample("10min").max() # Truth Labels

    # TODO: Make a separate weather dataset for the forecast data.

    # Join DemGen-data, with weather- and time-data and average over 1 hour of readings
    h_df = pd.DataFrame(index=pd.date_range(start=datetime.fromtimestamp(start), end=datetime.fromtimestamp(stop), freq='10min', closed="left"))
    h_df = h_df.join(demgen_df).join(hw_df).ffill().bfill()
    # Add relevant time and date information
    h_df["hour"] = [d.hour+1 for d in h_df.index]
    h_df["day"] = [d.day+1 for d in h_df.index]
    h_df["month"] = [d.month+1 for d in h_df.index]
    h_df["weekday"] = [d.weekday()+1 for d in h_df.index]

    # Align x and y by dropping times that are in one but not the other
    print(h_df.shape, f_df.shape, y_df.shape)

    y_df, xh_df = y_df.align(h_df, join="inner", axis=0, method="pad")
    xh_df, xf_df = xh_df.align(f_df, join="left", axis=0, method="pad")

    print(xh_df.shape, xf_df.shape, y_df.shape)

    # Go from dataframes to numpy arrays
    xh, xf, y = xh_df.values, xf_df.values, y_df.values

    if norms is None:
        # Normalize data and return norms
        xh, h_norms = normalizeData(xh)
        xf, f_norms = normalizeData(xf)
    else:
        # Unpack norms and apply to data
        h_norms, f_norms = norms
        xh = xh / h_norms
        xf = xf / f_norms

    # Make the timeseries from the historical data
    xts, xf, y = makeTimeseries(xh, xf, y)

    return xts, h_norms, xf, f_norms, y


def makeTimeseries(x_h,x_f,y):
    samples = len(x_h)-timesteps+1
    ts = np.zeros(shape=(samples, timesteps, x_h.shape[1]))
    for i in range(samples):
        ts[i] = x_h[i:i+timesteps]
    return ts, x_f[timesteps-1:], y[timesteps-1:]

def splitData(ts,f,y): return train_test_split(ts,f,y, test_size=0.1, shuffle=False)
def normalizeData(x): return normalize(x, axis=0, norm="l2", return_norm=True)
def reduceZones(y): return np.maximum.reduce(y, 1, initial=0)[:,np.newaxis]
def rebuildDate(x): return datetime(datetime.now().year, int(x[-2])-1, int(x[-3])-1, int(x[-4])-1) + timedelta(hours=hours_forecast)

def getPredictionData(hours=hours_forecast):
    print("hello!")
    # NOTE: You could also add previous ANM zone status' to the train input. Maybe that could make sense.
