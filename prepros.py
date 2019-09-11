import pymongo
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import requests as re
import config
import pickle

zone_names = ["Core Zone", "Zone 1", "Zone 1A", "Zone 2", "Zone 2A", "Zone 2B", "Zone 3", "Zone 4", "Zone 4A"]
bad_weather_date = datetime.strptime("2019-02-12", '%Y-%m-%d')
windgencorr = np.poly1d([-0.0262,0.3693,2.09,-1.626])
hours_forecast = 3
hours_history = 8
timesteps = hours_history*6
batch_size = 1


client = pymongo.MongoClient("mongodb://"+config.reader_user+":"+config.reader_pw+"@"+config.SERVER+"/sse-data")
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
        data[d]["ANM Generation"] = s["data"][2]["data"][1]
        data[d]["Non-ANM Generation"] = s["data"][3]["data"][1]

    df = pd.DataFrame.from_dict(data, orient="index")
    if len(df) == 0: raise ValueError("No data found for dates: "+ str(datetime.fromtimestamp(start)) +" to "+ str(datetime.fromtimestamp(end)))
    return df


def getANMStatus(start=0, end=0, hours=0):
    if end == 0: end = start+86400
    status = db["ANM_status"]
    data = dict()

    for s in status.find({"timestamp": {"$gt": start-1+(hours*3600), "$lt": end+(hours*3600)}}):
        d = getDate(s["timestamp"]) - timedelta(hours=hours)
        zs = dict()
        for z in zone_names:
            cs = int(isCurtailed(s[z]))*0.5
            ss = int(isStopped(s[z]))
            zs[z] = cs + ss
        data[d] = zs

    df = pd.DataFrame.from_dict(data, orient="index")
    if len(df) == 0: raise ValueError("No data found for dates: "+ str(datetime.fromtimestamp(start)) +" to "+ str(datetime.fromtimestamp(end)))
    return df


def getWeather(start=0, end=0, hours=0):
    if end == 0: end = start+86400
    weather = db["weather"]
    data = dict()

    for w in weather.find({"dt": {"$gt": start-1+(hours*3600), "$lt": end+(hours*3600)}}):
        time = getDate(w["dt"]) - timedelta(hours=hours)
        data[time] = weatherToDict(w)

    df = pd.DataFrame.from_dict(data, orient="index")
    if len(df) == 0: raise ValueError("No data found for dates: "+ str(datetime.fromtimestamp(start)) +" to "+ str(datetime.fromtimestamp(end)))
    return df

def getMetData(start=0, stop=0):
    start = datetime.strptime(start, '%Y-%m-%d').timestamp()
    stop = datetime.strptime(stop, '%Y-%m-%d').timestamp()
    if stop == 0: stop = start+86400
    status = db["metforecast"]
    data = []

    mph_to_mps = 0.44704

    for s in status.find({"timestamp": {"$gt": start-1, "$lt": stop}}):
        compute_time = datetime.strptime(s["SiteRep"]["DV"]["dataDate"], '%Y-%m-%dT%H:%M:%SZ')
        try:
            days = s["SiteRep"]["DV"]["Location"]["Period"]
            for daily in days:
                day = datetime.strptime(daily["value"], '%Y-%m-%dZ')
                for f in daily["Rep"]:
                    forecast_time = day + timedelta(minutes=int(f["dollar"]))
                    hours_forecast = forecast_time - compute_time
                    wind_speed = int(f["S"])*mph_to_mps
                    if forecast_time > compute_time:
                        d = dict()
                        d["forecast_time"] = forecast_time
                        d["compute_time"] = compute_time
                        d["hours_forecast"] = hours_forecast
                        d["wind_speed"] = wind_speed
                        data.append(d)
        except:
            pass

    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    if len(df) == 0: raise ValueError("No data found for dates: "+ str(datetime.fromtimestamp(start)) +" to "+ str(datetime.fromtimestamp(stop)))
    return df


def weatherToDict(w):
    d = dict()
    d["speed"] = w["wind"]["speed"]
    if "deg" in w["wind"].keys(): d["deg"] = w["wind"]["deg"]
    else: d["deg"] = -1
    d["pressure"] = w["main"]["pressure"]
    d["temp"] = round(w["main"]["temp"]-273.15,2) # Conversion from Kelvin to Celcius
    return d


def makeDataset(start, stop, hours_forecast=3, norms=None):
    xh_df, xf_df, y_df = getDataframes(start, stop, hours_forecast)

    # Go from dataframes to numpy arrays
    xh, xf, y = xh_df.values, xf_df.values, y_df.values

    # Normalize
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
    yr = reduceZones(y) # Make the reduced labels as well

    return xts, h_norms, xf, f_norms, y, yr

def getDataframes(start, stop, hours_forecast=3):
    start = datetime.strptime(start, '%Y-%m-%d').timestamp()
    stop = datetime.strptime(stop, '%Y-%m-%d').timestamp()
    if stop <= start: raise ValueError("Invalid date range: "+ str(datetime.fromtimestamp(start)) +" to "+ str(datetime.fromtimestamp(stop)))

    # Get dataframes and resample them.
    print("Getting Dem/Gen data")
    demgen_df = getDemandGen(start, stop).resample("10min").mean() # Demand Generation data
    print("Getting historical weather data")
    hw_df = getWeather(start, stop, 0).resample("10min").mean() # Historical Weather data
    print("Getting weather forecast data")
    f_df = getWeather(start, stop, hours_forecast).resample("10min").mean() # Weather Forecast data
    print("Getting ANM-status data")
    y_df = getANMStatus(start, stop, hours_forecast).resample("10min").max() # Truth Labels
    # NOTE: You could also add previous ANM zone status' to the train input. Maybe that could make sense.

    # Join DemGen-data, with weather- and time-data and average over readings
    h_df = pd.DataFrame(index=pd.date_range(start=datetime.fromtimestamp(start), end=datetime.fromtimestamp(stop), freq='10min', closed="left"))
    h_df = h_df.join(demgen_df).join(hw_df)
    f_df = f_df

    f_df = addTimeCols(f_df)

    # Align x and y by dropping times that are in one but not the other
    y_df, xh_df = y_df.align(h_df, join="inner", axis=0, method="pad")
    xh_df, xf_df = xh_df.align(f_df, join="left", axis=0, method="pad")
    # Fill in possible missing values
    xh_df = xh_df.ffill().bfill()
    xf_df = xf_df.ffill().bfill()

    return xh_df, xf_df, y_df


def getSingleDataframe(start="2018-12-01", stop="2019-03-01", fromPickle=False, clean=False, cleanGlitches=True):
    if not fromPickle:
        xh_df, xf_df, y_df = getDataframes(start, stop, 0)
        df = addTimeCols(xh_df).join(addReducedCol(y_df))
        pickle.dump(df, open(config.DATA_PATH+"SingleFrame"+start+"-"+stop, "wb"))
        print("Saved dataframe as pickle: SingleFrame"+start+"-"+stop)
    else:
        try:
            df = pickle.load(open(config.DATA_PATH+"SingleFrame"+start+"-"+stop, "rb"))
            print("Loaded dataframe as pickle: SingleFrame"+start+"-"+stop)
        except:
            xh_df, xf_df, y_df = getDataframes(start, stop, 0)
            df = addTimeCols(xh_df).join(addReducedCol(y_df))
            pickle.dump(df, open(config.DATA_PATH+"SingleFrame"+start+"-"+stop, "wb"))
            print("Saved dataframe as pickle: SingleFrame"+start+"-"+stop)

    if clean:
        print("Cleaning data...")
        df = cleanData(df)
        df = addReducedCol(df, clean=True)
        if cleanGlitches: df = removeGlitches(df)

    return df

def saveToCSV(df, name):
    df.to_csv(config.DATA_PATH+""+name)

def addReducedCol(df, clean=False):
    if "Curtailment" in df.columns: df = df.drop(["Curtailment"], axis=1)
    if not clean:
        r = df[zone_names].apply(np.maximum.reduce, axis=1)[["Core Zone"]].rename(columns={"Core Zone": "Curtailment"})
        return df.join(r)
    else:
        print("Applying De Minimis Level...")
        def de_minimis(z): return 1 if z > 1 else 0
        r = df[zone_names].sum(axis=1).apply(de_minimis)
        df["Curtailment"] = r
        return df

def removeGlitches(df, verbose=True):
    df = df.copy()
    print("Removing anomaly sections.")
    before = len(df)
    anomalies = [("2018-12-14","2018-12-19"),
                ("2018-12-28","2018-12-31"),
                ("2019-01-30","2019-02-02"),
                ("2019-02-10","2019-02-12"),
                ("2019-02-17","2019-02-21")]
    for a in anomalies:
        df = removePeriod(df, a[0], a[1])
        if verbose: print("Removed period:", a)
    if verbose: print("Removed {} data points".format(before-len(df)))
    return df

def removePeriod(df, start, stop):
    start = datetime.strptime(start, '%Y-%m-%d')
    stop = datetime.strptime(stop, '%Y-%m-%d')
    return df.drop(df.loc[start:stop].index)

def makeTimeseries(x_h,x_f,y):
    samples = len(x_h)-timesteps+1
    ts = np.zeros(shape=(samples, timesteps, x_h.shape[1]))
    for i in range(samples):
        ts[i] = x_h[i:i+timesteps]
    return ts, x_f[timesteps-1:], y[timesteps-1:]

def addTimeCols(df):
    # Add relevant time and date information
    df["hour"] = [d.hour+1 for d in df.index]
    df["day"] = [d.day+1 for d in df.index]
    df["month"] = [d.month+1 for d in df.index]
    df["weekday"] = [d.weekday()+1 for d in df.index]
    return df

def addTimeColsOneHot(df):
    # Add relevant time and date information as one hots
    df["hour"] = [toOneHot(d.hour,24) for d in df.index]
    df["day"] = [toOneHot(d.day-1,31) for d in df.index]
    df["month"] = [toOneHot(d.month-1,12) for d in df.index]
    df["weekday"] = [toOneHot(d.weekday(),7) for d in df.index]
    return df


def splitData(ts,f,y,yr): return train_test_split(ts,f,y,yr, test_size=0.1, shuffle=False)
def normalizeData(x): return normalize(x, axis=0, norm="l2", return_norm=True)
def reduceZones(y): return np.maximum.reduce(y, 1, initial=0)[:,np.newaxis]
def rebuildDate(x): return datetime(datetime.now().year, int(x[-2])-1, int(x[-3])-1, int(x[-4])-1) + timedelta(hours=hours_forecast)

def getForecastData():
    forecasts = re.get("http://api.openweathermap.org/data/2.5/forecast?lat=59.1885692&lon=-2.8229873&APPID="+config.API_KEY).json()
    forecast = forecasts["list"][0]
    dt = getDate(forecast["dt"])
    data = weatherToDict(forecast)
    data["hour"] = dt.hour+1
    data["day"] = dt.day+1
    data["month"] = dt.month+1
    data["weekday"] = dt.weekday()+1
    xf = np.array(list(data.values()))
    return xf, dt

def getLastTimeseries():
    stop = datetime.now()
    stop -= timedelta(minutes=stop.minute%10, seconds=stop.second, microseconds=stop.microsecond)
    start = stop - timedelta(hours=hours_history)
    start_stamp, stop_stamp = start.timestamp(), stop.timestamp()

    demgen = getDemandGen(start_stamp, stop_stamp).resample("10min").mean()
    hw = getWeather(start_stamp, stop_stamp, 0).resample("10min").mean()

    # Join DemGen-data, with weather- and time-data and average over readings
    hist = pd.DataFrame(index=pd.date_range(start=start, end=stop, freq='10min', closed="right"))
    hist = hist.join(demgen).join(hw).ffill().bfill()
    return hist.values

def getPredictionData():
    ts = getLastTimeseries()
    f, fdt = getForecastData()
    if len(ts) != timesteps: raise Exception("Timeseries has " + str(len(ts)) + " values, but should have " + str(timesteps))
    return ts,f,fdt

def cleanData(df, verbose=True):
    df = df.copy()
    dt = timedelta(hours=6)
    total_cleaned = 0
    for z in zone_names:
        b = df[z].sum()
        cleanCol(df,dt,z)
        r = b - df[z].sum()
        total_cleaned += r
        if verbose: print("Cleaned from",z,":",r/6,"hours")
    if verbose: print("Total Cleaned:",total_cleaned/6,"hours")
    return df

def howClean(df):
    print("How clean?")

    def printStats(df_new):
        print("Time with some curtailment in Orkney ANM: {:.2f} hours, which is {:.2f}%, {:.2f}% of full set"
        .format(df_new["Curtailment"].sum()/6,
        df_new["Curtailment"].sum()/len(df_new.index)*100,
        df_new["Curtailment"].sum()/len(df.index)*100))
        print("Time with curtailment in each zone combined: {:.2f} hours, which is {:.2f}%, {:.2f}% of full set"
        .format(df_new[zone_names].sum().sum()/6,
        df_new[zone_names].sum().sum()/(len(df_new.index)*len(zone_names))*100,
        df_new[zone_names].sum().sum()/(len(df.index)*len(zone_names))*100))

    print("----------")
    print("--- Original dataset:")
    printStats(df)
    print("----------")
    print("--- De Minimis on Reduced:")
    printStats(addReducedCol(df, clean=True))
    print("----------")
    print("--- Anomaly Detection per Zone:")
    cleaned = addReducedCol(cleanData(df, verbose=False))
    printStats(cleaned)
    print("----------")
    print("--- Both anomaly detections:")
    printStats(addReducedCol(cleaned, clean=True))
    print("----------")
    print("--- Remove glitch periods:")
    noGlitch = removeGlitches(df, verbose=False)
    printStats(noGlitch)
    print("----------")
    print("--- De Minimis on Reduced:")
    printStats(addReducedCol(noGlitch, clean=True))
    print("----------")
    print("--- Anomaly Detection per Zone:")
    #NOTE: Here we do the anomaly detection before removing the glitch periods,
    # since removing the glitches messes with the temporal aspect of the anomaly detection.
    noGlitch_cleaned = removeGlitches(cleaned, verbose=False)
    printStats(noGlitch_cleaned)
    print("----------")
    print("--- Both anomaly detections:")
    printStats(addReducedCol(noGlitch_cleaned, clean=True))

def cleanCol(df, threshold, col_name):
    c,d,e = False, False, False
    cs, ds = None, None
    for i, r in df.iterrows():
        if not c and r[col_name]:
            c, cs = True, i
        elif c and not r[col_name]:
            c,d = False, False
            if e:
                df.loc[cs:i,col_name] = 0
                e = False
        if c and not d and r["Demand"] > r["Generation"]:
            d, ds = True, i
        elif c and d and r["Demand"] <= r["Generation"]:
            d = False
        if not e and d and i-ds >= threshold:
            e = True

#NOTE: This is not working AT ALL. Use with caution
def estimateWindSpeeds(df):
    genToWind = np.zeros(200)
    print("Calculating wind speeds from generation")
    for i in range(200): genToWind[i] =  max(0,windgencorr(i/10))
    print("Done with lookup table")
    def getSpeed(gen): return np.abs(genToWind - gen).argmin() / 10
    for index, row in df.loc[:bad_weather_date,:].iterrows():
        est_wind = getSpeed(row["Generation"])
        df.loc[index,"speed"] = est_wind
    return df

def toOneHot(val,max):
    a = np.zeros(shape=(max,)).astype(int)
    a[val] = 1
    return a

def getEdayData():
    return pickle.load(open(config.DATA_PATH+"eday/eday-data.pickle", "rb"))
