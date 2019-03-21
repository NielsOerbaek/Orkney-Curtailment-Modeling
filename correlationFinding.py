import prepros as pp
from datetime import datetime
import pandas as pd
import calendar
import numpy as np


def makePlot(data, legend):
    print("\\addplot")
    print("coordinates {")
    for j, v in enumerate(data):
        print("(",j,",",round(v,2),")")
    print("};")
    print("\\addlegendentry{"+legend+"}")


def windGen(start, stop):
    start = datetime.strptime(start, '%Y-%m-%d').timestamp()
    stop = datetime.strptime(stop, '%Y-%m-%d').timestamp()
    if stop <= start: raise ValueError("Invalid date range: "+ str(datetime.fromtimestamp(start)) +" to "+ str(datetime.fromtimestamp(stop)))

    # Get dataframes and resample them.
    demgen_df = pp.getDemandGen(start, stop).resample("60min").mean() # Demand Generation data
    w_df = pp.getWeather(start, stop, 0).resample("60min").mean() # Weather data

    df = demgen_df.join(w_df[["speed"]], how="inner")
    df.dropna(inplace=True) # Remove NaN entries.
    df = pp.addTimeCols(df)

    c1 = 0
    c2 = 0
    #print(df.columns)
    for r in df.values:
        if r[2] <= 11545:
            print("(",r[2],",",round(r[1],2),")")
            #print(r[2],round(r[1],3))
            c2 +=1
        c1+=1

    #print(c2/c1*100)

def timeDem(start, stop):
    start = datetime.strptime(start, '%Y-%m-%d').timestamp()
    stop = datetime.strptime(stop, '%Y-%m-%d').timestamp()
    if stop <= start: raise ValueError("Invalid date range: "+ str(datetime.fromtimestamp(start)) +" to "+ str(datetime.fromtimestamp(stop)))

    # Get dataframes and resample them.
    df = pp.getDemandGen(start, stop).resample("10min").mean() # Demand Generation data
    df = pp.addTimeCols(df)

    #makePlot(df.groupby("weekday").median()["Demand"], "Median")

    if True:
        hours_mean = pd.DataFrame()
        hours_std = pd.DataFrame()
        hour_groups = df.groupby("hour")
        for hour, indices in hour_groups.groups.items():
            #print(hour-1)
            means = df[df.index.isin(indices)].groupby("weekday").median()["Demand"]
            stds = df[df.index.isin(indices)].groupby("weekday").std()["Demand"]
            hours_mean[hour-1] = means
            hours_std[hour-1] = stds

        for i, day in enumerate(hours_mean.values):
            #makePlot(day, calendar.day_abbr[i])
            print("[",", ".join(list(map(lambda x: str(round(x,3)),day))),"],")

    #print(hours_mean)
    #print(hours_std)

    #w_mean = df.groupby("weekday")
    #print(h_mean)
    #print(w_mean)
    #print(df.std(0))

    #c = 0
    #for i,r in zip(df.index, df.values):
    #    #print("(",i-1,",",round(r[0],2),")")
    #    #print(r[2],round(r[1],3))
    #    if r[0] <= 25 and r[0] >= 15:
    #        c += 1

    #print(c/len(df.values)*100)
    #print(len(df.values))





windGen("2019-02-11", "2019-03-01")
#timeDem("2018-01-12", "2019-03-01")
