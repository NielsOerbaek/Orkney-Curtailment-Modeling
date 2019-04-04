import prepros as pp
from datetime import datetime
import pandas as pd
import calendar
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib
import matplotlib.pyplot as plt


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

def windWind(start, stop):
    df_eday = pp.getEdayData()
    df_full = pp.getSingleDataframe(start, stop, fromPickle=True)

    start = datetime.strptime(start, '%Y-%m-%d')
    stop = datetime.strptime(stop, '%Y-%m-%d')

    df_eday = df_eday.loc[start:stop]
    df_full = df_full.loc[start:stop]

    df = df_full.join(df_eday, how="inner")[["speed", "Wind Mean (M/S)"]]

    model = LinearRegression()
    model.fit(df[["speed"]], df[["Wind Mean (M/S)"]])
    preds = model.predict(df[["speed"]])

    y_mean = [df[["Wind Mean (M/S)"]].mean()]*len(df["speed"].values)

    print("Coef and bias:", model.coef_[0][0], model.intercept_[0])
    print("R^2 score:", r2_score(df[["Wind Mean (M/S)"]],preds))
    print("R^2 score of mean:", r2_score(df[["Wind Mean (M/S)"]],y_mean))

    #Plot outputs
    plt.scatter(df[["speed"]], df[["Wind Mean (M/S)"]],  color='black')
    plt.plot(df[["speed"]], preds, color='blue', linewidth=3)
    plt.plot(df[["speed"]], y_mean, color='red', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()



    #for r in df.values:
    #    print("(",r[0],",",r[1],")")
    #    print(r[0],round(r[1],3))



windWind("2019-02-11", "2019-03-01")
#windWind("2018-12-01", "2019-03-01")
#timeDem("2018-01-12", "2019-03-01")
