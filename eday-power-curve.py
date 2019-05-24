import numpy as np
import prepros as pp
from scipy.interpolate import interp1d
from math import ceil
from datetime import datetime
import plotter
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def calculatePowerCurve():
    df = pp.getEdayData()
    df_full = pp.getSingleDataframe(fromPickle=True)

    #df = df_full.join(df,how="inner")
    #df = df[df["Curtailment"] == 0]
    #df = df[((df["Wind Mean (M/S)"] < 4) | (df["Wind Mean (M/S)"] > 22) | (df["Power Mean (Kw)"] > 10)) & ((df["Wind Mean (M/S)"] < 12) | (df["Wind Mean (M/S)"] > 22) | (df["Power Mean (Kw)"] > 700))]

    start = datetime.strptime("2018-12-01", '%Y-%m-%d')
    stop = datetime.strptime("2019-01-01", '%Y-%m-%d')

    df = df[["Wind Mean (M/S)", "Power Mean (Kw)"]].round().groupby("Wind Mean (M/S)").mean()

    for r in df.values:
        print(r[0])


def calculateLoss():
    df = pp.getEdayData()
    df_full = pp.getSingleDataframe(fromPickle=True)
    df = df_full.join(df,how="inner")

    eday_powercurve_discrete = [0,0,0.5,4,19,60,101,160,253,404,532,687,820,870,890,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900]
    eday_powercurve = interp1d(range(0,len(eday_powercurve_discrete)), eday_powercurve_discrete)

    start = datetime.strptime("2018-12-01", '%Y-%m-%d')
    stop = datetime.strptime("2019-01-01", '%Y-%m-%d')
    december_median = df.loc[start:stop][["Wind Mean (M/S)", "Power Mean (Kw)"]].round({"Wind Mean (M/S)": 0}).groupby("Wind Mean (M/S)").median().values[:,0]
    december_powercurve = interp1d(range(0,len(december_median)), december_median, bounds_error=False, fill_value=0)

    power_sum = 0 #716131
    for r in df["Power Mean (Kw)"].values:
        power_sum += r*(1/6)
    print("Power Output: {:.2f} kWh".format(power_sum))

    expected_power_sum = 0
    for r in df["Wind Mean (M/S)"].values:
        expected_power_sum += eday_powercurve(r)*(1/6)
    print("Expected Power Output: {:.2f} kWh".format(expected_power_sum))

    loss = expected_power_sum-power_sum
    print("Loss: {:.2f} kWh ({:.2f}%)".format(loss, loss/expected_power_sum*100))

    expected_power_sum = 0
    for r in df["Wind Mean (M/S)"].values:
        expected_power_sum += december_powercurve(r)*(1/6)
    print("Expected Power Output (new curve): {:.2f} kWh".format(expected_power_sum))

    loss = expected_power_sum-power_sum
    print("Loss (new curve): {:.2f} kWh ({:.2f}%)".format(loss, loss/expected_power_sum*100))

    eday_expected_power_sum = 0
    for r in df["Wind Mean (M/S)"].values:
        eday_expected_power_sum += eday_powercurve(ceil(r))*(1/6)
    print("Expected Power Output (eday method): {:.2f} kWh".format(eday_expected_power_sum))

    eday_loss = eday_expected_power_sum-power_sum
    print("Loss (eday method): {:.2f} kWh ({:.2f}%)".format(eday_loss, eday_loss/eday_expected_power_sum*100))

    print()
    print("------ Excluding measurements with no curtailment in Zone 1")
    df = df[df["Zone 1"] == 1]

    cur_power_sum = 0 #716131
    for r in df["Power Mean (Kw)"].values:
        cur_power_sum += r*(1/6)
    print("Power Output: {:.2f} kWh".format(cur_power_sum))

    cur_expected_power_sum = 0
    for r in df["Wind Mean (M/S)"].values:
        cur_expected_power_sum += eday_powercurve(ceil(r))*(1/6)
    print("Expected Power Output (eday method): {:.2f} kWh".format(cur_expected_power_sum))

    cur_loss = cur_expected_power_sum-cur_power_sum
    print("Loss (eday method): {:.2f} kWh ({:.2f}% locally, {:.2f}% of total)".format(cur_loss,cur_loss/cur_expected_power_sum*100, cur_loss/eday_expected_power_sum*100))

    print()

    print("Difference in loss: {:.2f} / {:.2f} = {:.2f}%".format(cur_loss, eday_loss, cur_loss / eday_loss * 100))

def highWinds():
    df = pp.getEdayData()
    all = len(df)
    high_wind = len(df[df["Wind Mean (M/S)"] > 25])
    print("All data points", all)
    print("Data points with wind over 25 m/s", high_wind)
    print("Which is {:.2f}%".format(high_wind/all*100))

calculateLoss()
#plotPowerCurves()
#orkneyPowerCurves()
#highWinds()
