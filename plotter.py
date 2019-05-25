import sys
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
import pandas as pd
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from matplotlib.lines import Line2D
from math import ceil
from matplotlib import rc
rc('font',**{'family':'serif','serif':['EB Garamond 12 Regular'], "size": 8})
rc('text', usetex=True)
from matplotlib import rcParams
rcParams['grid.linestyle'] = ':'
rcParams['grid.linewidth'] = 0.5
rcParams['axes.grid']=True
rcParams["legend.edgecolor"]="k"
rcParams["legend.fancybox"]=False
rcParams["legend.framealpha"]=0.7
rcParams["legend.borderpad"]=0.5
rcParams['axes.linewidth'] = 0.5 # set the value globally


from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.interpolate import spline

plt.style.use("seaborn-colorblind")

import prepros as pp
import descriptive as desc
import metoffice as met
import config


def buildModelGraph(start_limit=0, stop_limit=0, zones=0, filename="model-comparison", save_to_pdf=False):
    df_eday = pp.getEdayData()

    #Full DataSet, used for training
    try: df_full = pp.getSingleDataframe("2018-12-01", "2019-03-01", fromPickle=True)
    except FileNotFoundError: df_full = pp.getSingleDataframe("2018-12-01", "2019-03-01", fromPickle=False)
    df_full = df_full.join(df_eday, how="inner")
    df_full = pp.cleanData(df_full)
    df_full = pp.addReducedCol(df_full, clean=True)
    df_full = pp.removeGlitches(df_full)

    try: df = pp.getSingleDataframe(start_limit, stop_limit, fromPickle=True)
    except FileNotFoundError: df = pp.getSingleDataframe(start_limit, stop_limit, fromPickle=False)
    #df = df.join(df_eday, how="inner")
    df = pp.cleanData(df)
    df = pp.addReducedCol(df, clean=True)
    df = pp.removeGlitches(df)

    if start_limit != 0: start_limit = datetime.strptime(start_limit, '%Y-%m-%d').timestamp()
    if stop_limit != 0: stop_limit = datetime.strptime(stop_limit, '%Y-%m-%d').timestamp()

    # Adjust the amount of ticks to the data size
    if stop_limit - start_limit > 86400*8: tick_zoom = 24
    elif stop_limit - start_limit > 86400*4: tick_zoom = 12
    elif stop_limit - start_limit > 86400*2: tick_zoom = 6
    elif stop_limit - start_limit > 86400: tick_zoom = 3
    else: tick_zoom = 1

    model_names, accs = desc.evaluateDataframe(df_full, df)

    accs = accs[:,:-1]

    # Generate x ticks for the mesh plot
    meshxticks_major = []
    meshxticks_minor = []
    for i,d in enumerate(df.index):
        if d.hour == 0 and d.minute == 0: meshxticks_major.append(i)
        elif d.hour % tick_zoom == 0 and d.minute == 0: meshxticks_minor.append(i)

    plt.xticks(rotation=-60)

    fig = plt.figure()
    # Bottom plot
    ax1 = fig.add_axes([0.10,0.1,0.9,0.44])
    delta = (df["Generation"]-df["Demand"])#.rolling(3).mean()
    ax1.plot(df.index, delta,"k-", linewidth=1, alpha=0.8)
    plt.fill_between(df.index, delta, color="k", alpha=0.3)
    ax1.margins(x=0)
    ax1.set_ylabel("MegaWatt")
    ax1.set_ylim(-25, 25)
    ax1.set_yticks([-20,-10,0,10,20])
    ax1.grid(b=True, which="both", axis="y")
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter("%H:00"))
    for i,t in enumerate(ax1.xaxis.get_minor_ticks()):
        if i % 24 == 0: t.label.set_visible(False)
        if i % tick_zoom != 0: t.set_visible(False)
    ax1.tick_params(axis="x", which="minor")
    ax1.grid(b=True, which="major", axis="x", linestyle="-.")
    ax1.grid(b=True, which="minor", axis="x", linestyle="--")
    ax1.legend(["Generation relative to Demand"], loc=1)

    # Top plot
    cm = plt.get_cmap("binary")
    ax2 = fig.add_axes([0.10,0.56,0.9,0.44])
    ax2.pcolormesh(accs, alpha=1, cmap=cm, snap=True)
    ax2.set_xticks(meshxticks_major)
    ax2.set_xticks(meshxticks_minor, minor=True)
    ax2.xaxis.set_ticklabels([])
    ax2.grid(b=True, which="major", axis="x", linestyle="-.")
    ax2.grid(b=True, which="minor", axis="x", linestyle="--")
    ax2.set_yticks(np.arange(len(model_names))+0.5)
    ax2.set_yticks(np.arange(len(model_names)), minor=True)
    ax2.set_yticklabels(model_names, rotation=0, fontsize="8", va="center")
    ax2.grid(b=True, which="minor", axis="y")
    custom_lines = [Line2D([0], [0], color=cm(0), lw=4),
        Line2D([0], [0], color=cm(1.), lw=4)]
    ax2.legend(custom_lines, ["No curtailment", "Curtailment"], loc=1)
    #plt.title("Generation relative to demand for all of Orkney. \nAccuracies for models: " + ", ".join(model_names))

    fig.autofmt_xdate(which="both")

    fig.set_size_inches(8, 3)

    if save_to_pdf: fig.savefig("./plots/"+filename+".pdf")
    else: plt.show()
    plt.clf()

def buildDeltaZoneGraph(start_limit=0, stop_limit=0, zones=0, clean=False, save_to_pdf=False):
    zone_names = ["Curtailment","Core Zone", "Zone 1", "Zone 1A", "Zone 2", "Zone 2A", "Zone 2B", "Zone 3", "Zone 4", "Zone 4A"]

    file_name = "delta-zone-"+start_limit+"-"+stop_limit
    if clean: file_name += "-cleaned"

    df = pp.getSingleDataframe(start_limit, stop_limit, fromPickle=True, clean=clean, cleanGlitches=False)

    if start_limit != 0: start_limit = datetime.strptime(start_limit, '%Y-%m-%d').timestamp()
    if stop_limit != 0: stop_limit = datetime.strptime(stop_limit, '%Y-%m-%d').timestamp()

    # Adjust the amount of ticks to the data size
    if stop_limit - start_limit > 86400*8: tick_zoom = 24
    elif stop_limit - start_limit > 86400*4: tick_zoom = 12
    elif stop_limit - start_limit > 86400*2: tick_zoom = 6
    elif stop_limit - start_limit > 86400: tick_zoom = 3
    else: tick_zoom = 1

    if zones != 0: zone_names = zones

    # Generate x,y data for the mesh plot
    curtailments = np.zeros(shape=(len(zone_names),len(df.index)))
    for i, zone in enumerate(zone_names):
        for j, status in enumerate(df[zone]):
            curtailments[i,j] = status

    curtailments = curtailments[:,:-1]

    # Generate x ticks for the mesh plot
    meshxticks_major = []
    meshxticks_minor = []
    for i,d in enumerate(df.index):
        if d.hour == 0 and d.minute == 0: meshxticks_major.append(i)
        elif d.hour % tick_zoom == 0 and d.minute == 0: meshxticks_minor.append(i)


    fig = plt.figure()
    # Bottom plot
    ax1 = fig.add_axes([0.1,0.08,0.9,0.45])
    delta = (df["Generation"]-df["Demand"])#.rolling(3).mean()
    ax1.plot(df.index, delta,"k-", linewidth=1, alpha=0.8)
    plt.fill_between(df.index, delta, color="k", alpha=0.3)
    ax1.set_xlabel("Time")
    ax1.margins(x=0)
    ax1.set_ylabel("MegaWatt")
    ax1.set_ylim(-25, 25)
    ax1.set_yticks([-25,-20,-15,-10,-5,0,5,10,15,20,25])
    ax1.grid(b=True, which="both", axis="y")
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter("%H:00"))
    for i,t in enumerate(ax1.xaxis.get_minor_ticks()):
        if i % 24 == 0: t.label.set_visible(False)
        if i % tick_zoom != 0: t.set_visible(False)
    ax1.tick_params(axis="x", which="minor")
    ax1.grid(b=True, which="major", axis="x", linestyle="-.")
    ax1.grid(b=True, which="minor", axis="x", linestyle="--")
    ax1.legend(["Generation relative to Demand"], loc=1, fancybox=True, framealpha=0.5)

    # Top plot
    cm = plt.get_cmap("OrRd")
    ax2 = fig.add_axes([0.1,0.55,0.9,0.45])
    ax2.pcolormesh(curtailments, alpha=1, cmap=cm, snap=True)
    ax2.set_xticks(meshxticks_major)
    ax2.set_xticks(meshxticks_minor, minor=True)
    ax2.xaxis.set_ticklabels([])
    ax2.grid(b=True, which="major", axis="x", linestyle="-.")
    ax2.grid(b=True, which="minor", axis="x", linestyle="--")
    #ax2.set_ylabel("Zones")
    ax2.set_yticks(np.arange(len(zone_names))+0.5)
    ax2.set_yticks(np.arange(len(zone_names)), minor=True)
    ax2.set_yticklabels(zone_names, rotation=0, va="center")
    ax2.grid(b=True, which="minor", axis="y")
    custom_lines = [Line2D([0], [0], color=cm(0), lw=4),
        Line2D([0], [0], color=cm(.5), lw=4),
        Line2D([0], [0], color=cm(1.), lw=4)]
    ax2.legend(custom_lines, ["No curtailment in zone","Partial curtailment in zone", "Full stop in zone"], loc=1, fancybox=True, framealpha=0.5)

    fig.autofmt_xdate(which="both")

    fig.set_size_inches(8, 4.5)
    plt.xticks(rotation=-60)

    if save_to_pdf: fig.savefig("./plots/"+file_name+".pdf")
    else: plt.show()
    plt.clf()

def buildFirmNotFirmGraph(start_limit=0, stop_limit=0, zones=0, clean=False, save_to_pdf=False):
    zone_names = ["Core Zone", "Zone 1", "Zone 1A", "Zone 2", "Zone 2A", "Zone 2B", "Zone 3", "Zone 4", "Zone 4A"]

    file_name = "firm-not-firm-"+start_limit+"-"+stop_limit
    if clean: file_name += "-cleaned"

    try: df = pp.getSingleDataframe(start_limit, stop_limit, fromPickle=True, clean=clean, cleanGlitches=False)
    except FileNotFoundError: df = pp.getSingleDataframe(start_limit, stop_limit, fromPickle=False, clean=clean, cleanGlitches=False)

    if start_limit != 0: start_limit = datetime.strptime(start_limit, '%Y-%m-%d').timestamp()
    if stop_limit != 0: stop_limit = datetime.strptime(stop_limit, '%Y-%m-%d').timestamp()

    # Adjust the amount of ticks to the data size
    if stop_limit - start_limit > 86400*8: tick_zoom = 24
    elif stop_limit - start_limit > 86400*4: tick_zoom = 12
    elif stop_limit - start_limit > 86400*2: tick_zoom = 6
    elif stop_limit - start_limit > 86400: tick_zoom = 3
    else: tick_zoom = 1

    if zones != 0: zone_names = zones

    # Generate x,y data for the mesh plot
    curtailments = np.zeros(shape=(len(zone_names),len(df.index)))
    for i, zone in enumerate(zone_names):
        for j, status in enumerate(df[zone]):
            curtailments[i,j] = status

    curtailments = curtailments[:,:-1]

    # Generate x ticks for the mesh plot
    meshxticks_major = []
    meshxticks_minor = []
    for i,d in enumerate(df.index):
        if d.hour == 0 and d.minute == 0: meshxticks_major.append(i)
        elif d.hour % tick_zoom == 0 and d.minute == 0: meshxticks_minor.append(i)


    fig = plt.figure()
    # Bottom plot
    ax1 = fig.add_axes([0.05,0.18,0.9,0.40])
    ax1.plot(df.index, df["ANM Generation"],"k-", linewidth=1, alpha=0.8)
    ax1.plot(df.index, df["Non-ANM Generation"],"b--", linewidth=1, alpha=0.8)
    #ax1.set_xlabel("Time")
    ax1.margins(x=0)
    #ax1.set_ylabel("MegaWatt")
    ax1.grid(b=True, which="both", axis="y")
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter("%H:00"))
    for i,t in enumerate(ax1.xaxis.get_minor_ticks()):
        if i % 24 == 0: t.label.set_visible(False)
        if i % tick_zoom != 0: t.set_visible(False)
    ax1.tick_params(axis="x", which="minor")
    ax1.grid(b=True, which="major", axis="x", linestyle="-.")
    ax1.grid(b=True, which="minor", axis="x", linestyle="--")
    ax1.legend(["ANM Generation","Non-ANM Generation"], loc='upper center', bbox_to_anchor= (0.5, -0.20), ncol=2)

    #plt.xticks(rotation=-20)
    # Top plot
    cm = plt.get_cmap("OrRd")
    ax2 = fig.add_axes([0.05,0.60,0.90,0.40])
    ax2.pcolormesh(curtailments, alpha=1, cmap=cm, snap=True)
    ax2.set_xticks(meshxticks_major)
    ax2.set_xticks(meshxticks_minor, minor=True)
    ax2.xaxis.set_ticklabels([])
    ax2.grid(b=True, which="major", axis="x", linestyle="-.")
    ax2.grid(b=True, which="minor", axis="x", linestyle="--")
    #ax2.set_ylabel("Zones")
    ax2.set_yticks(np.arange(len(zone_names))+0.5)
    ax2.set_yticks(np.arange(len(zone_names)), minor=True)
    ax2.set_yticklabels(["C", "1", "1A", "2", "2A", "2B", "3", "4", "4A"], rotation=0, va="center", fontsize="8")
    ax2.grid(b=True, which="minor", axis="y")
    custom_lines = [Line2D([0], [0], color=cm(0), lw=4),
        Line2D([0], [0], color=cm(.5), lw=4),
        Line2D([0], [0], color=cm(1.), lw=4)]
    #ax2.legend(custom_lines, ["No curtailment in zone","Partial curtailment in zone", "Full stop in zone"], loc=1, fancybox=True, framealpha=0.5)

    fig.autofmt_xdate(which="both")

    fig.set_size_inches(4.9, 3)

    if save_to_pdf: fig.savefig("./plots/"+file_name+".pdf")
    else: plt.show()
    plt.clf()


def buildWindsGraph(start_limit=0, stop_limit=0, zones=0):
    df = pp.getSingleDataframe(start_limit, stop_limit, fromPickle=True).resample("10min").mean().interpolate(method='linear')

    if start_limit != 0: start_limit = datetime.strptime(start_limit, '%Y-%m-%d')
    if stop_limit != 0: stop_limit = datetime.strptime(stop_limit, '%Y-%m-%d')

    df_eday = pp.getEdayData()
    df_eday = df_eday.loc[start_limit:stop_limit]

    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.15,0.85,0.8])
    ax1.plot(df.index, df["speed"],"k-", linewidth=1, alpha=0.8)
    ax1.plot(df_eday.index, df_eday["Wind Mean (M/S)"],"b-", linewidth=1, alpha=0.8)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("M/S")
    ax1.grid(b=True, which="both", axis="y")
    ax1.tick_params(axis="x", which="minor")
    ax1.grid(b=True, which="major", axis="x", linestyle="-.")
    ax1.grid(b=True, which="minor", axis="x", linestyle="--")
    ax1.legend(["OpenWeatherMap", "Eday Turbine"], loc=1)

    plt.title("Wind speed comparison")

    fig.autofmt_xdate(which="both")

    fig.set_size_inches(15, 8)
    plt.xticks(rotation=-60)

    plt.show()
    plt.clf()

def buildEdayScatter(start_limit=0, stop_limit=0, zones=0, save_to_pdf=False,
                        filename="eday-scatter", curtail_code=0, color="k",
                        width=4, powercurve=False):
    if start_limit != 0: start_limit = datetime.strptime(start_limit, '%Y-%m-%d')
    if stop_limit != 0: stop_limit = datetime.strptime(stop_limit, '%Y-%m-%d')

    df = pp.getEdayData()
    df = df.loc[start_limit:stop_limit][["Wind Mean (M/S)","Power Mean (Kw)"]]

    df_full = pp.getSingleDataframe(fromPickle=True)
    df = df_full.join(df,how="inner")
    df = pp.addReducedCol(df, clean=True)
    if curtail_code == 1: df = df[df["Zone 1"] == 1]
    elif curtail_code == 2: df = df[df["Zone 1"] == 0]
    elif curtail_code == 2: df = df[df["Curtailment"] == 1]

    eday_curve = [0,0,0.5,4,19,60,101,160,253,404,532,687,820,870,890,900,900,900,900,900,900,900,900,900,900,900]

    plt.scatter(df["Wind Mean (M/S)"], df["Power Mean (Kw)"], c=color, alpha=0.5,s=2)
    if powercurve: plt.plot(eday_curve, "r-")
    plt.xlabel("Wind Speed (M/S)")
    plt.xlim(0, 40)
    plt.xticks([0,5,10,15,20,25,30,35,40])
    plt.ylabel("Power Generated (kW)")
    plt.yticks([0,100,200,300,400,500,600,700,800,900])
    if powercurve: plt.legend(["Power curve"], loc=1)
    #plt.title("Relation between windspeeds and generation for Eday 900kW Turbine")
    fig = plt.gcf()
    fig.set_size_inches(width, width*0.75)
    fig.tight_layout()
    if save_to_pdf: fig.savefig("./plots/eday/"+filename+".pdf")
    else: plt.show()
    plt.clf()

def buildEdayWindOrkneyGenScatter(start_limit=0, stop_limit=0, zones=0, save_to_pdf=False, filename="eday-scatter", curtail_code=0, color="k", wind_limit=40):
    if start_limit != 0: start_limit = datetime.strptime(start_limit, '%Y-%m-%d')
    if stop_limit != 0: stop_limit = datetime.strptime(stop_limit, '%Y-%m-%d')

    df = pp.getEdayData()
    df_full = pp.getSingleDataframe(fromPickle=True, clean=True)
    df = df_full.join(df,how="inner")
    df = df[df["Wind Mean (M/S)"] < wind_limit]
    if curtail_code == 1: df = df[df["Zone 1"] == 1]
    elif curtail_code == 2: df = df[df["Zone 1"] == 0]
    elif curtail_code == 3: df = df[df["Curtailment"] == 1]
    elif curtail_code == 4: df = df[df["Curtailment"] == 0]

    full_mean = df[["Wind Mean (M/S)", "Generation"]].round().groupby("Wind Mean (M/S)").median().values[:,0]
    powercurve = interp1d(range(0,len(full_mean)), full_mean, fill_value="extrapolate")
    r2 = r2_score(df[["Generation"]],df[["Wind Mean (M/S)"]].apply(powercurve))

    print(full_mean)
    print("R^2 score:", r2)

    plt.scatter(df["Wind Mean (M/S)"], df["Generation"], c=color, alpha=0.5, s=2, marker="x")
    plt.plot(full_mean, "bx-", markersize=4, linewidth=1)
    plt.xlabel("Wind Speed from ERE Turbine (M/S)")
    plt.xlim(0, 40)
    plt.xticks([0,5,10,15,20,25,30,35,40])
    plt.ylim(0, 40)
    plt.ylabel("Power Generated in Orkney (MW)")
    plt.legend(["Estimated Power Curve","Data"])
    fig = plt.gcf()
    fig.set_size_inches(3.9, 3.2)
    fig.tight_layout()
    #plt.title("Relation between windspeeds and generation for Eday 900kW Turbine")
    if save_to_pdf: fig.savefig("./plots/"+filename+".pdf")
    else: plt.show()
    plt.clf()

def buildWindWindScatter(start, stop, filename="wind-wind-scatter", save_to_pdf=False):
    df_eday = pp.getEdayData()
    df_full = pp.getSingleDataframe(start, stop, fromPickle=True)

    start = datetime.strptime(start, '%Y-%m-%d')
    stop = datetime.strptime(stop, '%Y-%m-%d')

    df_eday = df_eday.loc[start:stop]
    df_full = df_full.loc[start:stop]

    df = df_full.join(df_eday, how="inner")[["speed", "Wind Mean (M/S)"]]
    df = pp.removeGlitches(df)

    model = LinearRegression()
    model.fit(df[["speed"]], df[["Wind Mean (M/S)"]])
    preds = model.predict(df[["speed"]])

    coef = str(round(model.coef_[0][0],3))
    bias = str(round(model.intercept_[0],3))

    print("Coef and bias:", coef, bias)
    print("R^2 score:", r2_score(df[["Wind Mean (M/S)"]],preds))

    #Plot outputs
    plt.scatter(df[["speed"]], df[["Wind Mean (M/S)"]], color='black',alpha=0.5, s=2, marker="x")
    plt.plot(df[["speed"]], preds, color='blue', linewidth=1)
    plt.xlabel("Wind Speed from OpenWeatherMap")
    plt.ylabel("Wind Speed from ERE Turbine (M/S)")
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.legend(["$y = "+coef+"x + "+bias+"$","Data"])
    fig = plt.gcf()
    fig.set_size_inches(3.3, 3)
    fig.tight_layout()
    #plt.title("Relation between windspeeds and generation for Eday 900kW Turbine")
    if save_to_pdf: fig.savefig("./plots/"+filename+".pdf")
    else: plt.show()
    plt.clf()


def plotPowerCurves():
    df = pp.getEdayData()

    start = datetime.strptime("2018-12-01", '%Y-%m-%d')
    stop = datetime.strptime("2019-01-01", '%Y-%m-%d')

    full_median = df[["Wind Mean (M/S)", "Power Mean (Kw)"]].round().groupby("Wind Mean (M/S)").median()
    december_median = df.loc[start:stop][["Wind Mean (M/S)", "Power Mean (Kw)"]].round({"Wind Mean (M/S)": 0}).groupby("Wind Mean (M/S)").median().values[:,0]
    enercon_curve = [0.0,0,1,6,18,42,83,147,238,340,466,600,710,790,850,880,900,900,900,900,900,900,900,900,900,900]
    eday_curve = [0,0,0.5,4,19,60,101,160,253,404,532,687,820,870,890,900,900,900,900,900,900,900,900,900,900,900]


    plt.plot(enercon_curve, "s:", alpha=0.8, markersize=3)
    plt.plot(eday_curve, "x:", alpha=0.8, markersize=3)
    plt.plot(full_median, "o:", alpha=0.8, markersize=3)
    #plt.plot(full_median, "^-", alpha=0.8, markersize=3)
    plt.plot(december_median, "v:", alpha=0.8, markersize=3)
    plt.xlabel("Wind Speed (M/S)")
    plt.xlim(0, 40)
    plt.xticks([0,5,10,15,20,25,30,35,40])
    plt.ylabel("Power Generated (kW)")
    plt.yticks([0,100,200,300,400,500,600,700,800,900])
    plt.legend(["Enercon", "ERE", "Winter 2018/19", "December 2018"], loc=8)
    fig = plt.gcf()
    fig.set_size_inches(4.9, 3)
    fig.tight_layout()
    fig.savefig('./plots/eday_power_curves.pgf')
    plt.clf()

def glitchPlot(start, finish, filename):
    df = pp.getSingleDataframe(start, finish, fromPickle=True)
    rc('font',**{'family':'serif','serif':['EB Garamond 12 Regular'], "size": 10})

    fig = plt.figure()
    wind_ax = fig.add_axes([0.11,0.1,0.88,0.28])
    wind_ax.plot(df["speed"], "b-", linewidth=1)
    wind_ax.set_ylabel("Wind speed [m/s]")
    wind_ax.set_ylim(0, 20)
    plt.setp( wind_ax.xaxis.get_majorticklabels(), rotation=-60 )

    dem_ax = fig.add_axes([0.11,0.41,0.88,0.28])
    dem_ax.plot(df["Demand"], "r-", linewidth=1)
    dem_ax.set_xticklabels([])
    dem_ax.set_ylim(0, 40)
    dem_ax.set_ylabel("Demand [MW]")

    gen_ax = fig.add_axes([0.11,0.71,0.88,0.28])
    gen_ax.plot(df["Generation"], "g-", linewidth=1)
    gen_ax.set_xticklabels([])
    gen_ax.set_ylim(0, 40)
    gen_ax.set_ylabel("Generation [MW]")

    fig = plt.gcf()
    fig.set_size_inches(4.9, 7)
    fig.autofmt_xdate(which="both")
    #plt.show()
    fig.savefig("./plots/exceptions/"+filename+".pdf")
    del fig
    plt.clf()

def plotKModels(data, filename):
    index = data[0]
    values = data[1]
    names = data[2]
    styles = data[3]

    plt.xlim(-20,20)
    plt.ylim(0,100)

    plt.grid(True, axis="both")

    plt.xlabel("Values of $k$")
    plt.ylabel("Accuracy (\%)")

    for i,v in enumerate(values):
        plt.plot(index,v,styles[i], linewidth=1)

    plt.legend(names, loc="best")

    fig = plt.gcf()
    fig.set_size_inches(3.5, 3)
    fig.tight_layout()
    fig.savefig('./plots/'+filename+'.pgf')
    plt.clf()

def buildWindGenScatter(save_to_pdf=False, filename="wind-gen-scatter", api_only=False):

    df = pp.getSingleDataframe(fromPickle=True)
    if api_only: df = df.loc[datetime.strptime("2019-02-12", '%Y-%m-%d'):datetime.strptime("2019-03-01", '%Y-%m-%d')]

    powercurve = df[["speed", "Generation"]].round({"speed": 0}).groupby("speed").median().values[:,0]

    plt.scatter(df["speed"], df["Generation"], c="b", alpha=0.5, s=2, marker="x")
    plt.plot(powercurve, "r-")
    plt.xlabel("Wind Speed (M/S)")
    plt.xticks([0,5,10,15,20])
    plt.ylabel("Generation [MW]")
    plt.yticks([0,5,10,15,20,25,30,35,40])
    plt.legend(["Power curve", "Data"])
    fig = plt.gcf()
    fig.set_size_inches(3.5, 3)
    fig.tight_layout()
    if save_to_pdf: fig.savefig("./plots/"+filename+".pdf")
    else: plt.show()
    plt.clf()

def buildTimeDemScatter():
    df = pp.getSingleDataframe(fromPickle=True)
    df["hour"] -= 1
    mean = df[["Demand", "hour"]].groupby("hour").mean().values[:,0]

    plt.scatter(df["hour"], df["Demand"], c="b", alpha=0.5, s=2, marker="x")
    plt.plot(mean, "rs-", markersize=4)
    plt.xlabel("Time of Day [hours]")
    plt.xlim(-1,24)
    plt.xticks([0,6,12,18,24],["0:00","6:00","12:00","18:00","24:00"])
    plt.ylabel("Demand [MW]")
    plt.ylim(0,40)
    plt.legend(["Mean", "Data"], framealpha=0.8)
    fig = plt.gcf()
    fig.set_size_inches(3.5, 3)
    fig.tight_layout()
    fig.savefig("./plots/time-dem-plot.pdf")
    plt.clf()

def buildWeekdayHourPlot():
    df = pp.getSingleDataframe(fromPickle=True)
    df["hour"] -= 1
    weekday_groups = df.groupby("weekday")

    plots = []
    for day, indices in weekday_groups.groups.items():
        means = df[df.index.isin(indices)].groupby("hour").mean()["Demand"]
        x_smooth = np.linspace(0, 23, 200)
        y_smooth = spline(list(range(0,24)), means, x_smooth)
        plots.append(means)

    plt.plot(plots[0], "o-", linewidth=1, markersize=2)
    plt.plot(plots[1], "o-", linewidth=1, markersize=2)
    plt.plot(plots[2], "o-", linewidth=1, markersize=2)
    plt.plot(plots[3], "o-", linewidth=1, markersize=2)
    plt.plot(plots[4], "o-", linewidth=1, markersize=2)
    plt.plot(plots[5], "o:", linewidth=1, markersize=2)
    plt.plot(plots[6], "o:", linewidth=1, markersize=2)

    plt.xlabel("Time of Day [hours]")
    plt.xticks([0,6,12,18,24],["0:00","6:00","12:00","18:00","24:00"])
    plt.ylabel("Demand [MW]")
    plt.ylim(14,24)
    plt.legend(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], framealpha=0.8, loc='center left', bbox_to_anchor=(1, 0.5))
    fig = plt.gcf()
    fig.set_size_inches(4.9, 3)
    fig.tight_layout(rect=(0,0,0.97,1))
    fig.savefig("./plots/weekday-hour-plot.pdf")
    plt.clf()

def networkBoxplot():
    data = pd.read_csv(config.DATA_PATH+"network-stats.csv")
    data.boxplot()
    #plt.ylim(0,100)
    plt.xticks(rotation=-30)
    fig = plt.gcf()
    fig.set_size_inches(2, 2.8)
    fig.tight_layout(rect=(0,0,1,1))
    fig.savefig("./plots/networkBoxplot.pdf")
    plt.clf()

def buildTempDemScatter(start, stop, filename="temp-dem-scatter", save_to_pdf=False):
    df = pp.getSingleDataframe(start, stop, fromPickle=True, clean=True)

    df = pp.removeGlitches(df)
    df = df.round({'temp': 0})

    model = LinearRegression()
    model.fit(df[["temp"]], df[["Demand"]])
    preds = model.predict(df[["temp"]])

    coef = str(round(model.coef_[0][0],3))
    bias = str(round(model.intercept_[0],3))

    print("Coef and bias:", coef, bias)
    print("R^2 score:", r2_score(df[["Demand"]],preds))

    means = df.groupby("temp").mean()["Demand"]

    #Plot outputs
    plt.scatter(df[["temp"]], df[["Demand"]], color='black',alpha=0.5, s=2, marker="x")
    #plt.plot(df[["temp"]], preds, color='blue', linewidth=1)
    plt.plot(means, color='blue', linewidth=1)
    plt.xlabel("Temperature [C]")
    plt.ylabel("Demand [MW]")
    plt.xlim(-10,15)
    plt.legend(["Data"])
    #plt.legend(["$y = "+coef+"x + "+bias+"$","Data"])
    fig = plt.gcf()
    fig.set_size_inches(3.3, 3)
    fig.tight_layout()
    #plt.title("Relation between windspeeds and generation for Eday 900kW Turbine")
    if save_to_pdf: fig.savefig("./plots/"+filename+".pdf")
    else: plt.show()
    plt.clf()

def buildMetWindWindScatter(start="2019-03-01", stop="2019-05-01", filename="met-wind-wind-scatter", save_to_pdf=False):

    met_df = pp.getMetData(start, stop).set_index("forecast_time")
    anm_df = pp.getSingleDataframe(start, stop, fromPickle=True)

    df = anm_df.join(met_df, how="inner")

    x = "speed"
    y = "wind_speed"

    model = LinearRegression()
    model.fit(df[[x]], df[[y]])
    preds = model.predict(df[[x]])

    coef = str(round(model.coef_[0][0],3))
    bias = str(round(model.intercept_[0],3))

    print("Coef and bias:", coef, bias)
    print("R^2 score:", r2_score(df[[y]],preds))

    #Plot outputs
    plt.scatter(df[[x]], df[[y]], color='black',alpha=0.5, s=2, marker="x")
    plt.plot(df[[x]], preds, color='blue', linewidth=1)
    plt.ylabel("Wind Speed from Met Office")
    plt.xlabel("Wind Speed from OpenWeatherMap")
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.legend(["$y = "+coef+"x + "+bias+"$","Data"])
    fig = plt.gcf()
    fig.set_size_inches(3.3, 3)
    fig.tight_layout()
    #plt.title("Relation between windspeeds and generation for Eday 900kW Turbine")
    if save_to_pdf: fig.savefig("./plots/"+filename+".pdf")
    else: plt.show()
    plt.clf()


def metForecastPlot(start="2019-04-01",stop="2019-05-01",smooth=True,fromPickle=False,show=False,name="met-full-frame"):

    linestyles = ["-","-.","--",":"]

    df = met.getFullCombinedMetFrame()

    names = ["$WT_6$",
    "$WT_6$ - ERE power curve",
    "WT-Percep - ERE wind data",
    "WT-FFNN - ERE wind data",
    "WT-Percep",
    "WT-FFNN"]

    cols = ["prediction_correct",
    "ere_prediction_correct",
    "ere_percep_prediction_correct",
    "ere_wtnn_prediction_correct",
    "percep_prediction_correct",
    "wtnn_prediction_correct"]

    filename = "met-office-accs"
    if not smooth: filename += "-raw"

    hours = df.groupby("hours_forecast").mean()
    hours.index = [int(td.days*24+td.seconds/3600) for td in hours.index]
    hours = hours.iloc[:96]

    if smooth:
        cols.pop(4)
        cols.pop(4)

    for i,a in enumerate(cols):
        if smooth: hours[a] = hours[a].rolling(6, min_periods=1, center=True).mean()
        plt.plot(hours.index, hours[a], linewidth=1, linestyle=linestyles[i%4])

    plt.xlabel("Hours-ahead forecast")
    plt.xticks([0,12,24,36,48,60,72,84,96])
    plt.xlim(0,96)
    plt.ylabel("Accuracy [\%]")

    fig = plt.gcf()
    if smooth:
        plt.legend(names, loc='upper center', bbox_to_anchor= (0.5, -0.25))
        fig.set_size_inches(4.9, 3.3)
        fig.tight_layout(rect=(0,-0.05,1,1))
    else:
        plt.legend(names)
        fig.set_size_inches(4.9, 8)
        fig.tight_layout()
    if not show: fig.savefig("./plots/"+filename+".pdf")
    else: plt.show()
    del fig
    plt.clf()

def certaintyPlot(name=None,model="ere_percep",show=False):
    #df = met.ANNCertainty(fromPickle=False, clean=True, load_model=True)
    if name != None:
        df = pickle.load(open(config.DATA_PATH+""+name,"rb"))
    else:
        df = met.getFullCombinedMetFrame()

    df = df.loc[df["hours_forecast"] <= timedelta(days=4)]

    num_samples = df["hour"].count()
    num_predicted = df.loc[df[model+"_prediction"] > 0.5]["hour"].count()
    num_actual = df.loc[df["Curtailment"] == 1]["hour"].count()

    print("Total accuracy: {:.2f}%".format(df[model+"_prediction_correct"].mean()))
    print(df[model+"_prediction_correct"].describe())

    print("Curtailment ratio: {:.2f}%".format(num_actual/num_samples*100))
    print("Predicted curtailment ratio: {:.2f}%".format(num_predicted/num_samples*100))

    df_only_predicted = df.loc[df[model+"_prediction"] > 0.5]
    print("Hits on predictions: {:.2f}%".format(df_only_predicted[model+"_prediction_correct"].mean()))

    x = np.arange(0,1.05,0.05)

    bins = pd.cut(df[model+"_prediction"], x)
    counts_all = df.groupby(bins).count()[model+"_prediction"]

    df_only_curtailed = df.loc[df["Curtailment"] == round(df[model+"_prediction"])]
    bins_only_curtailed = pd.cut(df_only_curtailed[model+"_prediction"], x)
    counts = df_only_curtailed.groupby(bins_only_curtailed).count()[model+"_prediction"]

    x2 = x[:-1]

    chance_of_curtailment = df.groupby(bins).mean()["Curtailment"]
    fig, ax1 = plt.subplots()

    ax1.bar(x2-0.015,chance_of_curtailment.values*100, width=0.015, color=(0, 0, 0, 0.5), edgecolor="k", align="edge")
    ax1.set_xticks(x-0.025, minor=False)
    ax1.set_xticklabels(x.round(2), minor=False)
    for tick in ax1.get_xticklabels(): tick.set_rotation(90)
    ax1.set_ylabel("Probability [\%]")
    ax1.set_xlabel("Model output")
    ax1.grid(False, axis="x")

    ax2 = ax1.twinx()
    ax2.bar(x2,counts_all.values, width=0.015, color="w", edgecolor="k", align="edge")
    ax2.set_ylabel("Number of Samples")
    ax2.grid(False)

    fig.set_size_inches(4.9, 3.3)
    fig.tight_layout()

    fig.legend(["Probability of Curtailment","Number of Samples"], bbox_to_anchor=(0.1, 0.96), loc='upper left')
    if not show: fig.savefig("./plots/certainty-"+model+".pdf")
    else: plt.show()
    plt.clf()
    del fig


def buildAll():

    buildDeltaZoneGraph("2019-02-11", "2019-03-01", clean=False, save_to_pdf=True)
    buildDeltaZoneGraph("2019-01-14", "2019-01-21", clean=False, save_to_pdf=True)
    buildDeltaZoneGraph("2019-02-11", "2019-03-01", clean=True, save_to_pdf=True)
    buildDeltaZoneGraph("2019-03-03", "2019-03-05", clean=False, save_to_pdf=True)

    plotKModels(desc.evaluateModels("2018-12-01", "2019-03-01", clean=False, onlySCk=True), filename="sck-plot")
    plotKModels(desc.evaluateModels("2018-12-01", "2019-03-01", clean=False), filename="cck-plot")
    plotKModels(desc.evaluateModels("2018-12-01", "2019-03-01", clean=True), filename="clean-cck-plot")

    buildTempDemScatter("2018-12-01", "2019-03-01",save_to_pdf=True)
    buildTempDemScatter("2019-02-11", "2019-03-01",save_to_pdf=True, filename="temp-dem-scatter-only-api")


    buildModelGraph("2019-01-01", "2019-01-15", filename="models-january", save_to_pdf=True)
    buildModelGraph("2019-02-22", "2019-03-01", filename="models-february", save_to_pdf=True)
    buildModelGraph("2019-03-01", "2019-03-11", filename="models-march", save_to_pdf=True)
    buildTimeDemScatter()

    networkBoxplot()
    buildEdayScatter("2018-12-01", "2019-03-01", save_to_pdf=True, filename="eday-all-data", powercurve=False)
    buildEdayScatter("2018-12-01", "2019-03-01", save_to_pdf=True, filename="eday-with-curtailment-in-zone-1", curtail_code=1, color="b", width=3.5)
    buildEdayScatter("2018-12-01", "2019-03-01", save_to_pdf=True, filename="eday-without-curtailment-in-zone-1", curtail_code=2, color="b", width=3.5)
    buildEdayScatter("2018-12-01", "2019-01-01", save_to_pdf=True, filename="eday-december", color="r", width=3)
    buildEdayScatter("2019-01-01", "2019-02-01", save_to_pdf=True, filename="eday-january", color="r", width=3)
    buildEdayScatter("2019-02-01", "2019-03-01", save_to_pdf=True, filename="eday-february", color="r", width=3)

    buildEdayWindOrkneyGenScatter("2018-12-01", "2019-03-01", color="r", save_to_pdf=True, filename="eday-wind-orkney-gen")
    buildWindWindScatter("2019-02-11", "2019-03-01", save_to_pdf=True)


    glitchPlot("2018-12-14","2018-12-19", "decemberGenGlitch")
    glitchPlot("2018-12-28","2018-12-31", "decemberDemGlitch")
    glitchPlot("2019-02-09","2019-02-12", "februaryWindGlitch")
    glitchPlot("2019-02-17","2019-02-21", "februaryDemGlitch")
    glitchPlot("2019-01-29","2019-02-03", "januaryAllGlitch")

    buildWindGenScatter(save_to_pdf=True)
    buildWindGenScatter(save_to_pdf=True, filename="wind-gen-scatter-api", api_only=True)
    buildWeekdayHourPlot()

    certaintyPlot(model="ere_wtnn")
    certaintyPlot(model="ere_percep")
    certaintyPlot(model="percep")
    certaintyPlot(model="wtnn")
    metForecastPlot()
    metForecastPlot(smooth=False)
    metForecastPlot(smooth=False, fromPickle=True)
    plotPowerCurves()
    buildMetWindWindScatter(save_to_pdf=True)
    buildFirmNotFirmGraph("2019-01-26", "2019-01-29", clean=False, save_to_pdf=True)
