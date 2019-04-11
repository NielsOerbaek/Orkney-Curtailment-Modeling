import sys
import numpy as np
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
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

plt.style.use("seaborn-colorblind")

import prepros as pp
import descriptive as desc


def buildModelGraph(start_limit=0, stop_limit=0, zones=0):
    df_eday = pp.getEdayData()

    #Full DataSet, used for training
    df_full = pp.getSingleDataframe("2018-12-01", "2019-03-01", fromPickle=True)
    df_full = df_full.join(df_eday, how="inner")
    df_full = pp.cleanData(df_full)
    df_full = pp.addReducedCol(df_full, clean=True)
    df_full = pp.removeGlitches(df_full)

    df = pp.getSingleDataframe(start_limit, stop_limit, fromPickle=True)
    df = df.join(df_eday, how="inner")
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


    fig = plt.figure()
    # Bottom plot
    ax1 = fig.add_axes([0.08,0.1,0.9,0.39])
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
    cm = plt.get_cmap("binary")
    ax2 = fig.add_axes([0.08,0.51,0.9,0.4])
    ax2.pcolormesh(accs, alpha=1, cmap=cm, snap=True)
    ax2.set_xticks(meshxticks_major)
    ax2.set_xticks(meshxticks_minor, minor=True)
    ax2.xaxis.set_ticklabels([])
    ax2.grid(b=True, which="major", axis="x", linestyle="-.")
    ax2.grid(b=True, which="minor", axis="x", linestyle="--")
    ax2.set_ylabel("Models")
    ax2.set_yticks(np.arange(len(model_names))+0.5)
    ax2.set_yticks(np.arange(len(model_names)), minor=True)
    ax2.set_yticklabels(model_names, rotation=0, fontsize="10", va="center")
    ax2.grid(b=True, which="minor", axis="y")
    custom_lines = [Line2D([0], [0], color=cm(0), lw=4),
        Line2D([0], [0], color=cm(1.), lw=4)]
    ax2.legend(custom_lines, ["No curtailment", "Curtailment"], loc=1, fancybox=True, framealpha=0.5)
    plt.title("Generation relative to demand for all of Orkney. \nAccuracies for models: " + ", ".join(model_names))

    fig.autofmt_xdate(which="both")

    fig.set_size_inches(15, 8)
    plt.xticks(rotation=-60)

    #plt.savefig("./static/graphs/cleaned_"+file_name, orientation='landscape')
    #plt.savefig("./static/pdf/cleaned_"+file_name[:-3]+"pdf", orientation='landscape')
    plt.show()


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
    ax1.legend(["OpenWeatherMap", "Eday Turbine"], loc=1, fancybox=True, framealpha=0.5)

    plt.title("Wind speed comparison")

    fig.autofmt_xdate(which="both")

    fig.set_size_inches(15, 8)
    plt.xticks(rotation=-60)

    #plt.savefig("./static/graphs/cleaned_"+file_name, orientation='landscape')
    #plt.savefig("./static/pdf/cleaned_"+file_name[:-3]+"pdf", orientation='landscape')
    plt.show()

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
    if powercurve: plt.legend(["Power curve"], loc=1, fancybox=True, framealpha=0.5)
    #plt.title("Relation between windspeeds and generation for Eday 900kW Turbine")
    fig = plt.gcf()
    fig.set_size_inches(width, width*0.75)
    fig.tight_layout()
    if save_to_pdf: fig.savefig("../report/img/eday/"+filename+".pdf")
    else: plt.show()
    plt.clf()

def buildEdayWindOrkneyGenScatter(start_limit=0, stop_limit=0, zones=0, save_to_pdf=False, filename="eday-scatter", curtail_code=0, color="k", wind_limit=40):
    if start_limit != 0: start_limit = datetime.strptime(start_limit, '%Y-%m-%d')
    if stop_limit != 0: stop_limit = datetime.strptime(stop_limit, '%Y-%m-%d')

    df = pp.getEdayData()
    df_full = pp.getSingleDataframe(fromPickle=True)
    df = df_full.join(df,how="inner")
    df = pp.addReducedCol(df, clean=True)
    df = pp.removeGlitches(df)
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
    plt.legend(["Estimated Power Curve","Data"], fancybox=True, framealpha=0.5)
    fig = plt.gcf()
    fig.set_size_inches(3.9, 3.2)
    fig.tight_layout()
    #plt.title("Relation between windspeeds and generation for Eday 900kW Turbine")
    if save_to_pdf: fig.savefig("../report/plots/"+filename+".pdf")
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
    plt.legend(["$y = "+coef+"x + "+bias+"$","Data"], fancybox=True, framealpha=0.5)
    fig = plt.gcf()
    fig.set_size_inches(3.3, 3)
    fig.tight_layout()
    #plt.title("Relation between windspeeds and generation for Eday 900kW Turbine")
    if save_to_pdf: fig.savefig("../report/plots/"+filename+".pdf")
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
    plt.legend(["Enercon", "ERE", "Winter 2018", "December 2018"], loc=8, fancybox=True, framealpha=0.5)
    fig = plt.gcf()
    fig.set_size_inches(4.9, 3)
    fig.tight_layout()
    fig.savefig('../report/plots/eday_power_curves.pgf')
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
    dem_ax.set_xticks([])
    dem_ax.set_ylim(0, 40)
    dem_ax.set_ylabel("Demand [MW]")

    gen_ax = fig.add_axes([0.11,0.71,0.88,0.28])
    gen_ax.plot(df["Generation"], "g-", linewidth=1)
    gen_ax.set_xticks([])
    gen_ax.set_ylim(0, 40)
    gen_ax.set_ylabel("Generation [MW]")

    fig = plt.gcf()
    fig.set_size_inches(4.9, 7)
    fig.autofmt_xdate(which="both")
    #plt.show()
    fig.savefig("../report/img/exceptions/"+filename+".pdf")
    plt.clf()

def plotKModels(data):
    index = data[0]
    values = data[1]
    names = data[2]
    styles = data[3]

    plt.xlim(-20,20)
    plt.ylim(0,100)

    for i,v in enumerate(values):
        plt.plot(index,v,styles[i], linewidth=1)

    plt.legend(names, framealpha=0.5, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()



def buildAll():
    #buildEdayScatter("2018-12-01", "2019-03-01", save_to_pdf=True, filename="eday-all-data", powercurve=True)
    #buildEdayScatter("2018-12-01", "2019-03-01", save_to_pdf=True, filename="eday-with-curtailment-in-zone-1", curtail_code=1, color="b", width=3.5)
    #buildEdayScatter("2018-12-01", "2019-03-01", save_to_pdf=True, filename="eday-without-curtailment-in-zone-1", curtail_code=2, color="b", width=3.5)
    #buildEdayScatter("2018-12-01", "2019-01-01", save_to_pdf=True, filename="eday-december", color="r", width=3)
    #buildEdayScatter("2019-01-01", "2019-02-01", save_to_pdf=True, filename="eday-january", color="r", width=3)
    #buildEdayScatter("2019-02-01", "2019-03-01", save_to_pdf=True, filename="eday-february", color="r", width=3)

    #buildEdayWindOrkneyGenScatter("2018-12-01", "2019-03-01", color="r", save_to_pdf=True, filename="eday-wind-orkney-gen")
    #buildWindWindScatter("2019-02-11", "2019-03-01", save_to_pdf=True)

    #plotPowerCurves()

    #glitchPlot("2018-12-14","2018-12-19", "decemberGenGlitch")
    #glitchPlot("2018-12-28","2018-12-31", "decemberDemGlitch")
    #glitchPlot("2019-02-09","2019-02-12", "februaryWindGlitch")
    #glitchPlot("2019-02-17","2019-02-21", "februaryDemGlitch")

    plotKModels(desc.evaluateModels("2018-12-01", "2019-03-01", clean=False))
    #plotKModels(desc.evaluateModels("2018-12-01", "2019-03-01", clean=True))
