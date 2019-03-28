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

import prepros as pp
import descriptive as desc




def buildModelGraph(start_limit=0, stop_limit=0, zones=0):

    #Full DataSet, used for training
    df_full = pp.getSingleDataframe("2018-12-01", "2019-03-01", fromPickle=True).resample("10min").mean().interpolate(method='linear')
    df_full = pp.cleanData(df_full)
    df_full = pp.addReducedCol(df_full)

    df = pp.getSingleDataframe(start_limit, stop_limit, fromPickle=True).resample("10min").mean().interpolate(method='linear')
    df = pp.cleanData(df)
    df = pp.addReducedCol(df)

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
