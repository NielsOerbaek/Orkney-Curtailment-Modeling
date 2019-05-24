import prepros as pp
import descriptive as desc
import model as m
import pickle
import pandas as pd
import numpy as np
from keras.backend import clear_session

def evaluateMetForecast(start="2019-04-01",stop="2019-05-01"):

    print("Making Met-ANM dataset")
    met_df = pp.getMetData(start, stop).set_index("forecast_time")
    anm_df = pp.getSingleDataframe(start, "2019-05-31", fromPickle=True, clean=True)

    df = anm_df.join(met_df, how="inner")

    df["prediction"] = [int(desc.correlationModelKCurve(d["wind_speed"],i.weekday()+1,i.hour+1,6)) for i,d in df.iterrows()]
    df["prediction_correct"] = [int(d["prediction"] == d["Curtailment"])*100 for i,d in df.iterrows()]
    #df["prediction_corrected"] = [int(desc.correlationModelKCurve(d["wind_speed"]/0.875-3.1,i.weekday()+1,i.hour+1,6)) for i,d in df.iterrows()]
    #df["prediction_corrected_correct"] = [int(d["prediction_corrected"] == d["Curtailment"])*100 for i,d in df.iterrows()]
    df["ere_prediction"] = [int(desc.correlationModelKCurveEday(d["wind_speed"],i.weekday()+1,i.hour+1,6)) for i,d in df.iterrows()]
    df["ere_prediction_correct"] = [int(d["ere_prediction"] == d["Curtailment"])*100 for i,d in df.iterrows()]
    df["speed_delta"] = [d["wind_speed"]-d["speed"] for i,d in df.iterrows()]


    df_train = pp.getEdayData()
    df_full = pp.getSingleDataframe(fromPickle=True)
    df_train = df_full.join(df_train,how="inner")

    ere_wtnn = m.train_and_save_simple(df_train[["Wind Mean (M/S)", "weekday", "hour"]].values, df_train[["Curtailment"]].values,kfold=False,filename="WT-FFNN-ERE")
    print("Doing ERE WT-FFNN predictions...")
    df["ere_wtnn_prediction"] = [round(ere_wtnn.predict([[d[["wind_speed", "weekday","hour"]].values]])[0][0]) for i,d in df.iterrows()]
    df["ere_wtnn_prediction_correct"] = [int(d["ere_wtnn_prediction"] == d["Curtailment"])*100 for i,d in df.iterrows()]

    print("Clearing Keras session")
    clear_session()

    wtnn = m.train_and_save_simple(df_train[["speed", "weekday", "hour"]].values, df_train[["Curtailment"]].values,kfold=False)
    print("Doing WT-FFNN predictions...")
    df["wtnn_prediction"] = [round(wtnn.predict([[d[["wind_speed", "weekday","hour"]].values]])[0][0]) for i,d in df.iterrows()]
    df["wtnn_prediction_correct"] = [int(d["wtnn_prediction"] == d["Curtailment"])*100 for i,d in df.iterrows()]

    pickle.dump(df, open("data/met-full-frame","wb"))

    hours = df.groupby("hours_forecast")

    accs = []
    accs.append(hours["prediction_correct"].describe())
    #accs.append(hours["prediction_corrected_correct"].describe())
    accs.append(hours["ere_prediction_correct"].describe())
    accs.append(hours["wtnn_prediction_correct"].describe())
    accs.append(hours["ere_wtnn_prediction_correct"].describe())

    names = ["$WT_6$", "$WT_6$ - ERE power curve", "WT-FFNN", "WT-FFNN - ERE wind data"]

    return accs, names

def makeForecastAccTable():
    if True:
        accs = pickle.load(open("./met-accs", "rb"))
        names = pickle.load(open("./met-acc-names", "rb"))
        print("loaded met accs from pickle")
    else:
        accs, names = evaluateMetForecast()

    print(accs[0].iloc[:96]["count"].sum())

    df = pd.DataFrame()

    for i,acc in enumerate(accs):
        df[names[i]] = acc["mean"]
        print(acc["mean"])

    #df["Count"] = accs[0]["count"]

    df.index = [int(td.days*24+td.seconds/3600) for td in df.index]
    df = df.iloc[:96]

    df.columns = ["$WT_6$", "$WT_6$ - ERE", "WT-FFNN", "WT-FFNN - ERE"]

    bins = pd.cut(df.index, np.arange(0,97,12))
    intervals = df.groupby(bins).mean().round(2)

    print(intervals.to_latex())

    total = df.mean().round(2)

    print(total.to_latex())

def ANNCertainty(start="2019-04-01",stop="2019-05-01",fromPickle=False, clean=True):
    if clean: filename = "ANNCertainty"
    else: filename = "ANNCertainty-uncleaned"
    if fromPickle:
        print("Loaded", filename)
        return pickle.load(open("data/"+filename,"rb"))
    else:
        print("Making Met-ANM dataset for certaintyPlot", filename)
        met_df = pp.getMetData(start, stop).set_index("forecast_time")
        anm_df = pp.getSingleDataframe(start, "2019-05-31", fromPickle=True)

        df = anm_df.join(met_df, how="inner")

        df_train = pp.getEdayData()
        df_full = pp.getSingleDataframe(fromPickle=True, clean=clean)
        df_train = df_full.join(df_train,how="inner")

        ere_wtnn = m.train_and_save_simple(df_train[["Wind Mean (M/S)", "weekday", "hour"]].values, df_train[["Curtailment"]].values,kfold=False,filename="WT-FFNN-ERE")
        print("Doing ERE WT-FFNN predictions...")
        df["ere_wtnn_prediction"] = [ere_wtnn.predict([[d[["wind_speed", "weekday","hour"]].values]])[0][0] for i,d in df.iterrows()]
        df["ere_wtnn_prediction_correct"] = [int(round(d["ere_wtnn_prediction"]) == d["Curtailment"])*100 for i,d in df.iterrows()]

        pickle.dump(df, open("data/"+filename,"wb"))
        return df
