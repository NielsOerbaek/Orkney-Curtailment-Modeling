import prepros as pp
import descriptive as desc
import model as m
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta
from keras.backend import clear_session
from math import ceil

def evaluateMetForecast(start="2019-04-01",stop="2019-05-01", name="met-full-frame", code=0):
    if True:
        if False:
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
        else:
            df = pickle.load(open("data/"+name,"rb"))



        df_train = pp.getEdayData()
        df_full = pp.getSingleDataframe(fromPickle=True, clean=True)
        df_train = df_full.join(df_train,how="inner")

        if code == 1 or code == 0:
            percep = m.train_and_save_perceptron(df_train[["speed", "weekday", "hour"]].values, df_train[["Curtailment"]].values,kfold=False,filename="WT-Percep-"+name)
            print("Doing WT-Percep predictions...")
            df["percep_prediction"] = [percep.predict([[d[["wind_speed", "weekday","hour"]].values]])[0][0] for i,d in df.iterrows()]
            df["percep_prediction_correct"] = [int(round(d["percep_prediction"]) == ceil(d["Curtailment"]))*100 for i,d in df.iterrows()]

            print("Clearing Keras session")
            del percep
            clear_session()

        if code == 2 or code == 0:
            wtnn = m.train_and_save_simple(df_train[["speed", "weekday", "hour"]].values, df_train[["Curtailment"]].values,kfold=False,filename="WT-FFNN-"+name)
            print("Doing WT-FFNN predictions...")
            df["wtnn_prediction"] = [wtnn.predict([[d[["wind_speed", "weekday","hour"]].values]])[0][0] for i,d in df.iterrows()]
            df["wtnn_prediction_correct"] = [int(round(d["wtnn_prediction"]) == ceil(d["Curtailment"]))*100 for i,d in df.iterrows()]

            print("Clearing Keras session")
            del wtnn
            clear_session()

        if code == 3 or code == 0:
            ere_percep = m.train_and_save_perceptron(df_train[["Wind Mean (M/S)", "weekday", "hour"]].values, df_train[["Curtailment"]].values,kfold=False,filename="WT-Percep-ERE-"+name)
            print("Doing ERE WT-Percep predictions...")
            df["ere_percep_prediction"] = [ere_percep.predict([[d[["wind_speed", "weekday","hour"]].values]])[0][0] for i,d in df.iterrows()]
            df["ere_percep_prediction_correct"] = [int(round(d["ere_percep_prediction"]) == ceil(d["Curtailment"]))*100 for i,d in df.iterrows()]

            print("Clearing Keras session")
            del ere_percep
            clear_session()

        if code == 4 or code == 0:
            ere_wtnn = m.train_and_save_simple(df_train[["Wind Mean (M/S)", "weekday", "hour"]].values, df_train[["Curtailment"]].values,kfold=False,filename="WT-FFNN-ERE-"+name)
            print("Doing ERE WT-FFNN predictions...")
            df["ere_wtnn_prediction"] = [ere_wtnn.predict([[d[["wind_speed", "weekday","hour"]].values]])[0][0] for i,d in df.iterrows()]
            df["ere_wtnn_prediction_correct"] = [int(round(d["ere_wtnn_prediction"]) == ceil(d["Curtailment"]))*100 for i,d in df.iterrows()]

            print("Clearing Keras session")
            del ere_wtnn
            clear_session()


        pickle.dump(df, open("data/"+name,"wb"))

    else:
        print("Loading full met frame from","data/"+name)
        df = pickle.load(open("data/"+name,"rb"))

    hours = df.groupby("hours_forecast")

    accs = []
    accs.append(hours["prediction_correct"].describe())
    #accs.append(hours["prediction_corrected_correct"].describe())
    accs.append(hours["ere_prediction_correct"].describe())
    accs.append(hours["percep_prediction_correct"].describe())
    accs.append(hours["ere_percep_prediction_correct"].describe())
    accs.append(hours["wtnn_prediction_correct"].describe())
    accs.append(hours["ere_wtnn_prediction_correct"].describe())

    names = ["$WT_6$",
        "$WT_6$ - ERE power curve",
        "WT-Percep",
        "WT-Percep - ERE wind data",
        "WT-FFNN",
        "WT-FFNN - ERE wind data"]

    pickle.dump(accs, open("data/"+name+"-describes","wb"))
    pickle.dump(names, open("data/"+name+"-describes-name","wb"))

    return accs, names

def makeForecastAccTable(name="met-full-frame"):
    if True:
        accs = pickle.load(open("data/"+name+"-describes", "rb"))
        names = pickle.load(open("data/"+name+"-describes-name", "rb"))
        print("loaded met accs from pickle")
    else:
        accs, names = evaluateMetForecast()

    print(accs[0].iloc[:96]["count"].sum())

    df = pd.DataFrame()

    for i,acc in enumerate(accs):
        df[names[i]] = acc["mean"]

    #df["Count"] = accs[0]["count"]

    df.index = [int(td.days*24+td.seconds/3600) for td in df.index]
    df = df.iloc[:96]

    df.columns = ["$WT_6$", "$WT_6$ - ERE", "WT-Percep", "WT-Percep - ERE", "WT-FFNN", "WT-FFNN - ERE"]

    bins = pd.cut(df.index, np.arange(0,97,12))
    intervals = df.groupby(bins).mean().round(2)

    print(intervals.to_latex())

    total = df.mean().round(2)

    print(total.to_latex())

def ANNCertainty(start="2019-04-01",stop="2019-05-01",fromPickle=False, clean=True, load_model=False):
    if clean: filename = "ANNCertainty"
    else: filename = "ANNCertainty-uncleaned"
    if fromPickle:
        print("Loaded", filename)
        return pickle.load(open("data/"+filename,"rb"))
    else:
        print("Making Met-ANM dataset for certaintyPlot", filename)
        met_df = pp.getMetData(start, stop).set_index("forecast_time")
        anm_df = pp.getSingleDataframe(start, "2019-05-31", fromPickle=True, clean=clean)

        df = anm_df.join(met_df, how="inner")

        if load_model:
            ere_wtnn = m.load(filename=filename)
        else:
            df_train = pp.getEdayData()
            df_full = pp.getSingleDataframe(fromPickle=True, clean=clean)
            df_train = df_full.join(df_train,how="inner")
            ere_wtnn = m.train_and_save_simple(df_train[["Wind Mean (M/S)", "weekday", "hour"]].values, df_train[["Curtailment"]].values,kfold=False,filename=filename)

        print("Doing ERE WT-FFNN predictions...")
        df["ere_wtnn_prediction"] = [ere_wtnn.predict([[d[["wind_speed", "weekday","hour"]].values]])[0][0] for i,d in df.iterrows()]
        df["ere_wtnn_prediction_correct"] = [int(round(d["ere_wtnn_prediction"]) == d["Curtailment"])*100 for i,d in df.iterrows()]

        print(df["ere_wtnn_prediction_correct"].mean())

        pickle.dump(df, open("data/"+filename,"wb"))
        return df

def hitAccuracy():
    df = pickle.load(open("data/met-full-frame-all","rb"))
    df = df.loc[df["hours_forecast"] <= timedelta(days=4)]

    names = ["prediction", "wtnn_prediction", "ere_prediction", "ere_wtnn_prediction"]

    num_samples = df["deg"].count()
    num_actual = df.loc[df["Curtailment"] == 1]["hour"].count()

    for name in names:
        num_predicted = df.loc[df[name] > 0.5]["hour"].count()

        print("--------",name,"---------")
        print("Total accuracy: {:.2f}%".format(df[name+"_correct"].mean()))
        print("Curtailment ratio: {:.2f}%".format(num_actual/num_samples*100))
        print("Predicted curtailment ratio: {:.2f}%".format(num_predicted/num_samples*100))
        df_only_predicted = df.loc[df[name] > 0.5]
        print("Hits on predictions: {:.2f}%".format(df_only_predicted[name+"_correct"].mean()))

def getFullCombinedMetFrame():
    df = pickle.load(open("data/met-full-frame-all","rb"))
    df_clean = pickle.load(open("data/met-full-frame-all-clean","rb"))
    df["percep_prediction"] = df_clean["percep_prediction"]
    df["wtnn_prediction"] = df_clean["wtnn_prediction"]
    df["percep_prediction_correct"] = df_clean["percep_prediction_correct"]
    df["wtnn_prediction_correct"] = df_clean["wtnn_prediction_correct"]
    #df = df.loc[df["hours_forecast"] <= timedelta(days=4)]
    return df
