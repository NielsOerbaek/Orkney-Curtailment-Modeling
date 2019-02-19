import pickle
from datetime import datetime
import pymongo

import prepros as pp
import model as m
import config

client = pymongo.MongoClient("mongodb://"+config.writer_user+":"+config.writer_pw+"@"+config.SERVER+"/sse-data")
db = client["sse-data"]


model_files = ["predictor_30","predictor_60","predictor_90","predictor_120","predictor_150","predictor_180"]
models = dict()

def preloadModels():
    print("Loading models: ", end="", flush=True)
    for f in model_files:
        print(f, end=", ", flush=True)
        model = m.load(f)
        norms = pickle.load(open("./data/"+f+".norms", "rb"))
        models[f] = (model,norms)
    print(" - Done!")

def getPrediction(ts,f,dt):
    time_to_forecast = dt - datetime.now()
    model_number = max(round(time_to_forecast.seconds / 60 / 30)-1, 0)
    model_name = model_files[model_number]

    if model_name in models.keys(): model, norms = models[model_name]
    else:
        model = m.load(model_name)
        norms = pickle.load(open("./data/"+model_name+".norms", "rb"))

    # Apply norms
    ts_norm, f_norm = norms
    ts, f = ts / ts_norm, f / f_norm

    ts, f = [ts], [f] # Wrap input
    predictions = model.predict([ts,f])
    zone_prediction = predictions[0][0]
    reduced_prediction = predictions[1][0][0]

    print("Curtailment prediction for", dt, ", which is in ", time_to_forecast)
    print("Zone names:", pp.zone_names)
    print("Zone prediction:", zone_prediction)
    print("Reduced prediction:", reduced_prediction)

    return zone_prediction, reduced_prediction, dt, time_to_forecast

def storePrediction(zp,rp,dt,delta):
    pred = dict()
    pred["zones"] = zp.tolist()
    pred["overall"] = float(rp)
    pred["target_time"] = dt.timestamp()
    pred["prediction_time"] = (dt + delta).timestamp()

    pred_col = db["predictions"]
    pred_col.insert_one(pred)
