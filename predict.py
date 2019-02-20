import pickle
from datetime import datetime
import pymongo
from keras.backend import clear_session

import prepros as pp
import model as m
import config

client = pymongo.MongoClient("mongodb://"+config.writer_user+":"+config.writer_pw+"@"+config.SERVER+"/sse-data")
db = client["sse-data"]

model_files = ["predictor_30","predictor_60","predictor_90","predictor_120","predictor_150","predictor_180"]
models = dict()

# NOTE: Apparently keras cannot handle more that one model at a time.
# Then you need to set multiple tf sessions manually.
# For now we'll just load a model and kill it again each time. Slow, but works.
# TODO: Implement the above solution.
def preloadModels():
    print("Loading models: ", end="", flush=True)
    for f in model_files:
        print(f, end=", ", flush=True)
        model = m.load(f)
        norms = pickle.load(open("./data/"+f+".norms", "rb"))
        models[f] = (model,norms)
    print(" - Done!")

def makePrediction(ts,f,dt):
    print("----------------------")
    time_to_forecast = dt - datetime.utcnow()
    model_number = min(max(round(time_to_forecast.seconds / 60 / 30)-1, 0),5)
    model_name = model_files[model_number]
    print("Making curtailment forecast for", dt, ", which is in ", time_to_forecast)

    print("Loading model:", model_name, "...")
    model = m.load(model_name)
    norms = pickle.load(open("./data/"+model_name+".norms", "rb"))

    # Apply norms
    ts_norm, f_norm = norms
    ts, f = ts / ts_norm, f / f_norm
    ts, f = [ts], [f] # Wrap input

    print("Do prediction...")
    predictions = model.predict([ts,f])
    zone_prediction = predictions[0][0]
    reduced_prediction = predictions[1][0][0]

    print()
    print("Overall prediction:", reduced_prediction)
    print("Zone names:", pp.zone_names)
    print("Zone predictions:", zone_prediction)

    print("Clearing session and deleting model.")
    # Clear the Keras session and kill the model. Should be removed if we can handle the
    clear_session()
    del model

    return zone_prediction, reduced_prediction, dt, time_to_forecast

def storePrediction(zp,rp,dt,delta):
    pred = dict()
    pred["zones"] = zp.tolist()
    pred["overall"] = float(rp)
    pred["target_time"] = dt.timestamp()
    pred["prediction_time"] = (dt - delta).timestamp()

    pred_col = db["predictions"]
    pred_col.insert_one(pred)

def getAllPredictions():
    return list(db["predictions"].find({}, { "_id": 0 }))
