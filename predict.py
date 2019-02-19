import prepros as pp
from datetime import datetime
import model as m
import pickle

def getPrediction():
    ts, f, dt = pp.getPredictionData()

    model_files = ["predictor_30","predictor_60","predictor_90","predictor_120","predictor_150","predictor_180"]

    time_to_forecast = dt - datetime.now()
    model_number = max(round(time_to_forecast.seconds / 60 / 30)-1, 0)
    model_filename = model_files[model_number]

    # Apply norms
    ts_norm, f_norm = pickle.load(open("./data/"+model_filename+".norms", "rb"))
    ts, f = ts / ts_norm, f / f_norm

    model = m.load(model_filename)

    ts, f = [ts], [f] # Wrap input
    predictions = model.predict([ts,f])
    zone_prediction = predictions[0][0]
    reduced_prediction = predictions[1][0][0]

    print("Curtailment prediction for", dt, ", which is in ", time_to_forecast)
    print("Zone names:", pp.zone_names)
    print("Zone prediction:", zone_prediction)
    print()
    print("Reduced prediction:", reduced_prediction)

    return dt, zone_prediction, reduced_prediction

getPrediction()
