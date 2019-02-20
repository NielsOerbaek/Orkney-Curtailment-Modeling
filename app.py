from flask import Flask, render_template, json
import predict as p

#predict.preloadModels()

app = Flask(__name__)

@app.route('/')
def get_index():
    ps = json.dumps(p.getAllPredictions())
    return render_template("index.html", predictions=ps)

@app.route('/forecast')
def get_forecast():
    pred = p.getAndStorePrediction()
    pred["_id"] = 0
    return json.jsonify(pred)


#TODO: Add list of previous predictions.
