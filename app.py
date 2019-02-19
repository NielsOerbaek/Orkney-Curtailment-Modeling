from flask import Flask, render_template, json
import predict

#predict.preloadModels()

app = Flask(__name__)

@app.route('/')
def get_index():
    ps = json.dumps(predict.getAllPredictions())
    return render_template("index.html", predictions=ps)
