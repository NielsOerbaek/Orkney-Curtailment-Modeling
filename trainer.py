import pickle
import prepros as pp
import model as m

# ------ PARAMETERS --------
file_name = "predictor"
# --------------------------

def trainer(start="2018-12-01", stop="2019-02-20", forecast=180, file_name=file_name):
    file_name += "_"+str(forecast)
    xts, ts_norms, xf, f_norms, y, yr = pp.makeDataset("2018-12-01", "2019-02-01", hours_forecast=(forecast/60))
    #Save the norms for later use
    pickle.dump((ts_norms, f_norms), open("./data/"+file_name+".norms", "wb"))

    m.train_and_save(xts, xf, y, yr, epochs=3, filename=file_name)

trainer(forecast=180)
trainer(forecast=150)
trainer(forecast=120)
trainer(forecast=90)
trainer(forecast=60)
trainer(forecast=30)
