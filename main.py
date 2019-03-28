import pickle
import prepros as pp
import model as m
import pprint
import numpy as np
import config

# ------ PARAMETERS --------
load_model = False
file_name = "descriptor"
minutes_forecast = 00
# TODO: Move this to cmi arguments?
# --------------------------

file_name += "_"+str(minutes_forecast)

xts, ts_norms, xf, f_norms, y, yr = pp.makeDataset("2018-12-01", "2019-03-01", hours_forecast=(minutes_forecast/60))
#Save the norms for later use
pickle.dump((ts_norms, f_norms), open(config.DATA_PATH+file_name+".norms", "wb"))

print("Size of timeseries data:", xts.nbytes)
print(xts.shape, ts_norms.shape, xf.shape, f_norms.shape, y.shape, yr.shape)

# Load model or train new model
if load_model: model = m.load(file_name)
else: model = m.train_and_save(xts, xf, y, yr, epochs=2, filename=file_name)

# Make test dataset and predict
xts_test, ts_norms, xf_test, f_norms, y_test, yr_test = pp.makeDataset("2019-03-01", "2019-03-21", hours_forecast=(minutes_forecast/60), norms=(ts_norms, f_norms))

zone_acc = np.zeros(len(pp.zone_names))
reduced_acc = 0
zone_predictions, reduced_predictions = model.predict([xts_test, xf_test])

for i, p in enumerate(zip(zone_predictions, reduced_predictions)):
    if round(p[1][0]) == yr_test[i]: reduced_acc += 1
    for j,z in enumerate(p[0]):
        if y_test[i][j] == round(z):
            zone_acc[j] += 1

print("Overall accuracy:", reduced_acc / len(yr_test) * 100)
pprint.pprint(list(zip(pp.zone_names, (zone_acc / len(y_test) * 100).astype(int))))
