import pickle
import prepros as pp
import model as m
import pprint
import numpy as np

load_model = True
file_name = "all_zones_DecJan-predictor"

x,y = pp.makeDataset("2018-12-01", "2019-02-01")
# y = pp.reduceZones(y)

x_train_raw, x_val_raw, y_train, y_val = pp.splitData(x,y)

x_train, norms = pp.normalizeData(x_train_raw)
x_val = x_val_raw / norms
pickle.dump(norms, open("./data/"+file_name+".norms", "wb"))

data_dim = x_train.shape[1]
num_classes = y_train.shape[1]

seq_train = pp.makeTimeseries(x_train, y_train)
seq_val = pp.makeTimeseries(x_val, y_val)

# TODO: Maybe try to predict one zone at a time?

# Load model or train new model
if load_model: model = m.load(file_name+".h5")
else: model = m.train_and_save(seq_train, seq_val, data_dim, num_classes, 1, file_name+".h5")

# Make test dataset and predict
x_test_raw, y_test = pp.makeDataset("2019-02-01", "2019-02-15")
x_test = x_test_raw / norms
# y_test = pp.reduceZones(y_test)

zone_acc = np.zeros(len(pp.zone_names))
num_timeseries = len(x_test)-pp.timesteps-1

for p in range(num_timeseries):
    p_end = p+pp.timesteps
    pred = model.predict([[x_test[p:p_end]]])
    for i,z in enumerate(pred[0]):
        if y_test[p_end-1][i] == round(z):
            zone_acc[i] += 1

pprint.pprint(list(zip(pp.zone_names, (zone_acc / num_timeseries * 100).astype(int))))
