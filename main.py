import pickle
import prepros as pp
import model as m
import pprint
import numpy as np

load_model = True
file_name = "all_zones_DecJan_HybridPredictor"

xts, ts_norms, xf, f_norms, y = pp.makeDataset("2018-12-01", "2019-02-01")
# y = pp.reduceZones(y)

#Save the norms for later use
pickle.dump((ts_norms, f_norms), open("./data/"+file_name+".norms", "wb"))

print(xts.shape, ts_norms.shape, xf.shape, f_norms.shape, y.shape)

xts_train, xts_val, xf_train, xf_val, y_train, y_val = pp.splitData(xts, xf, y)


# TODO: Maybe try to predict one zone at a time?

# Load model or train new model
if load_model: model = m.load(file_name)
else: model = m.train_and_save(xts_train, xf_train, y_train, xts_val, xf_val, y_val, 10, file_name)

# Make test dataset and predict
xts_test, ts_norms, xf_test, f_norms, y_test = pp.makeDataset("2019-02-01", "2019-02-15", norms=(ts_norms, f_norms))

zone_acc = np.zeros(len(pp.zone_names))
predictions = model.predict([xts_test, xf_test])

for i, p in enumerate(predictions):
    for j,z in enumerate(p):
        if y_test[i][j] == round(z):
            zone_acc[j] += 1

pprint.pprint(list(zip(pp.zone_names, (zone_acc / len(y_test) * 100).astype(int))))
