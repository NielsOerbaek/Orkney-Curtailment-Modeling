import pickle
import prepros as pp
import model as m

x,y = pp.makeDataset("2018-12-01", "2019-02-01")
#x,y = pp.makeDataset("2018-12-01", "2018-12-03")
x_train_raw, x_val_raw, y_train, y_val = pp.splitData(x,y)

x_train, norms = pp.normalizeData(x_train_raw)
x_val = x_val_raw / norms

pickle.dump(norms, open("./data/norms", "wb"))

data_dim = x_train.shape[1]
num_classes = y_train.shape[1]

seq_train = pp.makeTimeseries(x_train, y_train)
seq_val = pp.makeTimeseries(x_val, y_val)

# TODO: Maybe try to predict one zone at a time?

model = m.train_and_save(seq_train, seq_val, data_dim, num_classes, 5, "predictor.h5")

x_test_raw, y_test = pp.makeDataset("2019-02-01", "2019-02-07")
x_test = x_test_raw / norms

# TODO: Do some prediction!!!
