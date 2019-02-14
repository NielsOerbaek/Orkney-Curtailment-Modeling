from keras.layers import Input,LSTM,Dense
from keras.models import Model
import tensorflow

import prepros as pp

x,y = pp.makeDataset("2018-12-01", "2019-02-01")
#x,y = pp.makeDataset("2018-12-01", "2018-12-02")
x_train_raw, x_val_raw, y_train, y_val = pp.splitData(x,y)

x_train, norms = pp.normalizeData(x_train_raw)
x_val = x_val_raw / norms

data_dim = x_train.shape[1]
num_classes = y_train.shape[1]

seq_train = pp.makeTimeseries(x_train, y_train)
seq_val = pp.makeTimeseries(x_val, y_val)

# TODO: Normalize columns. Find parameters on train set and reuse themself.
# TODO: Maybe try to predict one zone at a time?

input = Input(shape=(pp.timesteps,data_dim))
lstm_1 = LSTM(128, return_sequences=True)(input)
lstm_2 = LSTM(128)(lstm_1)
output = Dense(num_classes, activation="sigmoid")(lstm_2)
model = Model(inputs=input, outputs=output)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())

model.fit_generator(seq_train, epochs=100, validation_data=seq_val)
