from keras.layers import Input,LSTM,Dense, concatenate
from keras.models import Model, load_model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

import prepros as pp
import config

def train_and_save(xts_train, xf_train, y_train, yr_train, xts_val=None, xf_val=None, y_val=None, yr_val=None, epochs=10, filename="predictor"):
    ts_features = xts_train.shape[-1]
    f_features = xf_train.shape[-1]
    num_classes = y_train.shape[-1]
    # Timeseries going through LSTMs
    ts_input = Input(shape=(pp.timesteps,ts_features), name="TimeSeriesInput")
    lstm_1 = LSTM(128, return_sequences=True, name="RNN1")(ts_input)
    lstm_2 = LSTM(128, name="RNN2")(lstm_1)
    # Forecast data going through FFNN
    f_input = Input(shape=(f_features,), name="ForecastInput")
    dense_1 = Dense(128, name="FullyConnectedLayer")(f_input)
    # Concatenate the two and go through a last layer.
    merged = concatenate([lstm_2, dense_1], name="ConcatenateTensors")
    dense_2 = Dense(128, name="FullyConnectedLayer")(merged)
    # Output layer
    zone_output = Dense(num_classes, activation="sigmoid", name="PerZone",)(dense_2)
    reduced_output = Dense(1, activation="sigmoid", name="AnyZone")(dense_2)
    model = Model(inputs=[ts_input, f_input], outputs=[zone_output, reduced_output])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(model.summary())
    plot_model(model)
    if xts_val is None: model.fit([xts_train, xf_train], [y_train, yr_train], epochs=epochs, batch_size=1)
    else: model.fit([xts_train, xf_train], [y_train, yr_train], epochs=epochs, batch_size=1, validation_data=([xts_val, xf_val], [y_val,yr_val]))
    model.save(config.DATA_PATH+filename+".h5")
    return model


def load(filename="predictor"):
    return load_model(config.DATA_PATH+filename+".h5")


def train_and_save_simple(x, y, epochs=5, filename="predictor"):
    features = x.shape[-1]

    def getModel():
        input = Input(shape=(features,))
        dense_1 = Dense(128)(input)
        dense_2 = Dense(128)(dense_1)
        output = Dense(1, activation="sigmoid")(dense_2)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    print("Performing 10-fold cross-validation:")
    evaluateModel(getModel, x, y, epochs=epochs, features=features)
    print("Training model on all data:")
    model = getModel()
    model.fit(x,y, epochs=epochs, batch_size=32, verbose=0)
    model.save(config.DATA_PATH+filename+".h5")
    print("Done and saved to", config.DATA_PATH+filename+".h5")
    return model

def evaluateModel(getModelFunction, x, y, epochs=5, features=32):
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cvscores = []
    fold = 0
    for train, test in kfold.split(x, y):
        print("Fold " + str(fold) +": ", end="", flush=True)
        fold += 1
        model = getModelFunction()
        model.fit(x[train],y[train], epochs=epochs, batch_size=32, verbose=0)
        scores = model.evaluate(x[test], y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("----- FINAL ACC: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
