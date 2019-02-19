from keras.layers import Input,LSTM,Dense, concatenate
from keras.models import Model, load_model

import prepros as pp


def train_and_save(xts_train, xf_train, y_train, xts_val, xf_val, y_val, epochs=10, filename="predictor"):
    ts_features = xts_train.shape[-1]
    f_features = xf_train.shape[-1]
    num_classes = y_train.shape[-1]
    # Timeseries going through LSTMs
    ts_input = Input(shape=(pp.timesteps,ts_features))
    lstm_1 = LSTM(128, return_sequences=True)(ts_input)
    lstm_2 = LSTM(128)(lstm_1)
    # Forecast data going through FFNN
    f_input = Input(shape=(f_features,))
    dense_1 = Dense(128)(f_input)
    # Concatenate the two and go through a last layer.
    merged = concatenate([lstm_2, dense_1])
    dense_2 = Dense(128)(merged)
    # Output layer
    output = Dense(num_classes, activation="sigmoid")(dense_2)
    model = Model(inputs=[ts_input, f_input], outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(model.summary())
    model.fit([xts_train, xf_train], y_train, epochs=epochs, validation_data=([xts_val, xf_val], y_val))
    model.save("./data/"+filename+".h5")
    return model


def load(filename="predictor"):
    return load_model("./data/"+filename+".h5")
