from keras.layers import Input,LSTM,Dense
from keras.models import Model, load_model
import tensorflow

import prepros as pp

def train_and_save(x_train, y_train, x_val, y_val, num_features, num_classes, epochs=10, filename="predictor.h5"):

    input = Input(shape=(pp.timesteps,num_features))
    lstm_1 = LSTM(128, return_sequences=True)(input)
    lstm_2 = LSTM(128)(lstm_1)
    output = Dense(num_classes, activation="sigmoid")(lstm_2)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print(model.summary())
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
    model.save("./data/"+filename)
    return model

def load(filename="predictor.h5"):
    return load_model("./data/"+filename)
