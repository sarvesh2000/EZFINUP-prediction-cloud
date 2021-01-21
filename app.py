from flask import  Flask, render_template, request
import os
import numpy as np
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
# import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from newsapi import NewsApiClient

import json
from json import JSONEncoder

app = Flask(__name__)

@app.route('/')
def index():
    import numpy as np

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    df = pdr.get_data_yahoo(request.args.get('stockname'))
    df1 = df.reset_index()['Close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    training_size = int(len(df1) * 0.65)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    # import numpy
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    ### Create the Stacked LSTM model
    import tensorflow as tf
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM

    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=10, batch_size=64, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    ##Transformback to original form --- rescaling
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    ### Plotting
    # shift train predictions for plotting
    look_back = 100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

    x_input = test_data[331:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    # demonstrate prediction for next 10 days
    from numpy import array

    lst_output = []
    n_steps = 100
    i = 0
    while (i < 3):

        if (len(temp_input) > 100):

            x_input = np.array(temp_input[1:])

            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)

            temp_input.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i = i + 1

    day_new = np.arange(1, 101)  # testdata 100indexes
    day_pred = np.arange(101, 104)  # 101-131-predicted

    # Serialization
    numpyData1 = day_new
    encodedNumpyData1 = json.dumps(numpyData1, cls=NumpyArrayEncoder)  # use dump() to write array into file
    # print("Printing JSON serialized NumPy array")
    # print(encodedNumpyData1)

    numpyData2 = day_pred
    encodedNumpyData2 = json.dumps(numpyData2, cls=NumpyArrayEncoder)  # use dump() to write array into file

    numpyData3 = scaler.inverse_transform(df1[1131:])
    encodedNumpyData3 = json.dumps(numpyData3, cls=NumpyArrayEncoder)  # use dump() to write array into file

    numpyData4 = scaler.inverse_transform(lst_output)
    encodedNumpyData4 = json.dumps(numpyData4, cls=NumpyArrayEncoder)  # use dump() to write array into file

    daynew = {"x": encodedNumpyData1, "y": encodedNumpyData2}
    daypred = {"x": encodedNumpyData3, "y": encodedNumpyData4}

    data = []
    data.append(daynew)
    data.append(daypred)

    final_data = {"data": data}

    print(final_data)
    return final_data
if __name__ == '__main__':
    app.run(port=5000, debug=False)