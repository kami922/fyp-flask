import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import Sequential
import yfinance as yf
import joblib

'''
https://www.youtube.com/watch?v=GFSiL6zEZF0
'''

print("working dependencies")
crypto = "BTC"
base_currency = "USD"

start = dt.datetime(2022, 1, 1)
end = dt.datetime.now()

data = yf.download(f"{crypto}-{base_currency}", start=start, end=end)

sclar = MinMaxScaler(feature_range=(0, 1))
scaled_data = sclar.fit_transform(data["Close"].values.reshape(-1, 1))

prediction_days = 60
x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# neural network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss="mean_squared_error")
model.fit(x_train, y_train, epochs=25, batch_size=32)

# testing the model

# watch testing the model from start
test_start = dt.datetime(2023, 1, 1)
test_end = dt.datetime.now()

print("epoch done/ trading done now\nexecuting tests")

test_data = yf.download(f"{crypto}-{base_currency}", start=start, end=end)
actual_price = test_data["Close"].values


total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = sclar.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_price = model.predict(x_test)
prediction_price = sclar.inverse_transform(prediction_price)

plt.plot(actual_price, color="Black", label="Actual prices")
plt.plot(prediction_price, color="Green", label="predicted prices")
plt.title(f"{crypto} price prediction")
plt.xlabel("time")
plt.ylabel("price")
plt.legend(loc="upper left")
plt.show()
# watch testing the model from start

real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs)+1,0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = sclar.inverse_transform(prediction)
print(prediction)
print("end")
print("exporting to job lib")
joblib.dump(model, 'bitcoin_price_prediction_model.joblib')