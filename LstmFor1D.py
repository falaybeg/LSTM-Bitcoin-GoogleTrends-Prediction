import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM

# Crate 1D Data into Time-Series
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)

#Load Dataset
df = pd.read_csv("Bitcoin1D.csv")
# We convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Reindex all of dataset by Date column
df = df.reindex(index= df.index[::-1])

zaman = np.arange(1, len(df) + 1, 1)
OHCL_avg = df.mean(axis=1)
plt.plot(zaman, OHCL_avg)
plt.show()
#print(OHCL_avg.head())

# Normalize dataset
OHCL_avg = np.reshape(OHCL_avg.values, (len(OHCL_avg),1)) #7288 data
scaler = MinMaxScaler(feature_range=(0,1))
OHCL_avg = scaler.fit_transform(OHCL_avg)
#print(OHCL_avg)

#Train-Test SPLIT dataset
train_OHLC = int(len(OHCL_avg)*0.56)
test_OHLC = len(OHCL_avg) - train_OHLC
train_OHLC, test_OHLC = OHCL_avg[0:train_OHLC,:], OHCL_avg[train_OHLC:len(OHCL_avg),:]

# We create 1D dimension dataset from mean OHLV
trainX, trainY = new_dataset(train_OHLC,1)
testX, testY = new_dataset(test_OHLC,1)

# Reshape dataset for LSTM in 3D Dimension
trainX = np.reshape(trainX, (trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
step_size = 1

# LSTM Model is created
model = Sequential()
model.add(LSTM(128, input_shape=(1, step_size)))
model.add(Dropout(0.1))
# model.add(LSTM(64))
# model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=25, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-Normalizing for plotting 
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Performance Measure RMSE is calculated for predicted train dataset
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print("Train RMSE: %.2f" % (trainScore))

# Performance Measure RMSE is calculated for predicted test dataset
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print("Test RMSE: %.2f" % (testScore))

# Converted predicted train dataset for plotting
trainPredictPlot = np.empty_like(OHCL_avg)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size,:] = trainPredict

# Converted predicted test dataset for plotting
testPredictPlot = np.empty_like(OHCL_avg)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHCL_avg)-1,:] = testPredict


# Finally predicted values are visualized
OHCL_avg = scaler.inverse_transform(OHCL_avg)
plt.plot(OHCL_avg, 'g', label='Orginal Dataset')
plt.plot(trainPredictPlot, 'r', label='Training Set')
plt.plot(testPredictPlot, 'b', label='Predicted price/test set')
plt.title("Hourly Bitcoin Predicted Prices")
plt.xlabel('Hourly Time', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.legend(loc='upper right')
plt.show()