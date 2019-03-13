Summary
---
An LSTM (Long Short-Term Network) is kind of Recurrent Neural Network. Traditional neural networks can't remember previous inputs. But Recurrent Neural Networks enable us to learn from previous sequence input datas.

In this repository was written a Bitcoin Price Prediction project based Google Trend keyword by using LSTM algorithm and Python 3.6 version. Here we tried to determine, "Does LSTM algorithm predict Bitcoin Close price by adding many keywords volume from Google Trends". Bitcoin price dataset was downloaded hourly using coinapi.io API and Google Trends keywords were downloaded hourly using Python pytrend library. Finally chosen Bitcoin, BTC, Blockchain, Cryptocurrency and Iota keywords were added as columns into dataset.
Consequently LSTM algorithm predicted Bitcoin Close prices better than we expected by improving its learning in every epoch. The result images are shown following.

Images
---
Prediction Plot
<img src="/images/3D-PredictionPlot.png" width="60%" height="60%" />
Result of loss value after every epoch
<img src="/images/3D-LossValuePlot.png" width="60%" height="60%" />
Distribution of columns 
<img src="/images/3D-DistributionColumns.png" width="60%" height="60%" />


Recurrent Neural Network (RNN), LSTM (Long Short-Time Memory), Bitcoin, Google Trends, Prediction

