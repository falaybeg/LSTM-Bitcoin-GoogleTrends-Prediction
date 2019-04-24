Summary
---
[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) (Long Short-Term Network) is a kind of Recurrent Neural Network which used in the field of deep learning. Traditional neural networks can't remember previous inputs. But Recurrent Neural Networks enable us to learn from previous sequence input datas. A LSTM unit is composed of a cell, an input gate, an output gate and a forget gate.

In this repository was written a Bitcoin Price Prediction project based on Google Trend keywords by using LSTM algorithm and Python 3.6 version. Here we tried to determine, "Does LSTM algorithm predict Bitcoin Close price by adding many keywords volume from Google Trends". Bitcoin price dataset was downloaded hourly using coinapi.io API and Google Trends keywords were downloaded hourly using Python pytrend library. Finally chosen Bitcoin, BTC, Blockchain, Cryptocurrency and Iota keywords were added as columns into dataset.
Consequently LSTM algorithm predicted Bitcoin Close prices better than we expected by improving its learning in every epoch. The result images are shown following.

Images
---
Prediction Plot<br/>
<img src="/images/3D-PredictionPlot.png"  />
Result of loss value after every epoch<br/>
<img src="/images/3D-LossValuePlot.png" />
Distribution of columns <br/>
<img src="/images/3D-DistributionColumns.png" />


Recurrent Neural Network (RNN), LSTM (Long Short-Time Memory), Artifical Intelligence, Deep Learning, Prediction, Bitcoin, Google Trends 
