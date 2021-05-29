import pandas as pd
from neuralprophet import NeuralProphet, LSTM, DeepAR, TFT, NBeats
from neuralprophet.utils.utils import set_log_level

# set_log_level("ERROR")
df = pd.read_csv("example_data/yosemite_temps.csv")
df.head(3)
df = df.iloc[:1000]

# runs NeuralProphet on sample data

m = NeuralProphet(n_lags=12, n_forecasts=3, epochs=10, learning_rate=1)
metrics_NP = m.fit(df, freq="5min", validate_each_epoch=True)

# print(metrics)

m = LSTM(n_lags=12, n_forecasts=3, num_hidden_layers=1, d_hidden=64, learning_rate=1, epochs=10)
metrics_LSTM = m.fit(df, freq="5min", validate_each_epoch=True)



m = NBeats(n_lags=12, n_forecasts=3, epochs=10, learning_rate=1)
metrics_NBeats = m.fit(df, freq="5min")



m = DeepAR(n_lags=12, n_forecasts=3, epochs=10, learning_rate=1)
metrics_DeepAR = m.fit(df, freq="5min")



m = TFT(n_lags=12, n_forecasts=3, epochs=10, learning_rate=1)
metrics_TFT = m.fit(df, freq="5min")


print("Metrics during train NP")
print(metrics_NP)
print("Metrics during train LSTM")
print(metrics_LSTM)
print("Metrics during train NBeats")
print(metrics_NBeats)
print("Metrics during train DeepAR")
print(metrics_DeepAR)
print("Metrics during train TFT")
print(metrics_TFT)