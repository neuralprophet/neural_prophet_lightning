# Stat-of-the-art models

Some state-of-the-art models are available in the NeuralProphet: LSTM, DeepAR, TemporalFusionTransformer and NBeats.
These models are introduced under the same API as NeuralProphet, for easy compairson.

### LSTM
We introduced LSTM model, as a part of additional models modules.
It is based on refactored into Pytorch Lightning LSTM model from pure Pytorch.
For LSTM, the main class has the same functionality as the NeuralProphet and supports all main methods.
An example of LSTM usage can be found in `example_notebooks/LSTM_example.ipynb` notebook.

Here is the example of using LSTM module:
```python
m = LSTM(n_lags = 10,
         n_forecasts=7,
         num_hidden_layers=1,
         d_hidden=64,
         learning_rate=0.1,
         epochs=20)
metrics_df = m.fit(df, freq = '5min')     
future = m.make_future_dataframe(df, preiods = 7, n_historic_predictions=True)
forecast = m.predict(future)
```

### NBeats
We also introduced NBeats model. 
It is based on the Pytorch Forecasting model implementation. 
We refactored existing model code to support NeuralProphet Metric class. 
We also created a new wrapper class, which has the same main modules as NeuralProphet and can be used with the same API. 
An example of NBeats usage can be found in `example_notebooks/NBeats_example.ipynb` notebook.

Here is the example of using NBeats module:

```python
m = NBeats(
        n_lags = 150,
        n_forecasts = 1,
        epochs = 100,
        auto_lr_find=True)
m.fit(df, freq = '5min')
future = m.make_future_dataframe(df, preiods = 1, n_historic_predictions=10)
forecast = m.predict(future)
```



### DeepAR
We also introduced DeepAR model. 
It is based on the Pytorch Forecasting model implementation. 
We refactored existing model code to support NeuralProphet Metric class. 
We also created a new wrapper class, which has the same main modules as NeuralProphet 
and can be used with the same API. 
An example of DeepAR usage can be found in `example_notebooks/DeepAR_example.ipynb` notebook.

Here is the example of using NBeats module:

```python
m = DeepAR(
        n_lags=60,
        n_forecasts=20,
        epochs = 100,
        hidden_size=32,
        rnn_layers=2)
m.fit(df, freq = '5min')
future = m.make_future_dataframe(df, preiods = 20, n_historic_predictions=10)
forecast = m.predict(future)
```


### TFT
We also introduced TFT model. 
It is based on the Pytorch Forecasting model implementation. 
We refactored existing model code to support NeuralProphet Metric class. 
We also created a new wrapper class, which has the same main modules as NeuralProphet 
and can be used with the same API. 
An example of TFT usage can be found in `example_notebooks/TFT_example.ipynb` notebook.

Here is the example of using NBeats module:

```python
m = TFT(
        n_lags=60,
        n_forecasts=20,
        epochs = 100,
        attention_head_size=1,
        hidden_continuous_size=8)
m.fit(df, freq = '5min')
future = m.make_future_dataframe(df, preiods = 20, n_historic_predictions=10)
forecast = m.predict(future)
```

### LIBRA framework
Libra is a benchmarking framework, that we use to benchmark Neural Prophet and all additional models.

For more information about evaluation procedure, please follow [LIBRA documentation](../libre.md)


