import os
import pandas as pd
import numpy as np

from neuralprophet import NBeats, LSTM, DeepAR, NeuralProphet, TFT
from neuralprophet.utils.df_utils import split_df
from neuralprophet.tools.metrics_libra import *

def get_datasets(usecase, data_loc = '../example_data/LIBRA/'):
    datasets = os.listdir(data_loc)
    datasets_names = [dataset for dataset in datasets if usecase in dataset]
    datasets = {}

    for dataset in datasets_names:
        datasets.update({dataset: pd.read_csv(data_loc + dataset)})

    frequencies = pd.read_csv(data_loc + 'freq.csv')
    return datasets, frequencies


mapping_interpretable_frequencies = {
    1: 'D',
    4: 'Q',
    12: 'M',
    52: 'W',
    24: 'H',
    7: 'W',
    91: 'D',
    364: 'D',
    360: 'D',
    168: 'D',
    672: 'D',
    96: 'H',
    288: 'D',
    28: 'D',
    6: 'D',
    30: 'D',
    720: 'D'
}

def mapping(x):
    try:
        return mapping_interpretable_frequencies[x]
    except:
        return 'D'


def get_parameters(df, dataset_name, frequencies, method, usecase):
    idx_ts = int(dataset_name.split('.')[0].split('_')[-1]) - 1
    n_lags = frequencies.iloc[idx_ts][[col for col in frequencies.columns if usecase in col][0]]
    freq = mapping(frequencies.iloc[idx_ts][[col for col in frequencies.columns if usecase in col][0]])

    if method == 'onestep':
        n_forecasts = 1
    elif method == 'multistep':
        n_forecasts = int(np.amin((int(0.2 * len(df)), n_lags)))

    params = {'n_lags': n_lags,
              'freq': freq,
              'n_forecasts':n_forecasts}
    return params


def benchmark(model, method, df, params, one_dataset_metrics, n_epochs):
    valid_p = 0.2
    n_lags = params['n_lags']
    freq = params['freq']
    n_forecasts = params['n_forecasts']
    try:
        if model == 'LSTM':
            m = LSTM(n_lags=n_lags,
                     n_forecasts=n_forecasts,
                     learning_rate=0.01,
                     epochs=n_epochs)
        elif model == 'NP':
            m = NeuralProphet(n_lags=n_lags,
                              n_forecasts=n_forecasts,
                              learning_rate=0.01,
                              epochs=n_epochs)
        elif model == 'DeepAR':
            m = DeepAR(n_lags=int(n_lags),
                       n_forecasts=n_forecasts,
                       learning_rate=0.001,
                       epochs=n_epochs,
                       auto_lr_find=False)
        elif model == 'NBeats':
            m = NBeats(n_lags=int(n_lags),
                       n_forecasts=n_forecasts,
                       learning_rate=0.01,
                       epochs=n_epochs)
        elif model == 'TFT':
            m = TFT(n_lags=int(n_lags),
                    n_forecasts=n_forecasts,
                    learning_rate=0.01,
                    epochs=n_epochs)


        tr, vl = split_df(df, n_lags=n_lags, n_forecasts=n_forecasts, valid_p=valid_p)
        m.fit(tr, freq=freq)
        future = m.make_future_dataframe(vl, periods=0, n_historic_predictions=True)
        forecast = m.predict(future)
        fold = forecast.iloc[n_lags:][[f'yhat{i}' for i in range(1, n_forecasts + 1)]]

        y_predicted = [np.array(fold).diagonal(offset=-i) for i in range(len(fold) - n_forecasts + 1)]
        y = np.array(vl[n_lags:]["y"])
        y_rolled = [y[i: i + n_forecasts] for i in range(len(y) - n_forecasts + 1)]

        y_naive = np.array(vl[n_lags - 1:-1]["y"])
        y_naive_rolled = [y_naive[i: i + n_forecasts] for i in range(len(y_naive) - n_forecasts + 1)]

        smapes = np.mean([smape(y_rolled[i], y_predicted[i]) for i in range(len(y_rolled))])
        mases = np.mean([mase(y_rolled[i], y_predicted[i], y_naive_rolled[i]) for i in range(len(y_rolled))])
        mueses = np.mean([mues(y_rolled[i], y_predicted[i]) for i in range(len(y_rolled))])
        moeses = np.mean([moes(y_rolled[i], y_predicted[i]) for i in range(len(y_rolled))])
        muases = np.mean([muas(y_rolled[i], y_predicted[i]) for i in range(len(y_rolled))])
        moases = np.mean([moas(y_rolled[i], y_predicted[i]) for i in range(len(y_rolled))])

        one_dataset_metrics.update({
            f'smape_{model}_{method}': smapes,
            f'mase_{model}_{method}': mases,
            f'mues_{model}_{method}': mueses,
            f'moes_{model}_{method}': moeses,
            f'muas_{model}_{method}': muases,
            f'moas_{model}_{method}': moases
        })

    except:
        print('error')


def libra(n_datasets, datasets, frequencies, method, n_epochs, usecase, save_res=True):
    metrics = {}
    for i, (dataset_name, df) in enumerate(datasets.items()):
        if i >= n_datasets:
            break
        one_dataset_metrics = {}
        models = ['LSTM', 'NP', 'DeepAR', 'NBeats', 'TFT']
        for model in models:
            params = get_parameters(df, dataset_name, frequencies, method, usecase)
            benchmark(model, method, df, params, one_dataset_metrics, n_epochs)
        metrics.update({dataset_name: one_dataset_metrics})

    if save_res:
        loc_res = '../results_benchmarking/'
        pd.DataFrame(metrics).to_csv(loc_res + f'results_libra_{method}_{usecase}.csv')
    return metrics


if __name__ == '__main__':
    usecase = 'economics'
    datasets, frequencies = get_datasets(usecase)
    methods = ['onestep', 'multistep']
    metrics = libra(2, datasets, frequencies, methods[0], n_epochs=2, usecase=usecase)
    print(metrics)