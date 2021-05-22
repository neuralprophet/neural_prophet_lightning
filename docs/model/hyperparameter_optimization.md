# Hyperparameter optimization
We introduced a new module for hyperparameter tuning.
It relies on Ray Tune library, as it has an easy integration with Pytorch Lightning models and allows distributed training.
This module is located in `neuralprophet/hyperparameter_tuner.py` file.
It has two modes: auto and manual.
The automated mode runs hyperparameter tuning on predefined by us sets of hyperparameters.
It is useful as a first step for new users and fast tuning.
The manual mode allows users to define their own hyperparameter spaces to tune over. 
In `example_notebooks/hyperparameter_optimization_example.ipynb` we provide users with a basic example of how to use this module.

Here is an example of hyperparameter optimization usage:
```
best_params, results_df = tune_hyperparameters('NP', df, freq)
```

An example output of this function looks as follows: 
```
{'growth': 'off',
 'n_changepoints': 100,
 'changepoints_range': 0.8,
 'trend_reg': 0.0,
 'yearly_seasonality': False,
 'weekly_seasonality': False,
 'daily_seasonality': False,
 'seasonality_mode': 'additive',
 'seasonality_reg': 0.5,
 'n_lags': 100,
 'd_hidden': 8,
 'num_hidden_layers': 2,
 'ar_sparsity': 0.8,
 'learning_rate': 0.010444235692186717,
 'loss_func': 'Huber',
 'normalize': 'minmax'}
```
It can be used further directly into NeuralProphet configuration initialization.