from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune import CLIReporter
import pytorch_lightning as pl

from neuralprophet import LSTM
from neuralprophet import NeuralProphet
from neuralprophet import DeepAR
from neuralprophet import TFT
from neuralprophet import NBeats
import numpy as np


def tune_hyperparameters(
    model_name,  # NP, LSTM
    df,
    freq,
    num_epochs=100,
    mode="auto",  #'manual'
    config=None,
    num_samples=10,  # number of samples from hyperparameter space
    resources_per_trial={"cpu": 1},
    return_results=True,
):
    """
    Args:
        model_name: str, ['NP', 'LSTM', 'NBeats'] — The model name, which hyperparameters to tune over
        df: pandas.DataFrame — DataFrame containing column 'ds', 'y' with all data
        freq: str — Data step sizes. Frequency of data recording,
            Any valid frequency for pd.date_range, such as '5min', 'D' or 'MS'
        num_epochs: int — number of epochs to train.
        mode: str, ['auto', 'manual'] — a mode of hyperparameter tuning.
            'auto' uses the default configuration of hyperparameters
            'manual' uses the configuration, provided by the user
        config: dict — Dictionary of hyperparameters and tune spaces, from which to chose
            If you provide a config, please, use 'manual' mode.
            Hyperparameter spaces must be provided in ray.tune format:
            (tune.choice, tune.loguniform, tune.grid_search etc.)
        num_samples: int — Number of samples from hyperparameter spaces to check.
            Note, that if you use tune.grid_search, this parameter will be overwritten by the model
        resources_per_trial: dict — Resources per trial setting for ray.tune.run, {'cpu': 1, 'gpu': 2} for example
        return_results: bool — whether to return dataframe with results of num_samples trials
    Returns:
        best_config: dict — dictionary with best chosen parameters
        result_df: pandas.DataFrame — DataFrame with results of each trial (only if return_results == True)
    """

    if mode == "manual":
        assert type(config) != type(
            None
        ), """If mode is set to manual 
        you must provide a config"""
    if mode == "auto":
        assert type(config) == type(
            None
        ), """If mode is set to auto, 
        your config will not be used. Please, use the mode \"auto\" if you wish to use your own config"""

    def train_NP_tune(config, num_epochs=num_epochs):
        m = NeuralProphet(**config, epochs=num_epochs)
        train_loader, val_loader, model = m._hyperparameter_optimization(df, freq)
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            progress_bar_refresh_rate=0,
            num_sanity_val_steps=0,
            callbacks=TuneReportCallback({"loss": "val_loss"}, on="validation_end"),
            checkpoint_callback=False,
            logger=False,
        )
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    def train_LSTM_tune(config, num_epochs=num_epochs):
        m = LSTM(**config, epochs=num_epochs)
        train_loader, val_loader, model = m._hyperparameter_optimization(df, freq)

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            progress_bar_refresh_rate=0,
            num_sanity_val_steps=0,
            callbacks=TuneReportCallback({"loss": "val_loss"}, on="validation_end"),
            checkpoint_callback=False,
            logger=False,
        )
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    def train_NBeats_tune(config, num_epochs=num_epochs):
        m = NBeats(**config, epochs=num_epochs)
        train_loader, val_loader, model = m._hyperparameter_optimization(df, freq)
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            progress_bar_refresh_rate=0,
            num_sanity_val_steps=0,
            callbacks=TuneReportCallback({"loss": "val_loss"}, on="validation_end"),
            checkpoint_callback=False,
            logger=False,
        )
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    def train_TFT_tune(config, num_epochs=num_epochs):
        m = TFT(**config, epochs=num_epochs)
        train_loader, val_loader, model = m._hyperparameter_optimization(df, freq)
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            progress_bar_refresh_rate=0,
            num_sanity_val_steps=0,
            callbacks=TuneReportCallback({"loss": "val_loss"}, on="validation_end"),
            checkpoint_callback=False,
            logger=False,
        )
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    def train_DeepAR_tune(config, num_epochs=num_epochs):
        m = DeepAR(**config, epochs=num_epochs)
        train_loader, val_loader, model = m._hyperparameter_optimization(df, freq)
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            progress_bar_refresh_rate=0,
            num_sanity_val_steps=0,
            callbacks=TuneReportCallback({"loss": "val_loss"}, on="validation_end"),
            checkpoint_callback=False,
            logger=False,
        )
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    if model_name == "NP":
        train_func = train_NP_tune
        if mode == "auto":
            config = {
                "growth": tune.choice(["off", "linear"]),
                "n_changepoints": tune.choice([5, 10, 100]),
                "changepoints_range": tune.choice([0.5, 0.8, 0.9]),
                "trend_reg": tune.choice([0, 0.5, 1, 10]),
                "yearly_seasonality": tune.choice([True, False]),
                "weekly_seasonality": tune.choice([True, False]),
                "daily_seasonality": tune.choice([True, False]),
                "seasonality_mode": tune.choice(["additive", "multiplicative"]),
                "seasonality_reg": tune.choice([0, 0.5, 1, 10]),
                "n_lags": tune.choice([10, 30, 100]),
                "d_hidden": tune.choice([8, 64, 128]),
                "num_hidden_layers": tune.choice([2, 8, 16]),
                "ar_sparsity": tune.choice([0.1, 0.3, 0.8]),
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "loss_func": tune.choice(["Huber", "MSE"]),
                "normalize": tune.choice(["auto", "soft", "off", "minmax", "standardize"]),
            }

    elif model_name == "LSTM":
        train_func = train_LSTM_tune
        if mode == "auto":
            config = {
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "d_hidden": tune.choice([8, 64, 128]),
                "n_lags": tune.choice([10, 30, 100]),
                "num_hidden_layers": tune.choice([2, 8, 16]),
                "lstm_bias": tune.choice([False, True]),
                "lstm_bidirectional": tune.choice([False, True]),
            }

    elif model_name == "NBeats":
        train_func = train_NBeats_tune
        if mode == "auto":
            config = {
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "n_lags": tune.choice([30, 100]),
            }

    elif model_name == "TFT":
        train_func = train_TFT_tune
        if mode == "auto":
            config = {
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "n_lags": tune.choice([10, 30, 100]),
                "hidden_size": tune.choice([8, 16, 32]),
                "attention_head_size": tune.choice([1, 2]),
            }
    elif model_name == "DeepAR":
        train_func = train_DeepAR_tune
        if mode == "auto":
            config = {
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "n_lags": tune.choice([10, 30, 100]),
                "hidden_size": tune.choice([8, 16, 32]),
                "rnn_layers": tune.choice([1, 2]),
            }

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=np.min([10, num_epochs]), reduction_factor=2)
    reporter = CLIReporter(parameter_columns=list(config.keys()), metric_columns=["loss", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(train_func, num_epochs=num_epochs),
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        verbose=False,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_hyperparameters",
        log_to_file=False,
        checkpoint_freq=0,
        raise_on_failed_trial=False,
    )

    if return_results:
        return analysis.best_config, analysis.results_df
    else:
        return analysis.best_config
