from neuralprophet.models.DeepAR import LightDeepAR

import numpy as np
import pandas as pd

import torch
import logging

from neuralprophet.utils import df_utils
from neuralprophet.tools import metrics

from neuralprophet.tools.plot_forecast import plot
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.data.encoders import GroupNormalizer

from pytorch_lightning.callbacks import EarlyStopping

import pytorch_lightning as pl

log = logging.getLogger("AdditionalModels.DeepAR")


class DeepAR:
    def __init__(
            self,
            n_lags=60,
            n_forecasts=20,
            batch_size=None,
            epochs=100,
            num_gpus=0,
            patience_early_stopping=10,
            early_stop=True,
            learning_rate=3e-2,
            auto_lr_find=False,
            num_workers=3,
            loss_func="normaldistributionloss",
            hidden_size=32,
            rnn_layers=2,
            dropout=0.1,
    ):
        '''
        Args:
            n_lags:
            n_forecasts:
            batch_size:
            epochs:
            num_gpus:
            patience_early_stopping:
            early_stop:
            learning_rate:
            auto_lr_find:
            num_workers:
            loss_func:
            hidden_size:
            rnn_layers:
            dropout:
        '''

        self.batch_size = batch_size

        self.epochs = epochs
        self.num_gpus = num_gpus
        self.patience_early_stopping = patience_early_stopping
        self.early_stop = early_stop
        self.learning_rate = learning_rate
        self.auto_lr_find = auto_lr_find
        self.num_workers = num_workers

        self.context_length = n_lags
        self.prediction_length = n_forecasts

        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.loss_func = loss_func

        self.fitted = False
        self.freq = None

        if type(self.loss_func) == str:
            if self.loss_func.lower() in ["huber", "smoothl1", "smoothl1loss"]:
                self.loss_func = torch.nn.SmoothL1Loss()
            elif self.loss_func.lower() in ["mae", "l1", "l1loss"]:
                self.loss_func = torch.nn.L1Loss()
            elif self.loss_func.lower() in ["mse", "mseloss", "l2", "l2loss"]:
                self.loss_func = torch.nn.MSELoss()
            elif self.loss_func.lower() in ["normaldistloss", "ndl", "normaldistributionloss"]:
                self.loss_func = NormalDistributionLoss()
            else:
                raise NotImplementedError("Loss function {} name not defined".format(self.loss_func))
        elif callable(self.loss_func):
            pass
        elif hasattr(torch.nn.modules.loss, self.loss_func.__class__.__name__):
            pass
        else:
            raise NotImplementedError("Loss function {} not found".format(self.loss_func))

        self.metrics = metrics.MetricsCollection(
            metrics=[metrics.LossMetric(torch.nn.SmoothL1Loss()), metrics.MAE(), metrics.MSE(), ],
            value_metrics=[
                # metrics.ValueMetric("Loss"),
            ],
        )

        self.val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])

    def _init_model(self, training, train_dataloader):
        model = LightDeepAR.from_dataset(
            training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            loss=self.loss_func,
        )

        if self.auto_lr_find:
            res = self.trainer.tuner.lr_find(model, train_dataloader=train_dataloader, min_lr=1e-5, max_lr=1e2)
            model.hparams.learning_rate = res.suggestion()
            self.learning_rate = res.suggestion()

        return model

    def set_auto_batch_epoch(
            self, n_data: int, min_batch: int = 16, max_batch: int = 256, min_epoch: int = 40, max_epoch: int = 400,
    ):
        assert n_data >= 1
        log_data = np.log10(n_data)
        if self.batch_size is None:
            self.batch_size = 2 ** int(2 + log_data)
            self.batch_size = min(max_batch, max(min_batch, self.batch_size))
            self.batch_size = min(n_data, self.batch_size)

    def _create_dataset(self, df, valid_p=0.2):
        df = df_utils.check_dataframe(df)
        df = self._handle_missing_data(df)
        df = df[["ds", "y"]]
        df["time_idx"] = range(df.shape[0])
        df["series"] = 0
        self.n_data = df.shape[0]
        self.set_auto_batch_epoch(self.n_data)

        training_cutoff = df.shape[0] - int(valid_p * df.shape[0])

        training = TimeSeriesDataSet(
            df.iloc[:training_cutoff],
            time_idx="time_idx",
            target="y",
            categorical_encoders={"series": NaNLabelEncoder().fit(df.series)},
            group_ids=["series"],
            min_encoder_length=self.context_length,
            max_encoder_length=self.context_length,
            max_prediction_length=self.prediction_length,
            min_prediction_length=self.prediction_length,
            time_varying_unknown_reals=["y"],
            target_normalizer=GroupNormalizer(groups=["series"]),
            randomize_length=None,
            add_relative_time_idx=False,
            add_target_scales=False,
        )

        validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff)
        train_dataloader = training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=self.num_workers)
        val_dataloader = validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=self.num_workers)

        return training, train_dataloader, val_dataloader

    def _handle_missing_data(self, df, predicting=False):
        """Checks, auto-imputes and normalizes new data

        Args:
            df (pd.DataFrame): raw data with columns 'ds' and 'y'
            predicting (bool): when no lags, allow NA values in 'y' of forecast series or 'y' to miss completely

        Returns:
            pre-processed df
        """

        impute_limit_linear = 5
        impute_rolling = 20

        df, missing_dates = df_utils.add_missing_dates_nan(df, freq=self.freq)

        # impute missing values
        data_columns = []
        data_columns.append("y")

        for column in data_columns:
            sum_na = sum(df[column].isnull())
            if sum_na > 0:
                df.loc[:, column], remaining_na = df_utils.fill_linear_then_rolling_avg(
                    df[column], limit_linear=impute_limit_linear, rolling=impute_rolling,
                )
                if remaining_na > 0:
                    raise ValueError(
                        "More than {} consecutive missing values encountered in column {}. "
                        "{} NA remain. Please preprocess data manually.".format(
                            2 * impute_limit_linear + impute_rolling, column, remaining_na
                        )
                    )
        return df

    def _train(self, training, train_dataloader, val_dataloader, hyperparameter_optim=False):
        callbacks = []
        if self.early_stop:
            early_stop_callback = EarlyStopping(
                monitor="val_loss", min_delta=1e-4, patience=self.patience_early_stopping, verbose=False, mode="min"
            )
            callbacks = [early_stop_callback]

        self.trainer = pl.Trainer(
            max_epochs=self.epochs,
            gpus=self.num_gpus,
            weights_summary="top",
            gradient_clip_val=0.1,
            callbacks=callbacks,
            checkpoint_callback=False,
            logger=False,
            num_sanity_val_steps=0,
        )

        self.model = self._init_model(training, train_dataloader)
        self.model.set_forecaster(self)

        self.metrics.reset()
        self.val_metrics.reset()

        if hyperparameter_optim:
            return self.model
        else:

            self.trainer.fit(
                self.model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
            )
            self.fitted = True

            metrics_df = self.metrics.get_stored_as_df()
            metrics_df_val = self.val_metrics.get_stored_as_df()
            for col in metrics_df_val.columns:
                metrics_df["{}_val".format(col)] = metrics_df_val[col]

            return metrics_df

    def fit(self, df, freq, valid_p=0.2):
        self.freq = freq
        training, train_dataloader, val_dataloader = self._create_dataset(df, valid_p)
        metrics_df = self._train(training, train_dataloader, val_dataloader)
        return metrics_df

    def _hyperparameter_optimization(self, df, valid_p=0.2):
        training, train_dataloader, val_dataloader = self._create_dataset(df, valid_p)
        model = self._train(training, train_dataloader, val_dataloader, hyperparameter_optim=True)
        return train_dataloader, val_dataloader, model

    def make_future_dataframe(self, df, periods=0, n_historic_predictions=0):
        """
        Creates a dataframe for prediction
        Args:
            periods: number of future periods to forecast
            n_historic_predictions: number of historic_predictions to include in forecast

        Returns:
            future_dataframe: DataFrame, used further for prediction
        """

        if isinstance(n_historic_predictions, bool):
            if n_historic_predictions:
                n_historic_predictions = len(df) - self.context_length
            else:
                n_historic_predictions = 0
        elif not isinstance(n_historic_predictions, int):
            log.error("non-integer value for n_historic_predictions set to zero.")
            n_historic_predictions = 0
        if periods == 0 and n_historic_predictions == 0:
            raise ValueError("Set either history or future to contain more than zero values.")

        if len(df) < self.context_length:
            raise ValueError("Insufficient data for a prediction")
        elif len(df) < self.context_length + n_historic_predictions:
            log.warning(
                "Insufficient data for {} historic forecasts, reduced to {}.".format(
                    n_historic_predictions, len(df) - self.context_length
                )
            )
            n_historic_predictions = len(df) - self.context_length

        if periods > 0 and periods != self.prediction_length:
            periods = self.prediction_length
            log.warning(
                "Number of forecast steps is defined by n_forecasts. " "Adjusted to {}.".format(self.prediction_length)
            )

        self.periods = periods

        self.n_historic_predictions = n_historic_predictions

        df = df.copy(deep=True)

        df = df_utils.check_dataframe(df)
        df = self._handle_missing_data(df)
        df = df[["ds", "y"]]
        df["time_idx"] = range(df.shape[0])
        df["series"] = 0
        self.n_data = df.shape[0]

        encoder_data = df[lambda x: x.time_idx > x.time_idx.max() - (self.context_length + n_historic_predictions)]
        if periods != 0:
            last_data = df[lambda x: x.time_idx == x.time_idx.max()]
            decoder_data = pd.concat(
                [last_data.assign(ds=lambda x: x.ds + pd.offsets.MonthBegin(i)) for i in range(1, periods + 1)],
                ignore_index=True,
            )
            decoder_data["time_idx"] = range(
                decoder_data["time_idx"].iloc[0] + 1, decoder_data["time_idx"].iloc[0] + periods + 1
            )
            decoder_data["ds"] = pd.date_range(start=encoder_data["ds"].iloc[-1], periods=periods + 1, freq=self.freq)[
                                 1:]
            future_dataframe = pd.concat([encoder_data, decoder_data], ignore_index=True)
        elif periods == 0:
            future_dataframe = encoder_data
        return future_dataframe

    def predict(self, future_dataframe):
        """
        Predicts based on the future_dataframe. Should be called only after make_future_dataframe is called
        Args:
            future_dataframe: DataFrame form make_future_dataframe function
        Returns:
            forecast dataframe
        """

        if self.fitted is False:
            log.warning("Model has not been fitted. Predictions will be random.")

        future_dataframe = future_dataframe.copy(deep=True)

        testing = TimeSeriesDataSet(
            future_dataframe,
            time_idx="time_idx",
            target="y",
            categorical_encoders={"series": NaNLabelEncoder().fit(future_dataframe.series)},
            group_ids=["series"],
            min_encoder_length=self.context_length,
            max_encoder_length=self.context_length,
            max_prediction_length=self.prediction_length,
            min_prediction_length=self.prediction_length,
            time_varying_unknown_reals=["y"],
            target_normalizer=GroupNormalizer(groups=["series"]),
            randomize_length=None,
            add_relative_time_idx=False,
            add_target_scales=False,
        )

        y_predicted = self.model.predict(testing, mode="prediction")  # , return_x=True)

        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get("padder", np.nan)
            vector[: pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value

        y_pred_padded = np.pad(y_predicted, self.prediction_length, pad_with)[
                        self.prediction_length: -1, self.prediction_length: -self.prediction_length
                        ]
        y_pred_padded = np.vstack([np.roll(y_pred_padded[:, i], i, axis=0) for i in range(y_pred_padded.shape[1])]).T

        result = pd.DataFrame(
            np.ones(shape=(len(future_dataframe), (2 + self.prediction_length))) * np.nan,
            columns=["ds", "y"] + [f"yhat{i}" for i in range(1, self.prediction_length + 1)],
        )
        result["ds"] = future_dataframe["ds"]

        result.loc[: len(future_dataframe) - (self.periods + 1), "y"] = (
            future_dataframe["y"].iloc[: len(future_dataframe) - (self.periods)].values
        )

        first_part = result.iloc[: self.context_length]
        second_part = result.iloc[self.context_length:]

        second_part.loc[:, [col for col in second_part.columns[2:]]] = y_pred_padded
        result = pd.concat([first_part, second_part])
        for i in range(1, self.prediction_length + 1):
            result[f"residual{i}"] = result[f"yhat{i}"] - result["y"]

        return result

    def plot(self, fcst, ax=None, xlabel="ds", ylabel="y", figsize=(10, 6)):
        """Plot the NeuralProphet forecast, including history.

        Args:
            fcst (pd.DataFrame): output of self.predict.
            ax (matplotlib axes): Optional, matplotlib axes on which to plot.
            xlabel (string): label name on X-axis
            ylabel (string): label name on Y-axis
            figsize (tuple):   width, height in inches. default: (10, 6)

        Returns:
            A matplotlib figure.
        """

        return plot(fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize, )
