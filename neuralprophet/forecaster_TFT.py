from neuralprophet.models.TFT import LightTFT

import numpy as np
import pandas as pd

import torch
import logging


from neuralprophet.utils import df_utils
from neuralprophet.tools import metrics
from neuralprophet.tools.plot_forecast import plot
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

from pytorch_forecasting.data.encoders import GroupNormalizer

from pytorch_lightning.callbacks import EarlyStopping

import pytorch_lightning as pl

log = logging.getLogger("AdditionalModels.forecaster")




class TemporalFusionTransformerNP:
    def __init__(
            self,
            context_length=60,
            output_size=20,
            batch_size=None,
            epochs=100,
            num_gpus=0,
            patience_early_stopping=10,
            early_stop=True,
            learning_rate=3e-2,
            auto_lr_find=True,
            num_workers=3,
            loss_func="QuantileLoss",
            hidden_size=32,
            attention_head_size=1,
            hidden_continuous_size=8,
            # rnn_layers=2,
            dropout=0.1,
    ):

        self.batch_size = batch_size

        self.epochs = epochs
        self.num_gpus = num_gpus
        self.patience_early_stopping = patience_early_stopping
        self.early_stop = early_stop
        self.learning_rate = learning_rate
        self.auto_lr_find = auto_lr_find
        self.num_workers = num_workers

        self.context_length = context_length
        self.output_size = output_size

        self.hidden_size = hidden_size
        # self.rnn_layers = rnn_layers
        self.attention_head_size = attention_head_size
        self.hidden_continuous_size = hidden_continuous_size
        self.dropout = dropout
        self.loss_func = loss_func

        self.fitted = False

        if type(self.loss_func) == str:
            if self.loss_func.lower() in ["huber", "smoothl1", "smoothl1loss"]:
                self.loss_func = torch.nn.SmoothL1Loss()
            elif self.loss_func.lower() in ["mae", "l1", "l1loss"]:
                self.loss_func = torch.nn.L1Loss()
            elif self.loss_func.lower() in ["mse", "mseloss", "l2", "l2loss"]:
                self.loss_func = torch.nn.MSELoss()
            elif self.loss_func.lower() in ["quantileloss"]:
                self.loss_func = QuantileLoss()
            else:
                raise NotImplementedError("Loss function {} name not defined".format(self.loss_func))
        elif callable(self.loss_func):
            pass
        elif hasattr(torch.nn.modules.loss, self.loss_func.__class__.__name__):
            pass
        else:
            raise NotImplementedError("Loss function {} not found".format(self.loss_func))

        self.metrics = metrics.MetricsCollection(
            metrics=[
                metrics.LossMetric(torch.nn.SmoothL1Loss()),
                metrics.MAE(),
                metrics.MSE(),
            ],
            value_metrics=[
                # metrics.ValueMetric("Loss"),
            ],
        )

        self.val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])

    def _init_model(self, training, train_dataloader):

        model = LightTFT.from_dataset(
            training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=self.loss_func,
        )

        if self.auto_lr_find:
            res = self.trainer.tuner.lr_find(model, train_dataloader=train_dataloader, min_lr=1e-5, max_lr=1e2)
            model.hparams.learning_rate = res.suggestion()
            self.learning_rate = res.suggestion()

        return model

    def set_auto_batch_epoch(
            self,
            n_data: int,
            min_batch: int = 16,
            max_batch: int = 256,
            min_epoch: int = 40,
            max_epoch: int = 400,
    ):
        assert n_data >= 1
        log_data = np.log10(n_data)
        if self.batch_size is None:
            self.batch_size = 2 ** int(2 + log_data)
            self.batch_size = min(max_batch, max(min_batch, self.batch_size))
            self.batch_size = min(n_data, self.batch_size)

    def _create_dataset(self, df, freq, valid_p=0.2):
        df = df_utils.check_dataframe(df)
        df = self._handle_missing_data(df, freq)
        df = df[["ds", "y"]]
        df["time_idx"] = range(df.shape[0])
        df["series"] = 0
        self.n_data = df.shape[0]
        self.set_auto_batch_epoch(self.n_data)

        training_cutoff = df.shape[0] - int(valid_p * df.shape[0])

        # max_encoder_length = 36

        training = TimeSeriesDataSet(
            df.iloc[:training_cutoff],
            time_idx="time_idx",
            target="y",
            categorical_encoders={"series": NaNLabelEncoder().fit(df.series)},
            group_ids=["series"],
            min_encoder_length=self.context_length,
            max_encoder_length=self.context_length,
            max_prediction_length=self.output_size,
            min_prediction_length=1,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["y"],
            target_normalizer=GroupNormalizer(groups=["series"], transformation="softplus", center=False),
            # randomize_length=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff)
        train_dataloader = training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=self.num_workers)
        val_dataloader = validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=self.num_workers)

        return training, train_dataloader, val_dataloader

    def _handle_missing_data(sefl, df, freq, predicting=False):
        """Checks, auto-imputes and normalizes new data
        Args:
            df (pd.DataFrame): raw data with columns 'ds' and 'y'
            freq (str): data frequency
            predicting (bool): when no lags, allow NA values in 'y' of forecast series or 'y' to miss completely
        Returns:
            pre-processed df
        """

        impute_limit_linear = 5
        impute_rolling = 20

        df, missing_dates = df_utils.add_missing_dates_nan(df, freq=freq)

        # impute missing values
        data_columns = []
        data_columns.append("y")

        for column in data_columns:
            sum_na = sum(df[column].isnull())
            if sum_na > 0:
                df.loc[:, column], remaining_na = df_utils.fill_linear_then_rolling_avg(
                    df[column],
                    limit_linear=impute_limit_linear,
                    rolling=impute_rolling,
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

        self.trainer.fit(
            self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        self.metrics.reset()
        self.val_metrics.reset()

        if hyperparameter_optim:
            return self.model
        else:

            self.trainer.fit(
                self.model,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            self.fitted = True

            metrics_df = self.metrics.get_stored_as_df()
            metrics_df_val = self.val_metrics.get_stored_as_df()
            for col in metrics_df_val.columns:
                metrics_df["{}_val".format(col)] = metrics_df_val[col]

            return metrics_df

    def fit(self, df, freq, valid_p=0.2):

        training, train_dataloader, val_dataloader = self._create_dataset(df, freq, valid_p)
        # print(next(iter(train_dataloader)))
        metrics_df = self._train(training, train_dataloader, val_dataloader)
        return metrics_df

    def _hyperparameter_optimization(self, df, freq, valid_p=0.2):
        train_dataloader, val_dataloader = self._create_dataset(self, df, freq, valid_p)
        self.model = self._init_model(train_dataloader)

    def make_future_dataframe(self, df, freq, periods=0, n_historic_predictions=0):
        """
        Creates a dataframe for prediction
        Args:
            periods: number of future periods to forecast
            n_historic_predictions: number of historic_predictions to include in forecast
        Returns:
            future_dataframe: DataFrame, used further for prediction
        """

        self.periods = periods
        self.n_historic_predictions = n_historic_predictions

        df = df.copy(deep=True)

        df = df_utils.check_dataframe(df)
        df = self._handle_missing_data(df, freq)
        df = df[["ds", "y"]]
        df["time_idx"] = range(df.shape[0])
        df["series"] = 0
        self.n_data = df.shape[0]

        encoder_data = df[lambda x: x.time_idx > x.time_idx.max() - (self.context_length + n_historic_predictions)]
        last_data = df[lambda x: x.time_idx == x.time_idx.max()]
        decoder_data = pd.concat(
            [last_data.assign(ds=lambda x: x.ds + pd.offsets.MonthBegin(i)) for i in range(1, periods + 1)],
            ignore_index=True,
        )
        decoder_data["time_idx"] = range(
            decoder_data["time_idx"].iloc[0] + 1, decoder_data["time_idx"].iloc[0] + periods + 1
        )
        decoder_data["ds"] = pd.date_range(start=encoder_data["ds"].iloc[-1], periods=periods + 1, freq=freq)[1:]
        future_dataframe = pd.concat([encoder_data, decoder_data], ignore_index=True)

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
            max_prediction_length=self.periods + self.n_historic_predictions,
            min_prediction_length=self.periods + self.n_historic_predictions,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["y"],
            target_normalizer=GroupNormalizer(groups=["series"], transformation="softplus", center=False),
            randomize_length=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        new_raw_predictions, new_x = self.model.predict(testing, mode="raw", return_x=True)
        y_predicted = self.model.to_prediction(new_raw_predictions).detach().cpu()[0, : new_x["decoder_lengths"][0]]
        y_predicted = y_predicted.detach().numpy()

        future_dataframe.loc[len(future_dataframe) - self.periods:, "y"] = None
        future_dataframe["yhat1"] = None
        future_dataframe.loc[len(future_dataframe) - len(y_predicted):, "yhat1"] = y_predicted
        cols = ["ds", "y", "yhat1"]  # cols to keep from df
        df_forecast = pd.concat((future_dataframe[cols],), axis=1)
        df_forecast["residual1"] = df_forecast["yhat1"] - df_forecast["y"]

        return df_forecast

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

        return plot(
            fcst=fcst,
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
        )