from neuralprophet.additional_models import LightLSTM, NBeats
import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from neuralprophet import configure
from neuralprophet import time_dataset
from neuralprophet import df_utils
from neuralprophet import utils
from neuralprophet import metrics
from neuralprophet.plot_forecast import plot
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

import pytorch_lightning as pl



log = logging.getLogger("AdditionalModels.forecaster")


class LSTM:
    """LSTM forecaster."""

    def __init__(
        self,
        n_lags=10,
        n_forecasts=1,
        num_hidden_layers=1,
        d_hidden=10,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        loss_func="Huber",
        optimizer="AdamW",
        train_speed=None,
        normalize="auto",
        impute_missing=True,
        lstm_bias=True,
        lstm_bidirectional=False,
    ):
        """
        Args:

            ## Model Config
            n_forecasts (int): Number of steps ahead of prediction time step to forecast.
            num_hidden_layers (int): number of hidden layer to include in AR-Net. defaults to 0.
            d_hidden (int): dimension of hidden layers of the AR-Net. Ignored if num_hidden_layers == 0.

            ## Train Config
            learning_rate (float): Maximum learning rate setting for 1cycle policy scheduler.
                default: None: Automatically sets the learning_rate based on a learning rate range test.
                For manual values, try values ~0.001-10.
            epochs (int): Number of epochs (complete iterations over dataset) to train model.
                default: None: Automatically sets the number of epochs based on dataset size.
                    For best results also leave batch_size to None.
                For manual values, try ~5-500.
            batch_size (int): Number of samples per mini-batch.
                default: None: Automatically sets the batch_size based on dataset size.
                    For best results also leave epochs to None.
                For manual values, try ~1-512.
            loss_func (str, torch.nn.modules.loss._Loss, 'typing.Callable'):
                Type of loss to use: str ['Huber', 'MSE'],
                or torch loss or callable for custom loss, eg. asymmetric Huber loss

            ## Data config
            normalize (str): Type of normalization to apply to the time series.
                options: ['auto', 'soft', 'off', 'minmax, 'standardize']
                default: 'auto' uses 'minmax' if variable is binary, else 'soft'
                'soft' scales minimum to 0.1 and the 90th quantile to 0.9
            impute_missing (bool): whether to automatically impute missing dates/values
                imputation follows a linear method up to 10 missing values, more are filled with trend.

            ## LSTM specific
            bias (bool): If False, then the layer does not use bias weights b_ih and b_hh. Default: True
            bidirectional (bool): If True, becomes a bidirectional LSTM. Default: False

        """

        kwargs = locals()

        # General
        self.name = "LSTM"
        self.n_forecasts = n_forecasts
        self.n_lags = n_lags

        # Data Preprocessing
        self.normalize = normalize
        self.impute_missing = impute_missing
        self.impute_limit_linear = 5
        self.impute_rolling = 20

        # Training
        self.config_train = configure.from_kwargs(configure.Train, kwargs)

        self.metrics = metrics.MetricsCollection(
            metrics=[
                metrics.LossMetric(self.config_train.loss_func),
                metrics.MAE(),
                metrics.MSE(),
            ],
            value_metrics=[
                # metrics.ValueMetric("Loss"),
            ],
        )

        # Model
        self.config_model = configure.from_kwargs(configure.Model, kwargs)

        # LSTM specific
        self.lstm_bias = lstm_bias
        self.lstm_bidirectional = lstm_bidirectional

        # set during fit()
        self.data_freq = None

        # Set during _train()
        self.fitted = False
        self.data_params = None
        self.optimizer = None
        self.scheduler = None
        self.model = None

        # set during prediction
        self.future_periods = None
        # later set by user (optional)
        self.highlight_forecast_step_n = None
        self.true_ar_weights = None

    def _init_model(self):
        """Build Pytorch model with configured hyperparamters.

        Returns:
        """
        self.model = LightLSTM(
            input_size=self.n_lags,
            hidden_size=self.config_model.d_hidden,
            num_layers=self.config_model.num_hidden_layers,
            bias=self.lstm_bias,
            bidirectional=self.lstm_bidirectional,
            n_forecasts=self.n_forecasts,
        )

        self.model.set_loss_func(self.config_train.loss_func)
        self.model.set_forecaster(self)

        log.debug(self.model)
        return self.model

    def _create_dataset(self, df, predict_mode):
        """Construct dataset from dataframe.

        (Configured Hyperparameters can be overridden by explicitly supplying them.
        Useful to predict a single model component.)

        Args:
            df (pd.DataFrame): containing original and normalized columns 'ds', 'y', 't', 'y_scaled'
            predict_mode (bool): False includes target values.
                True does not include targets but includes entire dataset as input
        Returns:
            TimeDataset
        """
        return time_dataset.TimeDataset(
            df,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            predict_mode=predict_mode,
        )

    def _handle_missing_data(self, df, freq, predicting=False):
        """Checks, auto-imputes and normalizes new data

        Args:
            df (pd.DataFrame): raw data with columns 'ds' and 'y'
            freq (str): data frequency
            predicting (bool): when no lags, allow NA values in 'y' of forecast series or 'y' to miss completely

        Returns:
            pre-processed df
        """

        # add missing dates for autoregression modelling
        df, missing_dates = df_utils.add_missing_dates_nan(df, freq=freq)
        if missing_dates > 0:
            if self.impute_missing:
                log.info("{} missing dates added.".format(missing_dates))
            else:
                raise ValueError(
                    "{} missing dates found. Please preprocess data manually or set impute_missing to True.".format(
                        missing_dates
                    )
                )

        # impute missing values
        data_columns = []
        data_columns.append("y")

        for column in data_columns:
            sum_na = sum(df[column].isnull())
            if sum_na > 0:
                if self.impute_missing:
                    df.loc[:, column], remaining_na = df_utils.fill_linear_then_rolling_avg(
                        df[column],
                        limit_linear=self.impute_limit_linear,
                        rolling=self.impute_rolling,
                    )
                    log.info("{} NaN values in column {} were auto-imputed.".format(sum_na - remaining_na, column))
                    if remaining_na > 0:
                        raise ValueError(
                            "More than {} consecutive missing values encountered in column {}. "
                            "{} NA remain. Please preprocess data manually.".format(
                                2 * self.impute_limit_linear + self.impute_rolling, column, remaining_na
                            )
                        )
                else:  # fail because set to not impute missing
                    raise ValueError(
                        "Missing values found. " "Please preprocess data manually or set impute_missing to True."
                    )
        return df

    def _init_train_loader(self, df):
        """Executes data preparation steps and initiates training procedure.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with training data

        Returns:
            torch DataLoader
        """
        if not self.fitted:
            self.data_params = df_utils.init_data_params(df, normalize=self.normalize)

        df = df_utils.normalize(df, self.data_params)
        self.config_train.set_auto_batch_epoch(n_data=len(df))
        self.config_train.apply_train_speed(batch=True, epoch=True)
        dataset = self._create_dataset(df, predict_mode=False)  # needs to be called after set_auto_seasonalities
        loader = DataLoader(dataset, batch_size=self.config_train.batch_size, shuffle=True)
        self.loader_size = len(loader)

        if not self.fitted:
            self.model = self._init_model()  # needs to be called after set_auto_seasonalities

        assert self.config_train.learning_rate is not None, "Please, provide a learning rate"

        self.config_train.apply_train_speed(lr=True)
        self.optimizer = self.config_train.get_optimizer(self.model.parameters())
        ######
        self.model.set_optimizer(self.optimizer)
        self.scheduler = self.config_train.get_scheduler(self.optimizer, steps_per_epoch=len(loader))
        self.model.set_scheduler(self.scheduler)
        ######
        return loader

    def _init_val_loader(self, df):
        """Executes data preparation steps and initiates evaluation procedure.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with validation data

        Returns:
            torch DataLoader
        """
        df = df_utils.normalize(df, self.data_params)
        dataset = self._create_dataset(df, predict_mode=False)
        loader = DataLoader(dataset, batch_size=min(1024, len(dataset)), shuffle=False, drop_last=False)
        return loader

    def _train(self, df, df_val=None, progress_bar=True, plot_live_loss=False, hyperparameter_optim=False):
        """Execute model training procedure for a configured number of epochs.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with training data
            df_val (pd.DataFrame): containing column 'ds', 'y' with validation data
            progress_bar (bool): display updating progress bar
            plot_live_loss (bool): plot live training loss,
                requires [live] install or livelossplot package installed.
        Returns:
            df with metrics
        """
        if plot_live_loss:
            try:
                from livelossplot import PlotLosses
            except:
                plot_live_loss = False
                log.warning(
                    "To plot live loss, please install neuralprophet[live]."
                    "Using pip: 'pip install neuralprophet[live]'"
                    "Or install the missing package manually: 'pip install livelossplot'",
                    exc_info=True,
                )

        loader = self._init_train_loader(df)
        val = df_val is not None
        ## Metrics
        if self.highlight_forecast_step_n is not None:
            self.metrics.add_specific_target(target_pos=self.highlight_forecast_step_n - 1)
        if not self.normalize == "off":
            self.metrics.set_shift_scale((self.data_params["y"].shift, self.data_params["y"].scale))
        if val:
            val_loader = self._init_val_loader(df_val)
            val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])

            self.val_metrics = val_metrics

        ## Run
        start = time.time()
        if progress_bar:
            training_loop = tqdm(
                range(self.config_train.epochs), total=self.config_train.epochs, leave=log.getEffectiveLevel() <= 20
            )
        else:
            training_loop = range(self.config_train.epochs)
        if plot_live_loss:
            live_out = ["MatplotlibPlot"]
            if not progress_bar:
                live_out.append("ExtremaPrinter")
            live_loss = PlotLosses(outputs=live_out)

        self.metrics.reset()
        if val:
            self.val_metrics.reset()

        self.trainer = Trainer(
            max_epochs=self.config_train.epochs,
            checkpoint_callback=False,
            logger=False
            # logger = log
        )

        if hyperparameter_optim:
            return loader, val_loader, self.model
        else:
            if val:
                self.trainer.fit(self.model, train_dataloader=loader, val_dataloaders=val_loader)
            else:
                self.trainer.fit(self.model, train_dataloader=loader)

            ## Metrics
            log.debug("Train Time: {:8.3f}".format(time.time() - start))
            log.debug("Total Batches: {}".format(self.metrics.total_updates))

            metrics_df = self.metrics.get_stored_as_df()

            if val:
                metrics_df_val = self.val_metrics.get_stored_as_df()
                for col in metrics_df_val.columns:
                    metrics_df["{}_val".format(col)] = metrics_df_val[col]
            return metrics_df

    def _evaluate(self, loader):
        """Evaluates model performance.

        Args:
            loader (torch DataLoader):  instantiated Validation Dataloader (with TimeDataset)
        Returns:
            df with evaluation metrics
        """
        test_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])
        if self.highlight_forecast_step_n is not None:
            test_metrics.add_specific_target(target_pos=self.highlight_forecast_step_n - 1)
        ## Run

        self.test_metrics = test_metrics
        self.trainer.test(self.model, test_dataloaders=loader, ckpt_path=None, verbose=False)

        test_metrics_dict = self.test_metrics.compute(save=True)

        log.info("Validation metrics: {}".format(utils.print_epoch_metrics(test_metrics_dict)))
        val_metrics_df = self.test_metrics.get_stored_as_df()
        return val_metrics_df

    def split_df(self, df, freq, valid_p=0.2):
        """Splits timeseries df into train and validation sets.

        Prevents overbleed of targets. Overbleed of inputs can be configured.
        Also performs basic data checks and fills in missing data.

        Args:
            df (pd.DataFrame): data
            freq (str):Data step sizes. Frequency of data recording,
                Any valid frequency for pd.date_range, such as '5min', 'D' or 'MS'
            valid_p (float): fraction of data to use for holdout validation set
                Targets will still never be shared.

        Returns:
            df_train (pd.DataFrame):  training data
            df_val (pd.DataFrame): validation data
        """
        df = df.copy(deep=True)
        df = df_utils.check_dataframe(df, check_y=False)
        df = self._handle_missing_data(df, freq=freq, predicting=False)
        df_train, df_val = df_utils.split_df(
            df,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            valid_p=valid_p,
            inputs_overbleed=True,
        )
        return df_train, df_val

    def crossvalidation_split_df(self, df, freq, k=5, fold_pct=0.1, fold_overlap_pct=0.5):
        """Splits timeseries data in k folds for crossvalidation.

        Args:
            df (pd.DataFrame): data
            freq (str):Data step sizes. Frequency of data recording,
                Any valid frequency for pd.date_range, such as '5min', 'D' or 'MS'
            k: number of CV folds
            fold_pct: percentage of overall samples to be in each fold
            fold_overlap_pct: percentage of overlap between the validation folds.

        Returns:
            list of k tuples [(df_train, df_val), ...] where:
                df_train (pd.DataFrame):  training data
                df_val (pd.DataFrame): validation data
        """
        df = df.copy(deep=True)
        df = df_utils.check_dataframe(df, check_y=False)
        df = self._handle_missing_data(df, freq=freq, predicting=False)
        folds = df_utils.crossvalidation_split_df(
            df,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            k=k,
            fold_pct=fold_pct,
            fold_overlap_pct=fold_overlap_pct,
        )
        return folds

    def fit(
        self, df, freq, epochs=None, validate_each_epoch=False, valid_p=0.2, progress_bar=True, plot_live_loss=False
    ):
        """Train, and potentially evaluate model.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with all data
            freq (str):Data step sizes. Frequency of data recording,
                Any valid frequency for pd.date_range, such as '5min', 'D' or 'MS'
            epochs (int): number of epochs to train.
                default: if not specified, uses self.epochs
            validate_each_epoch (bool): whether to evaluate performance after each training epoch
            valid_p (float): fraction of data to hold out from training for model evaluation
            progress_bar (bool): display updating progress bar (tqdm)
            plot_live_loss (bool): plot live training loss,
                requires [live] install or livelossplot package installed.
        Returns:
            metrics with training and potentially evaluation metrics
        """
        self.data_freq = freq
        if epochs is not None:
            default_epochs = self.config_train.epochs
            self.config_train.epochs = epochs
        if self.fitted is True:
            log.warning("Model has already been fitted. Re-fitting will produce different results.")
        df = df_utils.check_dataframe(df, check_y=True)
        df = self._handle_missing_data(df, freq=self.data_freq)
        if validate_each_epoch:
            df_train, df_val = df_utils.split_df(df, n_lags=self.n_lags, n_forecasts=self.n_forecasts, valid_p=valid_p)
            metrics_df = self._train(df_train, df_val, progress_bar=progress_bar, plot_live_loss=plot_live_loss)
        else:
            metrics_df = self._train(df, progress_bar=progress_bar, plot_live_loss=plot_live_loss)
        if epochs is not None:
            self.config_train.epochs = default_epochs
        self.fitted = True

        return metrics_df

    def _hyperparameter_optimization(self, df, freq, epochs=None, validate_each_epoch=True, valid_p=0.2):

        self.data_freq = freq
        if epochs is not None:
            default_epochs = self.config_train.epochs
            self.config_train.epochs = epochs
        if self.fitted is True:
            log.warning("Model has already been fitted. Re-fitting will produce different results.")
        df = df_utils.check_dataframe(df, check_y=True)
        df = self._handle_missing_data(df, freq=self.data_freq)
        if validate_each_epoch:
            df_train, df_val = df_utils.split_df(df, n_lags=self.n_lags, n_forecasts=self.n_forecasts, valid_p=valid_p)
            tr_loader, val_loader, model = self._train(df_train, df_val, hyperparameter_optim=True)

        return tr_loader, val_loader, model

    def test(self, df):
        """Evaluate model on holdout data.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with holdout data
        Returns:
            df with evaluation metrics
        """
        if self.fitted is False:
            log.warning("Model has not been fitted. Test results will be random.")
        df = df_utils.check_dataframe(df, check_y=True)
        df = self._handle_missing_data(df, freq=self.data_freq)
        loader = self._init_val_loader(df)
        val_metrics_df = self._evaluate(loader)
        return val_metrics_df

    def make_future_dataframe(self, df, periods=None, n_historic_predictions=0):
        df = df.copy(deep=True)

        n_lags = 0 if self.n_lags is None else self.n_lags
        if periods is None:
            periods = 1 if n_lags == 0 else self.n_forecasts
        else:
            assert periods >= 0

        if isinstance(n_historic_predictions, bool):
            if n_historic_predictions:
                n_historic_predictions = len(df) - n_lags
            else:
                n_historic_predictions = 0
        elif not isinstance(n_historic_predictions, int):
            log.error("non-integer value for n_historic_predictions set to zero.")
            n_historic_predictions = 0

        if periods == 0 and n_historic_predictions == 0:
            raise ValueError("Set either history or future to contain more than zero values.")

        last_date = pd.to_datetime(df["ds"].copy(deep=True)).sort_values().max()

        if len(df) < n_lags:
            raise ValueError("Insufficient data for a prediction")
        elif len(df) < n_lags + n_historic_predictions:
            log.warning(
                "Insufficient data for {} historic forecasts, reduced to {}.".format(
                    n_historic_predictions, len(df) - n_lags
                )
            )
            n_historic_predictions = len(df) - n_lags
        if (n_historic_predictions + n_lags) == 0:
            df = pd.DataFrame(columns=df.columns)
        else:
            df = df[-(n_lags + n_historic_predictions) :]

        if len(df) > 0:
            if len(df.columns) == 1 and "ds" in df:
                assert n_lags == 0
                df = df_utils.check_dataframe(df, check_y=False)
            else:
                df = df_utils.check_dataframe(df, check_y=n_lags > 0)
                df = self._handle_missing_data(df, freq=self.data_freq, predicting=True)
            df = df_utils.normalize(df, self.data_params)

        # future data
        # check for external events known in future

        if n_lags > 0:
            if periods > 0 and periods != self.n_forecasts:
                periods = self.n_forecasts
                log.warning(
                    "Number of forecast steps is defined by n_forecasts. " "Adjusted to {}.".format(self.n_forecasts)
                )

        if periods > 0:
            future_df = df_utils.make_future_df(
                df_columns=df.columns,
                last_date=last_date,
                periods=periods,
                freq=self.data_freq,
            )
            future_df = df_utils.normalize(future_df, self.data_params)
            if len(df) > 0:
                df = df.append(future_df)
            else:
                df = future_df
        df.reset_index(drop=True, inplace=True)
        return df

    def predict(self, df):
        """Runs the model to make predictions.

        and compute stats (MSE, MAE)
        Args:
            df (pandas DataFrame): Dataframe with columns 'ds' datestamps, 'y' time series values and
                other external variables

        Returns:
            df_forecast (pandas DataFrame): columns 'ds', 'y', 'trend' and ['yhat<i>']
        """
        # TODO: Implement data sanity checks?
        if self.fitted is False:
            log.warning("Model has not been fitted. Predictions will be random.")
        dataset = self._create_dataset(df, predict_mode=True)
        loader = DataLoader(dataset, batch_size=min(1024, len(df)), shuffle=False, drop_last=False)

        predicted_vectors = list()
        component_vectors = None
        with torch.no_grad():
            self.model.eval()
            for inputs, _ in loader:
                predicted = self.model.forward(inputs)
                predicted_vectors.append(predicted.detach().numpy())

        predicted = np.concatenate(predicted_vectors)

        scale_y, shift_y = self.data_params["y"].scale, self.data_params["y"].shift
        predicted = predicted * scale_y + shift_y

        cols = ["ds", "y"]  # cols to keep from df
        df_forecast = pd.concat((df[cols],), axis=1)

        # create a line for each forecast_lag
        # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
        for i in range(self.n_forecasts):
            forecast_lag = i + 1
            forecast = predicted[:, forecast_lag - 1]
            pad_before = self.n_lags + forecast_lag - 1
            pad_after = self.n_forecasts - forecast_lag
            yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
            df_forecast["yhat{}".format(i + 1)] = yhat
            df_forecast["residual{}".format(i + 1)] = yhat - df_forecast["y"]

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
        if self.n_lags > 0:
            num_forecasts = sum(fcst["yhat1"].notna())
            if num_forecasts < self.n_forecasts:
                log.warning(
                    "Too few forecasts to plot a line per forecast step." "Plotting a line per forecast origin instead."
                )
                return self.plot_last_forecast(
                    fcst,
                    ax=ax,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    figsize=figsize,
                    include_previous_forecasts=num_forecasts - 1,
                    plot_history_data=True,
                )
        return plot(
            fcst=fcst,
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            highlight_forecast=self.highlight_forecast_step_n,
        )





class NBeatsNP:
    def __init__(
            self,
            max_encoder_length=150,
            batch_size=None,
            epochs=100,
            num_gpus=0,
            patience_early_stopping=10,
            early_stop=True,
            weight_decay=1e-2,
            learning_rate=3e-2,
            auto_lr_find=True,
            num_workers=3,
            loss_func='Huber'
    ):

        self.batch_size = batch_size

        self.max_encoder_length = max_encoder_length

        self.epochs = epochs
        self.num_gpus = num_gpus
        self.patience_early_stopping = patience_early_stopping
        self.early_stop = early_stop
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.auto_lr_find = auto_lr_find
        self.num_workers = num_workers

        self.loss_func = loss_func
        self.fitted = False

        if type(self.loss_func) == str:
            if self.loss_func.lower() in ["huber", "smoothl1", "smoothl1loss"]:
                self.loss_func = torch.nn.SmoothL1Loss()
            elif self.loss_func.lower() in ["mae", "l1", "l1loss"]:
                self.loss_func = torch.nn.L1Loss()
            elif self.loss_func.lower() in ["mse", "mseloss", "l2", "l2loss"]:
                self.loss_func = torch.nn.MSELoss()
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
                metrics.LossMetric(self.loss_func),
                metrics.MAE(),
                metrics.MSE(),
            ],
            value_metrics=[
                # metrics.ValueMetric("Loss"),
            ],
        )

        self.val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])

    def _init_model(self, training, train_dataloader):
        model = NBeats.from_dataset(training,
                                    learning_rate=self.learning_rate,
                                    log_gradient_flow=False,
                                    weight_decay=self.weight_decay,
                                    )
        if self.auto_lr_find:
            res = self.trainer.tuner.lr_find(model,
                                             train_dataloader=train_dataloader,
                                             min_lr=1e-5,
                                             max_lr=1e2)

            model.hparams.learning_rate = res.suggestion()
            self.learning_rate = res.suggestion()

        return model

    def set_auto_batch_epoch(self,
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

    def _create_dataset(self, df, freq, valid_p):
        df = df_utils.check_dataframe(df)
        df = self._handle_missing_data(df, freq)
        df = df[['ds', 'y']]
        df['time_idx'] = range(df.shape[0])
        df['series'] = 0
        self.n_data = df.shape[0]
        self.set_auto_batch_epoch(self.n_data)

        max_prediction_length = int(valid_p * df.shape[0])
        training_cutoff = df.shape[0] - int(valid_p * df.shape[0])

        self.context_length = self.max_encoder_length
        self.prediction_length = max_prediction_length

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
            randomize_length=None,
            add_relative_time_idx=False,
            add_target_scales=False,
        )

        validation = TimeSeriesDataSet.from_dataset(training,
                                                    df,
                                                    min_prediction_idx=training_cutoff)

        train_dataloader = training.to_dataloader(train=True,
                                                  batch_size=self.batch_size,
                                                  num_workers=self.num_workers)
        val_dataloader = validation.to_dataloader(train=False,
                                                  batch_size=self.batch_size,
                                                  num_workers=self.num_workers)

        return training, train_dataloader, val_dataloader

    def _train(self, training, train_dataloader, val_dataloader, hyperparameter_optim=False):
        callbacks = []
        if self.early_stop:
            early_stop_callback = EarlyStopping(monitor="val_loss",
                                                min_delta=1e-4,
                                                patience=self.patience_early_stopping,
                                                verbose=False,
                                                mode="min")
            callbacks = [early_stop_callback]

        self.trainer = pl.Trainer(
            max_epochs=self.epochs,
            gpus=self.num_gpus,
            weights_summary="top",
            gradient_clip_val=0.1,
            callbacks=callbacks,
            checkpoint_callback=False,
            logger=False,
        )

        self.model = self._init_model(training, train_dataloader)
        self.model.set_forecaster(self)

        self.metrics.reset()
        self.val_metrics.reset()

        if hyperparameter_optim:
            return self.model
        else:

            self.trainer.fit(self.model,
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
        metrics_df = self._train(training, train_dataloader, val_dataloader)
        return metrics_df

    def _hyperparameter_optimization(self, df, freq, valid_p=0.2):
        training, train_dataloader, val_dataloader = self._create_dataset(df, freq, valid_p)
        model = self._train(training, train_dataloader, val_dataloader, hyperparameter_optim=True)

    def make_future_dataframe(self, df, freq, periods=0, n_historic_predictions=0):

        self.periods = periods
        self.n_historic_predictions = n_historic_predictions

        df = df.copy(deep=True)

        df = df_utils.check_dataframe(df)
        df = self._handle_missing_data(df, freq)
        df = df[['ds', 'y']]
        df['time_idx'] = range(df.shape[0])
        df['series'] = 0
        self.n_data = df.shape[0]

        encoder_data = df[lambda x: x.time_idx > x.time_idx.max() - (self.max_encoder_length + n_historic_predictions)]
        last_data = df[lambda x: x.time_idx == x.time_idx.max()]
        decoder_data = pd.concat(
            [last_data.assign(ds=lambda x: x.ds + pd.offsets.MonthBegin(i)) for i in range(1, periods + 1)],
            ignore_index=True,
        )
        decoder_data['time_idx'] = range(decoder_data['time_idx'].iloc[0] + 1,
                                         decoder_data['time_idx'].iloc[0] + periods + 1)
        decoder_data['ds'] = pd.date_range(start=encoder_data['ds'].iloc[-1],
                                           periods=periods + 1, freq=freq)[1:]
        future_dataframe = pd.concat([encoder_data, decoder_data], ignore_index=True)

        return future_dataframe

    def predict(self, future_dataframe):

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
            time_varying_unknown_reals=["y"],
            randomize_length=None,
            add_relative_time_idx=False,
            add_target_scales=False,
        )

        new_raw_predictions, new_x = self.model.predict(testing, mode="raw", return_x=True)
        y_predicted = self.model.to_prediction(new_raw_predictions).detach().cpu()[0, : new_x["decoder_lengths"][0]]
        y_predicted = y_predicted.detach().numpy()

        future_dataframe.loc[len(future_dataframe) - self.periods:, 'y'] = None
        future_dataframe['yhat1'] = None
        future_dataframe.loc[len(future_dataframe) - len(y_predicted):, 'yhat1'] = y_predicted
        cols = ["ds", "y", "yhat1"]  # cols to keep from df
        df_forecast = pd.concat((future_dataframe[cols],), axis=1)
        df_forecast["residual1"] = df_forecast['yhat1'] - df_forecast["y"]

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