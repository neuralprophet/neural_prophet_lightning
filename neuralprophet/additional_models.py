import torch.nn as nn
import pytorch_lightning as pl
import logging


log = logging.getLogger("NP.additional_models")


class LightLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, bias, bidirectional, n_forecasts):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            bidirectional=bidirectional,
            batch_first=False,
        )
        if bidirectional:
            self.linear = nn.Linear(hidden_size*num_layers, n_forecasts)
        else:
            self.linear = nn.Linear(hidden_size, n_forecasts)
        # Metrics live
        self.metrics_live = {}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer  ##### todo add this to init

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler  ##### todo add this to init

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func  ##### todo add this to init

    def set_forecaster(self, self_forecaster):
        self.forecaster = self_forecaster

    def forward(self, x):
        x = x["lags"]
        x.resize_((x.size()[0], 1, x.size()[1]))

        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)
        self.forecaster.metrics.update(predicted=y_hat.detach(), target=y.detach(), values={"Loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss)
        self.forecaster.val_metrics.update(predicted=y_hat.detach(), target=y.detach())

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('test_loss', loss)
        self.forecaster.test_metrics.update(predicted=y_hat.detach(), target=y.detach())

        return loss

    def training_epoch_end(self, outputs):
        epoch_metrics = self.forecaster.metrics.compute(save=True)
        self.metrics_live["{}".format(list(epoch_metrics)[0])] = epoch_metrics[list(epoch_metrics)[0]]

        self.forecaster.metrics.reset()

    def validation_epoch_end(self, validation_step_outputs):
        val_epoch_metrics = self.forecaster.val_metrics.compute(save=True)
        self.metrics_live["val_{}".format(list(val_epoch_metrics)[0])] = val_epoch_metrics[list(val_epoch_metrics)[0]]
        self.forecaster.val_metrics.reset()

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
