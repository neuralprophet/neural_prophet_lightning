from pytorch_forecasting import NBeats
import logging

log = logging.getLogger("NP.NBeats")

class LightNBeats(NBeats):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forecaster = None

    def set_forecaster(self, self_forecaster):
        self.forecaster = self_forecaster

    def training_step(self, batch, batch_idx):
        """
        Train on batch.
        """
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        y_hat = self.to_prediction(out)
        y = y[0]

        # log loss
        # assert len(log["loss"].size()) >= 1

        self.log("train_loss", log["loss"], on_step=True, on_epoch=True, prog_bar=True)
        if type(self.forecaster) != type(None):
            self.forecaster.metrics.update(predicted=y_hat.detach(), target=y.detach(), values={"Loss": log["loss"]})
        return log

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs)
        if type(self.forecaster) != type(None):
            epoch_metrics = self.forecaster.metrics.compute(save=True)
            self.forecaster.metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log, out = self.step(x, y, batch_idx)  # log loss
        y_hat = self.to_prediction(out)
        y = y[0]

        self.log("val_loss", log["loss"], on_step=False, on_epoch=True, prog_bar=True)
        if type(self.forecaster) != type(None):
            self.forecaster.val_metrics.update(predicted=y_hat.detach(), target=y.detach())
        return log

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs)
        if type(self.forecaster) != type(None):
            val_epoch_metrics = self.forecaster.val_metrics.compute(save=True)
            self.forecaster.val_metrics.reset()
