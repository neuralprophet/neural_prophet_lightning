import logging

log = logging.getLogger("NP")
log.setLevel("INFO")
# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("logs.log", "w+")
# c_handler.setLevel("WARNING")
# f_handler.setLevel("INFO")
# Create formatters and add it to handlers
c_format = logging.Formatter("%(levelname)s - (%(name)s.%(funcName)s) - %(message)s")
f_format = logging.Formatter("%(asctime)s; %(levelname)s; %(name)s; %(funcName)s; %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
# Add handlers to the logger
log.addHandler(c_handler)
log.addHandler(f_handler)

logging.captureWarnings(True)
warnings_log = logging.getLogger("py.warnings")
warnings_log.addHandler(c_handler)
warnings_log.addHandler(f_handler)

from .forecasters.forecaster import NeuralProphet
from .forecasters.forecaster_LSTM import LSTM
from .forecasters.forecaster_NBeats import NBeats
from .forecasters.forecaster_DeepAR import DeepAR
from .forecasters.forecaster_TFT import TFT

from .utils.utils import set_random_seed, set_log_level
from .utils.df_utils import split_df
#from .utils import utils

from neuralprophet.utils.df_utils import split_df
#from .df_utils import split_df