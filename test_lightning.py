from neuralprophet import NeuralProphet
import pandas as pd
import numpy as np

m = NeuralProphet(n_lags = 10, epochs = 15)
df = pd.DataFrame()
df['ds'] = pd.date_range(start = '2020-01-01', periods = 100)
df['y'] = np.random.randint(0, 100, size = len(df))
a = m.fit(df, freq = 'D')

print(a)