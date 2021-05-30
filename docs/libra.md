# LIBRA

LIBRA is a benchmarking framework, that we use to benchmark Neural Prophet and all additional models.

You can ran LIBRA on four datasets containing 100 time series each from four different usecases:
* `economics` — (gas, sales, unemployment, etc.)
* `finance` — (stocks, sales prices, exchange rate, etc.)
* `human` — (calls, SMS, Internet, etc.)
* `nature` — (rain, birth, death, etc.)

Datasets and their frequencies can be loaded with the function `get_datasets`:

```python
from neuralprophet.libra import get_datasets, libra

datasets, frequencies = get_datasets(usecase='economics', data_loc='../example_data/LIBRA/')
``` 

Benchmarking can be performed with two methods:
* `onestep` - one-step-ahead forecasting constitutes to forecasting one period ahead from the set date.
* `multistep` - multi-step-ahead forecasting constitutes to forecasting several steps ahead in the future.

You can run benchmarking by using the function `libra`.
By changing parameter `n_datasets`, you can vary number of time series from 1 to 100.

```python
metrics = libra(n_datasets=2, 
                datasets=datasets, 
                frequencies=frequencies, 
                method='onestep', 
                n_epochs=5, 
                usecase='economics',
                save_res=True)
``` 

For each time series for each model 6 metrics are calculated:
* `smape` — symmetrical mean absolute percentage error
* `mase` — mean absolute scaled error
* `mues` — mean under-estimation shares
* `moes` — mean over-estimation shares
* `muas` — mean under-accuracy shares
* `moas` — mean over-accuracy shares

Results can be saved in the `.csv` file. Here is a small example on two time series:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>economics_33.csv</th>
      <th>economics_23.csv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>smape_LSTM_onestep</td>
      <td>7.855039</td>
      <td>113.709703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mase_LSTM_onestep</td>
      <td>3.005221</td>
      <td>11.786924</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mues_LSTM_onestep</td>
      <td>0.916667</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>moes_LSTM_onestep</td>
      <td>0.083333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>muas_LSTM_onestep</td>
      <td>0.072298</td>
      <td>0.723410</td>
    </tr>
    <tr>
      <th>5</th>
      <td>moas_LSTM_onestep</td>
      <td>0.001859</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>smape_NP_onestep</td>
      <td>50.435280</td>
      <td>1419.567313</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mase_NP_onestep</td>
      <td>28.257673</td>
      <td>29.190349</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mues_NP_onestep</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>moes_NP_onestep</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>muas_NP_onestep</td>
      <td>0.000000</td>
      <td>1.772128</td>
    </tr>
    <tr>
      <th>11</th>
      <td>moas_NP_onestep</td>
      <td>0.694518</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>smape_DeepAR_onestep</td>
      <td>5.725982</td>
      <td>13.837990</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mase_DeepAR_onestep</td>
      <td>2.019481</td>
      <td>2.227683</td>
    </tr>
    <tr>
      <th>14</th>
      <td>mues_DeepAR_onestep</td>
      <td>0.541667</td>
      <td>0.576923</td>
    </tr>
    <tr>
      <th>15</th>
      <td>moes_DeepAR_onestep</td>
      <td>0.458333</td>
      <td>0.423077</td>
    </tr>
    <tr>
      <th>16</th>
      <td>muas_DeepAR_onestep</td>
      <td>0.037479</td>
      <td>0.086398</td>
    </tr>
    <tr>
      <th>17</th>
      <td>moas_DeepAR_onestep</td>
      <td>0.018422</td>
      <td>0.044979</td>
    </tr>
    <tr>
      <th>18</th>
      <td>smape_NBeats_onestep</td>
      <td>5.539724</td>
      <td>13.668021</td>
    </tr>
    <tr>
      <th>19</th>
      <td>mase_NBeats_onestep</td>
      <td>2.167093</td>
      <td>2.205612</td>
    </tr>
    <tr>
      <th>20</th>
      <td>mues_NBeats_onestep</td>
      <td>0.500000</td>
      <td>0.461538</td>
    </tr>
    <tr>
      <th>21</th>
      <td>moes_NBeats_onestep</td>
      <td>0.500000</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>22</th>
      <td>muas_NBeats_onestep</td>
      <td>0.034671</td>
      <td>0.073146</td>
    </tr>
    <tr>
      <th>23</th>
      <td>moas_NBeats_onestep</td>
      <td>0.019655</td>
      <td>0.060271</td>
    </tr>
    <tr>
      <th>24</th>
      <td>smape_TFT_onestep</td>
      <td>81.920169</td>
      <td>14.710118</td>
    </tr>
    <tr>
      <th>25</th>
      <td>mase_TFT_onestep</td>
      <td>23.308894</td>
      <td>2.358982</td>
    </tr>
    <tr>
      <th>26</th>
      <td>mues_TFT_onestep</td>
      <td>0.333333</td>
      <td>0.730769</td>
    </tr>
    <tr>
      <th>27</th>
      <td>moes_TFT_onestep</td>
      <td>0.666667</td>
      <td>0.269231</td>
    </tr>
    <tr>
      <th>28</th>
      <td>muas_TFT_onestep</td>
      <td>0.333333</td>
      <td>0.115717</td>
    </tr>
    <tr>
      <th>29</th>
      <td>moas_TFT_onestep</td>
      <td>0.173972</td>
      <td>0.016877</td>
    </tr>
  </tbody>
</table>
</div>


```python

```
