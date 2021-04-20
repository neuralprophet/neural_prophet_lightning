
This project aims to contribute to the open-source library NeuralProphet by adding state-of-the-art models and refactoring the code to adopt the best machine learning engineering practices. 



keywords: Time series forecasting\and Neural Prophet \and Neural forecasting


% Why NeuralProphet is impotent
NeuralProphet is a new library for time series forecasting built on PyTorch.
It is inspired by widely known Facebook library for time series forecasting called Prophet (\cite{Taylor2018}) and DeepAR model (\cite{Salinas2020}).
However, while Prophet is additive model focused mostly on seasonal components and holiday effects, NeuralProphet includes AutoRegression components. 
Moreover, NeuralProphet is built on PyTorch, which allows to configure the model more precisely.
% Why should we change it
In this project we aim to improve existing NeuralProphet library to allow even more possibilities for its users.
Currently, NeuralProphet is written on pure PyTorch.
However, there is a framework called PyTorch Lighting which allows more possibilities for research purposes.
Moreover, NeuralProphet API is rather specific in terms of initial data format and outputs.
This means that for comparison with other models, users are required to write a lot of additional code. 
Therefore it will be exceptionally useful to add some of state-of-the-art models into NeuralProphet library, which will allow users to use them with the same API.
