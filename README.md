# NeuralProphet project for Theoretical Foundations of DataScience course

---
This is the repository for NeuralProphet project for Theoretical Foundations of DataScience course. Contributors are Alexey Voskoboinikov and Polina Pilyugina.The main aim of this project is to improve NeuralProphet Library.

### Project outline
- Refactor the code with PyTorch Lightning in accordance with existing API
- Adapt and include existing implementations of SOTA models for time series forecasting under the NeuralProphet API
- Add hyperparameter tuning with Ray Tune as additional module
- Recreate LIBRA framework for benchmarking in Python and run it on NeuralProphet and our additionally included models
- Add neccessary tests and documentation for introduced functional

### Baseline solutions
The main source of the code of this work is original [NeuralProphet library](https://github.com/ourownstory/neural_prophet).
Firstly, we will refactor the code to support [PyTorch Lightning](https://www.pytorchlightning.ai).
This includes refactoring the model itself and all additional parts.
The main goal of refactoring is to structure the code in a reusable way and separate research modules from engineering parts. 
Additionally, we seek to introduce functional for distributed training from PyTorch Lightning.

For model implementations we will use modules from [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/latest/index.html).
They are already built in PyTorch Lightning.
We will add data preprocessing steps for these models.
Additionally, we will add wrappers so that these models rely on the same API and produce results in the same format as NeuralProphet

For hyperparameter tuning we will add a module using Ray Tune functional.

## Distribution of roles and roadmap

![PT to PL](roadmap.jpg)
