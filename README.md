# ML benchmark
Training time benchmark for Machine Learning algorithms

## Installation steps

1. **Clone the repo** locally
```console
git clone git@github.com:Ludecan/ml_benchmark.git
```
2. Install pyenv following the instructions here: https://github.com/pyenv/pyenv
3. Install poetry following the instructions here: https://python-poetry.org/docs/#installation
4. Install `python 3.10.11` using `pyenv` and use it
```console
pyenv install 3.10.11
pyenv shell 3.10.11
```
5. Tell poetry to use the recently installed Python version
```console
pyenv which python | xargs poetry env use
```
6. Configure poetry to keep venvs locally (if you use VSCode, it will detect it out of the box):
```console
poetry config virtualenvs.in-project true
```
7. Install dependencies (this will also create the `venv`):
```console
poetry install
```

## Running the benchmark
Inside the poetry virtualenv run:
```console
python regression_benchmark.py
```

## Considerations about results
The main purpose of this benchmark is to compare the training time of different CPU implementations of ML Regression Algorithms for datasets of varying sizes under different hardware configurations.
It creates random, all float in the [-10, 10] range datasets with several row and column numbers and a synthetic target is created from them using the following generalization of the Rosenbrock to multiple input dimensions (thanks for it ChatGPT!)

```
f(x) = Σ[ c * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 ] for i in [0, N-2]
```

Notice the function was requested to be non-linear and have interactions between all pairs of consecutive features, allowing non linear models to show their strengths (and utterly defeating linear models).
The accuracy metrics (ME, MAE, RMSE, R^2) are provided for reference but generalization of these results to other datasets is not advised without proper testing. It is relatively simple to switch the random datasets used in this benchmark for your own dataset if you want to try these models yourself.


TODO:
- Dockerize installation to ensure common base libs
- Profile maximum memory usage during execution of each model
- Different targets/noise?
