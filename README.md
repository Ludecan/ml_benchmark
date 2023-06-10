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


TODO: Dockerize installation to ensure common base libs
