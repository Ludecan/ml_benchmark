# %%
import os
import gc
import time
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from catboost import CatBoost
from lightgbm import LGBMRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from results_table import ResultsTable
from datetime import datetime
import shutil
import re

# Disable Tensorflow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow_decision_forests as tfdf

np.random.seed(42)

# TODO: move these to a config file and don't use global variables
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest(n_jobs=-1)": RandomForestRegressor(n_jobs=-1),
    "XGBoost": XGBRegressor(),
    'XGBoost(tree_method="hist")': XGBRegressor(tree_method="hist"),
    "LightGBM": LGBMRegressor(),
    "CatBoost": CatBoost(params={"logging_level": "Silent"}),
    'TabNet(device_name="cpu")': TabNetRegressor(verbose=0, device_name="cpu"),
    "AutoGluon": None,
    "TFDecisionForest": None,
}
dataset_sizes = product(
    (
        1000,
        10000,
        100000,
        500000,
        1000000,
        2000000,
    ),
    (
        10,
        25,
        50,
        100,
    ),
)


def timed_print(msg: str):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {msg}")


# %%
def create_random_dataset(nrows: int, ncols: int) -> tuple[np.ndarray, np.ndarray]:
    # Create a numpy array of uniform(-10, 10) values of size nrows x ncols
    X = np.random.uniform(low=-10, high=10, size=nrows * ncols).reshape((nrows, ncols))

    # Define a generalized Rosenbrock-like function to generate the target array
    def target_function(x: np.ndarray, c: float = 100) -> np.ndarray:
        N = x.shape[1]
        return np.sum(
            [
                c * (x[:, i + 1] - x[:, i] ** 2) ** 2 + (1 - x[:, i]) ** 2
                for i in range(N - 1)
            ],
            axis=0,
        )

    # Apply the function to the input array to generate the target array
    y = target_function(X)

    return X, y


def load_random_datasets():
    for nrows, ncols in dataset_sizes:
        # * 1.25 to account for the train/test split
        yield (f"{nrows} x {ncols}", create_random_dataset(int(nrows * 1.25), ncols))


def load_datasets():
    datasets = {
        # "Boston Housing": load_boston(return_X_y=True),
        "Diabetes": load_diabetes(return_X_y=True),
        "California Housing": fetch_california_housing(return_X_y=True),
        "Ames Housing": fetch_openml(
            name="house_prices", as_frame=True, return_X_y=True
        ),
    }
    return datasets


def get_metrics(y_pred, y_test) -> tuple[float, float, float, float]:
    me = (y_pred - y_test).mean()
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return me, rmse, mae, r2


def evaluate_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float, float, float]:
    # These models don't finish their training properly on datasets bigger
    # than these in my 64 GB RAM computer so I skip their training to avoid
    # wasting time
    max_df_size = [
        ('TabNet(device_name="cpu")', 1000000 * 500),
        ("AutoGluon", 1000000 * 500),
        ("TabNet", 100000 * 50),
    ]

    for mname, max_size in max_df_size:
        if (
            model_name.startswith(mname)
            and X_train.shape[0] * X_train.shape[1] > max_size
        ):
            return (
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            )

    if model_name == 'TabNet(device_name="cpu")':
        start_time = time.time()
        model.fit(X_train, y_train.reshape(-1, 1))
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)
    elif model_name.startswith("AutoGluon"):
        regex_match = re.search(r'presets="([^"]*)"', model_name)
        if regex_match:
            preset = regex_match.group(1)
        else:
            # Default value of the preset parameter
            preset = "medium_quality"

        ag_path = "./AutogluonModels"
        if os.path.isdir(ag_path):
            shutil.rmtree(ag_path)
        start_time = time.time()
        train_data = pd.DataFrame(
            np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
        )
        model = TabularPredictor(
            label=train_data.columns[-1], problem_type="regression", path=ag_path
        ).fit(train_data, verbosity=0)
        train_time = time.time() - start_time
        test_data = pd.DataFrame(X_test)
        y_pred = model.predict(test_data).values
        if os.path.isdir(ag_path):
            shutil.rmtree(ag_path)
    elif model_name == "TFDecisionForest":
        start_time = time.time()
        train_data = pd.DataFrame(
            np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1),
            columns=[f"f{i}" for i in range(X_train.shape[1])] + ["y"],
        )
        # Convert the data to TensorFlow tensors
        train_data = tfdf.keras.pd_dataframe_to_tf_dataset(
            train_data,
            task=tfdf.keras.Task.REGRESSION,
            label=train_data.columns[-1],
            fix_feature_names=False,
        )
        test_data = pd.DataFrame(
            np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1),
            columns=[f"f{i}" for i in range(X_test.shape[1])] + ["y"],
        )
        # Convert the data to TensorFlow tensors
        test_data = tfdf.keras.pd_dataframe_to_tf_dataset(
            test_data,
            task=tfdf.keras.Task.REGRESSION,
            label=test_data.columns[-1],
            fix_feature_names=False,
        )

        # Train the model
        model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION, verbose=0)
        model.fit(train_data)
        train_time = time.time() - start_time

        # Predict
        y_pred = model.predict(test_data)
        y_pred = y_pred.squeeze()  # Remove the extra dimension
    else:
        start_time = time.time()
        model.fit(X_train, y_train.ravel())
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)

    me, rmse, mae, r2 = get_metrics(y_pred, y_test)

    return me, rmse, mae, r2, train_time


def time_execution(
    model_name: str,
    dataset_name: str,
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    time_threshold_seconds: float = 2.0,
    max_num_executions: int = 5,
) -> tuple[float, float, float, float, float]:
    timed_print(f"  Evaluating {model_name}...")

    me, rmse, mae, r2, train_time = (
        np.inf,
        np.inf,
        np.inf,
        np.inf,
        np.inf,
    )
    train_time = 0
    num_executions = 0
    train_times = np.full(shape=max_num_executions, fill_value=np.nan, dtype=np.float64)

    # If training takes less than time_threshold_seconds, time it
    # max_num_executions times and take the median measurement
    while (train_time < time_threshold_seconds) and (
        num_executions < max_num_executions
    ):
        # Delete and reload train/test data to ensure fair caching among models
        del X_train, X_test, y_train, y_test
        npzfile = np.load("./temp.npz")
        X_train, X_test, y_train, y_test = (
            npzfile["X_train"],
            npzfile["X_test"],
            npzfile["y_train"],
            npzfile["y_test"],
        )
        # Run garbage collector to avoid it kicking in during training
        gc.collect()

        try:
            me, rmse, mae, r2, train_time = evaluate_model(
                model_name, model, X_train, X_test, y_train, y_test
            )
            train_times[num_executions] = train_time
            train_time = np.nanmedian(train_times)
            num_executions += 1
        except Exception as e:
            timed_print(
                f"Exception training model {model_name} for dataset {dataset_name}.\n{e}"
            )
            me, rmse, mae, r2, train_time = (
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            )
            num_executions = max_num_executions

    return me, rmse, mae, r2, train_time


# %%
def main():
    start_dt = datetime.now()

    full_results = ResultsTable()
    # dataset_name, (X, y) = next(load_random_datasets())
    for dataset_name, (X, y) in load_random_datasets():
        results = ResultsTable()

        timed_print(f"Dataset: {dataset_name}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Save train/test data to file so they are loaded fresh before running each model
        data_file_path = "./temp.npz"
        np.savez(
            data_file_path,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        # model_name, model = list(models.items())[-1]
        for model_name, model in models.items():
            me, rmse, mae, r2, train_time = time_execution(
                model_name, dataset_name, model, X_train, X_test, y_train, y_test
            )

            results.add_row(
                model_name,
                dataset_name,
                X_train.shape[0],
                X_train.shape[1],
                me,
                rmse,
                mae,
                r2,
                train_time,
            )
        results.print_table()
        print("\n\n")

        if os.path.exists(data_file_path):
            os.unlink(data_file_path)

        full_results.add_table(results)

    full_results.results.to_csv(
        f"{start_dt.strftime('%Y%m%d_%H%M%S')}_benchmark_results.csv"
    )
    timed_print("Finished running benchmark.")


# %%
if __name__ == "__main__":
    main()

# %%
