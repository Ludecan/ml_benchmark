# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from autogluon.tabular import TabularPredictor
from catboost import CatBoost
import time

from itertools import product

from results_table import ResultsTable


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
    for nrows, ncols in product((1000,), (10, 20, 50, 100)):
        # * 1.25 to account for the train/test split
        yield (f"{nrows} x {ncols}", create_random_dataset(int(nrows * 1.25), ncols))


def load_datasets():
    datasets = {
        # "Boston Housing": load_boston(return_X_y=True),
        "Diabetes": load_diabetes(return_X_y=True),
        "California Housing": fetch_california_housing(return_X_y=True),
        # "Ames Housing": fetch_openml(
        #    name="house_prices", as_frame=True, return_X_y=True
        # ),
    }
    return datasets


def get_metrics(y_pred, y_test) -> tuple[float, float, float, float]:
    me = (y_pred - y_test).mean()
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return me, rmse, mae, r2


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    if model_name == "TabNet (CPU)":
        start_time = time.time()
        model.fit(X_train, y_train.reshape(-1, 1))
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)
    elif model_name == "AutoGluon":
        train_data = pd.DataFrame(
            np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
        )
        start_time = time.time()
        model = TabularPredictor(
            label=train_data.columns[-1], problem_type="regression"
        ).fit(train_data, verbosity=0)
        train_time = time.time() - start_time
        test_data = pd.DataFrame(X_test)
        y_pred = model.predict(test_data).values
    else:
        start_time = time.time()
        model.fit(X_train, y_train.ravel())
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)

    me, rmse, mae, r2 = get_metrics(y_pred, y_test)

    return me, rmse, mae, r2, train_time


# %%
def main():
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(),
        "XGBoost (hist)": XGBRegressor(tree_method="hist"),
        "LightGBM": LGBMRegressor(),
        "CatBoost": CatBoost(params={"logging_level": "Silent"}),
        "TabNet (CPU)": TabNetRegressor(verbose=0, device_name="cpu"),
        "AutoGluon": None,
    }

    full_results = ResultsTable()
    # dataset_name, (X, y) = next(load_random_datasets())
    for dataset_name, (X, y) in load_random_datasets():
        results = ResultsTable()

        print(f"Dataset: {dataset_name}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_name, model = list(models.items())[0]
        for model_name, model in models.items():
            print(f"  Evaluating {model_name}...")
            me, rmse, mae, r2, train_time = evaluate_model(
                model_name, model, X_train, X_test, y_train, y_test
            )

            results.add_row(
                model_name,
                dataset_name,
                X.shape[0],
                X.shape[1],
                me,
                rmse,
                mae,
                r2,
                train_time,
            )
        full_results.add_table(results)

        results.print_table()


if __name__ == "__main__":
    main()

# %%
