import pandas as pd
from IPython.display import display


class ResultsTable:
    results: pd.DataFrame

    def __init__(self) -> None:
        self.results = pd.DataFrame(
            data={
                "Model": [],
                "Dataset": [],
                "Rows": [],
                "Columns": [],
                "ME": [],
                "RMSE": [],
                "MAE": [],
                "R2": [],
                "Training Time (s)": [],
            }
        )

    """
    def _min_max_column(self, col_idx: int) -> tuple[float, float]:
        values = [float(cell) for cell in self.table.columns[col_idx].cells]
        return min(values), max(values)

    def _get_gradient_style(value: float, min_value: float, max_value: float) -> Style:
        start_color = Color.from_rgb(255, 0, 0)  # Red
        end_color = Color.from_rgb(0, 255, 0)  # Green

        ratio = (value - min_value) / (max_value - min_value)
        blended_color = start_color.blend(end_color, ratio)

        return Style(color=blended_color)
    """

    def add_row(
        self,
        model_name: str,
        dataset_name: str,
        nrows: int,
        ncolumns: int,
        me: float,
        rmse: float,
        mae: float,
        r2: float,
        train_time: float,
    ):
        self.results.loc[self.results.shape[0]] = [
            model_name,
            dataset_name,
            nrows,
            ncolumns,
            me,
            rmse,
            mae,
            r2,
            train_time,
        ]

    def add_table(self, results_table: "ResultsTable"):
        self.results = pd.concat([self.results, results_table.results])

    def print_table(self, ndigits: int = 2, styled: bool = False):
        if styled:
            # Custom format to remove trailing zeros
            custom_format = "{:." + str(ndigits) + "f}"
            formatter = {
                column: custom_format
                for column in ["ME", "RMSE", "MAE", "R2", "Training Time (s)"]
            }

            styled_df = (
                self.results.set_index(["Model", "Dataset"])
                .round(ndigits)
                .style.background_gradient(
                    cmap="RdYlGn",
                    subset=["R2"],
                    axis=0,
                    # high=1,
                    # low=0,
                )
                .background_gradient(
                    cmap="RdYlGn_r",
                    subset=["ME", "RMSE", "MAE", "Training Time (s)"],
                    axis=0,
                    # high=1,
                    # low=0,
                )
                .format(formatter)
            )

            display(styled_df)
        else:
            display(self.results.set_index(["Model", "Dataset"]).round(ndigits))
