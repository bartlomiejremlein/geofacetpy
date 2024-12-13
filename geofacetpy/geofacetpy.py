import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, Optional, Dict


class GridLayoutValidator:
    """
    Validator for grid layout configurations.

    Provides methods to validate and adjust grid layout DataFrames.
    """

    @staticmethod
    def validate_columns(grid_layout: pd.DataFrame) -> None:
        """
        Validate that the grid layout contains the required columns.

        Args:
            grid_layout (pd.DataFrame): Grid layout DataFrame to validate.

        Raises:
            ValueError: If required columns are missing.
        """
        required_columns = {"name", "row", "col"}
        missing = required_columns - set(grid_layout.columns)
        if missing:
            raise ValueError(f"Grid layout is missing required columns: {missing}")

    @staticmethod
    def adjust_indexing(grid_layout: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically adjust the grid layout to 0-based indexing if needed.

        Args:
            grid_layout (pd.DataFrame): Input grid layout.

        Returns:
            pd.DataFrame: Grid layout with 0-based indexing.
        """
        grid_layout = grid_layout.copy()

        if (grid_layout["row"] > 0).all() and (grid_layout["col"] > 0).all():
            grid_layout["row"] -= 1
            grid_layout["col"] -= 1

        return grid_layout


class DataValidator:
    """
    Validator for input data in geofaceted plotting.
    """

    @staticmethod
    def validate_data(
        data: pd.DataFrame, group_column: str, optional_columns: Optional[list] = None
    ) -> None:
        """
        Ensure the provided data contains the required group column.

        Args:
            data (pd.DataFrame): Input data to validate.
            group_column (str): Column name to check in the DataFrame.
            optional_columns (list, optional): Additional columns to validate.

        Raises:
            ValueError: If the group column is not found or optional columns are missing.
        """
        if group_column not in data.columns:
            raise ValueError(
                f"Column '{group_column}' not found in the data. "
                f"Available columns: {list(data.columns)}"
            )

        if optional_columns:
            missing_cols = [col for col in optional_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing optional columns: {missing_cols}")


class GeoFacetPlotter:
    """
    A comprehensive geofaceted plotting class that handles grid layout, data validation,
    and plot generation.
    """

    def __init__(
        self,
        grid_layout: pd.DataFrame,
        data: pd.DataFrame,
        group_column: str,
        plotting_function: Callable,
        figsize: tuple = (12, 8),
        grid_spacing: tuple = (0.5, 0.5),
        tick_placement: Optional[Dict[str, str]] = None,
        sharex: bool = False,
        sharey: bool = False,
        **plot_kwargs,
    ):
        """
        Initialize the GeoFacetPlotter.

        Args:
            grid_layout (pd.DataFrame): Grid layout with 'name', 'row', 'col' columns.
            data (pd.DataFrame): Data to plot.
            group_column (str): Column matching grid layout names.
            plotting_function (Callable): Function to plot individual grid cells.
            figsize (tuple, optional): Overall figure size. Defaults to (12, 8).
            grid_spacing (tuple, optional): Spacing between grid rows/columns. Defaults to (0.5, 0.5).
            tick_placement (dict, optional): Controls tick placement.
            sharex (bool, optional): Share x-axis across subplots. Defaults to False.
            sharey (bool, optional): Share y-axis across subplots. Defaults to False.
            **plot_kwargs: Additional arguments for plotting function.
        """
        GridLayoutValidator.validate_columns(grid_layout)
        grid_layout = GridLayoutValidator.adjust_indexing(grid_layout)
        DataValidator.validate_data(data, group_column)

        self.grid_layout = grid_layout
        self.data = data
        self.group_column = group_column
        self.plotting_function = plotting_function
        self.figsize = figsize
        self.grid_spacing = grid_spacing
        self.tick_placement = tick_placement or {"x": "bottom", "y": "first"}
        self.sharex = sharex
        self.sharey = sharey
        self.plot_kwargs = plot_kwargs

        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare data by converting columns to consistent string type.
        """
        self.data[self.group_column] = self.data[self.group_column].astype(str)
        self.grid_layout["name"] = self.grid_layout["name"].astype(str)

    def plot(self) -> plt.Figure:
        """
        Generate the geofaceted plot.

        Returns:
            plt.Figure: Matplotlib figure with geofaceted plots.
        """
        max_row = self.grid_layout["row"].max() + 1
        max_col = self.grid_layout["col"].max() + 1

        fig, axes = plt.subplots(
            max_row,
            max_col,
            figsize=self.figsize,
            squeeze=False,
            sharex=self.sharex,
            sharey=self.sharey,
        )
        fig.subplots_adjust(hspace=self.grid_spacing[0], wspace=self.grid_spacing[1])

        for _, entry in self.grid_layout.iterrows():
            row, col = entry["row"], entry["col"]
            ax = axes[row, col]
            subset = self.data[self.data[self.group_column] == entry["name"]]
            self.plotting_function(
                ax=ax, data=subset, group_name=entry["name"], **self.plot_kwargs
            )
            ax.set_title(entry["name"])

        self._customize_ticks(axes)
        self._hide_empty_subplots(axes)

        return fig

    def _customize_ticks(self, axes):
        """
        Adjust tick placement based on configuration.

        Args:
            axes (ndarray): Array of matplotlib Axes objects.
        """
        first_in_row = self.grid_layout.groupby("row")["col"].idxmin()
        first_indices = [
            (self.grid_layout.loc[idx, "row"], self.grid_layout.loc[idx, "col"])
            for idx in first_in_row
        ]

        last_in_col = self.grid_layout.groupby("col")["row"].idxmax()
        last_indices = [
            (self.grid_layout.loc[idx, "row"], self.grid_layout.loc[idx, "col"])
            for idx in last_in_col
        ]

        for i, row_axes in enumerate(axes):
            for j, ax in enumerate(row_axes):
                if not ax.get_visible():
                    continue
                ax.tick_params(axis="y", labelleft=(i, j) in first_indices)
                ax.tick_params(axis="x", labelbottom=(i, j) in last_indices)

    def _hide_empty_subplots(self, axes):
        """
        Hide subplots that do not contain any data.

        Args:
            axes (ndarray): Array of matplotlib Axes objects.
        """
        for row_axes in axes:
            for ax in row_axes:
                if not ax.lines and not ax.has_data():
                    ax.set_visible(False)


class GridLayoutPreviewer:
    """
    Utility class to preview grid layout for visualization and debugging.
    """

    @staticmethod
    def preview(grid_layout: pd.DataFrame, show: bool = True):
        """
        Visualize the grid layout.

        Args:
            grid_layout (pd.DataFrame): Grid layout with 'name', 'row', and 'col' columns.

        Raises:
            ValueError: If required columns are missing.
        """
        GridLayoutValidator.validate_columns(grid_layout)
        grid_layout = GridLayoutValidator.adjust_indexing(grid_layout)

        max_row = grid_layout["row"].max() + 1
        max_col = grid_layout["col"].max() + 1
        fig_width, fig_height = max_col * 1.5, max_row * 1.2

        _, ax = plt.subplots(figsize=(fig_width, fig_height))
        for _, row in grid_layout.iterrows():
            ax.text(
                row["col"],
                row["row"],
                row["name"],
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"
                ),
            )
        ax.set_xlim(-0.5, max_col - 0.5)
        ax.set_ylim(-0.5, max_row - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(max_col))
        ax.set_yticks(range(max_row))
        plt.grid(visible=True, linestyle="--", alpha=0.7)


def geofacet(
    grid_layout: pd.DataFrame,
    data: pd.DataFrame,
    group_column: str,
    plotting_function: Callable,
    figsize=(12, 8),
    grid_spacing=(0.5, 0.5),
    tick_placement: Optional[Dict[str, str]] = None,
    sharex: bool = False,
    sharey: bool = False,
    **plot_kwargs,
) -> plt.Figure:
    """
    Convenience function for creating a geofaceted plot.

    Args:
        grid_layout (pd.DataFrame): Grid layout with 'name', 'row', 'col' columns.
        data (pd.DataFrame): Data to plot.
        group_column (str): Column matching grid layout names.
        plotting_function (Callable): Function to plot individual grid cells.
        figsize (tuple, optional): Overall figure size. Defaults to (12, 8).
        grid_spacing (tuple, optional): Spacing between grid rows/columns. Defaults to (0.5, 0.5).
        tick_placement (dict, optional): Controls tick placement.
        sharex (bool, optional): Share x-axis across subplots. Defaults to False.
        sharey (bool, optional): Share y-axis across subplots. Defaults to False.
        **plot_kwargs: Additional arguments for plotting function.

    Returns:
        plt.Figure: Geofaceted plot figure
    """
    plotter = GeoFacetPlotter(
        grid_layout=grid_layout,
        data=data,
        group_column=group_column,
        plotting_function=plotting_function,
        figsize=figsize,
        grid_spacing=grid_spacing,
        tick_placement=tick_placement,
        sharex=sharex,
        sharey=sharey,
        **plot_kwargs,
    )
    return plotter.plot()


def preview_grid(grid_layout: pd.DataFrame, show: bool = True):
    """
    Convenience function for previewing grid layout.

    Args:
        grid_layout (pd.DataFrame): Grid layout to preview
    """
    GridLayoutPreviewer.preview(grid_layout)
    if show:
        plt.show()
