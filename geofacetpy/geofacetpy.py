import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, Optional, Dict


def _validate_columns(grid_layout: pd.DataFrame) -> None:
    """
    Validates that the grid layout contains the required columns.

    Args:
        grid_layout (pd.DataFrame): Grid layout DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"name", "row", "col"}
    missing = required_columns - set(grid_layout.columns)
    if missing:
        raise ValueError(f"Grid layout is missing required columns: {missing}")


def _adjust_indexing(grid_layout: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically adjusts the grid layout to 0-based indexing if needed.

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


def _validate_data(data: pd.DataFrame, group_column: str = None) -> None:
    """
    Ensures the provided data contains the required group column.

    Args:
        data (pd.DataFrame): Input data to validate.
        group_column (str): Column name to check in the DataFrame.

    Raises:
        ValueError: If the group column is not found.
    """
    if group_column not in data.columns:
        raise ValueError(
            f"Column '{group_column}' not found in the data. "
            f"Available columns: {list(data.columns)}"
        )


def _customize_ticks(axes, grid_layout, tick_placement, sharex, sharey):
    """
    Adjusts tick placement based on configuration.

    Args:
        axes (ndarray): Array of matplotlib Axes objects.
        grid_layout (pd.DataFrame): Grid layout.
        tick_placement (dict): Controls tick placement.
        sharex (bool): Whether x-axes are shared across subplots.
        sharey (bool): Whether y-axes are shared across subplots.
    """
    if not tick_placement:
        tick_placement = {}

    last_visible_in_col = grid_layout.groupby("col")["row"].idxmax()
    first_visible_in_row = grid_layout.groupby("row")["col"].idxmin()
    first_col = grid_layout["col"].min()

    for i, row_axes in enumerate(axes):
        for j, ax in enumerate(row_axes):
            if not ax.get_visible():
                continue

            if sharey:
                y_tick_placement = tick_placement.get("y")
                if y_tick_placement == "left":
                    if j == grid_layout.loc[first_visible_in_row[i], "col"]:
                        ax.tick_params(axis="y", labelleft=True)
                    else:
                        ax.tick_params(axis="y", left=False, labelleft=False)

                elif y_tick_placement == "first_col":
                    if not j == first_col:
                        ax.tick_params(axis="y", left=False, labelleft=False)

            if sharex:
                x_tick_placement = tick_placement.get("x")
                if x_tick_placement == "last_row":
                    if i == grid_layout["row"].max():
                        ax.tick_params(axis="x", labelbottom=True)
                    else:
                        ax.tick_params(axis="x", labelbottom=False, bottom=False)
                elif x_tick_placement == "bottom":
                    if i == grid_layout.loc[last_visible_in_col[j], "row"]:
                        ax.tick_params(axis="x", labelbottom=True)
                    else:
                        ax.tick_params(axis="x", bottom=False, labelbottom=False)


def _hide_empty_subplots(axes):
    """
    Hides subplots that do not contain any data.

    Args:
        axes (ndarray): Array of matplotlib Axes objects.
    """
    for row_axes in axes:
        for ax in row_axes:
            if not ax.lines and not ax.patches and not ax.has_data():
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")


def geofacet(
    grid_layout: pd.DataFrame,
    data: pd.DataFrame,
    group_column: str,
    plotting_function: Callable,
    figsize=(12, 8),
    grid_spacing=(0.5, 0.5),
    tick_placement: Optional[Dict[str, str]] = {"x": "bottom", "y": "left"},
    sharex: bool = False,
    sharey: bool = False,
) -> plt.Figure:
    """
    Create a geofaceted plot.

    Args:
        grid_layout (pd.DataFrame): Grid layout with 'name', 'row', 'col' columns.
        data (pd.DataFrame): Data to plot.
        group_column (str): Column matching grid layout names.
        plotting_function (Callable): Function to plot individual grid cells.
        figsize (tuple, optional): Overall figure size. Defaults to (12, 8).
        grid_spacing (tuple, optional): Spacing between grid rows/columns. Defaults to (0.5, 0.5).
        tick_placement (dict, optional): Controls tick placement Default to {"x": "bottom", "y": "left"}.
        sharex (bool, optional): Share x-axis across subplots. Defaults to False.
        sharey (bool, optional): Share y-axis across subplots. Defaults to False.

    Returns:
        plt.Figure: Geofaceted plot figure.
        axes (ndarray): Array of matplotlib Axes objects.
    """
    _validate_columns(grid_layout)
    grid_layout = _adjust_indexing(grid_layout)
    _validate_data(data, group_column)

    max_row = grid_layout["row"].max() + 1
    max_col = grid_layout["col"].max() + 1
    fig, axes = plt.subplots(
        max_row,
        max_col,
        figsize=figsize,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )
    fig.subplots_adjust(hspace=grid_spacing[0], wspace=grid_spacing[1])

    for _, entry in grid_layout.iterrows():
        row, col = entry["row"], entry["col"]
        ax = axes[row, col]
        subset = data[data[group_column] == entry["name"]]
        plotting_function(ax=ax, data=subset, group_name=entry["name"])

    _customize_ticks(axes, grid_layout, tick_placement, sharex, sharey)
    _hide_empty_subplots(axes)
    return fig, axes


def preview_grid(grid_layout: pd.DataFrame, show: bool = True):
    """
    Preview grid layout for visualization and debugging.

    Args:
        grid_layout (pd.DataFrame): Grid layout with 'name', 'row', 'col' columns.
        show (bool): Whether to display the preview. Defaults to True.
    """
    _validate_columns(grid_layout)
    grid_layout = _adjust_indexing(grid_layout)

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

    if show:
        plt.show()
