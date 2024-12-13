import pytest
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from geofacetpy import (
    GridLayoutValidator,
    DataValidator,
    GeoFacetPlotter,
    preview_grid,
    geofacet,
)

matplotlib.use("Agg")


# Sample test data for validation
def sample_grid_layout():
    return pd.DataFrame({"name": ["A", "B", "C"], "row": [0, 1, 2], "col": [0, 1, 0]})


def sample_data():
    return pd.DataFrame({"group": ["A", "B", "C"], "value": [10, 20, 30]})


def sample_plot_function(ax, data, group_name, **kwargs):
    ax.bar(data.index, data["value"])
    ax.set_title(group_name)


def test_validate_columns():
    valid_grid = sample_grid_layout()
    GridLayoutValidator.validate_columns(valid_grid)

    invalid_grid = pd.DataFrame({"row": [0, 1], "col": [0, 1]})
    with pytest.raises(ValueError):
        GridLayoutValidator.validate_columns(invalid_grid)


def test_adjust_indexing():
    grid = pd.DataFrame({"name": ["A", "B"], "row": [1, 2], "col": [1, 2]})
    adjusted = GridLayoutValidator.adjust_indexing(grid)
    assert (adjusted["row"] == [0, 1]).all()
    assert (adjusted["col"] == [0, 1]).all()


def test_validate_data():
    data = sample_data()
    DataValidator.validate_data(data, "group")

    with pytest.raises(ValueError):
        DataValidator.validate_data(data, "nonexistent")

    with pytest.raises(ValueError):
        DataValidator.validate_data(data, "group", ["missing_column"])


def test_geofacet_plotter():
    grid_layout = sample_grid_layout()
    data = sample_data()

    plotter = GeoFacetPlotter(
        grid_layout=grid_layout,
        data=data,
        group_column="group",
        plotting_function=sample_plot_function,
    )
    fig = plotter.plot()

    assert isinstance(fig, plt.Figure)


def test_geofacet():
    grid_layout = sample_grid_layout()
    data = sample_data()

    fig = geofacet(
        grid_layout=grid_layout,
        data=data,
        group_column="group",
        plotting_function=sample_plot_function,
    )
    assert isinstance(fig, plt.Figure)


def test_preview_grid():
    grid_layout = sample_grid_layout()
    preview_grid(grid_layout, show=False)
