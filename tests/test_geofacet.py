import pytest
import pandas as pd
from matplotlib.figure import Figure
from geofacetpy import geofacet, preview_grid


@pytest.fixture
def sample_grid_layout():
    return pd.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "row": [1, 1, 2, 2],
            "col": [1, 2, 1, 2],
        }
    )


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "name": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "value": [10, 20, 30, 40, 15, 25, 35, 45],
        }
    )


@pytest.fixture
def sample_plotting_function():
    def plot(ax, data, group_name):
        ax.bar(data.index, data["value"], label=group_name)
        ax.set_title(group_name)

    return plot


def test_geofacet_valid_input(
    sample_grid_layout, sample_data, sample_plotting_function
):
    fig, axes = geofacet(
        grid_layout=sample_grid_layout,
        data=sample_data,
        group_column="name",
        plotting_function=sample_plotting_function,
        figsize=(8, 6),
        sharex=True,
        sharey=True,
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) > 0


def test_geofacet_missing_columns(sample_data, sample_plotting_function):
    invalid_grid_layout = pd.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "row": [1, 1, 2, 2],
        }
    )
    with pytest.raises(ValueError, match="Grid layout is missing required columns"):
        geofacet(
            grid_layout=invalid_grid_layout,
            data=sample_data,
            group_column="name",
            plotting_function=sample_plotting_function,
        )


def test_geofacet_missing_group_column(
    sample_grid_layout, sample_data, sample_plotting_function
):
    invalid_data = sample_data.drop(columns=["name"])
    with pytest.raises(ValueError, match="Column 'name' not found in the data"):
        geofacet(
            grid_layout=sample_grid_layout,
            data=invalid_data,
            group_column="name",
            plotting_function=sample_plotting_function,
        )


def test_preview_grid_valid_input(sample_grid_layout):
    """Test preview_grid runs without error for valid input."""
    preview_grid(sample_grid_layout, show=False)


def test_preview_grid_missing_columns():
    """Test preview_grid raises ValueError for missing columns."""
    invalid_grid_layout = pd.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "row": [1, 1, 2, 2],
        }
    )
    with pytest.raises(ValueError, match="Grid layout is missing required columns"):
        preview_grid(invalid_grid_layout, show=False)
