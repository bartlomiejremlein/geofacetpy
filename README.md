# geofacetpy

geofacetpy is a Python library built to simplify the creation of geofaceted plots using [matplotlib](https://matplotlib.org/). It allows to easily map data to a grid layout and visualize trends across different regions using matplotlib and seaborn.

This library was heavily inspired by the [R library geofacet](https://github.com/hafen/geofacet).

## Installation
```bash
pip install geofacetpy
```

## Usage

### Before you start [IMPORTANT]
#### Grid 
Currently, the grid has to be a pandas `DataFrame` with specific columns - `name`, `row`, and `col`.
There's a large repository of grids, which follow the same data structure at [hafen/grid-desginer](https://github.com/hafen/grid-designer/tree/master/grids). 

#### Custom plotting function
The custom plotting function, that is supplied to the `geofacet()` must take the following arguments
- `ax` (`Axes` object), 
- `data`
- `group_name` (name of the grid element)
- and all other arguments, such as column names to be plotted

#### geofacet
`group_column` argument in the `geofacet()` is the name of the column in the `data` that is equivalent to the `name` in the grid layout and is the basis for the placement on the grid. 

### Examples

#### Previewing Grid Layout

```python
from geofacet import preview_grid
import pandas as pd

grid = pd.read_csv("grid.csv")
preview_grid(grid)
```
![image](images/grid.png)

#### Creating a geofacet plot

```python
from geofacet import geofacet
import pandas as pd
import matplotlib.pyplot as plt

# Load data and grid layout
data = pd.read_csv("data_grouped.csv")
grid = pd.read_csv("grid.csv")

# Define a custom plotting function
def custom_plot(ax, data, group_name, col_x, col_y):
    ax.bar(data[col_x], data[col_y], color="blue")
    ax.set_title(group_name.replace(" ", "\n"), fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5)

# Create the geofaceted plot
fig = geofacet(
    grid_layout=grid,
    data=data,
    group_column="district",
    plotting_function=custom_plot,
    figure_size=(11, 9),
    grid_spacing=(0.5, 0.5),
    sharex=True,
    sharey=True,
)

# Add titles and labels
fig.suptitle("Example Geofaceted Plot")
fig.supxlabel("Year")
fig.supylabel("Count")
plt.tight_layout()
plt.show()
```

#### Creating a Geofacet Plot with Seaborn

```python
from geofacet import geofacet
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data and grid layout
data = pd.read_csv("data_grouped.csv")
grid = pd.read_csv("grid.csv")

# Define a custom plotting function using Seaborn
def seaborn_plot(ax, data, group_name, col_x, col_y):
    sns.lineplot(ax=ax, data=data, x=col_x, y=col_y, marker="o")
    ax.set_title(group_name.replace(" ", "\n"), fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5)

# Create the geofaceted plot
fig = geofacet(
    grid_layout=grid,
    data=data,
    group_column="district",
    plotting_function=seaborn_plot,
    col_x="submission_year",
    col_y="count",
    figure_size=(11, 9),
    grid_spacing=(0.5, 0.5),
    sharex=True,
    sharey=True,
)

# Add titles and labels
fig.suptitle("Geofaceted Plot with Seaborn")
fig.supxlabel("Year")
fig.supylabel("Count")
plt.tight_layout()
plt.show()
```

### Output Example

![alt text](images/example1.png)

## Contributing

Feel free to open an issue for suggestions, report bugs, or submit a pull request to improve the library.&#x20;

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

