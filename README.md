# geofacetpy

geofacetpy is a Python library built to simplify the creation of geofaceted plots using [matplotlib](https://matplotlib.org/). It allows to easily map data to a grid layout and visualize trends across different regions using matplotlib and seaborn.

![image](images/us.png)

This library was heavily inspired by the [R library geofacet](https://github.com/hafen/geofacet).

## Installation
```
pip install geofacetpy
```

## Usage

### Before you start [IMPORTANT]
#### Grid 
Currently, the grid has to be a pandas `DataFrame` with specific columns - `row`, `col` and string column that serves as a label (default `name`).

|row|col|name      |
|---|---|----------|
|6  |7  |Alabama   |
|1  |1  |Alaska    |
|6  |2  |Arizona   |
|6  |5  |Arkansas  |
|6  |1  |California|

There's a large repository of grids, which follow the same data structure at [hafen/grid-desginer](https://github.com/hafen/grid-designer/tree/master/grids). 

#### Custom plotting function
The custom plotting function, that is supplied to the `geofacet()` must take the following arguments
- `ax` (`Axes` object), 
- `data`
- `group_name` (name of the column in data that corresponds to string column with label in grid)

```python
def custom_plot(ax, data, group_name):
    ax.bar(data['col_x'], data['col_y'], color="blue")
    ax.set_title(group_name, fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5)
```

#### geofacet
To create a geofaceted plot, use `geofacet()`. 
Supply the following arguments:
- `grid_layout`: pd.DataFrame with grid
- `data`: pd.DataFrame with data
- `group_column`: column name in `data` to be used as a facet, basis for placement on the grid
- `grid_col`: column name in `grid_layout` with label (optional, if different than `name`) 
- `plotting_function`: callable, function to draw a plot for each grid element

```python
from geofacetpy import geofacet

fig, axes = geofacet(
    grid_layout=grid,
    data=data,
    group_column="district",
    plotting_function=custom_plot,
    sharex=True,
    sharey=True,
)
```

### Examples


#### Creating a geofacet plot

```python
from geofacet import geofacet
import pandas as pd
import matplotlib.pyplot as plt

# Load data and grid layout
data = pd.read_csv("data_grouped.csv")
grid = pd.read_csv("grid.csv")

# Define a custom plotting function
def custom_plot(ax, data, group_name):
    ax.bar(data['col_x'], data['col_y'], color="blue")
    ax.set_title(group_name.replace(" ", "\n"), fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5)

# Create the geofaceted plot
fig, axes = geofacet(
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
def seaborn_plot(ax, data, group_name):
    sns.lineplot(ax=ax, data=data, x='col_x', y='col_y', marker="o")
    ax.set_title(group_name, fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5)

# Create the geofaceted plot
fig, axes = geofacet(
    grid_layout=grid,
    data=data,
    group_column="district",
    plotting_function=seaborn_plot,
    figure_size=(11, 9),
    grid_spacing=(0.5, 0.5),
    sharex=True,
    sharey=True,
)

# Add titles and labels
fig.suptitle("Geofaceted Plot with Seaborn")
fig.supxlabel("Year")
fig.supylabel("Count")
plt.show()
```

### Output Example

![alt text](images/example1.png)
![alt text](images/europe.png)

#### Previewing Grid Layout

If the label column is not `name`, pass it in `grid_col` argument. 

```python
from geofacet import preview_grid
import pandas as pd

grid = pd.read_csv("grid.csv")
preview_grid(grid)
```
or
```python
grid_2 = pd.read_csv("grid.csv")
preview_grid(grid_2, grid_col="label")
```
![image](images/grid.png)


## Contributing

Feel free to open an issue for suggestions, report bugs, or submit a pull request to improve the library.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

