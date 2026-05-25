# <img src="https://github.com/Aristle12/TuLIP/blob/TuLIPClass/flower_668095.png" width="30" height="30"> TuLIP



Thermally understanding Large Igneous Provinces (TuLIP)

## Description

This project combines heat transfer solvers with functions for emplacing sills in geological models. The solvers handle steady-state and time-varying heat diffusion in anisotropic media, while the sill emplacement functions allow for the simulation of magmatic intrusions. The project is implemented in Python and utilizes libraries such as NumPy, SciPy, Pandas, and PyPardiso for efficient computation.
Documentation is outdated and funky. Currently working on updating the documentaiton!!

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Heat Transfer Solvers](#heat-transfer-solvers)
  - [Sill Emplacement Functions](#sill-emplacement-functions)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Aristle12/TuLIP.git
   ```
2. Install required packages
   ```sh
   pip install numpy scipy pandas pypardiso numba
   ```

## Usage

### Heat Transfer Solvers

#### Example Usage

```python
import numpy as np
from heat_flux import heat_flux

# Define parameters
a = 10  # number of rows
b = 10  # number of columns
dx = 1  # spacing in x direction
dy = 1  # spacing in y direction
k = np.ones((a, b))  # diffusivity field
Tnow = np.zeros((a, b))  # initial temperature field

# Solve for steady-state heat flux
Tf = heat_flux(k, a, b, dx, dy, Tnow, method='straight')
print(Tf)
```

#### Available Solvers

- `heat_flux`: Solves the steady-state heat flux equation.
- `perm_smoothed_solve`: Solves the time-varying heat diffusion equation using an averaged permeability field.
- `perm_chain_solve`: Solves the time-varying heat diffusion equation using the chain rule.
- `conv_chain_solve`: Solves the heat diffusion equation using the convolution method with the chain rule.
- `conv_smooth_solve`: Solves the heat diffusion equation using the convolution method with smoothed permeability.
- `diff_solve`: Switch function to call particular solver and pass parameters.

### Sill Emplacement Functions

#### Example Usage

```python
import numpy as np
from sill_emplacement import single_sill

# Define parameters
T_field = np.zeros((10, 10))  # temperature field
x_space = 5  # x-coordinate for sill center
height = 5  # y-coordinate for sill center
width = 4  # width of the sill
thick = 2  # thickness of the sill
T_mag = 1000  # temperature of the sill

# Emplace a single sill
T_field = single_sill(T_field, x_space, height, width, thick, T_mag)
print(T_field)
```

#### Available Functions

- `to_emplace`: Determines if a temperature is above a threshold for emplacement.
- `single_sill`: Emplaces a simple rectangular sill without a dike tail.
- `circle_sill`: Emplaces a simple circular sill without a dike tail.
- `randn_heights`: Generates random emplacement heights over a normal distribution.
- `x_spacings`: Generates random x-coordinate spacings over a normal distribution.
- `uniform_heights`: Generates random heights spacing from a uniform distribution.
- `uniform_x`: Generates random x-coordinate spacings from a uniform distribution.
- `empirical_CDF`: Generates random numbers from a specific empirical distribution.
- `get_scaled_dims`: Linearly scaled dimensions with height plus noise.
- `randn_dims`: Random normal distribution of dimensions for aspect ratio and shape.
- `uniform_dims`: Random uniform distribution of dimensions for aspect ratio and shape.
- `mult_sill`: Emplaces multiple sills with various shapes and options.
- `get_H`: Calculates external heat sources generated through latent heat of crystallization and radioactive heat generation.
- `get_diffusivity`: Calculates diffusivity based on temperature and lithology.
- `sill_3Dcube`: Generates sills in 3D space for flux control.
- `emplace_3Dsill`: Emplaces a sill into a 2D slice from a 3D sill array.
- `lithology_3Dsill`: Tracks lithology changes in a 2D array due to sill emplacement.
- `cmb_3Dsill`: Updates the crust-mantle boundary due to sill emplacement.


### Simulation Controller (sill_controls)

The `sill_controls` class orchestrates the model setup, execution, and property management. It serves as the interface between the thermal solvers, the geometric rules, and the simulation state.

#### Example Usage

```python
from TuLIP import sill_controls

# Initialize controls with required dimensions
sc = sill_controls(x=300000, y=12000, dx=50, dy=50,
                   T_liquidus=1200, sill_cube_dir='my_simulation/')
```

#### Class Arguments

**Required Arguments:**

*   `x` (float): Horizontal extent of the crust (meters).
*   `y` (float): Vertical thickness of the crust (meters).
*   `dx` (float): Grid spacing in the x-direction (meters).
*   `dy` (float): Grid spacing in the y-direction (meters).

**Optional Arguments:**

*   `T_liquidus` (float, default=`1250`): Magma liquidus temperature (C).
*   `T_solidus` (float, default=`800`): Magma solidus temperature (C).
*   `include_external_heat` (bool, default=`True`): If True, explicitly calculates latent and radiogenic heat contributions.
*   `k_const` (bool, default=`True`): If True, uses a constant thermal conductivity specified by `kc_val`.
*   `kc_val` (float, default=`7.884e7`): Constant thermal conductivity value (used if `k_const=True`).
*   `cp` (float, default=`1`): Initial specific heat capacity reference value.
*   `cp_const` (bool, default=`True`): If True, uses constant specific heat or simple lookup.
*   `calculate_closest_sill` (bool, default=`False`): If True, calculates distance to the nearest already emplaced sill (for thermal interaction rules).
*   `calculate_all_sills_distances` (bool, default=`False`): If True, calculates distances to ALL sills (computationally expensive).
*   `calculate_at_all_times` (bool, default=`False`): If True, re-calculates spatial metrics at every time step.
*   `rock_prop_dict` (dict, default=`None`): Custom dictionary mapping rock names (str) to properties (Density, Porosity, TOC, Specific Heat).
*   `magma_prop_dict` (dict, default=`None`): Custom dictionary for magma properties (keys: Temperature, Lithology, Porosity, Density, Specific Heat, Latent Heat, TOC).
*   `lith_plot_dict` (dict, default=`None`): Dictionary mapping lithologies to integer codes for plotting/VTK export.
*   `sill_cube_dir` (str, default=`'sillcubes/'`): Output directory for saving simulation results and intermediate files.
*   `k_func` (callable, default=`None`): Custom function `f(T, ...)` to calculate thermal conductivity dynamically.
*   `cp_func` (callable, default=`None`): Custom function `f(T, ...)` to calculate specific heat dynamically.
*   `melt_rock` (str, default=`'basalt'`): The name of the host rock type that undergoes melting (used for phase changes).
*   `melt_fraction_function` (callable, default=`None`): Custom function to calculate melt fraction vs temperature.
*   `melt_function_args` (dict, default=`None`): Arguments to pass to the melt fraction function.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Your Name - [@Aristle12](https://github.com/Aristle12) - aristle.jm@psu.edu

Project Link: [https://github.com/Aristle12/TuLIP.git](https://github.com/Aristle12/TuLIP.git)


<a href="https://www.freepik.com/icons/tulip-flower">Icon by Stockio</a>
