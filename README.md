# <img src="https://github.com/Aristle12/TuLIP/blob/TuLIPClass/flower_668095.png" width="30" height="30"> TuLIP



Thermally understanding Large Igneous Provinces (TuLIP)

## Description

This project combines heat transfer solvers with functions for emplacing sills in geological models. The solvers handle steady-state and time-varying heat diffusion in anisotropic media, while the sill emplacement functions allow for the simulation of magmatic intrusions. The project is implemented in Python and utilizes libraries such as NumPy, SciPy, Pandas, and PyPardiso for efficient computation.

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


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Your Name - [@yourusername](https://github.com/yourusername) - your.email@example.com

Project Link: [https://github.com/Aristle12/TuLIP.git](https://github.com/Aristle12/TuLIP.git)


<a href="https://www.freepik.com/icons/tulip-flower">Icon by Stockio</a>
