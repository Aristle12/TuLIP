# <img src="https://github.com/Aristle12/TuLIP/blob/TuLIPClass/flower_668095.png" width="30" height="30"> TuLIP



Thermally understanding Large Igneous Provinces (TuLIP)

## Description

TuLIP simulates heat diffusion and carbon emissions from sill swarms in Large Igneous Provinces. Model magma emplacement, thermal evolution, and degassing.

TuLIP (Thermally Understanding Large Igneous Provinces) is a Python library for simulating the thermal evolution and carbon degassing of magmatic intrusions in the Earth’s crust. It combines high-performance 2D heat-transfer solvers with stochastic sill-emplacement algorithms and industry-standard carbon-emission kinetics models to help geoscientists understand how Large Igneous Provinces (LIPs) affect the environment over geological timescales.
## What TuLIP does

TuLIP lets you set up a 2D crustal cross-section, populate it with layered rock types, and then stochastically inject swarms of magmatic sills over millions of years. At each time step the library:
- Advances the temperature field using an implicit or convolution-based heat-transfer solver
- Emplaces sills according to user-defined flux, volume, depth, and geometry distributions
- Computes thermogenic carbon emissions (CO₂) as rising temperatures bake organic-rich sediments, using either the SILLi (Easy%Ro) or Sillburp multi-kerogen kinetics model
- Writes VTK snapshot files you can open in ParaView for 3D visualization

## Key capabilities

- **Multiple Solvers**: ADI, convolution, and fully implicit heat-transfer methods — choose speed or stability
- **Two Emissions Models**: SILLi and Sillburp kinetics give you flexibility to match your geological setting
- **Parallel Runs**: Built-in joblib integration for sweeping across flux and volume parameter spaces
- **VTK Output**: Export temperature, lithology, and emissions fields for 3D visualization in ParaView
- **Numba JIT**: Inner kinetics loops are JIT-compiled for fast time-stepping
- **HDF5 Checkpoints**: Save and reload simulation state for long-running runs

## How it’s organized

| Module | What it provides |
| :--- | :--- |
| `sill_controls` | High-level orchestrator — build sill cubes, run simulations, manage state |
| `cool` | Heat-transfer solvers and thermal-property lookups |
| `emit` | Carbon-emission models (SILLi and Sillburp) |
| `rules` | Sill geometry, stochastic samplers, and mesh-update helpers |
| `utilities` | Workflow helpers (`cooler`, `sillburp_cooler`, `cubemaker`) for parallel runs |

---

New to TuLIP? Visit the [TuLIP Documentation](https://tulip-14ca4526.mintlify.app/) to read the Quickstart and Core Concepts before customizing your own model.
<a href="https://www.freepik.com/icons/tulip-flower">Icon by Stockio</a>
