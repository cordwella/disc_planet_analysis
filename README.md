# Disc Planet Analysis


Python module for analysing the output of 2D and 3D planet-disc interaction simulations.
Currently this has been tested for 3D PLUTO spherical and 2D & 3D Athena++ spherical simulations.

## Installation

This has been designed as a standard Python3 module and in future will be made available from PyPI. You can either add this directory to your python path, or build and install from the source.

If you add this to your Python path you will also have to install the following dependencies: `numpy`, `scipy` and for Athena++ simulations `h5py`.

To build and install this package directly please run:
```
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install dist/disc_planet-0.0.1.tar.gz
```

## Citing this repository
If you use output from this module in your paper please cite this github repository directly as well as the relevant definitions/equations in:
Cordwell & Rafikov (2024) or Cordwell, Ziampras & Rafikov (2025, in prep),  for 2D and 3D conventions respectively

## Notational Convention

2D Cylindrical Co-ordinates ${R, \phi}$
3D Spherical   Co-ordinates ${R, \phi, \theta}$. Where $\theta$ is co-latitutde
3D Cylindrical Co-ordinates ${R_{cylindrical}, \phi, z}$

$\Phi_p$ is the potential of a planet (or generic a perturbing potential)

Velocity is $v$.


## Usage Examples 

### Process 2D simulation data

```
from disc_planet.athena import Athena2DSimulation
import pprint
new_orbit = Athena2DSimulation("test_data/2D_athena_test/", 10, 'athinput.potential')
summary = new_orbit.process_summary_outputs()
new_orbit.save_1d()
pprint.pp(summary)
```

This will save 1D outputs as a dictionary in `test_data/2D_athena_/1D_orbit_10.p` and print the below 
dictionary to the command line.
```
{'setup': {'R0': 1.0,
           'surface_density_slope': -1.5,
           'temperature_slope': -1,
           'sigma_0': 0,
           'stellar_mass': 1.0,
           'planet_mass': 3.125000000000001e-05,
           'dimension': 2,
           'R_p': 1.0,
           'omega0': np.float64(1.0),
           'H0': np.float64(0.05),
           'smoothing_length': np.float64(0.0),
           '2D_potential_type': 'BesselTypeForcing',
           'potential_type': 'NoneType'},
 'gap_timescale_2D': np.float64(-82784.70698165598),
 'total_torque_2D': np.float64(8.253767074045289e-07),
 'inner_torque_2D': np.float64(2.2380648254000206e-06),
 'outer_torque_2D': np.float64(3.052301702754556e-06),
 'gap_inner_loc_2D': np.float32(0.99908847),
 'gap_outer_loc_2D': np.float32(1.0141662),
 'gap_spacing_2D': np.float32(0.01507777),
 'gap_inner_timescale_2D': np.float64(-1.21272518096513e-05),
 'gap_outer_timescale_2D': np.float64(-1.5500776845279193e-05),
 'osl_torque_2D': np.float64(2.6451832640772883e-06)}
```


### 3D Athena++ Simulation
```
from disc_planet.athena import Athena3DSimulation
import pprint
new_orbit = Athena3DSimulation("test_data/3D_athena_test/", 10, 'athinput.potential_3d')
summary = new_orbit.process_summary_outputs()
new_orbit.save_1d()
pprint.pp(summary)
```

Which will produce:
```
{'setup': {'surface_density_slope': -1,
           'temperature_slope': 0,
           'stellar_mass': 1.0,
           'planet_mass': 3.125000000000001e-05,
           'dimension': 3,
           'R0': 1.0,
           'R_p': 1.0,
           'ramp_time': 10,
           'omega0': np.float64(1.0),
           'H0': np.float64(0.05),
           'sigma_0': 1,
           'rho_0': np.float64(7.978845608028654),
           'flaring_index': 0.5,
           'density_slope': -2.5,
           'smoothing_length': 0.01,
           '2D_potential_type': 'BesselTypePotential',
           'potential_type': 'SecondOrderSmoothedPotential'},
 'gap_timescale_2D': np.float64(-85052.09317268865),
 'total_torque_2D': np.float64(4.363565199597769e-07),
 'inner_torque_2D': np.float64(2.350073666873991e-06),
 'outer_torque_2D': np.float64(2.7820512293123616e-06),
 'gap_inner_loc_2D': np.float32(0.9989905),
 'gap_outer_loc_2D': np.float32(1.0133314),
 'gap_spacing_2D': np.float32(0.014340937),
 'gap_inner_timescale_2D': np.float64(85052.09317268865),
 'gap_outer_timescale_2D': np.float64(50516.21982772132),
 'osl_torque_2D': np.float64(2.566062448093176e-06),
 'total_torque': np.float64(6.91096060784992e-07),
 'inner_torque': np.float64(2.4708106681366017e-06),
 'outer_torque': np.float64(3.153336274747061e-06),
 'osl_torque': np.float64(2.8120734714418316e-06),
 'gap_inner_loc': np.float32(0.9989905),
 'gap_outer_loc': np.float32(1.0133314),
 'gap_spacing': np.float32(0.014340937),
 'gap_timescale': np.float64(52998.78857978705),
 'gap_inner_timescale': np.float64(52998.78857978705),
 'gap_outer_timescale': np.float64(48295.036476036345)}
```


Alternatively you can take the significantly slower, but more correct, route of calculating this through integration along columns (rather than the default of spherical shells).
```
from disc_planet.athena import Athena3DSimulation
import pprint
new_orbit = Athena3DSimulation("test_data/3D_athena_test/", 10, 'athinput.potential_3d', intergration_method = 'column', output_folder='test_data/3D_athena_test/col_')
summary = new_orbit.process_summary_outputs()
new_orbit.save_1d()
pprint.pp(summary)
```

### 3D PLUTO Simulation

```
from disc_planet.pluto import Pluto3DSimulation
folder = "../disk-with-slurm/ziampras_data/iso/"
new_orbit = Pluto3DSimulation(folder, 10)
summary = new_orbit.process_summary_outputs()
new_orbit.save_1d()
new_orbit.save_2d()
```


### Changing settings

By default this software will not calculate vortensity or horseshoe widths. These must be enabled by passing the relevant arguments
into your simulation object. e.g.

```
new_orbit = Pluto3DSimulation(folder, 10, include_horseshoe=True, include_vortensity=True)
```

There are additional settings for the calculation of horseshoe width that can be included. These will be documented before release
but for now please look at `simulation.py`.

## Outputs 
By default this code will save a python pickle with a dictionary containting 1D/2D data from the simulation.

2-Dimensional Outputs:
- 2D Vortensity (In the 3D case we use the definition from Masset & Benetize-Llambay (2016) equation 50)
- Averaged velocities
- Surface density

1-Dimensional Outputs:
- $\Sigma$ (surface density)
- $F_{wave}$ 
- $dT/dR$
- $F_{dep}$
- 1D Average of vortensity
- $d \sigma /dt$ using the theory of Cordwell & Rafikov 2024 (valid only for local isothermal discs)

0-Dimensional Outputs:
- Total planetary torque on the disc
- One sided Linblad torques
- Inner and Outer planetary torques
- Gap Width
- Horseshoe width

## Memory usage
By default this will load entire copies of your 3D data into memory, as such it is a very 
memory hungry application. For production size simulations it is unlikely to be able to run on local desktops.


## Planned Features
- [ ] Unit-tests and examples
- [x] PLUTO input class
- [x] 3D Vortensity Calculation
- [x] Horseshoe width calculation
- [ ] Uniform Logging
- [x] Klar type 3D potential
- [ ] FARGO3D Input class
- [ ] 2D Slices of 3D outputs at different scale heights
- [ ] Support for 3D simulations using cylindrical co-ordinates