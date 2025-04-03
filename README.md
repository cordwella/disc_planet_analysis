# Disc Planet Analysis


Python module for analysing the output of 2D and 3D planet-disc interaction simulations.
Currently this has been tested for 3D PLUTO spherical and 2D & 3D Athena++ spherical simulations.

Contributions via pull request are welcome.

## Installation

This has been designed as a standard Python3 module and will be made installible with pip.

Current dependencies are `numpy`, `scipy` and for Athena++ simulations `h5py`.

## Citing this repository
If you use output from this module in your paper please cite this github repository directly as well as the relevant definitions/equations in:
Cordwell & Rafikov (2024), and Cordwell, Ziampras & Rafikov (2025, in prep)
For 2D and 3D conventions respectively

## Notation Convention

2D Cylindrical Co-ordinates ${R, \phi}$
3D Spherical   Co-ordinates ${R, \phi, \theta}$. Where $\theta$ is co-latitutde
3D Cylindrical Co-ordinates ${R_{cylindrical}, \phi, z}$

$\Phi_p$ is the potential of a planet (or generic a perturbing potential)

Velocity is $v$

## Usage Example 


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
           '2D_potential_type': 'BesselTypeForcing'},
 'gap_timescale': np.float64(-7097.135954785253),
 'total_torque_2D': np.float64(8.423336435410304e-08),
 'inner_torque_2D': np.float64(5.010496707543883e-07),
 'outer_torque_2D': np.float64(4.7176811825816574e-07),
 'gap_inner_loc_2D': np.float32(0.9858743),
 'gap_outer_loc_2D': np.float32(1.0141662),
 'gap_spacing_2D': np.float32(0.02829194),
 'gap_inner_timescale_2D': np.float64(-8497.287093065679),
 'gap_outer_timescale_2D': np.float64(-8088.894285830293),
 'osl_torque_2D': np.float64(4.864088945062771e-07)}
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


## Outputs 
By default this will create a python pickle object detailing a single snapshot of the
simulation at a given orbital id number.

2-Dimensional Outputs:
- 2D Vortensity
- Averaged velocities
- Surface density

1-Dimensional Outputs:
- $\Sigma$ (surface density)
- $F_{wave}$ 
- $dT/dR$
- $F_{dep}$
- $d \sigma /dt$ using the theory of Cordwell & Rafikov 2024 (valid only for local isothermal discs)

0-Dimensional Outputs:
- Total planetary torque on the disc
- One sided Linblad torques
- Inner and Outer planetary torques
- Gap Width


## Memory usage
By default this will load entire copies of your 3D data into memory, as such it is a very 
memory hungry application. For production size simulations it is unlikely to be able to run on local desktops.


## Planned Features
- [ ] Unit-tests and examples
- [ ] PLUTO input class
- [ ] 3D Vortensity Calculation
- [ ] Horseshoe width calculation
- [ ] Logging
- [ ] Klar type 3D potential
- [ ] FARGO3D Input class