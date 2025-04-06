import logging
import numpy as np

from disc_planet.simulation import Simulation
from disc_planet import potentials

logger = logging.getLogger(__name__)


def parse_pluto_ini(filename):
	f = open(filename)
	a = f.read().split('\n')

	config = {}
	for line in a:
		if len(line) > 3 and line[0] != '[':
			line = line.split()

			if len(line) == 2:
				config[line[0]] = line[1]
			else:
				config[line[0]] = line[1:]
	return config


class Pluto3DSimulation(Simulation):
	"""
	Load a 3D Pluto simulation

	This makes a key assumptions - the most pressing of which is that 
	the output is configured as dbl multiple_files

	"""
	SIMULATION_SOURCE = 'PLUTO'
	dimension         = 3

	def __init__(self, folder, orbit_id, *args, **kwargs):
		self.folder = folder
		self.orbit_id = orbit_id

		self.config = parse_pluto_ini(folder + 'pluto.ini')

		# Background slopes etc
		self.setup = {
			'density_slope': float(self.config['PROFILE_P']),
			'temperature_slope': float(self.config['PROFILE_Q']),
			'stellar_mass': float(self.config['MASS_STAR']),
			'planet_mass': float(self.config['MASS_PLAN']),
			'dimension': 3,
			'R0': 1,
			'R_p': 1,
			'ramp_time': float(self.config['GROWTH_ORBITS']),
			'H0': float(self.config['ASPECT_RATIO']),
		}

		self.setup['rho_0'] = float(self.config['RHO0'])
		self.setup['sigma_0'] = self.setup['rho_0'] * (np.sqrt(2 * np.pi) * self.setup['H0'])

		self.setup['flaring_index'] = (1 + self.setup['temperature_slope'])/2
		self.setup['surface_density_slope'] = self.setup['density_slope'] + 1 + self.setup['flaring_index']

		# Need to double check this calculation against the ini file
		time_step = float(self.config['dbl'][0])
		self.time = orbit_id * time_step

		# Pluto default RSM is 0.03 R
		# Equation 55
		# https://www.aanda.org/articles/aa/pdf/2012/09/aa19557-12.pdf
		self.setup['smoothing_length'] = 0.5 * (self.setup['planet_mass']/(3 * self.setup['stellar_mass']))**(1/3)

		# self.config['Static Grid Output']

		# Okay this is actually not the best way to get it -> probably construct from 
		# ini instead

		_, R_min, NR, R_scale, R_max = self.config['X1-grid']
		_, theta_min, Ntheta, theta_scale, theta_max = self.config['X2-grid']
		_, phi_min, Nphi, phi_scale, phi_max = self.config['X3-grid']

		if R_scale == 'u':
			self.R =  np.linspace(R_min, R_max, NR)
		elif R_scale == 'l+':
			self.R = np.logspace(np.log10(float(R_min)), np.log10(float(R_max)), num=int(NR), base=10)
		else:
			raise Exception('Unable to reconsruct R grid')

		self.theta = np.linspace(float(theta_min), float(theta_max), int(Ntheta))
		self.phi = np.linspace(float(phi_min), float(phi_max), int(Nphi))

		NX1, NX2, NX3 = self.R.size, self.theta.size, self.phi.size

		data_dir = folder

		if self.config.get('output_dir'):
			data_dir = folder + self.config['output_dir'].strip('./') + '/'

		self.density = np.moveaxis(np.fromfile(f'{data_dir}rho.{orbit_id:04d}.dbl').reshape(NX3, NX2, NX1), [2, 1], [0, 2])		
		self.v_R = np.moveaxis(np.fromfile(f'{data_dir}vx1.{orbit_id:04d}.dbl').reshape(NX3, NX2, NX1),  [2, 1], [0, 2])
		self.v_phi = np.moveaxis(np.fromfile(f'{data_dir}vx3.{orbit_id:04d}.dbl').reshape(NX3, NX2, NX1),  [2, 1], [0, 2])

		self.setup['ramp_time'] = float(self.config['GROWTH_ORBITS']) * 2 * np.pi
		
		ramp = 1
		if self.time < self.setup['ramp_time']:
			# TODO: Figure out the ramp setup for this simulation
			logger.warning('Pluto ramping not yet setup')
			ramp = 1

		phi_p = 0

		self.potential = potentials.KlarTypePotential(
			self.setup['planet_mass'] * ramp, self.setup['R0'], phi_p,
			 np.pi/2, self.setup['smoothing_length'])
	
		self.potential_2D = potentials.BesselTypeForcing(
			self.setup['planet_mass'] * ramp, self.setup['R0'], phi_p,
			self.setup['H0'], self.setup['temperature_slope'],
			self.setup['smoothing_length'])

		super().__init__(*args, **kwargs)



class Pluto2DSimulation(Simulation):
	SIMULATION_SOURCE = 'PLUTO'
	dimension         = 2

	def __init__(self, folder, orbit_id, *args, **kwargs):
		self.folder = folder

		self.config = parse_pluto_ini(folder + 'pluto.ini')

		# Background slopes etc
		self.setup = {
			'surface_density_slope': float(self.config['PROFILE_P']),
			'temperature_slope': float(self.config.get('PROFILE_Q', 0)),
			'stellar_mass': float(self.config['MASS_STAR']),
			'planet_mass': float(self.config['MASS_PLAN']),
			'dimension': 2,
			'R0': 1,
			'R_p': 1,
			'ramp_time': float(self.config.get('GROWTH_ORBITS', 1)) * 2 * np.pi,
			'H0': float(self.config['ASPECT_RATIO']),
		}

		self.setup['sigma_0'] = float(self.config['RHO0'])
		## self.setup['sigma_0'] = self.setup['rho_0'] * (np.sqrt(2 * np.pi) * self.setup['H0'])

		self.setup['flaring_index'] = (1 + self.setup['temperature_slope'])/2
		# self.setup['surface_density_slope'] = self.setup['density_slope'] + 1 + self.setup['flaring_index']

		# Need to double check this calculation against the ini file
		time_step = float(self.config['dbl'][0])
		self.time = orbit_id * time_step

		# Pluto default RSM is 0.03 R
		# Equation 55
		# https://www.aanda.org/articles/aa/pdf/2012/09/aa19557-12.pdf
		self.setup['smoothing_length'] = 0.6 * self.setup['H0']

		_, R_min, NR, R_scale, R_max = self.config['X1-grid']
		_, phi_min, Nphi, phi_scale, phi_max = self.config['X2-grid']

		if R_scale == 'u':
			self.R =  np.linspace(R_min, R_max, NR)
		elif R_scale == 'l+':
			self.R = np.logspace(np.log10(float(R_min)), np.log10(float(R_max)), num=int(NR), base=10)
		else:
			raise Exception('Unable to reconsruct R grid')

		self.phi = np.linspace(float(phi_min), float(phi_max), int(Nphi))

		NX1, NX2 = self.R.size, self.phi.size

		data_dir = folder

		if self.config.get('output_dir'):
			data_dir = folder + self.config['output_dir'].strip('./') + '/'

		self.surface_density = np.moveaxis(np.fromfile(f'{data_dir}rho.{orbit_id:04d}.dbl').reshape(NX2, NX1), [1], [0])
		self.v_R_2D   = np.moveaxis(np.fromfile(f'{data_dir}vx1.{orbit_id:04d}.dbl').reshape(NX2, NX1), [1], [0])
		self.v_phi_2D = np.moveaxis(np.fromfile(f'{data_dir}vx3.{orbit_id:04d}.dbl').reshape(NX2, NX1), [1], [0])

		ramp = 1
		if self.time < self.setup['ramp_time']:
			ramp = 1

		phi_p = 0
	
		self.potential_2D = potentials.SecondOrderSmoothedPotential(
			self.setup['planet_mass'] * ramp, self.setup['R0'], phi_p, 0,
			self.setup['H0'], 
			self.setup['smoothing_length'])

		super().__init__(*args, **kwargs)
