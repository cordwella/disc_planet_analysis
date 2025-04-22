"""
athena.py

Configuration simulation details for Athena++
"""

import logging
import os
import os.path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from disc_planet.simulation import Simulation
from disc_planet.utils import non_local_change_in_density
from disc_planet import potentials
from disc_planet import athena_read

logger = logging.getLogger(__name__)


class Athena3DSimulation(Simulation):
	"""
	Setup using Amelia Cordwell's 3D Athena++ spherical codes

	See https://github.com/cordwella/disk-with-slurm/blob/potentials/athena_custom/src/pgen/diskplanet_3d_sph.cpp
	for detail
	"""

	SIMULATION_SOURCE = 'ATHENA'
	dimension		 = 3

	def __init__(self, folder, orbit_id, athinput_fn, *args, **kwargs):
		self.folder	   = folder.removesuffix('/') + '/'
		self.orbit_id = orbit_id
		self.athinput	 = athena_read.athinput(self.folder + athinput_fn.removeprefix('/'))

		# TODO: Implement alternative configuration
		fn = f'{self.folder}disksph.out1.{orbit_id:05d}.athdf'

		self.athena_data = athena_read.athdf(fn)
		if 'dens' in self.athena_data:
			self.density_key = 'dens'
		else:
			self.density_key = 'rho'
		# Load data
		self.time	= self.athena_data['Time']

		self.R	   = self.athena_data['x1v']
		self.theta   = self.athena_data['x2v']
		self.phi	 = self.athena_data['x3v']

		# by default Athena's output mom1 etc uses (theta, phi, R) ordering for its indi cies
		self.density = np.moveaxis(self.athena_data[self.density_key], [0, 1], [1, 2])
		self.v_R	 = np.moveaxis(self.athena_data['mom1'], [0, 1], [1, 2])/self.density
		self.v_theta = np.moveaxis(self.athena_data['mom2'], [0, 1], [1, 2])/self.density
		self.v_phi   = np.moveaxis(self.athena_data['mom3'], [0, 1], [1, 2])/self.density

		# Background slopes etc
		self.setup = {
			'surface_density_slope': self.athinput['problem']['sslope'],
			'temperature_slope': self.athinput['problem'].get('tslope', 0),
			'stellar_mass': self.athinput['problem']['GM'],
			'planet_mass': self.athinput['problem']['GMp'],
			'dimension': 3,
			'R0': self.athinput['problem']['r0'],
			'R_p': self.athinput['problem']['r0'],
			'ramp_time': self.athinput['problem']['tramp_dur'],
		}

		self.setup['omega0'] = np.sqrt(self.setup['stellar_mass'] * self.setup['R0']**(-3))

		self.setup['nx_R'] = len(self.R)

		# Setup h0 depending on the thermodynamics
		if self.density_key == 'dens':
			# Globally isothermal setup
			self.setup['H0'] = self.athinput['hydro']['iso_sound_speed']/self.setup['omega0']
		else:
			# setup using the
			p0_over_r0 = self.athinput['problem']['p0_over_rho0']
			self.setup['H0'] = p0_over_r0**(1/2)/self.setup['omega0']

		self.setup['sigma_0'] = self.athinput['problem'].get('sig0', 1)
		self.setup['rho_0'] = self.setup['sigma_0']/(np.sqrt(2 * np.pi) * self.setup['H0'])

		self.setup['flaring_index'] = (1 + self.setup['temperature_slope'])/2
		self.setup['density_slope'] = self.setup['surface_density_slope'] - 1 - self.setup['flaring_index']

		# Setup the planetary potential object
		self.setup['smoothing_length']   = self.athinput['problem'].get('rsm', 0)

		# Ramp setup
		ramp = 1
		if self.time < (self.athinput['problem']['tramp_dur'] + self.athinput['problem']['tramp_start']) * 2 * np.pi:
			ramp  = 0.5*(1.0 - np.cos(np.pi * (self.time - self.athinput['problem']['tramp_start']) / self.athinput['problem']['tramp_dur']))

		phi_p = self.athinput['problem'].get('phipl0', np.pi) + self.time * self.setup['omega0']

		self.potential = potentials.SecondOrderSmoothedPotential(
			self.setup['planet_mass'] * ramp, self.setup['R0'], phi_p, np.pi/2,
			self.setup['smoothing_length'])

		self.potential_2D = potentials.BesselTypePotential(
			self.setup['planet_mass'] * ramp, self.setup['R0'], phi_p,
			self.setup['H0'], self.setup['temperature_slope'],
			self.setup['smoothing_length'])
		super().__init__(*args, **kwargs)


class Athena2DSimulation(Simulation):
	"""
	Setup using Amelia Cordwell's 2D Athena++ cylindircal codes.

	By default this calculates values from the 2-dimensional outputs

	See: https://github.com/cordwella/disk-with-slurm/blob/potentials/athena_custom/src/pgen/diskplanet_alphabeta.cpp
	"""

	SIMULATION_SOURCE = 'ATHENA'
	dimension		 = 2

	def __init__(self, folder, orbit_id, athinput_fn, *args, **kwargs):
		self.folder	   = folder.removesuffix('/') + '/'
		self.orbit_id = orbit_id
		self.athinput	 = athena_read.athinput(self.folder + athinput_fn.removeprefix('/'))
		self.use_1d_athena_outputs = kwargs.get('use_1d_athena_outputs', False)

		fn = self.folder + 'diskplanet.out1.{:05d}.athdf'.format(orbit_id)

		self.athena_data = athena_read.athdf(fn)

		if 'dens' in self.athena_data:
			self.density_key = 'dens'
		else:
			self.density_key = 'rho'

		# Load data
		self.time = self.athena_data['Time']
		self.R   = self.athena_data['x1v']
		self.phi   = self.athena_data['x2v']

		# by default Athena's output mom1 etc uses (theta, phi, R) ordering for its indicies
		self.surface_density =	 np.moveaxis(self.athena_data[self.density_key][0], [0], [1])
		self.v_R_2D		  = np.moveaxis(self.athena_data['mom1'][0], [0], [1])/self.surface_density
		self.v_phi_2D	  = np.moveaxis(self.athena_data['mom2'][0], [0], [1])/self.surface_density

		# Read in the athinput file to get all of the data we need
		self.setup = {
			'R0': self.athinput['problem']['r0'],
			'surface_density_slope': self.athinput['problem']['dslope'],
			'temperature_slope': self.athinput['problem'].get('tslope', 0),
			'sigma_0': self.athinput['problem'].get('rho_0', 0),
			'stellar_mass': self.athinput['problem']['GM'],
			'planet_mass': self.athinput['problem']['GMp'],
			'dimension': 2,
			'R_p': self.athinput['problem']['r0'],
			'b': self.athinput['problem'].get('eps', 0),
			'n_phi': self.athinput['mesh']['nx2']
		}

		self.setup['omega0'] = np.sqrt(self.setup['stellar_mass'] * self.setup['R0']**(-3))

		if self.density_key == 'dens':
			# Globally isothermal setup
			self.setup['H0'] = self.athinput['hydro']['iso_sound_speed']/self.setup['omega0']
		else:
			# setup using the
			p0_over_r0 = self.athinput['problem']['p0_over_rho0']
			self.setup['H0'] = p0_over_r0**(1/2)/self.setup['omega0']

		ramp = 1
		if self.time < self.athinput['problem']['tramp'] * 2 * np.pi:
			# Ramp setup
			ramp  = 0.5*(1.0 - np.cos(np.pi * self.time / (self.athinput['problem']['tramp'] *  2 * np.pi)))

		omegap = np.sqrt((self.setup['stellar_mass'] + self.setup['planet_mass']) * self.athinput['problem']['r0']**(-3))
		phi_p = self.athinput['problem'].get('phipl0', np.pi) + self.time * omegap

		# Check potential type and setup the relevant thing
		pot_order = self.athinput['problem'].get('potential_order', 4)

		# Setup the planetary potential object
		self.setup['smoothing_length'] = self.athinput['problem'].get('eps', 0) * self.setup['H0']

		# smoothingB = 2 * self.setup['R0'] * self.athinput["mesh"]["x2max"]/self.athinput["mesh"]["nx2"]
		smoothingB = 2 * self.setup['R0'] * np.max(self.phi)/len(self.phi)
		# self.athinput["mesh"]["x2max"]/self.athinput["mesh"]["nx2"]

		if pot_order == 4:
			self.potential_2D = potentials.FourthOrderSmoothedPotential(
				self.setup['planet_mass'] * ramp, self.setup['R_p'], phi_p,
				self.setup['smoothing_length'])
		elif pot_order == 2:
			self.potential_2D = potentials.SecondOrderSmoothedPotential(
				self.setup['planet_mass'] * ramp, self.setup['R_p'], phi_p, 0,
				self.setup['smoothing_length'])
		elif pot_order == -1:
			self.potential_2D = potentials.BesselTypeForcing(
				self.setup['planet_mass'] * ramp, self.setup['R_p'], phi_p,
				self.setup['H0'], self.setup['temperature_slope'], smoothingB)
		elif pot_order == -2:
			self.potential_2D = potentials.BesselTypePotentialConstH(
				self.setup['planet_mass'] * ramp, self.setup['R_p'], phi_p,
				self.setup['H0'], smoothingB)
		elif pot_order == -3:
			self.potential_2D = potentials.BesselTypePotential(
				self.setup['planet_mass'] * ramp, self.setup['R_p'], phi_p,
				self.setup['H0'], self.setup['temperature_slope'], smoothingB)
		elif pot_order == -4:
			self.potential_2D = potentials.BesselLinModePotential(
				self.setup['planet_mass'] * ramp, self.setup['R_p'], phi_p,
				self.setup['H0'], self.setup['temperature_slope'], smoothingB)


		super().__init__(*args, **kwargs)


	def process_1d_outputs(self, *args, **kwargs):
		# Check if we can access out3 and out4
		# These are specified as outputs only in AJC's setups
		f_1 = self.folder + 'diskplanet.out3.{:04d}0.athdf'.format(self.orbit_id)
		f_2 = self.folder + 'diskplanet.out4.{:04d}0.athdf'.format(self.orbit_id)

		if self.use_1d_athena_outputs and os.path.isfile(f_1) and os.path.isfile(f_2):
			logger.info('Extracting 1D outputs from out3 and out4')

			a = athena_read.athdf(f_1)
			b = athena_read.athdf(f_2)

			dphi = 2 * np.pi / a['RootGridSize'][1]

			num_phi = np.shape(b[self.density_key])[1]
			total_phi = a['RootGridSize'][1]
			norm = total_phi/num_phi

			self.dTdR_2D = self.R**2 * np.sum(a['planet_src_mom2'][0], axis=0) * dphi
			self.T_2D = cumulative_trapezoid(self.dTdR_2D, self.R, initial=0)

			self.G_2D = self.R**2 * dphi * np.sum(a['visc_flux_mom2'][0], axis=0)
			self.dGdR_2D = self.R**2 * np.gradient(self.G_2D, self.R)

			self.F_wave_2D = - np.sum(a['f_wave'][0], axis=0) - self.G_2D
			self.F_dep_2D  = self.T_2D - self.F_wave_2D

			self.dF_depdR_2D = self.dTdR_2D + self.dGdR_2D - np.sum(a['df_wave_dr'][0], axis=0)
			self.dF_wavedR_2D = self.dGdR_2D + np.sum(a['df_wave_dr'][0], axis=0)
			self.dF_wavedR_2D[0] = 0
			self.dF_depdR_2D[0] = 0

			self.surface_density_1D = np.average(b[self.density_key][0], axis=0)/norm
			self.v_r_1D = np.average(b['mom1'][0], axis=0)/norm/self.surface_density_1D
			self.v_phi_1D = np.average(b['mom2'][0], axis=0)/norm/self.surface_density_1D

			self.dsigmadt_2D = non_local_change_in_density(
				self.R/self.setup['R0'], -1 * self.setup['surface_density_slope'], -1 *self.setup['temperature_slope'],
				self.setup['H0']/self.setup['R0'], self.dF_depdR_2D/self.surface_density_1D)

		else:
			logger.info('Extracting 1D outputs from 2D results')
			return super().process_1d_outputs(*args, **kwargs)
