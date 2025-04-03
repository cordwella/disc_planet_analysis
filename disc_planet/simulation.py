import logging
import pickle

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import RegularGridInterpolator

from disc_planet.analytic_gap import non_local_change_in_density


logger = logging.getLogger(__name__)

class Simulation(object):
	"""
	SimulationObject 

	Base class that all other references should inherit from.
	Inheriting objects should provide an __init__ function that can
	read in data from a specific format/set of simulation settings

	The format for v_R, v_phi, v_theta dens etc is a 3D numpy array 
	where the first index is along R, the second along phi, and the third
	along thetatheta_i	For consistency even 2D simulations must return the third column, 
	except in the 2D implementations.

	2D implemations should represent either mass weighted averaged
	velocities, or fully integrated values, and should be provided as 
	2D numpy arrays.

	Classes are free to implement their own preferred way of calculating
	these values (e.g. small angle approximation). Integration along 
	spheres under the small angle approximation may be done as default.
	"""

	dimension = None
	intergration_method = 'spherical'

	include_vortensity	  = False

	setup = {}
	orbit_id = None
	potential    = None 
	potential_2D = None

	# default data values
	R = None
	phi = None
	theta = None
	z = None
	R_cylindrical = None

	density = None
	v_R = None 
	v_phi = None 
	v_theta = None

	surface_density = None
	v_R_2D = None
	v_phi_2D = None 

	surface_density_1D = None
	v_R_1D = None
	v_phi_1D = None

	dsigmadt_2D = None
	dsigmadt = None

	def __init__(self, *args, **kwargs):
		self.output_folder = kwargs.get('output_folder', None)
		if self.output_folder is None:
			self.output_folder = self.folder
		self.intergration_method = kwargs.get('intergration_method', self.intergration_method)

		if self.R is None:
			raise Exception('Data not setup. This class is not designed to be used on its own and must be inherited from.')

	# Main computation logic will be contained in these
	def process_all_details(self, *args, compute_vortensity=False, **kwargs):
		self.reduce_to_2d(*args, **kwargs)

		if compute_vortensity:
			self.calculate_vortensity()
		
		self.process_1d_outputs(*args, **kwargs)
		return self.process_summary_outputs(*args, **kwargs)

	def process_1d_outputs(self, *args, **kwargs):
		# Depends on process all details having been calculated
		if self.surface_density is None:
			self.reduce_to_2d()

		# Calculate 1D averages
		self.surface_density_1D = np.average(self.surface_density, axis=1)
		self.v_R_1D   = np.average(
			self.v_R_2D * self.surface_density, axis=1)/self.surface_density_1D
		self.v_phi_1D = np.average(
			self.v_phi_2D * self.surface_density, axis=1)/self.surface_density_1D

		if self.dimension == 3:
			# Calculate dT/dR
			integrand = -1 * (self.density * self.R[:, None, None] * 
				self.potential.calculate_dPhidphi_3D(self.R, self.phi, self.theta))
			
			self.dTdR = np.trapezoid(
				self.intergrate_in_z(integrand),
				self.phi, axis=1)
			self.T = cumulative_trapezoid(self.dTdR, self.R, initial=0)

			# Calculate F_{wave}
			d_v_phi = self.v_phi - self.v_phi_1D[:, None, None]
			self.F_wave = self.R**2 * (np.trapezoid( 
				self.intergrate_in_z(self.density * self.v_R * (self.v_phi - self.v_phi_1D[:, None, None])), 
				axis=1, x=self.phi))
			self.dF_wavedR = np.gradient(self.F_wave, self.R)

			# Combine all properties
			self.F_dep    = self.T - self.F_wave
			self.dF_depdR = self.dTdR - self.dF_wavedR

			self.dsigmadt = non_local_change_in_density(
				self.R/self.setup['R0'], -1 * self.setup['surface_density_slope'], -1 *self.setup['temperature_slope'], self.setup['H0']/self.setup['R0'],
				self.dF_depdR/self.surface_density_1D)

		# Calculate dT/dR_2D
		self.dTdR_2D = -1 * np.trapezoid(
			self.surface_density * self.R[:, None] * 
			self.potential_2D.calculate_dPhidphi_2D(self.R, self.phi),
			self.phi, axis=1)

		self.T_2D = cumulative_trapezoid(self.dTdR_2D, self.R, initial=0)

		# Calculate F_{wave}_2D
		self.F_wave_2D = self.R**2 * np.trapezoid(
			self.surface_density * self.v_R_2D *(self.v_phi_2D - self.v_phi_1D[:, None]),
			axis=1, x=self.phi)

		self.dF_wavedR_2D = np.gradient(self.F_wave_2D, self.R)

		# Combine all properties
		self.F_dep_2D	= self.T_2D - self.F_wave_2D
		self.dF_depdR_2D = self.dTdR_2D - self.dF_wavedR_2D

		self.dsigmadt_2D = non_local_change_in_density(
			self.R/self.setup['R0'], -1 * self.setup['surface_density_slope'], -1 *self.setup['temperature_slope'], self.setup['H0']/self.setup['R0'],
			self.dF_depdR_2D/self.surface_density_1D)

	def process_summary_outputs(self, *args, **kwargs):
		# Depends on process_1d_outputs having run

		if self.dsigmadt_2D is None:
			self.process_1d_outputs()

		furthest_gap_distance = self.setup['H0'] * 0.3	

		gap_timescale = 1/self.dsigmadt_2D[np.argmin( np.abs(self.R - self.setup['R_p']) )]
		r_inner = self.R[np.logical_and(self.R < self.setup['R_p'], self.R > (self.setup['R_p'] - furthest_gap_distance))]
		r_outer = self.R[np.logical_and(self.R > self.setup['R_p'], self.R < (self.setup['R_p'] + furthest_gap_distance))]
		
		inner_gap_mask = np.logical_and(self.R < self.setup['R_p'], self.R > (self.setup['R_p'] - furthest_gap_distance))
		outer_gap_mask = np.logical_and(self.R > self.setup['R_p'], self.R < (self.setup['R_p'] + furthest_gap_distance))
		r_id_inner_gap = np.argmin(self.dsigmadt_2D[inner_gap_mask])
		r_id_outer_gap = np.argmin(self.dsigmadt_2D[outer_gap_mask])

		gap_inner_loc = self.R[inner_gap_mask][r_id_inner_gap]
		gap_outer_loc = self.R[outer_gap_mask][r_id_outer_gap]
		
		gap_inner_depth = self.dsigmadt_2D[inner_gap_mask][r_id_inner_gap]
		gap_outer_depth = self.dsigmadt_2D[outer_gap_mask][r_id_outer_gap]

		self.setup['2D_potential_type'] = self.potential_2D.__class__.__name__
		self.setup['potential_type'] = self.potential.__class__.__name__

		self.summary =  {
			'setup': self.setup,
			'gap_timescale_2D': gap_timescale,

			'total_torque_2D':  self.T_2D[-1],
			'inner_torque_2D':  np.abs(np.trapz(self.dTdR_2D[self.R < self.setup['R_p']], self.R[self.R < self.setup['R_p']])),
			'outer_torque_2D':  np.abs(np.trapz(self.dTdR_2D[self.R > self.setup['R_p']], self.R[self.R > self.setup['R_p']])),

			# Calculated from f_{dep}
			'gap_inner_loc_2D': gap_inner_loc,
			'gap_outer_loc_2D': gap_outer_loc,
			'gap_spacing_2D': gap_outer_loc - gap_inner_loc,
			'gap_inner_timescale_2D': gap_inner_depth,
			'gap_outer_timescale_2D': gap_outer_depth,
		}
		
		self.summary['osl_torque_2D'] = (self.summary['inner_torque_2D'] + self.summary['outer_torque_2D'])/2

		if self.dimension == 3:
			self.summary['total_torque'] = self.T[-1]
			self.summary['inner_torque'] =   np.abs(np.trapz(self.dTdR[self.R < self.setup['R_p']], self.R[self.R < self.setup['R_p']]))
			self.summary['outer_torque'] = np.abs(np.trapz(self.dTdR[self.R > self.setup['R_p']], self.R[self.R > self.setup['R_p']]))
	
			self.summary['osl_torque'] = (self.summary['inner_torque'] + self.summary['outer_torque'])/2

			# Gap properties

			gap_timescale = 1/self.dsigmadt[np.argmin( np.abs(self.R - self.setup['R_p']) )]
			r_inner = self.R[np.logical_and(self.R < self.setup['R_p'], self.R > (self.setup['R_p'] - furthest_gap_distance))]
			r_outer = self.R[np.logical_and(self.R > self.setup['R_p'], self.R < (self.setup['R_p'] + furthest_gap_distance))]
			
			inner_gap_mask = np.logical_and(self.R < self.setup['R_p'], self.R > (self.setup['R_p'] - furthest_gap_distance))
			outer_gap_mask = np.logical_and(self.R > self.setup['R_p'], self.R < (self.setup['R_p'] + furthest_gap_distance))
			r_id_inner_gap = np.argmin(self.dsigmadt[inner_gap_mask])
			r_id_outer_gap = np.argmin(self.dsigmadt[outer_gap_mask])

			gap_inner_loc = self.R[inner_gap_mask][r_id_inner_gap]
			gap_outer_loc = self.R[outer_gap_mask][r_id_outer_gap]
			
			gap_inner_depth = self.dsigmadt[inner_gap_mask][r_id_inner_gap]
			gap_outer_depth = self.dsigmadt[outer_gap_mask][r_id_outer_gap]

			self.summary['gap_inner_loc'] =  gap_inner_loc
			self.summary['gap_outer_loc'] =  gap_outer_loc
			self.summary['gap_spacing']   =  gap_outer_loc - gap_inner_loc
			self.summary['gap_timescale'] = gap_timescale
			self.summary['gap_inner_timescale'] = 1/gap_inner_depth
			self.summary['gap_outer_timescale'] = 1/gap_outer_depth

		return self.summary


	def calculate_vortensity(self):
		logger.warning('Currently only computing 2D vortensity equivalent')
		return (np.gradient( self.v_phi_2D * self.R[None, :], self.R, axis=1) - np.gradient(self.v_R_2D, self.phi, axis=0))/self.R[None, :]/self.surface_density

		if self.dimension == 2:
			return (np.gradient( self.v_phi_2D * self.R[None, :], self.R, axis=1) - np.gradient(self.v_R_2D, self.phi, axis=0))/self.R[None, :]/self.surface_density
		else:
			if self.intergration_method == 'spherical':
				# Allow the small angle approximation for this calculation
				return (np.gradient( self.v_phi_2D * self.R[None, :], self.R, axis=1) - np.gradient(self.v_R_2D, self.phi, axis=0))/self.R[None, :]/self.surface_density
			raise NotImplementedError('No 3D vortensity calculation yet implemented')


	# Z-Intergral Functions
	def intergrate_in_z(self, integrand):
		if self.intergration_method == 'spherical':
			return self.integrate_along_spherical_column(integrand)
		return self.intergrate_along_actual_column(integrand)

	def integrate_along_spherical_column(self, integrand):
		# Use the small angle approximation dz \approx r d \theta
		# to perform an approximate integral
		if self.z is None:
			# This is done with the assmuption that theta is monotonically increasing
			self.z = -1 * self.R[:, None] * np.cos(self.theta)

		output = np.zeros((self.R.size, self.phi.size))
		for phi_i in range(self.phi.shape[0]):
			output[:, phi_i] = np.trapezoid(
				integrand[:, phi_i, :], self.z, axis=1)
		return output

	def intergrate_along_actual_column(self, integrand):
		# NOTE: THIS METHOD IS VERY SLOW! 
		# I DON'T RECCOMEND ACTUALLY USING IT WITH THE EXCEPTION OF TESTS TO ENSURE 
		# THAT THE SMALL ANGLE APPROXIMATION IS VALID ENOUGH FOR YOUR APPLICATION

		if self.z is None:
			self.z = -1 * self.R[:, None] * np.cos(self.theta[None, :])

		if self.R_cylindrical is None:
			self.R_cylindrical = self.R[:, None] * np.sin(self.theta[None, :])
		
		R = self.R_cylindrical
		output = np.zeros((self.R.size, self.phi.size))

		for phi_i in range(self.phi.shape[0]):
			# construct the interpolator in spherical coordinates		
			interp	= RegularGridInterpolator((self.R, self.theta), 
				integrand[:, phi_i, :], fill_value=0, bounds_error=False)
			
			# Now for each possible R we need to calculate this seperately
			# this is a very slow approach but I also have a very slow brain
			for r_i in range(self.R.shape[0]):
				r_current = self.R[r_i]
				z_interior = np.linspace(np.min(self.z[r_i, :]), np.max(self.z[r_i, :]), 
					self.theta.shape[0])
				
				r_sph_current = np.sqrt(r_current**2 + z_interior**2)
				
				theta_reset = np.acos(z_interior/r_sph_current)
				a = interp(np.swapaxes(np.vstack([r_sph_current, theta_reset]), 0, 1))
				output[r_i, phi_i] = np.trapezoid(a, z_interior, axis=0)

		# Avoid numerical errors
		output[-1, :] = output[-2, :]
		return output

	# Perform all z-integrals
	def reduce_to_2d(self):
		if self.dimension == 2:
			logger.info('Already 2D')
			return

		self.surface_density = self.intergrate_in_z(self.density)
		self.v_R_2D   = self.intergrate_in_z(self.v_R   * self.density)/self.surface_density
		self.v_phi_2D = self.intergrate_in_z(self.v_phi * self.density)/self.surface_density

	# Data saving boilerplate
	def save_all(self, filename=None):
		if filename is None:
			filename = f'{self.output_folder}all_orbit_{self.orbit_id}.p'

		keys = ['setup',
				'R',
				'time', 
				'dsigmadt_2D',
				'surface_density_1D',
				'v_R_1D',
				'v_phi_1D',
				'dTdR_2D',
				'T_2D' ,
				'F_wave_2D', 
				'dF_wavedR_2D', 
				'F_dep_2D', 
				'dF_depdR_2D',
				'surface_density',
				'v_R_2D',
				'v_phi_2D'
				'summary']

		# add in the 2D modes of torque calculations
		if self.dimension == 3:
			keys = keys + ['dsigmadt', 'dTdR', 'F_wave',
						   'dF_wavedR', 'F_dep', 'dF_depdR']


		self.save_subset(keys, filename)

	def save_2d(self, filename=None):
		if filename is None:
			filename = f'{self.output_folder}2D_orbit_{self.orbit_id}.p'

		keys = ['setup',
				'R',
				'phi',
				'time', 
				'setup', 
				'surface_density',
				'v_R_2D',
				'v_phi_2D']

		if self.include_vortensity:
			keys = keys + ['vortensity']

		self.save_subset(keys, filename)


	def save_1d(self, filename=None):
		if filename is None:
			filename = f'{self.output_folder}1D_orbit_{self.orbit_id}.p'

		keys = ['setup',
				'R',
				'time', 
				'dsigmadt_2D',
				'surface_density_1D',
				'v_R_1D',
				'v_phi_1D',
				'dTdR_2D',
				'T_2D' ,
				'F_wave_2D', 
				'dF_wavedR_2D', 
				'F_dep_2D', 
				'dF_depdR_2D',
				'summary']

		# add in the 2D modes of torque calculations
		if self.dimension == 3:
			keys = keys + ['dsigmadt', 'dTdR', 'F_wave',
						   'dF_wavedR', 'F_dep', 'dF_depdR']

		self.save_subset(keys, filename)

	def save_subset(self, keys, filename):
		""" Save an aribtrary subset of processed data to a filename"""
		output_object = {}
		for key in keys:
			output_object[key] = getattr(self, key)

		file = open(filename, 'wb')
		pickle.dump(output_object, file)
		file.close()



class ProcessedData(Simulation):
	""" Load pre-processed data in but allow for all of the same outputs as simulation """
	possible_keys = [
		'setup',
		'R',
		'phi',
		'time', 
		'dsigmadt_2D',
		'surface_density_1D',
		'v_R_1D',
		'v_phi_1D',
		'dTdR', 
		'F_wave', 
		'dF_wavedR', 
		'F_dep', 
		'dF_depdR',
		'dTdR_2D', 'F_wave_2D', 'dF_wavedR_2D', 'F_dep_2D', 'dF_depdR_2D',
		'surface_density',
		'v_R_2D',
		'v_phi_2D',
	]


	def __init__(self, filename, *args, **kwargs):		
		file = open(filename, 'rb')
		data = pickle.load(file)
		file.close()

		for key, value in data.items():
			# Sanity check to not override a function
			if key in self.possible_keys:
				setattr(self, key, value)


class ProcessedDataMissing(Exception):
	""" Exception for when processed data does not yet exist """
	pass