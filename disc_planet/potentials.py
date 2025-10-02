""" 
potentials.py

Defines classes of various planetary potentials

All classes should inherit from PlanetaryPotential and 
implement functions __init__ and calculate_dPhidphi_3D(self, R, phi, theta) and/or
calculate_dPhidphi_2D(self, R, phi). Information on e.g. planetary location etc 
should be passed during the __init__ function.
"""

import numpy as np
from scipy.special import k0e, k1e

class PlanetaryPotential(object):
    """ Abstract base class to represent planetary potentials in a disc """

    def __init__(self):
        pass

    def calculate_dPhidphi_3D(self, R, phi, theta):
        raise NotImplementedError

    def calculate_dPhidphi_2D(self, R, phi):
        raise NotImplementedError


# Potentials that work both in 2 and 3 dimensions
class SecondOrderSmoothedPotential(PlanetaryPotential):
    def __init__(self, m_p, R_p, phi_p, theta_p, rsm):
        self.rsm        = rsm
        self.m_p        = m_p
        self.R_p        = R_p
        self.phi_p      = phi_p
        self.theta_p    = theta_p

    def calculate_dPhidphi_3D(self, R, phi, theta):
        R = R[:, None, None]
        theta = theta[None, None, :]
        phi = phi[None, :, None]

        denom = (R * R + self.R_p**2 
                 - 2 * R *self.R_p * np.sin(theta - (self.theta_p - np.pi/2)) * np.cos(phi - self.phi_p) 
                 + self.rsm**2)**(3/2)

        return self.m_p/denom * (R * self.R_p * np.sin(theta - (self.theta_p - np.pi/2)) 
                                 * np.sin(phi - self.phi_p))

    def calculate_dPhidphi_2D(self, R, phi):
        theta = 0
        R = R[:, None]
        phi = phi[None, :]

        dr2 = R * R - 2. * R * self.R_p * np.cos(phi - self.phi_p) + self.R_p**2
        dr2prsm2 = dr2+ (self.rsm * self.rsm)
        i32 = dr2prsm2**(-3/2)

        return self.m_p * i32 * R * self.R_p * np.sin(phi - self.phi_p)

class KlarTypePotential(PlanetaryPotential):
    # see: https://ui.adsabs.harvard.edu/abs/2006A%26A...445..747K/abstract

    def __init__(self, m_p, R_p, phi_p, theta_p, rsm):
        self.rsm        = rsm
        self.m_p        = m_p
        self.R_p        = R_p
        self.phi_p        = phi_p
        self.theta_p    = theta_p


    def calculate_dPhidphi_3D(self, R, phi, theta):
        R = R[:, None, None]
        theta = theta[None, None, :]
        phi = phi[None, :, None]

        dr2 = (R * R + self.R_p**2 - 2 * R *self.R_p * np.sin(theta) * np.cos(phi - self.phi_p))
        d = dr2**(1/2)

        dPhidPhi = self.m_p * R * self.R_p * np.sin(theta) * np.sin(phi - self.phi_p)  * dr2**(-3/2)

        inner_dPhidPhi = self.m_p/(2 * d**5) * (R * self.R_p * np.sin(theta) * np.sin(phi - self.phi_p)  
            * (d**2 - 9 * self.rsm**2 ) )    

        dPhidPhi[np.abs(d) < self.rsm] = inner_dPhidPhi[np.abs(d) < self.rsm]

        return dPhidPhi



## Various 2D potentials
class FourthOrderSmoothedPotential(PlanetaryPotential):
    def __init__(self, m_p, R_p, phi_p, rsm):
        self.rsm        = rsm
        self.m_p        = m_p
        self.R_p        = R_p
        self.phi_p      = phi_p

    def calculate_dPhidphi_2D(self, R, phi):
        R = R[:, None]
        phi = phi[None, :]

        dr2 = R*R - 2. * R * self.R_p * np.cos(phi - self.phi_p) + self.R_p**2
        dr2prsm2 = dr2 + self.rsm**2
        i32 = (dr2 + self.rsm**2)**(-3/2)
        smoothing = i32 * (1.5 * (dr2 + 1.5 * self.rsm**2) / dr2prsm2 - 1.0) #  Smoothing function
        return self.m_p*2.* R * self.R_p * np.sin(phi - self.phi_p) * smoothing


class BesselTypePotentialConstH(PlanetaryPotential):
    def __init__(self, m_p, R_p, phi_p, H_p, smoothingB=0.005):
        self.m_p   = m_p
        self.gmp_root_2_pi = m_p * (2 * np.pi)**(-1/2)
        self.R_p   = R_p
        self.phi_p = phi_p
        self.H_p = H_p
        self.smoothingB = smoothingB

    def calculate_dPhidphi_2D(self, R, phi):
        R = R[:, None]
        phi = phi[None, :]

        dr2 = R**2 - 2. * R * self.R_p * np.cos(phi - self.phi_p) + self.R_p**2 + self.smoothingB*self.smoothingB
        s2 = dr2/(4 * self.H_p * self.H_p)

        dPhids2 = - self.gmp_root_2_pi/self.H_p * (
            k0e(s2) - k1e(s2))
 
        return dPhids2 * R * self.R_p  * np.sin(phi - self.phi_p)/(2 * self.H_p * self.H_p)


class BesselTypePotential(PlanetaryPotential):
    def __init__(self, m_p, R_p, phi_p, H_p, temperature_slope, smoothingB=0.005):
        self.m_p = m_p
        self.gmp_root_2_pi = m_p * (2 * np.pi)**(-1/2)
        self.R_p = R_p
        self.phi_p = phi_p
        self.H_p = H_p
        self.height_slope = (1.5 + temperature_slope/2)
        self.smoothingB = smoothingB

    def calculate_dPhidphi_2D(self, R, phi):
        R = R[:, None]
        phi = phi[None, :]

        radpl2 = self.R_p * self.R_p
        local_scale_height = self.H_p * (R/self.R_p)**(self.height_slope)
        dr2 = R*R - 2. * R * self.R_p * np.cos(phi - self.phi_p) + self.R_p * self.R_p + self.smoothingB * self.smoothingB
        s2 = dr2/(4 * local_scale_height * local_scale_height)

        k_0 = k0e(s2)
        k_1 = k1e(s2)

        scale = - self.m_p/( local_scale_height * np.sqrt(2 * np.pi))

        ds2dr = (
            (1 - self.height_slope) * R - self.height_slope * self.R_p * self.R_p / R 
            - (1 - 2 * self.height_slope) * np.cos(phi - self.phi_p))/(2 * local_scale_height * local_scale_height)

        return scale * (k_0 - k_1) * R * self.R_p  * np.sin(phi - self.phi_p)/(2 * local_scale_height * local_scale_height)
        

class BesselLinModePotential(BesselTypePotential):
    pass

class BesselTypeForcing(PlanetaryPotential):
    def __init__(self, m_p, R_p, phi_p, H_p, temperature_slope, smoothingB=0.005):
        self.m_p = m_p
        self.gmp_root_2_pi = m_p * (2 * np.pi)**(-1/2)
        self.R_p = R_p
        self.phi_p = phi_p
        self.H_p = H_p
        self.height_slope = 1.5 + temperature_slope/2
        self.smoothingB = smoothingB


    def calculate_dPhidphi_2D(self, R, phi):
        R = R[:, None]
        phi = phi[None, :]

        local_scale_height = self.H_p * (R/self.R_p)**(self.height_slope)
        dr2 = R**2 - 2. * R * self.R_p * np.cos(phi - self.phi_p) + self.R_p**2 + self.smoothingB*self.smoothingB
        s2 = dr2/(4 * local_scale_height * local_scale_height)

        dPhids2 = - self.gmp_root_2_pi/local_scale_height * (k0e(s2) - k1e(s2));

        return dPhids2 * R * self.R_p * np.sin(phi - self.phi_p)/(2 * local_scale_height * local_scale_height)
