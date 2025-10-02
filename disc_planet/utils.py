import numpy as np
import logging

from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.special import iv, ive, kv, kve
from scipy.interpolate import RegularGridInterpolator


logger = logging.getLogger(__name__)

def non_local_change_in_density(R, p, q, h, f_dep):
    """
    Global solution for the initial evolution of a gap in a
    protoplanetary disc.

    See equation C4 in Cordwell & Rafikov (2024)

    Parameters
    ------------------------------
        R: 1D numpy array of floats
            input array of locations to evaluate
        p: float
            background gradient in surface density (Sigma_0 = R^(-p))
        q: float
            background gradient in disc temperature (c_s = c_{s, p} R^(-q) )
        h: float
            scale height of the disc at $R = 1$
        f_dep: 1D numpy array of floats
            Angular momentum deposition function. Defined as $\partial \F_{dep}/\partial R / \Sigma$

    Returns
    ------------------------------
        d\sigma / dt: 1D numpy array with the same shape as R
    """

    if q == 0.5:
        logger.warning('Approximating q == 0.5 as q = 0.55')
        q = 0.55
        # raise Exception("Solution not provided for q = 1/2")

    A = h

    S = 1/np.pi * np.gradient(R**(1/2 - p) * f_dep)/np.gradient(R)

    x = R**(q - 1/2)/( (q - 1/2) * A)
    x = x.astype(complex)

    beta = np.abs((p + 2 * q - 3)/(1 - 2 * q))

    I_3 = iv(beta, np.abs(x))
    K_3 = kv(beta, np.abs(x))

    if q < 1/2:
        i_int = cumulative_trapezoid((I_3 * S * R**(p/2 + q - 3/2))[::-1], R[::-1], initial=0)[::-1]
        k_int = cumulative_trapezoid((K_3 * S * R**(p/2 + q - 3/2)), R, initial=0)
    else:
        i_int = cumulative_trapezoid((I_3 * S * R**(p/2 + q - 3/2)), R, initial=0)
        k_int = cumulative_trapezoid((K_3 * S * R**(p/2 + q - 3/2))[::-1], R[::-1], initial=0)[::-1]

    return A **(-2) * R**(p/2 + q - 3/2) * (
        I_3 * k_int - K_3 * i_int)/(q - 1/2)


def get_horseshoe_width(R, phi, vr, vphi,
                        R_start = 0.99,
                        delR = 0.0005,
                        rmax = 1.1,
                        rmin = 0.9,
                        phi_min = 0.1,
                        phi_start = 0.2,
                        tmax = 1000,
                        iterations = 100,
                        max_step = 5,
                        background_vphi = None):
    """
    Algorithm to find the outer horseshoe width in 2D data. The azimuthal location of the 
    planet in the provided data must be np.pi

    Parameters
    ---------------
        R: 1D numpy array of floats
            Radial grid points
        phi: 1D numpy array of floats
            Azimuthal grid points
        vr: 2D numpy array of floats, shape must be (len(R), len(phi))
            Radial velocity
        vphi: 2D numpy array of floats, shape must be (len(R), len(phi))
            Azimuthal velocity

    Returns
    --------------
    horseshoe half width, inner horseshoe radial position, outer horseshoe radial position
    """

    if background_vphi is None:
        background_vphi = R[:, None]**(3/2)
    interp_vr    = RegularGridInterpolator((phi, R), np.swapaxes(vr, 0, 1), bounds_error=False, fill_value=0)
    interp_vphi  = RegularGridInterpolator((phi, R), np.swapaxes(vphi - background_vphi, 0 , 1), bounds_error=False, fill_value=0)


    def hit_bounds_phi(t, Y):
        # Check if it has hit the bounds of the domain in phi
        a = min(5 - Y[0], Y[0] - phi_min * 1.01)

        if a < 0.05:
            return 0
        return a


    def hit_bounds_r(t, Y):
        # Check if it has hit the bounds of the domain in R
        return min(rmax * 0.99 - Y[1], Y[1] - rmin * 1.01)

    # Exit at the edge of bounds
    hit_bounds_r.terminal = True
    hit_bounds_phi.terminal = True

    def velocity(t, y):
        return np.squeeze([interp_vphi(y), interp_vr(y)])

    r            = R_start
    prev_r_final = r
    prev_r       = r

    for n in range(iterations):
        sol = solve_ivp(velocity, [0, tmax], [phi_start, r], events=(hit_bounds_r, hit_bounds_phi), max_step=max_step)

        if sol.status == 0:
            logger.warning('ERROR: tmax not long enough')
            return 0, 0, 0

        # find out where we have ended up by looking at the final (terminal) point
        t_events = sol.t_events
        y_events = sol.y_events

        if t_events[0].size:
            logger.warning('ERROR: hit R bounds')
            return 0, 0, 0

        # Have we escaped the horseshoe region?
        final_position_r = sol.y[1][-1]
        final_position_phi = sol.y[0][-1]

        if final_position_phi > np.pi * 1.1:
            # We have escaped the horseshoe region
            logger.info('Horseshoe width found')
            return [(prev_r_final - prev_r)/2, prev_r_final, prev_r]

        prev_r = r
        prev_r_final = final_position_r
        r = r - delR

    if final_position_phi < np.pi:
        logger.warning('ERROR: Horseshoe width not found in max iterations')
        return 0, 0, 0
