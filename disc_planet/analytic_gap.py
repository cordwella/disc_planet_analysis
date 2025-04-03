import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import iv, ive, kv, kve

# Implements the analytic solutions for the initial evolution of a gap in a 
# protoplanetary disc as in Cordwell & Rafikov 2024 

def non_local_change_in_density(R, p, q, h, f_dep):
    """
    Global solution for the initial evolution of a gap in a 
    protoplanetary disc.
    
    See equation C4 in Cordwell & Rafikov (2024)
    
    Inputs:
    - R, input array of locations to evaluate
    - p, background gradient in surface density (Sigma_0 = R^(-p))
    - q, background gradient in disc temperature (c_s = c_{s, p} R^(-q) )
    - h, scale height of the disc at $R = 1$
    - f_dep, pre-evaluated angular momentum deposition function. Defined as $\partial \F_{dep}/\partial R / \Sigma$
    
    Returns:
    - d\sigma / dt, array with the same shape as R
    """
    
    if q == 0.5:
        print('Approximating q == 0.5 as q = 0.55')
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
