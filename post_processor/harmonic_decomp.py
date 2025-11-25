import numpy as np

def fourier_azimuth(v, mesh, model, nh=4):
    """
    Return coeffs c_m for m=0…nh-1 and m=-1…-(nh-1) for ur, uf, uz
    on the polar grid.
    """
    # placeholder – FFT along theta after interpolation
    return np.zeros(nh), np.zeros(nh), np.zeros(nh)