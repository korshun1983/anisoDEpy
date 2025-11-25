import numpy as np

def classify_by_energy(TE_rad, r, r_max, r_pml_start):
    idx_pml = np.searchsorted(r, r_pml_start)
    TE_tot = np.trapz(r*TE_rad, r)
    TE_pml = np.trapz(r[idx_pml:]*TE_rad[idx_pml:], r[idx_pml:])
    ratio = TE_pml/TE_tot if TE_tot else 1.0
    return ratio < 0.05