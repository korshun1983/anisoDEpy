# File 2: prepare_phys_prop_fluid_sp_safe.py
def prepare_phys_prop_fluid_sp_safe(comp_struct, ii_l):
    """
    Prepare physical properties for ideal fluid layer

    Inputs:
        comp_struct : dict
            Dictionary containing model parameters
        ii_l : int
            Layer number (0-indexed)

    Outputs:
        phys_prop : dict
            Dictionary containing fluid properties
    """
    phys_prop = {}

    # Extract fluid parameters (rho, lambda)
    # DomainParam is a list where each element is [rho, lambda]
    phys_prop['rho'] = comp_struct['Model']['DomainParam'][ii_l][0]
    phys_prop['lambda'] = comp_struct['Model']['DomainParam'][ii_l][1]

    return phys_prop