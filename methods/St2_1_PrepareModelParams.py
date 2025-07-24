import bigCompStruct from structures

def St2_1_PrepareModelParams(CompStruct: bigCompStruct):
    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   St2_1_PrepareModelParams_sp_SAFE.m M-file
    #      St2_1_PrepareModelParams_sp_SAFE.m prepares the specific set
    # of the parameters, which will be used for the computations
    # of the waveguide spectrum using spectral method.
    # It organizes input data in the form,
    # which is convenient for this problem.
    #
    # NB!!! This implementation is specific for spectrum computation
    # with the spectral method.
    #
    #   [T.Zharnikov, D.Syresin, SMR v0.3_08.2014]
    #
    # function [CompStruct] = St2_1_PrepareModelParams_sp_SAFE(CompStruct)
    #
    #  Inputs -
    #
    #       CompStuct - structure containing the parameters of the model
    #
    #  Outputs -
    #
    #       CompStuct - structure containing the parameters of the model after
    #                   update with some advanced parameters like unit
    #                   computation of positions of harmonics, the unit conversion, etc.
    #
    #  M-files required-
    #
    # Last Modified by Timur Zharnikov SMR v0.3_08.2014

    ################################################################################
    #
    #   Code for St2_1_PrepareModelParams_sp_SAFE
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # Retaining all necessary information
    # CompStruct = InputParam

    # ===============================================================================
    # Modification and preparation of input information
    # (especially model data) for further steps of computation.
    # Calculation of the additional parameters,
    # which will be necessary for the computations.
    # ===============================================================================

    # Define unit conversion factors
    # for the frequency
    match CompStruct.Config.FreqUnits:
        case 'Hz':
            CompStruct.Misc.F_conv = 1
        case 'kHz':
            CompStruct.Misc.F_conv = 1e+3

    # for the slowness
    match CompStruct.Config.SloUnits:
        case 'us/m':
            CompStruct.Misc.S_conv = 1e+3
        case 'us/ft':
            CompStruct.Misc.S_conv = 0.3048 * 1e+3

    # # Define the totatl number of azimuthal harmonics to compute simultaneously -
    # # necessary for anisotropy
    # CompStruct.Data.N_harmonics = max(size(CompStruct.Model.harmonics))
    # CompStruct.Data.Pl_modes_index = (CompStruct.Model.harmonics >= 0)
    # CompStruct.Data.Pl_modes_pos = find(CompStruct.Model.harmonics >= 0)
    # CompStruct.Data.Num_pl_modes = sum(CompStruct.Data.Pl_modes_index)

    # Compute number of computational domains
    CompStruct.Data.N_domain = max(size(CompStruct.Model.DomainType))

    # Compute positions of subdomains and boundaries
    # Domains
    # for ii_l = 1:CompStruct.Data.N_domains
    # end

    # Boundaries

    # Compute the array indicating the number of variables,
    # which are required for each layer
    for ii_d in range(1,1,CompStruct.Data.N_domain):
        match   CompStruct.Model.DomainType[ii_d]:
            case 'fluid':
                CompStruct.Data.DVarNum[ii_d] = 1
            case 'HTTI':
                CompStruct.Data.DVarNum[ii_d] = 3

    # Compute asymptotes, if necessary
    match CompStruct.Config.CheckAsymptote:
        case 'yes':
            CompStruct.Asymp = CompStruct.Methods.ComputeAsymptotes(CompStruct)
            # Adjust the range of phase speeds to study, if necessary
            if hasattr(CompStruct.Asymp, 'V_SH'):
                CompStruct.Advanced.V_min = CompStruct.Asymp.V_SH

            if hasattr(CompStruct.Asymp, 'V_qP'):
                CompStruct.Advanced.V_max = CompStruct.Asymp.V_qP

        case 'no':
            foo = 1
    # ===============================================================================
    # Assign types of layers (cell array)
    # ===============================================================================
    # CompStruct.Data.LayerType = CompStruct.Model.LayerType

    # ===============================================================================
    # Assign physical properties of the layers (cell array)
    # For the present time, take data from CompStruct.Model
    # ===============================================================================

    return CompStruct