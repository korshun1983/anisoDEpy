from structures import BigCompStruct
import math

class Asympt:
    pass

def ComputeAsymptotesSAFE(CompStruct: BigCompStruct):
    Asymptotes = Asympt()
    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    # ComputeAsymptotes M-file
    #      ComputeAsymptotes, by itself, computes various
    # asymptotes of dispersion curves, like V_mud, Stoneley,
    # anisotropic Stoneley (provided F.Karpfinger and R.Prioul permission),
    # low-frequency asymptotes of dipole normal modes, etc.
    #
    #   [T.Zharnikov, D.Syresin, SMR v0.12_12.2012]
    #
    # function [Asymptotes] =  ComputeAsymptotes(CompStruct)
    #
    #  Inputs -
    #       CompStruct - structure containing the information about model;
    #
    #  Outputs -
    #       Asymptotes - structure containing the asymptotes of dispersion
    #       curves
    #
    #  M-files required-
    #
    # Last Modified by Timur Zharnikov SMR v0.1_01.2012

    ################################################################################
    #
    #   Code for ComputeAsymptotes
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # Compute asymptotes, if necessary
    # Compute asymptote for mud layer, if it is present in the model
    match CompStruct.Model.mud_layer:
        case 0:
            foo = 1
        case _:
            # extract mud properties
            #        Mud_properties = cell2mat(CompStruct.Model.LayerParam(CompStruct.Model.mud_layer));
            Mud_properties = CompStruct.Model.LayerParam{CompStruct.Model.mud_layer}
            Rho_mud = Mud_properties(1)
            Lambda_mud = Mud_properties(2)
            # compute V_mud
            Asymptotes.V_mud = math.sqrt(Lambda_mud / Rho_mud)




    # Compute asymptotes for the outer formation, if it is TTI
    match CompStruct.Model.LayerType{CompStruct.Data.N_layers}:
        case 'HTTI':
            # extract outer formation parameters
            Formation_properties = CompStruct.Model.LayerParam{CompStruct.Data.N_layers}
            Rho = 1e+3 * Formation_properties(1)
            C_main = Formation_properties(2:6)
            Theta = Formation_properties(7)
            # compute Christoffel equation solution according to
            # the Rock Physics Handbook
            [V_qP, V_qSV, V_SH] = CompStruct.Methods.V_phase_VTI_exact_RPH(Rho, C_main, Theta)
            Asymptotes.V_qP = 1e-3 * V_qP
            Asymptotes.V_qSV = 1e-3 * V_qSV
            Asymptotes.V_SH = 1e-3 * V_SH
            # compute the Stoneley wave speed (low-frequency asymptote)
            # for VTI homogeneous formation
            if CompStruct.Model.mud_layer != 0:
                Asymptotes.V_St = 1 / math.sqrt(Rho_mud * (1 / Lambda_mud + 1 / C_main(5)))

        case _:

return Asymptotes