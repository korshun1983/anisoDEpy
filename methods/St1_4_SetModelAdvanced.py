from structures import BigCompStruct

def St1_4_SetModelAdvanced(InputParam: BigCompStruct):
    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   St1_4_SetModelAdvanced_sp_SAFE.m M-file
    #      St1_4_SetModelAdvanced_sp_SAFE.m sets the advanced parameters,
    # which are not supposed to be available to the regular user.
    # These parameters are for the development and for advanced users.
    # They are typically not obvious, less understood,
    # and generally require more careful handling, such as
    # the number of approximation points, outer radius of the model, etc.
    #
    # NB! Typically this script will not be modified by the user.
    #
    # NB! This implementation is specific to spectrum calculation
    # by the spectral method
    #
    #   [T.Zharnikov, D.Syresin, SMR v0.3_08.2014]
    #
    # function [InputParam] = St1_4_SetModelAdvanced_sp_SAFE(InputParam)
    #
    #
    #  Inputs -
    #
    #       InputParam - structure containing the input parameters:
    #
    #  Outputs -
    #
    #       InputParam - structure containing the input parameters,
    #               which is updated with Advanced structure
    #               with some advanced parameters of the model
    #

    #  M-files required-
    #
    # Last Modified by Timur Zharnikov SMR v0.3_08.2014

    ################################################################################
    #
    #   Code for St1_4_SetModelAdvanced_sp_SAFE
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # ===============================================================================
    # Setting the model
    # ===============================================================================

    # # Set the range of layers to take into account
    # #   when classifying the spectrum (modes)
    # InputParam.Advanced.LayerClassificationStart = 1
    # InputParam.Advanced.LayerClassificationStop = 1
    #
    # # Set the number of first eigenvalues
    # #   to consder for dispersion curves construction
    # InputParam.Advanced.m_output_max = 10
    #
    # # Set the default range of phase speeds to consider (km/s)
    # InputParam.Advanced.V_min = 0.5
    # InputParam.Advanced.V_max = 5.0
    #
    # # Set the default range of minimum and maximum speeds for spectrum analysis (km/s)
    # InputParam.Advanced.V_min_threshold = 0.2
    # InputParam.Advanced.V_max_threshold = 8.0
    #
    # # Set the default range of minimum and maximum speeds in case of using speed up options (km/s)
    # InputParam.Advanced.V_min_spup = 0.2
    # InputParam.Advanced.V_max_spup = 8.0
    #
    # # Set the default number of eigenvectors to estimate in case of using speed up options
    # InputParam.Advanced.eig_vec_num_spup = 5
    #
    # # Set reference frequency for adjustable outer radius of the model (approximate)
    # InputParam.Advanced.f_ref = ( InputParam.Model.f_min + InputParam.Model.f_max )/2 #Hz
    #
    # # Set the radial extension factor for adjustable radius of the outer layer
    # InputParam.Advanced.RadialExtensionFactor = 10
    #
    # # Set the PML parameters
    # InputParam.Advanced.PML_factor = 1.0
    # InputParam.Advanced.PML_exponent = 2.0
    #
    # # Set the ABC parameters
    # InputParam.Advanced.ABC_factor = 4.0
    # InputParam.Advanced.ABC_exponent = 3.0
    # InputParam.Advanced.ABC_adjust_factor = 10
    #
    # # Set frequency display limits
    # InputParam.Advanced.display_f_lo = 0 #kHz
    # InputParam.Advanced.display_f_up = 20 #kHz

    # Set the switch whether to visualize the mesh
    # visualization of the mesh - should be logical variable
    # (0 -no visualization, 1 - visualization)
    InputParam.Advanced.VisualizeMesh = True
    # Set the mesh options
    # InputParam.Advanced.MeshOptions.dhmax=0.25
    # InputParam.Advanced.MeshOptions.dhmax=0.0025
    # visualization of the mesh during its creation - should be logical
    # (0 -no visualization, 1 - visualization)
    # InputParam.Advanced.MeshOptions.output = (false)
    # OPTIONS is a structure array that allows some of the "tuning" parameters
    # used in the solver to be modified:
    #
    #   options.mlim   : The convergence tolerance. The maximum percentage
    #                    change in edge length per iteration must be less than
    #                    MLIM { 0.02, 2.0# }.
    #   options.maxit  : The maximum allowable number of iterations { 20 }.
    #   options.dhmax  : The maximum allowable (relative) gradient in the size
    #                    function { 0.3, 30.0# }.
    #   options.output : Displays the mesh and the mesh statistics upon
    #                    completion { TRUE }.

    # Set the number of nodes per element
    # 3 - the first order 6 - the second order 10 - the third order (cubic element)
    InputParam.Advanced.N_nodes = 10
    InputParam.Advanced.NEdge_nodes = 4

    # Moved to main input file
    # Set the number of eigenvalues to compute for each frequency
    # Input.num_eig=40# number of eigenvalues to be found for each k.
    # InputParam.Advanced.num_eig_max = 100

    # Set the accuracy of integration (number of integration points) NOT used anymore
    # Input.integration_acc=20 # accuracy for integation (number of integration points) NOT used anymore
    # InputParam.Advanced.IntAcc = 20

    # Moved to main input file
    # Set the starting velocity, in the vicinity of which to look for the
    # eigenvalues
    # Input.Starting_Velocity=1.5 # velocity of the first eigenvalue for eigs
    # InputParam.Advanced.EigSearchStart = 0.5 # starting velocity to search for the first eigenvalue for eigs

    # Set the eigs options
    InputParam.Advanced.EigsOptions.disp = 0
    InputParam.Advanced.EigsOptions.tol = 1e-8
    # InputParam.Advanced.EigsOptions.isreal = 1

    # Set the parameters of the source (Gaussian ring)
    # This set of parameters is necessary for DS way to classify the spectrum
    # (maximum of the excitation function)
    InputParam.Advanced.Source.xc = 0  # eccentricity of the source in x axis
    InputParam.Advanced.Source.yc = 0  # eccentricity of the source in y axis
    InputParam.Advanced.Source.r0x = 0.06  # x radius of the source
    InputParam.Advanced.Source.r0y = 0.06  # y radius of the source
    InputParam.Advanced.Source.theta_r = 0  # rotation of the source (geometrical)
    InputParam.Advanced.Source.theta0 = 0  # pi/4 # rotation of the source direction
    InputParam.Advanced.Source.sigma = 0.02  # width of the source
    InputParam.Advanced.Source.Plim = 5e-4  # constrain amplitude to consider oscillation in node (maximum amplitude is 1)
    InputParam.Advanced.Source.symmetry = 0  # Symmetry of the source (0 is monopole, 1 sym dipole, 2 asym dipole).Will be changed automatically.

    return InputParam