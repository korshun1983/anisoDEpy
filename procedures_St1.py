from structures import InputParams
import math

def  St1_3_SetModelUser_sp_SAFE(modelParams: InputParams):

      #Initialization
      #Setting the model

      # Set the geometry of the model - positions of layer boundaries and interfaces, in meters

      modelParams.DomainRx = [0.1016, 2.0] # positions of layer boundaries along x axis (from inner to outer) OH
      modelParams.DomainRy = [0.1016, 2.0] # positions of layer boundaries along y axis (from inner to outer) OH
      modelParams.DomainTheta = [0*math.pi/180, 0*math.pi/180] #rotation of each layer with respect to x? axis, in radians
      modelParams.DomainEcc = [0, 0] # eccentricity of each layer with respect to the coordinate origin, in meters
      modelParams.DomainEccAngle = [0, 0] # azimuthal shift of eccentricity direction
      # 2.0 - recommended outer radius, possibly adjusted automatically depending
      #           on configuration

      #===============================================================================
      # Set the types of the layers (medium used)
      #===============================================================================

      modelParams.DomainType = {'fluid', 'HTTI'}
      #  types of layers:
      # 'fluid' - ideal fluid
      #'HTTI' - TTI, homogeneous in cartesian coordinates
      # 0 - fluid,
      # 1 - solid elastic aniso C_ij
      #   - solid elastic Vp,Vs,epsilon,gamma,delta
      #   - solid viscoelastic, Biot, Pride, etc.


      #===============================================================================
      # Set physical properties of the layers' media
      #===============================================================================

      #Elastic
      #      Input.rho=[1000, 1000, 2230]; # density for each layer [kg/m3]
      #      Input.lambdaf=[ 2.25e9 2.25e9 0]; # bulk modulus for fluid
      #      Input.Elastic_prop=[0 0 0 0 0 0;... #
      #                          0 0 0 0 0 0;...
      #                                  22e9 14e9 2.4e9 3.1e9 12e9  0]; % C11,C33, C44, C66, C13, TI axis Inclination angle [rad]
      #                                  40.9e9 26.9e9 10.5e9 15.3e9 8.5e9  pi/2]; %  C11, C33, C44, C66, C13, TI axis Inclination angle [rad]

      modelParams.DomainParam = { [1.0, 2.25],[ 2.23 , 40.9 , 8.5 , 26.9 , 10.5 , 15.3 , 0, 0]}
      #...%  [ 2.23 - 0.1*1i, 40.9 - 0.1*1i, 8.5 - 0.1*1i, 26.9 - 0.1*1i, 10.5 - 0.1*1i, 15.3 - 0.1*1i, 0, 0]};

      #    [ 2.2, 22 12 14 2.4 3.1, 0, 0],...
      #    [ 2.2, 22 12 14 2.4 3.1, 0, 0]};

      #
      #   parameters for the layers:
      # - fluid - [density, lambda]
      # - solidTTI - [density, c11 c13 c33 c44 c66, relative dip, azimuth ] azimuth is optional

      #   dimensions of parameters:
      # - density 1e-3*kg/m^3 (kg/cm^3)
      # - Cij - in GPa
      # - relative dip (VTI axis inclination to that of the waveguide) in radians

      #===============================================================================
      # Set the physical properties of the reference layer
      # It will be used to determine the far field asymptotic values
      #===============================================================================

      modelParams.RefDomainType = 'HTTI'
      modelParams.RefDomainParam = { [ 2.23, 40.9, 8.5, 26.9, 10.5, 15.3, 0, 0] }

      # ===============================================================================
      #  Set the types of boundary and interface conditions
      # ===============================================================================

      modelParams.BCType = {'FS', 'rigid'}
      # % InputParam.Model.BCType = {'free',...%'FS','FS',
      # %     'FS','SSstiff','rigid'}; %OH+Pipe
      # %   types of boundary and interface conditions (BCs and ICs):
      # % 0 - free  - free surface (fluid, solid)
      # % 1 - rigid - rigid surface (fluid, solid)
      # % 2 - FS - natural fluid-solid or fluid-fluid contact
      # % 3 - SSstiff - stiff solid-solid
      # % 4 - SSslip - slip solid-solid
      # % 5 - SSfric - slip with friction solid-solid
      # %   - etc.
      #
      # %===============================================================================
      # % Set the frequency range of interest, kHz
      # %===============================================================================
      #
      # % Minimum frequency
      modelParams.f_min = 3.0 #kHz

      # Maximum frequency
      modelParams.f_max = 7.0 #kHz

      # %===============================================================================
      # % Set model discretization parameters
      # %===============================================================================
      # % NB! These parameters are available to the user,
      # %   but should be treated with care
      # %===============================================================================
      #
      # % Set the number of azimuthal nodes  to use for each layer
      modelParams.DomainNth = [12, 12 ] # number of azimuthal nodes for every layer

      # % number of discretization points %OH+Pipe
      # %   Typical number is ...
      #
      # % Set the layer, which contains drilling mud
      # % Input.Mud_subdomain=2;% the number of the fluid subdomain (needs to be defined in geometry file)
      # InputParam.Model.mud_domain = 1;
      # % Typically mud layer is:
      # %   - 1 for open hole (OH)
      # %   - 2 for heavy fluid model (HFM)
      #
      # % Set the number of omega (angular frequency) points,
      # %   in which the spectrum will be calculated to construct dispersion curves
      # % Input.num_points=18;%31; % number of points to calculate
      # % Input.Logparam= ((Input.fend - Input.fstrt)/(Input.num_points - 1) + Input.fstrt)/Input.fstrt; % coefficient of geometric progression: f(n)=Logparam*f(n-1);
      modelParams.N_disp = 2
      # % Typically:
      # %   - 5 for fast quick look
      # %   - 10-15 for more or less good impression (fairly smooth curve); usual
      # %   - 20-30+ for detailed dispersion curve
      return modelParams

def St1_4_SetModelAdvanced_sp_SAFE(modelParams: InputParams):

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
      #===============================================================================
      # Initialization
      #===============================================================================

      #===============================================================================
      # Setting the model
      #===============================================================================

      # # Set the range of layers to take into account
      # #   when classifying the spectrum (modes)
      # InputParam.Advanced.LayerClassificationStart = 1;
      # InputParam.Advanced.LayerClassificationStop = 1;
      #
      # # Set the number of first eigenvalues
      # #   to consder for dispersion curves construction
      # InputParam.Advanced.m_output_max = 10;
      #
      # # Set the default range of phase speeds to consider (km/s)
      # InputParam.Advanced.V_min = 0.5;
      # InputParam.Advanced.V_max = 5.0;
      #
      # # Set the default range of minimum and maximum speeds for spectrum analysis (km/s)
      # InputParam.Advanced.V_min_threshold = 0.2;
      # InputParam.Advanced.V_max_threshold = 8.0;
      #
      # # Set the default range of minimum and maximum speeds in case of using speed up options (km/s)
      # InputParam.Advanced.V_min_spup = 0.2;
      # InputParam.Advanced.V_max_spup = 8.0;
      #
      # # Set the default number of eigenvectors to estimate in case of using speed up options
      # InputParam.Advanced.eig_vec_num_spup = 5;
      #
      # # Set reference frequency for adjustable outer radius of the model (approximate)
      # InputParam.Advanced.f_ref = ( InputParam.Model.f_min + InputParam.Model.f_max )/2; #Hz
      #
      # # Set the radial extension factor for adjustable radius of the outer layer
      # InputParam.Advanced.RadialExtensionFactor = 10;
      #
      # # Set the PML parameters
      # InputParam.Advanced.PML_factor = 1.0;
      # InputParam.Advanced.PML_exponent = 2.0;
      #
      # # Set the ABC parameters
      # InputParam.Advanced.ABC_factor = 4.0;
      # InputParam.Advanced.ABC_exponent = 3.0;
      # InputParam.Advanced.ABC_adjust_factor = 10;
      #
      # # Set frequency display limits
      # InputParam.Advanced.display_f_lo = 0; #kHz
      # InputParam.Advanced.display_f_up = 20; #kHz


      # Set the switch whether to visualize the mesh
      # visualization of the mesh - should be logical variable
      # (0 -no visualization, 1 - visualization)
      modelParams.Advanced.VisualizeMesh = 'true'
      # Set the mesh options
      modelParams.Advanced.MeshOptions.dhmax=0.25;
      # visualization of the mesh during its creation - should be logical
      # (0 -no visualization, 1 - visualization)
      InputParam.Advanced.MeshOptions.output = (false);
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
      # 3 - the first order; 6 - the second order; 10 - the third order (cubic element)
      modelParams.Advanced.N_nodes = 10
      modelParams.Advanced.NEdge_nodes = 4

      # Set the number of eigenvalues to compute for each frequency
      # Input.num_eig=40;# number of eigenvalues to be found for each k.
      modelParams.Advanced.num_eig_max = 40

      # Set the accuracy of integration (number of integration points) NOT used anymore
      # Input.integration_acc=20; # accuracy for integration (number of integration points) NOT used anymore
      # InputParam.Advanced.IntAcc = 20;

      # Set the starting velocity, in the vicinity of which to look for the
      # eigenvalues
      # Input.Starting_Velocity=1500; # velocity of the first eigenvalue for eigs
      modelParams.Advanced.EigSearchStart = 1.5 # starting velocity to search for the first eigenvalue for eigs

      # Set the eigs options
      modelParams.Advanced.EigsOptions.disp = 0
      modelParams.Advanced.EigsOptions.tol = 1e-8
      # InputParam.Advanced.EigsOptions.isreal = 1

      # Set the parameters of the source (Gaussian ring)
      # This set of parameters is necessary for DS way to classify the spectrum
      # (maximum of the excitation function)
      modelParams.Advanced.Source.xc = 0  # eccentricity of the source in x axis
      modelParams.Advanced.Source.yc = 0  # eccentricity of the source in y axis
      modelParams.Advanced.Source.r0x = 0.06 # x radius of the source
      modelParams.Advanced.Source.r0y = 0.06 # y radius of the source
      modelParams.Advanced.Source.theta_r = 0 # rotation of the source (geometrical)
      modelParams.Advanced.Source.theta0 = 0#pi/4 # rotation of the source direction
      modelParams.Advanced.Source.sigma = 0.02 # width of the source
      modelParams.Advanced.Source.Plim = 5e-4  # constrain amplitude to consider oscillation in node (maximum amplitude is 1)
      modelParams.Advanced.Source.symmetry = 0 # Symmetry of the source (0 is monopole, 1 sym dipole, 2 asym dipole).Will be changed automatically.

      return modelParams