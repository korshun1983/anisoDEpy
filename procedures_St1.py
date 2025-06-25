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