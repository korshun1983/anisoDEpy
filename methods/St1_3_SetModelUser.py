import numpy as np
from structures import BigCompStruct


def St1_3_SetModelUser_sp_SAFE(InputParam: BigCompStruct):
    #   Code for St1_3_SetModelUser_sp_SM
    ################################################################################
    #===============================================================================
    # Setting the model
    #===============================================================================

    #===============================================================================
    # Set the geometry of the model - positions of layer boundaries and interfaces, in meters or wavelength
    #===============================================================================
    InputParam.Model.DomainRx = [0.1 , 2.0] # positions of layer boundaries along x axis (from inner to outer)
    InputParam.Model.DomainRy = [0.1 , 2.0] # positions of layer boundaries along y axis (from inner to outer)
    InputParam.Model.DomainTheta = [0 , 0 ] # rotation of each layer with respect to x axis, in radians
    InputParam.Model.DomainEcc = [0 , 0] # eccentricity of each layer with respect to the coordinate origin, in meters
    InputParam.Model.DomainEccAngle = [0 , 0] # azimuthal shift of eccentricity direction, in radians

    # DomainRx and DomainRy of last layer are given in wavelength of V_SH (yes)
    # or in meters (none) (2.0 - recommended outer radius)
    InputParam.Model.LDomain_in_LSH='yes',

    #===============================================================================
    # Set the types of the layers (medium used)
    #===============================================================================
    # 'fluid' - ideal fluid
    # 'HTTI' - TTI, homogeneous in Cartesian coordinates
    InputParam.Model.DomainType = {'fluid',  'HTTI'}

    #===============================================================================
    # Set parameters of additional. It is initial physical properties are equal to the last layer.
    #===============================================================================
    InputParam.Model.AddDomainLoc='ext'    # Location can be 'ext' --- external or 'int' --- internal (outside or inside of last domain)
    InputParam.Model.AddDomainType='abc'   # pml, abc, abc+pml or pml+abc, same (as last domain), none
    InputParam.Model.AddDomainL=1.         # layer's length is specified in wavelength of V_SH if Model.LDomain_in_LSH='yes',
                                            # otherwise --- in meters

    #===============================================================================
    # PML layer. Now gamma(r) complex funtion is specified in KM_el_matrix_HTTI_PML.m
    # Due to PML realization for simplicity should be InputParam.Model.DomainRx(end)=InputParam.Model.DomainRy(end)
    #===============================================================================
    InputParam.Model.PML_factor = 10
    InputParam.Model.PML_degree = 2.0
    InputParam.Model.PML_method = 2.   # 1 is used differentiation with respect to r, Circle PML
                                        # 2 is used differentiation with respect to x and y, Rectangle PML
    #===============================================================================
    # ABC layer. C1_ij = C_ij*(1-i*factor*((x-d)/h)^degree
    InputParam.Model.ABC_factor = 0.1
    InputParam.Model.ABC_degree = 1.0
    InputParam.Model.ABC_account_r='yes'

    #===============================================================================
    # Set physical properties of the layers' media
    #===============================================================================
    InputParam.Model.DomainParam = { [1.0, 2.25],  [ 2.23 , 40.9 , 8.5 , 26.9 , 10.5 , 15.3 , 0 , 0] }
    #
    # parameters for the layers:
    # fluid - [density, lambda]
    # solidTTI - [density, c11 c13 c33 c44 c66, relative dip, azimuth ], azimuth is optional
    #
    # dimensions of parameters:
    # density 1e-3*kg/m^3 (kg/cm^3)
    # Cij - in GPa
    # relative dip (VTI axis inclination to that of the waveguide) in radians

    #===============================================================================
    # Set the physical properties of the reference layer,
    # whcih used to determine the far field asymptotic values
    #===============================================================================
    InputParam.Model.RefDomainType = {'HTTI'}

    #InputParam.Model.RefDomainParam = { [2.23 , 40.9 , 8.5 , 26.9 , 10.5 , 15.3 , 0 , 0] };
    InputParam.Model.RefDomainParam =  InputParam.Model.DomainParam[2]
    InputParam.Model.RefDomainParam[1][7]=0
    #===============================================================================
    # Set the types of boundary and interface conditions
    #===============================================================================
    InputParam.Model.BCType = {'FS', 'rigid'}
    # types of boundary and interface conditions (BCs and ICs):
    # 1 - rigid - rigid surface (fluid, solid)
    # 2 - FS - natural fluid-solid or fluid-fluid contact

    #===============================================================================
    # Set the frequency range of interest in kHz
    #===============================================================================
    InputParam.Model.f_array = np.arange([0.5,0.25,15.0])

    #===============================================================================
    # Set model discretization parameters
    #===============================================================================
    # NB! These parameters are available to the user,
    #     but should be treated with care
    #===============================================================================

    # Set the number of azimuthal main nodes to use for each layer
    # Usually, you do not have to change it
    InputParam.Model.DomainNth = [12 , 12]

    # Set the layer, which contains drilling mud
    InputParam.Model.mud_domain = 1
    # Typically mud layer is:
    #  1 - for open hole (OH)
    #  2 - for heavy fluid model (HFM)

    # Set the number of eigenvalues to compute for each frequency
    # which is estimated experimentally at present time
    InputParam.Advanced.num_eig_max = 50

    # Set the starting velocity, in the vicinity of which to look for the eigenvalues,
    # which is estimated experimentally at present time
    # The running time of the program essentially depends on this parameter
    InputParam.Advanced.EigSearchStart = 1.0

    #===============================================================================
    # Set some usefull mesh program options,
    # if they are commneted then we use defaults
    #===============================================================================
    #InputParam.Mesh.hmax=0.05;    # Max allowable global element size (h/(5*L)); 0.025
    #InputParam.Mesh.dhmax=0.25;   # The maximum allowable (relative) gradient
    InputParam.Mesh.output='no';  # Displays the mesh
    #InputParam.Mesh.ext_boundary_shape='cir';  # external boubdary is rectangle, default --- circle ('cir')
                                               # now - no rotation, no eccentricit, no azimuthal shift of lasr layer
    return InputParam