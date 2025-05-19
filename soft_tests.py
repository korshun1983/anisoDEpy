from pprint import pprint

import numpy as np
from ast import literal_eval
import re

class inputModel():
    pass

class advanced():
    pass

class mesh():
    pass

inputParams = inputModel()
inputParams.advanced = advanced()
inputParams.mesh = mesh()

# # Set the geometry of the model - positions of layer boundaries and interfaces, in meters or wavelength
# inputParams.DomainRx = [0.1, 2.0] #positions of layer boundaries along x axis(from inner to outer)
# inputParams.DomainRy = [0.1, 2.0] #positions of layer boundaries along y axis(from inner to outer)
# inputParams.DomainTheta = [0, 0] #rotation  of each layer with respect to x axis, in radians
# inputParams.DomainEcc = [0, 0] #eccentricity of each layer with respect to the coordinate origin, in meters
# inputParams.DomainEccAngle = [0, 0]
# # DomainRx and DomainRy of last layer are given in wavelength of V_SH (yes) or in meters (none) (2.0 - recommended outer radius)
# inputParams.LDomain_in_LSH = 'yes'
# #Set the types of the layers(medium used)
# #'fluid' - ideal fluid, 'HTTI' - TTI, homogeneous in Cartesian coordinates
# inputParams.DomainType = {'fluid', 'HTTI'}
# # ===============================================================================
# # Set parameters of additional. It is initial physical properties are equal to the last layer.
# # ===============================================================================
# inputParams.AddDomainLoc = 'ext'  # Location can be 'ext' --- external or 'int' --- internal (outside or inside of last domain)
# inputParams.AddDomainType = 'abc'  # pml, abc, abc+pml or pml+abc, same (as last domain), none
# inputParams.AddDomainL = 1.  # layer's length is specified in wavelength of V_SH if Model.LDomain_in_LSH='yes',
# # otherwise --- in meters
#
# # ===============================================================================
# # PML layer. Now gamma(r) complex function is specified in KM_el_matrix_HTTI_PML.m
# # Due to PML realization for simplicity should be inputParams.DomainRx(end)=inputParams.DomainRy(end)
# # ===============================================================================
# inputParams.PML_factor = 10
# inputParams.PML_degree = 2.0
# inputParams.PML_method = 2.  # 1 is used differentiation with respect to r, Circle PML
# # 2 is used differentiation with respect to x and y, Rectangle PML
# # ===============================================================================
# # ABC layer. C1_ij = C_ij*(1-i*factor*((x-d)/h)^degree
# inputParams.ABC_factor = 0.1
# inputParams.ABC_degree = 1.0
# inputParams.ABC_account_r = 'yes'
#
# # ===============================================================================
# # Set physical properties of the layers' media
# # ===============================================================================
# inputParams.DomainParam = ([1.0, 2.25], [2.23, 40.9, 8.5, 26.9, 10.5, 15.3, 0, 0])
#
# #
# # parameters for the layers:
# # fluid - [density, lambda]
# # solidTTI - [density, c11 c13 c33 c44 c66, relative dip, azimuth ], azimuth is optional
# #
# # dimensions of parameters:
# # density 1e-3*kg/m^3 (kg/cm^3)
# # Cij - in GPa
# # relative dip (VTI axis inclination to that of the waveguide) in radians
#
# # ===============================================================================
# # Set the physical properties of the reference layer,
# # whcih used to determine the far field asymptotic values
# # ===============================================================================
# inputParams.RefDomainType = {'HTTI'}
#
# # inputParams.RefDomainParam = { [2.23 , 40.9 , 8.5 , 26.9 , 10.5 , 15.3 , 0 , 0] };
# # inputParams.RefDomainParam = {inputParams.DomainParam{2}}
# # inputParams.RefDomainParam{1}(7) = 0
# # ===============================================================================
# # Set the types of boundary and interface conditions
# # ===============================================================================
# inputParams.BCType = {'FS', 'rigid'}
# # types of boundary and interface conditions (BCs and ICs):
# # 1 - rigid - rigid surface (fluid, solid)
# # 2 - FS - natural fluid-solid or fluid-fluid contact
#
# # ===============================================================================
# # Set the frequency range of interest in kHz
# # ===============================================================================
# inputParams.f_array = np.arange(0.5,15.25,0.25)
#
# # ===============================================================================
# # Set model discretization parameters
# # ===============================================================================
# # NB! These parameters are available to the user,
# #     but should be treated with care
# # ===============================================================================
#
# # Set the number of azimuthal main nodes to use for each layer
# # Usually, you do not have to change it
# inputParams.DomainNth = [12, 12]
#
# # Set the layer, which contains drilling mud
# inputParams.mud_domain = 1
# # Typically mud layer is:
# #  1 - for open hole (OH)
# #  2 - for heavy fluid model (HFM)
#
# # Set the number of eigenvalues to compute for each frequency
# # which is estimated experimentally at present time
# inputParams.Advanced = advanced()
# inputParams.Advanced.num_eig_max = 50
#
# # Set the starting velocity, in the vicinity of which to look for the eigenvalues,
# # which is estimated experimentally at present time
# # The running time of the program essentially depends on this parameter
# inputParams.Advanced.EigSearchStart = 1.0
#
# # ===============================================================================
# # Set some usefull mesh program options,
# # if they are commneted then we use defaults
# # ===============================================================================
# # InputParam.Mesh.hmax=0.05;    # Max allowable global element size (h/(5*L)); 0.025
# # InputParam.Mesh.dhmax=0.25;   # The maximum allowable (relative) gradient
# inputParams.Mesh = mesh()
# inputParams.Mesh.MeshOutput = 'no'  # Displays the mesh
# # InputParam.Mesh.ext_boundary_shape='cir';  # external boubdary is rectangle, default --- circle ('cir')
# # now - no rotation, no eccentricit, no azimuthal shift of lasr layer

def get_data_from_line(parsed_line):
    data = parsed_line.split(" = ")
    data = data[1]
    data = data.rstrip()
    return data

def get_array_from_string(linestr):
    return np.array(literal_eval(linestr))

def extract_float_vectors(linestr):
    data = re.findall(r"[-+]?(?:\d*\.*\d+)", linestr)
    data_out = ([data[0], data[1]], [data[2], data[3],data[4], data[5],data[6], data[7],data[8], data[9]])
    return data_out

def get_ranged_vector_from_string(linestr):
    vals = re.findall(r"[-+]?(?:\d*\.*\d+)", linestr)
    vals = np.arange(float(vals[0]),float(vals[2])+float(vals[1]),float(vals[1]))
    return vals

with open("Bakken-B-00.m") as fid:
    for line in fid:
        if line.startswith('Name'):
            inputParams.Name = get_data_from_line(line)
        elif line.startswith('DomainRx'):
            inputParams.DomainRx = get_array_from_string(get_data_from_line(line))
        elif line.startswith('DomainRy'):
            inputParams.DomainRy = get_array_from_string(get_data_from_line(line))
        elif line.startswith('DomainTheta'):
            inputParams.DomainTheta = get_array_from_string(get_data_from_line(line))
        elif line.startswith('DomainEcc'):
            inputParams.DomainEcc = get_array_from_string(get_data_from_line(line))
        elif line.startswith('DomainEccAngle'):
            inputParams.DomainEccAngle = get_array_from_string(get_data_from_line(line))
        elif line.startswith('LDomain_in_LSH'):
            inputParams.LDomain_in_LSH = get_data_from_line(line)
        elif line.startswith('DomainType'):
            inputParams.DomainType = get_data_from_line(line).split()
        elif line.startswith('AddDomainLoc'):
            inputParams.AddDomainLoc = get_data_from_line(line)
        elif line.startswith('AddDomainType'):
            inputParams.AddDomainType = get_data_from_line(line)
        elif line.startswith('AddDomainL'):
            inputParams.AddDomainL = float(get_data_from_line(line))
        elif line.startswith('PML_factor'):
            inputParams.PML_factor = float(get_data_from_line(line))
        elif line.startswith('PML_degree'):
            inputParams.PML_degree = float(get_data_from_line(line))
        elif line.startswith('PML_method'):
            inputParams.PML_method = float(get_data_from_line(line))
        elif line.startswith('ABC_factor'):
            inputParams.ABC_factor = float(get_data_from_line(line))
        elif line.startswith('ABC_degree'):
            inputParams.ABC_degree = float(get_data_from_line(line))
        elif line.startswith('ABC_account_r'):
            inputParams.ABC_account_r = get_data_from_line(line)
        elif line.startswith('DomainParam'):
            inputParams.DomainParam = extract_float_vectors(get_data_from_line(line))
        elif line.startswith('RefDomainType'):
            inputParams.RefDomainType = get_data_from_line(line)
        elif line.startswith('RefDomainParam'):
            inputParams.RefDomainParam = get_data_from_line(line)
            if inputParams.RefDomainParam == 'same':
                inputParams.RefDomainParam = inputParams.DomainParam
            else:
                inputParams.RefDomainParam = extract_float_vectors(inputParams.RefDomainParam)
        elif line.startswith('BCType'):
            inputParams.BCType = get_data_from_line(line).split(',')
        elif line.startswith('f_array'):
            inputParams.f_array = get_ranged_vector_from_string(get_data_from_line(line))
        elif line.startswith('DomainNth'):
            inputParams.DomainNth = get_array_from_string(get_data_from_line(line))
        elif line.startswith('mud_domain'):
            inputParams.mud_domain = int(get_data_from_line(line))
        elif line.startswith('Advanced.num_eig_max'):
            inputParams.advanced.num_eig_max = int(get_data_from_line(line))
        elif line.startswith('Advanced.EigSearchStart'):
            inputParams.advanced.EigSearchStart = float(get_data_from_line(line))
        elif line.startswith('Mesh.hmax'):
            inputParams.mesh.hmax = float(get_data_from_line(line))
        elif line.startswith('Mesh.dhmax'):
            inputParams.mesh.dhmax = float(get_data_from_line(line))
        elif line.startswith('Mesh.output'):
            inputParams.mesh.output = get_data_from_line(line)
        elif line.startswith('Mesh.ext_boundary_shape'):
            inputParams.mesh.ext_boundary_shape = get_data_from_line(line)

#easy test
print("The parameters read from file are:", vars(inputParams))