# core/config.py
"""
===============================================================================
SAFE Parameter Structures - Python Dataclasses
Replicates MATLAB InputParam structure hierarchy
===============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import numpy as np


# -----------------------------------------------------------------------------
# Sub-structure definitions
# -----------------------------------------------------------------------------

@dataclass
class ConfigParameters:
    """Replicates InputParam.Config"""
    ProblemType: str = 'spectrum'  # 'spectrum', 'source', 'ExFun'
    NumMethod: str = 'SAFE'  # 'SAFE', 'SM', 'Riccati'
    SpeedUp: str = 'no'  # 'no' for SAFE (sparse automatic)
    SaveData: str = 'yes'  # Save intermediate .mat files
    OuterBC: str = 'fixed'  # 'fixed', 'adjust', 'PML', 'ABC', 'TTBC'
    PML: str = 'r2'  # PML formulation: 'none', 'r2'
    Eccentricity: str = 'no'  # 'yes', 'no'
    Symmetry: str = 'none'  # 'none', 'plane0', 'plane_pi2'
    EigenVar: str = 'k'  # 'k' (wavenumber), 'omega'
    SloUnits: str = 'us/ft'  # Slowness units
    FreqUnits: str = 'kHz'  # Frequency units
    PressureUnits: str = 'GPa'  # Moduli units
    CheckAsymptote: str = 'yes'  # Compare with analytical asymptotes
    DisplayAttenuation: str = 'yes'  # Plot attenuation curves
    root_path: Path = field(default_factory=Path)  # Project root directory
    solver_path: Path = field(default_factory=Path)  # Auto-set by pipeline


@dataclass
class MeshParameters:
    """Replicates InputParam.Mesh"""
    hmax: float = 0.16  # Max element size (fraction of wavelength)
    hmax_absolute: float = None  # NEW: Computed absolute size in meters
    dhmax: float = 0.25  # Max relative gradient
    output: str = 'no'  # Display mesh: 'yes', 'no'
    ext_boundary_shape: str = 'cir'  # 'cir' or 'rect'
    mlim: float = 0.02  # Mesh quality tolerance (2%)
    maxit: int = 10  # Max mesh iterations
    # Mesh visualization handle (runtime)
    fig_handle: Optional[Any] = None


@dataclass
class AdvancedSourceParameters:
    """Replicates InputParam.Advanced.Source"""
    xc: float = 0.0  # Source x-eccentricity
    yc: float = 0.0  # Source y-eccentricity
    r0x: float = 0.06  # Source radius x
    r0y: float = 0.06  # Source radius y
    theta_r: float = 0.0  # Source rotation (geometric)
    theta0: float = 0.0  # Source direction rotation
    sigma: float = 0.02  # Gaussian width
    Plim: float = 5e-4  # Amplitude threshold
    symmetry: int = 0  # 0=monopole, 1=dipole


@dataclass
class AdvancedEigsOptions:
    """Replicates InputParam.Advanced.EigsOptions"""
    tol: float = 1e-8  # Eigenvalue solver tolerance
    disp: int = 0  # Display iterations (0=no, 1=yes)
    isreal: Optional[int] = None  # Deprecated in modern scipy


@dataclass
class AdvancedParameters:
    """Replicates InputParam.Advanced"""
    N_nodes: int = 10  # Nodes per element (3, 6, 10)
    NEdge_nodes: int = 4  # Edge nodes for cubic elements
    num_eig_max: int = 10  # Max eigenvalues to compute
    EigSearchStart: float = 1.0  # Starting velocity for eigensearch
    VisualizeMesh: bool = True  # Enable mesh visualization
    max_harm_limit: float = 0.3  # Symmetry classification threshold
    use_class_TE_NT: str = 'n'  # Nguyen classification
    use_class_ABCPML_HTTI: str = 'y'  # ABC/PML classification

    # Nested structures
    EigsOptions: AdvancedEigsOptions = field(default_factory=AdvancedEigsOptions)
    Source: AdvancedSourceParameters = field(default_factory=AdvancedSourceParameters)

    # Mesh options (flattened from MATLAB structure)
    MeshOptions_mlim: float = 0.02
    MeshOptions_maxit: int = 20
    MeshOptions_dhmax: float = 0.25
    MeshOptions_output: bool = False


@dataclass
class MiscParameters:
    """Replicates InputParam.Misc"""
    isDryRun: bool = False
    save_mat: bool = True
    plot_intermediate: bool = False
    F_conv: float = 1.0          # Frequency conversion factor
    S_conv: float = 1.0          # Slowness conversion factor

@dataclass
class DataParameters:
   """New - stores runtime data"""
   N_domain: int = 0  # Number of domains
   DVarNum: List[int] = field(default_factory=list)  # Variables per domain
   N_interface: int = 0  # Number of interfaces


# -----------------------------------------------------------------------------
# Main InputParam structure
# -----------------------------------------------------------------------------

@dataclass
class InputParam:
    """Main parameter container - replicates MATLAB InputParam"""
    Config: ConfigParameters = field(default_factory=ConfigParameters)
    Mesh: MeshParameters = field(default_factory=MeshParameters)
    Advanced: AdvancedParameters = field(default_factory=AdvancedParameters)
    Misc: MiscParameters = field(default_factory=MiscParameters)

    # Model parameters (dynamic, loaded from JSON)
    Model: Dict[str, Any] = field(default_factory=dict)
    Methods: Dict[str, Callable] = field(default_factory=dict)

    # Python-specific additions
    _json_file: Optional[Path] = None

    Data: DataParameters = field(default_factory=DataParameters)
    Asymp: Dict[str, Any] = field(default_factory=dict)  # Asymptote data

@dataclass
class CompStruct(InputParam):
    """Computation structure extends InputParam with runtime data"""
    if_grid: int = 0
    f_grid: np.ndarray = field(default_factory=lambda: np.array([]))

# -----------------------------------------------------------------------------
# Methods container (Python-specific)
# -----------------------------------------------------------------------------

@dataclass
class MethodsContainer:
    """
    Python alternative to MATLAB function handles.
    Stores direct function references for pipeline stages.
    """
    # Stage 1
    St1_4_SetModelAdvanced: Callable = None

    # Stage 2
    St2_PrepareModel: Callable = None
    St2_1_PrepareModelParams: Callable = None
    St2_2_PrepareModelMethods: Callable = None

    # Stage 3
    St3_PrepareBasicMatrices: Callable = None
    St3_PrepareMesh: Callable = None

    # Stage 4
    St4_ComputeSolution: Callable = None

    # Utilities
    ComputeAsymptotes: Callable = None
    em_tensor_VTI: Callable = None
    rot_c_ij: Callable = None
    rot_matrix: Callable = None
    V_phase_VTI_exact_RPH: Callable = None
    MeshFaces: Callable = None


# -----------------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------------

def create_default_input() -> InputParam:
    """Create default parameter structure"""
    return InputParam()


def create_input_from_json(json_data: Dict[str, Any]) -> InputParam:
    """
    Convert JSON data (nested dict) to InputParam dataclass.
    Handles nested structures recursively.
    """
    input_param = InputParam()

    # Flatten nested JSON into dataclass fields
    def set_nested_field(obj, field_path, value):
        parts = field_path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    # Map JSON keys to dataclass fields
    mapping = {
        'Model.f_array_range': lambda: np.arange(
            json_data['Model']['f_array_range']['start'],
            json_data['Model']['f_array_range']['end'] + json_data['Model']['f_array_range']['step'] / 2,
            json_data['Model']['f_array_range']['step']
        ).tolist(),
        'Model.DomainRx': 'Model.DomainRx',
        'Model.DomainRy': 'Model.DomainRy',
        'Model.DomainTheta': 'Model.DomainTheta',
        'Model.DomainEcc': 'Model.DomainEcc',
        'Model.DomainEccAngle': 'Model.DomainEccAngle',
        'Model.DomainType': 'Model.DomainType',
        'Model.DomainParam': 'Model.DomainParam',
        'Model.LDomain_in_LSH': 'Model.LDomain_in_LSH',
        'Model.AddDomainLoc': 'Model.AddDomainLoc',
        'Model.AddDomainType': 'Model.AddDomainType',
        'Model.AddDomainL': 'Model.AddDomainL',
        'Model.PML_factor': 'Model.PML_factor',
        'Model.PML_degree': 'Model.PML_degree',
        'Model.PML_method': 'Model.PML_method',
        'Model.ABC_factor': 'Model.ABC_factor',
        'Model.ABC_degree': 'Model.ABC_degree',
        'Model.ABC_account_r': 'Model.ABC_account_r',
        'Model.BCType': 'Model.BCType',
        'Model.DomainNth': 'Model.DomainNth',
        'Model.mud_domain': 'Model.mud_domain',
        'Model.RefDomainType': 'Model.RefDomainType',
        'Model.RefDomainParam': 'Model.RefDomainParam',
        'Advanced.num_eig_max': 'Advanced.num_eig_max',
        'Advanced.EigSearchStart': 'Advanced.EigSearchStart',
        'Mesh.output': 'Mesh.output',
    }

    # Manual field assignment for nested structures
    if 'Model' in json_data:
        model = json_data['Model']
        input_param.Model = model.copy()

        # Convert f_array_range to actual array
        if 'f_array_range' in model:
            far = model['f_array_range']
            input_param.Model['f_array'] = np.arange(
                far['start'], far['end'] + far['step'] / 2, far['step']
            ).tolist()
            input_param.Model['N_disp'] = len(input_param.Model['f_array'])

        # Convert domain parameters to proper Python lists
        for key in ['DomainRx', 'DomainRy', 'DomainTheta', 'DomainEcc', 'DomainEccAngle', 'DomainNth']:
            if key in model:
                input_param.Model[key] = list(model[key])

        # Convert string lists
        if 'DomainType' in model:
            input_param.Model['DomainType'] = list(model['DomainType'])
        if 'BCType' in model:
            input_param.Model['BCType'] = list(model['BCType'])

    # Advanced parameters
    if 'Advanced' in json_data:
        adv = json_data['Advanced']
        if 'num_eig_max' in adv:
            input_param.Advanced.num_eig_max = adv['num_eig_max']
        if 'EigSearchStart' in adv:
            input_param.Advanced.EigSearchStart = adv['EigSearchStart']

    # Mesh parameters
    if 'Mesh' in json_data:
        mesh = json_data['Mesh']
        if 'output' in mesh:
            input_param.Mesh.output = mesh['output']

    return input_param