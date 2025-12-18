#!/usr/bin/env python3
"""
Data structures for ANISO_SAFE project.
Analogous to MATLAB structs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import numpy as np


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Config:
    """Configuration parameters (from St1_1_SetModelConfig.m)"""
    ProblemType: str = 'spectrum'
    NumMethod: str = 'SAFE'
    SpeedUp: str = 'no'
    SaveData: str = 'yes'
    OuterBC: str = 'fixed'
    PML: str = 'r2'
    Eccentricity: str = 'no'
    Symmetry: str = 'none'
    EigenVar: str = 'k'
    SloUnits: str = 'us/ft'
    FreqUnits: str = 'kHz'
    PressureUnits: str = 'GPa'
    CheckAsymptote: str = 'yes'
    DisplayAttenuation: str = 'yes'
    root_path: str = ''
    solver_path: str = ''


@dataclass
class ModelConfig:
    """Model parameters (from St1_3_SetModelUser + JSON)"""
    DomainRx: List[float] = field(default_factory=list)
    DomainRy: List[float] = field(default_factory=list)
    DomainTheta: List[float] = field(default_factory=list)
    DomainEcc: List[float] = field(default_factory=list)
    DomainEccAngle: List[float] = field(default_factory=list)
    DomainType: List[str] = field(default_factory=list)
    BCType: List[str] = field(default_factory=list)
    LDomain_in_LSH: str = 'yes'
    AddDomainLoc: str = 'ext'
    AddDomainType: str = 'none'
    AddDomainL: float = 1.0
    AddDomain_Exist: str = 'none'
    PML_factor: float = 10.0
    PML_degree: float = 2.0
    PML_method: float = 2.0
    ABC_factor: float = 0.1
    ABC_degree: float = 1.0
    ABC_account_r: str = 'yes'
    DomainParam: List[List[float]] = field(default_factory=list)
    DomainNth: List[int] = field(default_factory=list)
    mud_domain: int = 1
    RefDomainType: List[str] = field(default_factory=list)
    RefDomainParam: List[List[float]] = field(default_factory=list)
    f_array: np.ndarray = field(default_factory=lambda: np.array([]))
    N_disp: int = 0
    f_min: float = 0.0
    f_max: float = 0.0


@dataclass
class AdvancedConfig:
    """Advanced parameters (from St1_4_SetModelAdvanced_sp_SAFE.m)"""
    VisualizeMesh: bool = True
    N_nodes: int = 10
    NEdge_nodes: int = 4
    EigsOptions: Dict[str, Any] = field(default_factory=lambda: {'disp': 0, 'tol': 1e-8})
    Source: Dict[str, float] = field(default_factory=lambda: {
        'xc': 0.0, 'yc': 0.0, 'r0x': 0.06, 'r0y': 0.06,
        'theta_r': 0.0, 'theta0': 0.0, 'sigma': 0.02,
        'Plim': 5e-4, 'symmetry': 0
    })
    # Parameters that can be overridden by user in JSON
    num_eig_max: int = 10
    EigSearchStart: float = 1.0


@dataclass
class Methods:
    """Method handles container"""
    # Step 1 methods
    St1_3_SetModelUser: Optional[Callable] = None
    St1_4_SetModelAdvanced: Optional[Callable] = None

    # Step 2 methods
    St2_PrepareModel: Optional[Callable] = None
    St2_1_PrepareModelParams: Optional[Callable] = None
    St2_2_PrepareModelMethods: Optional[Callable] = None

    # Step 3 methods
    St3_PrepareBasicMatrices: Optional[Callable] = None

    # Step 4 methods
    St4_ComputeSolution: Optional[Callable] = None

    # Utility methods
    chebdif: Optional[Callable] = None
    em_tensor_VTI: Optional[Callable] = None
    rot_c_ij: Optional[Callable] = None
    rot_matrix: Optional[Callable] = None
    ComputeAsymptotes: Optional[Callable] = None
    V_phase_VTI_exact_RPH: Optional[Callable] = None
    MeshFaces: Optional[Callable] = None


@dataclass
class InputParam:
    """Main input parameter structure (analogous to MATLAB InputParam)"""
    Config: Config = field(default_factory=Config)
    Model: ModelConfig = field(default_factory=ModelConfig)
    Advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    Methods: Methods = field(default_factory=Methods)


@dataclass
class CompStruct:
    """Computational structure (preparation for Step 2)"""
    Config: Config = field(default_factory=Config)
    Model: ModelConfig = field(default_factory=ModelConfig)
    Advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    Methods: Methods = field(default_factory=Methods)
    f_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    if_grid: int = 0
    Mesh: Dict[str, Any] = field(default_factory=dict)
    ModelInitial: Optional[ModelConfig] = None