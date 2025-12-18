"""
ANISO_SAFE - Python conversion of MATLAB toolbox for acoustic waveguide analysis.
"""

__version__ = "0.1.0"
__author__ = "T. Zharnkikov, A. Kozlov"

# Main imports
from .gen_aniso import gen_aniso
from .config.structures import (
    Config,
    ModelConfig,
    AdvancedConfig,
    InputParam,
    CompStruct
)

__all__ = [
    'gen_aniso',
    'Config',
    'ModelConfig',
    'AdvancedConfig',
    'InputParam',
    'CompStruct'
]