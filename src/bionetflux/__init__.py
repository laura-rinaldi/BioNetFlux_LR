
"""BioNetFlux: Multi-Domain Biological Network Flow Simulation Framework"""

__version__ = "1.0.0"

# Main exports for convenience
from .core.problem import Problem
from .geometry.domain_geometry import DomainGeometry, DomainInfo
from .visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

__all__ = ["Problem", "DomainGeometry", "DomainInfo", "LeanMatplotlibPlotter"]

