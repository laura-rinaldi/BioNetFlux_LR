
import numpy as np
import warnings
from typing import List, Dict, Optional, Callable, Union

from .problem import Problem
from .discretization import Discretization
from .bulk_data import BulkData
from bionetflux.utils.elementary_matrices import ElementaryMatrices

def plot_bulk(
                                  bulk_solution: BulkData,
                                  problem: Problem,
                                  discretization: Discretization,
                                  time: float) -> Dict[int, Union[float, None]]:
        
        neq = problem.neq
        n_elements = discretization.n_elements
        nodes = discretization.nodes
        
        domain_errors = {}
        
        for eq_idx in range(neq):
            # Integrate over each element using 4-point quadrature
                for elem_idx in range(n_elements):
                    x_left = nodes[elem_idx]
                    x_right = nodes[elem_idx + 1]
                    h_elem = x_right - x_left
                    
                    # Get bulk coefficients for this element and equation
                    element_coeffs = bulk_solution.get_element_data(elem_idx)
                    c0 = element_coeffs[2 * eq_idx]      # Left coefficient
                    c1 = element_coeffs[2 * eq_idx + 1]  # Right coefficient
                    
                    # Map quadrature nodes to physical element
                    xi_01 = (self.quad_nodes + 1) / 2  # Map [-1,1] to [0,1]
                    mapped_nodes = x_left + xi_01 * h_elem
                    
                    # Evaluate numerical solution at quadrature points
                    numerical_values = c0 * (1 - xi_01) + c1 * xi_01
                    

                
                domain_errors[eq_idx] = numerical_values
        return domain_errors