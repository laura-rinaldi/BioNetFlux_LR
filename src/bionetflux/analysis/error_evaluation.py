"""
L2 Error Evaluation Module for BioNetFlux

This module provides functionality to compute L2 errors between numerical and analytical solutions,
perform convergence analysis, and evaluate solution quality metrics.
"""

import numpy as np
from typing import List, Callable, Optional, Dict, Tuple
import warnings


def retrieve_analytical_solution(problems: List) -> Dict:
    """
    Extract analytical solutions from a list of problem objects.
    
    Args:
        problems: List of problem objects that may contain analytical solutions
    
    Returns:
        Dictionary with analytical functions organized by domain index
        Format: {'domain_0': [func1, func2, ...], 'domain_1': [...], ...}
    """
    analytical_solutions = {}
    
    for i, problem in enumerate(problems):
        domain_key = f'domain_{i}'
        
        if hasattr(problem, 'solution') and problem.solution is not None:
            # Use the actual analytical solution from the problem
            if callable(problem.solution):
                # Single function - replicate for all equations in this problem
                analytical_solutions[domain_key] = [problem.solution] * problem.neq
            elif isinstance(problem.solution, (list, tuple)):
                # Multiple functions for multiple equations
                if len(problem.solution) == problem.neq:
                    analytical_solutions[domain_key] = list(problem.solution)
                else:
                    warnings.warn(f"Problem {i}: Number of solution functions ({len(problem.solution)}) "
                                f"doesn't match number of equations ({problem.neq}). Using zero analytical solution.")
                    analytical_solutions[domain_key] = None
            else:
                warnings.warn(f"Problem {i}: solution attribute exists but is not callable or list. "
                            f"Type: {type(problem.solution)}. Using zero analytical solution.")
                analytical_solutions[domain_key] = None
        else:
            warnings.warn(f"Problem {i}: No solution attribute found. Using zero analytical solution.")
            analytical_solutions[domain_key] = None
    
    return analytical_solutions


class ErrorEvaluator:
    """
    Evaluates L2 errors between numerical and analytical solutions.
    Supports both pointwise and integrated error measures.
    """
    
    def __init__(self, problems: List, discretizations: List):
        """
        Initialize the L2 error evaluator.
        
        Args:
            problems: List of problem objects with analytical solutions
            discretizations: List of spatial discretization objects
        """
        self.problems = problems
        self.discretizations = discretizations
        self.n_domains = len(problems)
        
        # Validate inputs
        if len(discretizations) != self.n_domains:
            raise ValueError("Number of discretizations must match number of problems")
        
        # Automatically extract analytical solutions from problems
        self.analytical_solutions = retrieve_analytical_solution(problems)
    
    def compute_trace_error(self, 
                        numerical_solutions: List[np.ndarray], 
                        time: float,
                        analytical_functions: Optional[List[List[Callable]]] = None) -> Dict:
        """
        Compute L2 errors between numerical and analytical solutions.
        
        Args:
            numerical_solutions: List of numerical trace solutions for each domain
            time: Current time for analytical solution evaluation
            analytical_functions: Optional list of analytical functions per domain/equation
                                 If None, uses automatically extracted solutions from problems
        
        Returns:
            Dictionary with error metrics per domain and equation
        """
        # Use provided analytical functions or fall back to extracted ones
        if analytical_functions is None:
            analytical_functions = [self.analytical_solutions.get(f'domain_{i}', None) 
                                   for i in range(self.n_domains)]
        
        results = {
            'domain_errors': [],
            'equation_errors': {},  # New: organized by (domain_idx, eq_idx)
            'global_error_per_equation': [],  # New: global error for each equation
            'global_error': 0.0,
            'max_error': 0.0,
            'time': time
        }
        
        # Track errors per equation across all domains
        max_equations = max(problem.neq for problem in self.problems)
        global_error_squared_per_eq = [0.0] * max_equations
        global_solution_norm_squared_per_eq = [0.0] * max_equations
        max_pointwise_error = 0.0
        
        for domain_idx in range(self.n_domains):
            problem = self.problems[domain_idx]
            discretization = self.discretizations[domain_idx]
            numerical_sol = numerical_solutions[domain_idx]
            
            domain_result = self._compute_domain_l2_error(
                problem, discretization, numerical_sol, time, 
                analytical_functions[domain_idx] if analytical_functions and domain_idx < len(analytical_functions) else None
            )
            
            results['domain_errors'].append(domain_result)
            
            # Store individual equation errors with domain/equation indexing
            for eq_error in domain_result['equation_errors']:
                eq_idx = eq_error['equation_idx']
                results['equation_errors'][(domain_idx, eq_idx)] = eq_error
                
                # Accumulate global error per equation
                if eq_idx < len(global_error_squared_per_eq):
                    global_error_squared_per_eq[eq_idx] += eq_error['l2_error_squared']
                    global_solution_norm_squared_per_eq[eq_idx] += eq_error['solution_norm_squared']
            
            max_pointwise_error = max(max_pointwise_error, domain_result['max_pointwise_error'])
        
        # Compute global errors per equation
        for eq_idx in range(max_equations):
            global_l2_error = np.sqrt(global_error_squared_per_eq[eq_idx])
            if global_solution_norm_squared_per_eq[eq_idx] > 1e-14:
                relative_error = np.sqrt(global_error_squared_per_eq[eq_idx] / global_solution_norm_squared_per_eq[eq_idx])
            else:
                relative_error = np.inf
                
            results['global_error_per_equation'].append({
                'equation_idx': eq_idx,
                'global_l2_error': global_l2_error,
                'global_relative_error': relative_error,
                'global_solution_norm': np.sqrt(global_solution_norm_squared_per_eq[eq_idx])
            })
        
        # Overall global error (sum of all equations)
        total_error_squared = sum(global_error_squared_per_eq)
        total_solution_norm_squared = sum(global_solution_norm_squared_per_eq)
        
        results['global_error'] = np.sqrt(total_error_squared)
        results['max_error'] = max_pointwise_error
        
        if total_solution_norm_squared > 1e-14:
            results['relative_global_error'] = np.sqrt(total_error_squared / total_solution_norm_squared)
        else:
            results['relative_global_error'] = np.inf
            
        return results
    
    def _compute_domain_l2_error(self, 
                                problem, 
                                discretization, 
                                numerical_sol: np.ndarray, 
                                time: float,
                                analytical_functions: Optional[List[Callable]] = None) -> Dict:
        """
        Compute L2 error for a single domain.
        
        Args:
            problem: Problem object for the domain
            discretization: Spatial discretization for the domain
            numerical_sol: Numerical solution array
            time: Current time
            analytical_functions: List of analytical functions per equation
        
        Returns:
            Dictionary with domain-specific error metrics (WITHOUT accumulating equations)
        """
        nodes = discretization.nodes
        n_nodes = len(nodes)
        neq = problem.neq
        
        # Get analytical functions
        if analytical_functions is None:
            analytical_functions = self._get_analytical_functions(problem)
        
        max_pointwise_error = 0.0
        equation_errors = []
        
        for eq_idx in range(neq):
            # Extract numerical solution for this equation
            eq_start = eq_idx * n_nodes
            eq_end = eq_start + n_nodes
            numerical_values = numerical_sol[eq_start:eq_end, 0] if numerical_sol.ndim == 2 else numerical_sol[eq_start:eq_end]
            
            # Compute analytical solution at nodes
            if analytical_functions and eq_idx < len(analytical_functions):
                analytical_values = np.array([analytical_functions[eq_idx](x, time) for x in nodes])
            else:
                # Fallback: assume zero analytical solution with warning
                analytical_values = np.zeros_like(numerical_values)
                warnings.warn(f"No analytical solution available for equation {eq_idx}, using zero")
            
            # Compute pointwise errors
            pointwise_errors = numerical_values - analytical_values
            
            # Integrate using trapezoidal rule for L2 norm
            eq_error_squared = self._integrate_trapezoidal(nodes, pointwise_errors**2)
            eq_solution_norm_squared = self._integrate_trapezoidal(nodes, analytical_values**2)
            
            # Track maximum pointwise error
            eq_max_error = np.max(np.abs(pointwise_errors))
            max_pointwise_error = max(max_pointwise_error, eq_max_error)
            
            # Store equation-specific results
            equation_errors.append({
                'equation_idx': eq_idx,
                'l2_error': np.sqrt(eq_error_squared),
                'l2_error_squared': eq_error_squared,
                'solution_norm': np.sqrt(eq_solution_norm_squared),
                'solution_norm_squared': eq_solution_norm_squared,
                'max_pointwise_error': eq_max_error,
                'relative_error': np.sqrt(eq_error_squared / eq_solution_norm_squared) if eq_solution_norm_squared > 1e-14 else np.inf,
                'numerical_values': numerical_values.copy(),
                'analytical_values': analytical_values.copy(),
                'pointwise_errors': pointwise_errors.copy()
            })
        
        return {
            'domain_idx': getattr(problem, 'domain_idx', 0),
            'max_pointwise_error': max_pointwise_error,
            'equation_errors': equation_errors,
            'nodes': nodes.copy(),
            'n_equations': neq
        }

    def _integrate_trapezoidal(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Integrate using trapezoidal rule.
        
        Args:
            x: Coordinate points
            y: Function values at points
        
        Returns:
            Integrated value
        """
        if len(x) != len(y):
            raise ValueError("x and y arrays must have same length")
        if len(x) < 2:
            return 0.0
        
        return np.trapz(y, x)
    
    def _get_analytical_functions(self, problem) -> Optional[List[Callable]]:
        """
        WARNING: This method is wrong, it should be removed.  Use retrieve_analytical_solution instead.
        Extract analytical functions from problem object.
        
        Args:
            problem: Problem object
        
        Returns:
            List of analytical functions or None if not available
        """
        warnings.warn(
            "The _get_analytical_functions method is deprecated and incorrect. "
            "Use retrieve_analytical_solution() instead. "
            "This method will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Try common attribute names for analytical solutions
        for attr_name in ['analytical_solution', 'exact_solution', 'analytical_functions']:
            if hasattr(problem, attr_name):
                analytical = getattr(problem, attr_name)
                if callable(analytical):
                    # Single function - wrap in list
                    return [analytical]
                elif isinstance(analytical, (list, tuple)):
                    # Multiple functions
                    return list(analytical)
        
        return None
    
    def compute_convergence_rate(self, 
                                errors: List[float], 
                                mesh_sizes: List[float]) -> Tuple[float, float]:
        """
        Compute convergence rate from L2 errors and mesh sizes.
        
        Args:
            errors: List of L2 errors for different mesh sizes
            mesh_sizes: List of corresponding mesh sizes (h values)
        
        Returns:
            Tuple of (convergence_rate, correlation_coefficient)
        """
        if len(errors) != len(mesh_sizes) or len(errors) < 2:
            raise ValueError("Need at least 2 error/mesh_size pairs")
        
        # Remove zero or negative errors (log will fail)
        valid_indices = [i for i, err in enumerate(errors) if err > 1e-16]
        if len(valid_indices) < 2:
            warnings.warn("Insufficient valid errors for convergence analysis")
            return 0.0, 0.0
        
        valid_errors = [errors[i] for i in valid_indices]
        valid_mesh_sizes = [mesh_sizes[i] for i in valid_indices]
        
        # Linear regression in log-log space: log(error) = p * log(h) + c
        log_h = np.log(valid_mesh_sizes)
        log_error = np.log(valid_errors)
        
        # Fit: log_error = p * log_h + c
        A = np.vstack([log_h, np.ones(len(log_h))]).T
        p, c = np.linalg.lstsq(A, log_error, rcond=None)[0]
        
        # Correlation coefficient
        correlation = np.corrcoef(log_h, log_error)[0, 1]
        
        return p, correlation
    
    def generate_error_report(self, error_results: Dict) -> str:
        """
        Generate a formatted error analysis report.
        
        Args:
            error_results: Results from compute_trace_error
        
        Returns:
            Formatted string report
        """
        report = []
        report.append("="*60)
        report.append("L2 ERROR ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"Time: {error_results['time']:.6f}")
        report.append(f"Overall Global L2 Error: {error_results['global_error']:.6e}")
        report.append(f"Overall Relative Global Error: {error_results.get('relative_global_error', 'N/A'):.6e}")
        report.append(f"Maximum Pointwise Error: {error_results['max_error']:.6e}")
        report.append("")
        
        # Report global errors per equation
        report.append("Global Errors per Equation:")
        for eq_result in error_results['global_error_per_equation']:
            eq_idx = eq_result['equation_idx']
            report.append(f"  Equation {eq_idx + 1}:")
            report.append(f"    Global L2 Error: {eq_result['global_l2_error']:.6e}")
            report.append(f"    Global Relative Error: {eq_result['global_relative_error']:.6e}")
            report.append(f"    Global Solution Norm: {eq_result['global_solution_norm']:.6e}")
        report.append("")
        
        # Report domain-wise breakdown
        report.append("Domain-wise Breakdown:")
        for domain_result in error_results['domain_errors']:
            domain_idx = domain_result['domain_idx']
            report.append(f"Domain {domain_idx + 1} (Max pointwise error: {domain_result['max_pointwise_error']:.6e}):")
            
            for eq_error in domain_result['equation_errors']:
                eq_idx = eq_error['equation_idx']
                report.append(f"    Equation {eq_idx + 1}:")
                report.append(f"      L2 Error: {eq_error['l2_error']:.6e}")
                report.append(f"      Relative Error: {eq_error['relative_error']:.6e}")
                report.append(f"      Max Pointwise Error: {eq_error['max_pointwise_error']:.6e}")
                report.append(f"      Solution Norm: {eq_error['solution_norm']:.6e}")
            report.append("")
        
        return "\n".join(report)

    def get_equation_error(self, error_results: Dict, domain_idx: int, equation_idx: int) -> Optional[Dict]:
        """
        Get error results for a specific equation in a specific domain.
        
        Args:
            error_results: Results from compute_trace_error
            domain_idx: Domain index
            equation_idx: Equation index
        
        Returns:
            Error dictionary for the specified equation or None if not found
        """
        return error_results['equation_errors'].get((domain_idx, equation_idx), None)
    
    def get_global_equation_error(self, error_results: Dict, equation_idx: int) -> Optional[Dict]:
        """
        Get global error results for a specific equation across all domains.
        
        Args:
            error_results: Results from compute_trace_error
            equation_idx: Equation index
        
        Returns:
            Global error dictionary for the specified equation or None if not found
        """
        if equation_idx < len(error_results['global_error_per_equation']):
            return error_results['global_error_per_equation'][equation_idx]
        return None

    def get_analytical_solutions(self) -> Dict:
        """
        Get the automatically extracted analytical solutions.
        
        Returns:
            Dictionary with analytical functions per domain
        """
        return self.analytical_solutions.copy()
    
    def has_analytical_solution(self, domain_idx: int) -> bool:
        """
        Check if a domain has analytical solutions available.
        
        Args:
            domain_idx: Index of the domain to check
        
        Returns:
            True if analytical solutions are available, False otherwise
        """
        domain_key = f'domain_{domain_idx}'
        return (domain_key in self.analytical_solutions and 
                self.analytical_solutions[domain_key] is not None)

    def compute_bulk_error(self, 
                          bulk_solutions: List, 
                          time: float,
                          analytical_functions: Optional[List[List[Callable]]] = None) -> Dict:
        """
        Compute L2 errors between numerical bulk solutions and analytical solutions.
        Handles discontinuous Galerkin bulk solutions.
        
        Args:
            bulk_solutions: List of BulkData objects for each domain
            time: Current time for analytical solution evaluation
            analytical_functions: Optional list of analytical functions per domain/equation
                                 If None, uses automatically extracted solutions from problems
        
        Returns:
            Dictionary with bulk error metrics per domain and equation
        """
        # Use provided analytical functions or fall back to extracted ones
        if analytical_functions is None:
            analytical_functions = [self.analytical_solutions.get(f'domain_{i}', None) 
                                   for i in range(self.n_domains)]
        
        results = {
            'domain_errors': [],
            'equation_errors': {},  # organized by (domain_idx, eq_idx)
            'global_error_per_equation': [],  # global error for each equation
            'global_error': 0.0,
            'max_error': 0.0,
            'time': time,
            'error_type': 'bulk'
        }
        
        # Track errors per equation across all domains
        max_equations = max(problem.neq for problem in self.problems)
        global_error_squared_per_eq = [0.0] * max_equations
        global_solution_norm_squared_per_eq = [0.0] * max_equations
        max_pointwise_error = 0.0
        
        for domain_idx in range(self.n_domains):
            problem = self.problems[domain_idx]
            discretization = self.discretizations[domain_idx]
            bulk_data = bulk_solutions[domain_idx]
            
            # Extract the numpy array from BulkData object
            bulk_sol = bulk_data.data if hasattr(bulk_data, 'data') else bulk_data
            
            domain_result = self._compute_domain_bulk_error(
                problem, discretization, bulk_sol, time, 
                analytical_functions[domain_idx] if analytical_functions else None
            )
            
            results['domain_errors'].append(domain_result)
            
            # Store individual equation errors with domain/equation indexing
            for eq_error in domain_result['equation_errors']:
                eq_idx = eq_error['equation_idx']
                results['equation_errors'][(domain_idx, eq_idx)] = eq_error
                
                # Accumulate global error per equation
                if eq_idx < len(global_error_squared_per_eq):
                    global_error_squared_per_eq[eq_idx] += eq_error['l2_error_squared']
                    global_solution_norm_squared_per_eq[eq_idx] += eq_error['solution_norm_squared']
            
            max_pointwise_error = max(max_pointwise_error, domain_result['max_pointwise_error'])
        
        # Compute global errors per equation
        for eq_idx in range(max_equations):
            global_l2_error = np.sqrt(global_error_squared_per_eq[eq_idx])
            if global_solution_norm_squared_per_eq[eq_idx] > 1e-14:
                relative_error = np.sqrt(global_error_squared_per_eq[eq_idx] / global_solution_norm_squared_per_eq[eq_idx])
            else:
                relative_error = np.inf
                
            results['global_error_per_equation'].append({
                'equation_idx': eq_idx,
                'global_l2_error': global_l2_error,
                'global_relative_error': relative_error,
                'global_solution_norm': np.sqrt(global_solution_norm_squared_per_eq[eq_idx])
            })
        
        # Overall global error (sum of all equations)
        total_error_squared = sum(global_error_squared_per_eq)
        total_solution_norm_squared = sum(global_solution_norm_squared_per_eq)
        
        results['global_error'] = np.sqrt(total_error_squared)
        results['max_error'] = max_pointwise_error
        
        if total_solution_norm_squared > 1e-14:
            results['relative_global_error'] = np.sqrt(total_error_squared / total_solution_norm_squared)
        else:
            results['relative_global_error'] = np.inf
            
        return results
    
    def _compute_domain_bulk_error(self, 
                                  problem, 
                                  discretization, 
                                  bulk_sol: np.ndarray, 
                                  time: float,
                                  analytical_functions: Optional[List[Callable]] = None) -> Dict:
        """
        Compute L2 error for bulk solution in a single domain.
        Handles discontinuous Galerkin basis functions.
        
        Args:
            problem: Problem object for the domain
            discretization: Spatial discretization for the domain
            bulk_sol: Bulk solution array (2*neq, n_elements) - each column has coefficients for one element
            time: Current time
            analytical_functions: List of analytical functions per equation
        
        Returns:
            Dictionary with domain-specific bulk error metrics
        """
        n_elements = discretization.n_elements
        neq = problem.neq
        
        # Get analytical functions
        if analytical_functions is None:
            analytical_functions = self._get_analytical_functions(problem)
        
        max_pointwise_error = 0.0
        equation_errors = []
        
        # Validate bulk solution structure
        expected_shape = (2 * neq, n_elements)
        if bulk_sol.shape != expected_shape:
            warnings.warn(f"Expected bulk solution shape {expected_shape}, got {bulk_sol.shape}")
            # Try to reshape if possible
            if bulk_sol.size == 2 * neq * n_elements:
                bulk_sol = bulk_sol.reshape(expected_shape)
            else:
                raise ValueError(f"Cannot reshape bulk solution from {bulk_sol.shape} to {expected_shape}")
        
        for eq_idx in range(neq):
            # Extract coefficients for this equation across all elements
            # For equation eq_idx, coefficients are at rows [2*eq_idx, 2*eq_idx+1]
            eq_coeff_row_start = 2 * eq_idx
            eq_coeff_row_end = eq_coeff_row_start + 2
            eq_coeffs = bulk_sol[eq_coeff_row_start:eq_coeff_row_end, :]  # Shape: (2, n_elements)
            
            # Compute error using Gaussian quadrature on each element
            eq_error_squared = 0.0
            eq_solution_norm_squared = 0.0
            eq_max_error = 0.0
            
            # Gaussian quadrature points and weights for [-1, 1]
            gauss_points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            gauss_weights = np.array([1.0, 1.0])
            
            for elem_idx in range(n_elements):
                # Element domain
                x_left = discretization.nodes[elem_idx]
                x_right = discretization.nodes[elem_idx + 1]
                h_elem = x_right - x_left
                
                # Element coefficients for this equation: eq_coeffs[:, elem_idx] -> [coeff0, coeff1]
                elem_coeffs = eq_coeffs[:, elem_idx]  # Shape: (2,)
                
                # Integrate over element using Gaussian quadrature
                for gp, gw in zip(gauss_points, gauss_weights):
                    # Map Gauss point from [-1, 1] to physical element
                    x_phys = 0.5 * ((1 - gp) * x_left + (1 + gp) * x_right)
                    
                    # Evaluate DG basis functions at Gauss point (linear basis)
                    # phi_0(xi) = (1 - xi)/2, phi_1(xi) = (1 + xi)/2
                    phi_0 = (1 - gp) / 2
                    phi_1 = (1 + gp) / 2
                    basis_values = np.array([phi_0, phi_1])
                    
                    # Numerical solution at Gauss point
                    numerical_value = np.dot(elem_coeffs, basis_values)
                    
                    # Analytical solution at Gauss point
                    if analytical_functions and eq_idx < len(analytical_functions):
                        analytical_value = analytical_functions[eq_idx](x_phys, time)
                    else:
                        analytical_value = 0.0
                        if elem_idx == 0:  # Warn only once per equation
                            warnings.warn(f"No analytical solution available for bulk equation {eq_idx}, using zero")
                    
                    # Pointwise error
                    pointwise_error = numerical_value - analytical_value
                    
                    # Accumulate L2 error (with Jacobian = h_elem/2)
                    jacobian = h_elem / 2
                    eq_error_squared += (pointwise_error**2) * gw * jacobian
                    eq_solution_norm_squared += (analytical_value**2) * gw * jacobian
                    
                    # Track maximum pointwise error
                    eq_max_error = max(eq_max_error, abs(pointwise_error))
            
            max_pointwise_error = max(max_pointwise_error, eq_max_error)
            
            # Store equation-specific results
            equation_errors.append({
                'equation_idx': eq_idx,
                'l2_error': np.sqrt(eq_error_squared),
                'l2_error_squared': eq_error_squared,
                'solution_norm': np.sqrt(eq_solution_norm_squared),
                'solution_norm_squared': eq_solution_norm_squared,
                'max_pointwise_error': eq_max_error,
                'relative_error': np.sqrt(eq_error_squared / eq_solution_norm_squared) if eq_solution_norm_squared > 1e-14 else np.inf,
                'bulk_coefficients': eq_coeffs.copy()  # Shape: (2, n_elements)
            })
        
        return {
            'domain_idx': getattr(problem, 'domain_idx', 0),
            'max_pointwise_error': max_pointwise_error,
            'equation_errors': equation_errors,
            'n_elements': n_elements,
            'n_equations': neq,
            'error_type': 'bulk'
        }

def create_analytical_solutions_example() -> Dict:
    """
    THIS FUNCTION IS FOR TESTING PURPOSES ONLY. DO NOT USE IN PRODUCTION.
    Example analytical solutions for testing purposes.
    
    Returns:
        Dictionary with example analytical functions
    """
    warnings.warn(
        "The create_analytical_solutions_example function is for testing purposes only. "
        "Do not use this function in production code. "
        "Use retrieve_analytical_solution() to get real analytical solutions from problem objects.",
        UserWarning,
        stacklevel=2
    )
    
    def pressure_analytical(x: float, t: float) -> float:
        """Example: exponentially decaying pressure wave"""
        return np.exp(-t) * np.sin(np.pi * x)
    
    def flow_analytical(x: float, t: float) -> float:
        """Example: flow derived from pressure gradient"""
        return -np.pi * np.exp(-t) * np.cos(np.pi * x)
    
    return {
        'domain_0': [pressure_analytical, flow_analytical],
        # Add more domains as needed
    }
