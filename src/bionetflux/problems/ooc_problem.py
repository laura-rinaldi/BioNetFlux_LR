#!/usr/bin/env python3
"""
Custom Problem Template
Based on MATLAB TestProblem.m structure with same physical parameters as OoC_grid_new.

This template extracts all the physical parameters and mathematical functions
from the MATLAB reference, with custom grid geometry specified.
"""

import numpy as np
import sys
import os
from typing import Optional

# Handle both relative imports (when used as module) and direct execution
try:
    from ..core.problem import Problem
    from ..core.discretization import Discretization, GlobalDiscretization
    from ..core.constraints import ConstraintManager
    from ..geometry.domain_geometry import DomainGeometry, EXTERIOR_BOUNDARY, build_grid_geometry
except ImportError:
    # If relative imports fail, add the src directory to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, src_dir)
    
    from bionetflux.core.problem import Problem
    from bionetflux.core.discretization import Discretization, GlobalDiscretization
    from bionetflux.core.constraints import ConstraintManager
    from bionetflux.geometry.domain_geometry import DomainGeometry, EXTERIOR_BOUNDARY, build_grid_geometry


def build_default_geometry():
    """
    Build the default OoC grid geometry with vertical segments and horizontal connectors.
    
    Returns:
        DomainGeometry: Default grid geometry instance
    """
    return build_grid_geometry()


def setup_constraints_from_geometry(geometry: DomainGeometry, problems, neq: int) -> ConstraintManager:
    """
    Generate constraint manager from geometry connections.
    
    Args:
        geometry: DomainGeometry instance with connections
        problems: List of Problem instances
        neq: Number of equations per domain
    
    Returns:
        ConstraintManager with appropriate constraints
    """
    print("Setting up constraints from geometry connections...")
    
    constraint_manager = ConstraintManager()
    
    # Get all connections
    boundary_connections = geometry.get_boundary_connections()
    interior_connections = geometry.get_interior_connections()
    
    print(f"  Processing {len(boundary_connections)} boundary connections...")
    
    # Add homogeneous Neumann boundary conditions for all boundary connections
    for conn in boundary_connections:
        domain_idx = conn.domain1_id
        parameter = conn.parameter1
        boundary_type = conn.get_boundary_type()
        
        if conn.is_exterior_boundary():
            # Add homogeneous Neumann BC for all equations at this boundary point
            for eq_idx in range(neq):
                constraint_manager.add_neumann(
                    equation_index=eq_idx,
                    domain_index=domain_idx,
                    position=parameter,
                    data_function=lambda t: 0.0  # Homogeneous Neumann: zero flux
                )
            print(f"    Added homogeneous Neumann BC: domain {domain_idx} @ {parameter:.3f} ({boundary_type})")
    
    print(f"  Processing {len(interior_connections)} interior connections...")
    
    # Add trace continuity constraints for all interior connections
    for conn in interior_connections:
        domain1_idx = conn.domain1_id
        domain2_idx = conn.domain2_id
        parameter1 = conn.parameter1
        parameter2 = conn.parameter2
        
        # Add trace continuity for all equations
        for eq_idx in range(neq):
            try:
                constraint_manager.add_trace_continuity(
                    equation_index=eq_idx,
                    domain1_index=domain1_idx,
                    domain2_index=domain2_idx,
                    position1=parameter1,
                    position2=parameter2
                )
            except Exception as e:
                print(f"    ERROR adding trace continuity eq {eq_idx}, domains {domain1_idx}↔{domain2_idx}: {e}")
        
        print(f"    Added trace continuity: domain {domain1_idx}@{parameter1:.3f} ↔ domain {domain2_idx}@{parameter2:.3f}")
    
    # Calculate constraint statistics
    total_boundary_conditions = len(boundary_connections) * neq
    total_continuity_conditions = len(interior_connections) * neq
    
    print(f"✓ Constraint setup completed from geometry:")
    print(f"  - Boundary conditions: {total_boundary_conditions} ({len(boundary_connections)} locations × {neq} equations)")
    print(f"  - Continuity conditions: {total_continuity_conditions} ({len(interior_connections)} connections × {neq} equations)")
    print(f"  - Total constraints: {total_boundary_conditions + total_continuity_conditions}")
    
    return constraint_manager


def create_global_framework(geometry: Optional[DomainGeometry] = None):
    """
    Custom problem with grid geometry following MATLAB TestProblem.m structure.
    
    Physical parameters and functions match OoC_grid_new exactly.
    Custom grid geometry with vertical segments and horizontal connectors.
    
    Args:
        geometry: Optional pre-defined DomainGeometry instance. If None, creates default geometry.
    
    Returns:
        Tuple: (problems, global_discretization, constraint_manager, problem_name)
    """
    
    # ============================================================================
    # SECTION 1: PHYSICAL PARAMETERS (From MATLAB TestProblem.m)
    # ============================================================================
    
    print("Setting up physical parameters from MATLAB TestProblem.m...")
    
    # Global problem configuration
    neq = 4  # 4-equation OrganOnChip system (u, omega, v, phi)
    problem_name = "OoC_Grid_Problem"
    
    # Time discretization parameters
    T = 1.0     # Final time
    dt = 0.1    # Time step
    
    # Physical parameters following MATLAB TestProblem.m exactly:
    # Viscosity parameters
    nu = 1.0        # MATLAB: nu = 1.
    mu = 2.0        # MATLAB: mu = 2.
    epsilon = 1.0   # MATLAB: epsilon = 1.
    sigma = 1.0     # MATLAB: sigma = 1.
    
    # Reaction parameters  
    a = 0.0     # MATLAB: a = 0.
    c = 0.0     # MATLAB: c = 0.
    
    # Coupling parameters
    b = 1.0     # MATLAB: b = 1.
    d = 1.0     # MATLAB: d = 1.
    chi = 1.0   # MATLAB: chi = 1.
    
    
    # Combine into parameter array (matches MATLAB order)
    parameters = np.array([nu, mu, epsilon, sigma, a, b, c, d, chi])
    
    print(f"✓ Physical parameters configured:")
    print(f"  Viscosity: nu={nu}, mu={mu}, epsilon={epsilon}, sigma={sigma}")
    print(f"  Reactions: a={a}, c={c}")
    print(f"  Coupling: b={b}, d={d}, chi={chi}")
    
    # ============================================================================
    # SECTION 2: MATHEMATICAL FUNCTIONS (From MATLAB TestProblem.m)
    # ============================================================================
    
    print("Defining mathematical functions from MATLAB TestProblem.m...")

    def constant_function(x):
        """Constant function returning ones, matching MATLAB constant_function"""
        return np.ones_like(x)
    
    # Nonlinear coupling function (MATLAB: lambda = @(x) constant_function(x))
    def lambda_func(x):
        """Nonlinear coupling function lambda(x) - matches MATLAB constant_function"""
        return np.ones_like(x)  # Constant function
    
    def dlambda_func(x):
        """Derivative of lambda function"""
        return np.zeros_like(x)  # Derivative of constant is zero
    
    # Initial conditions (MATLAB: problem.u0{i})
    def initial_u(s, t=0.0):
        """Initial condition for u (equation 1) - MATLAB: sin(2*pi*x)"""
        return np.sin(2 * np.pi * s)
    
    def initial_omega(s, t=0.0):
        """Initial condition for omega (equation 2) - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    def initial_v(s, t=0.0):
        """Initial condition for v (equation 3) - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    def initial_phi(s, t=0.0):
        """Initial condition for phi (equation 4) - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    # Source terms (MATLAB: force{i})
    def source_u(s, t):
        """Source term for u equation - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    def source_omega(s, t):
        """Source term for omega equation - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    def source_v(s, t):
        """Source term for v equation - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    def source_phi(s, t):
        """Source term for phi equation - MATLAB: zeros(size(x))"""
        return np.zeros_like(s)
    
    print("✓ Mathematical functions defined:")
    print("  - Nonlinear coupling: lambda_func (constant), dlambda_func")
    print("  - Initial: u=sin(2πs), ω=0, v=0, φ=0 matching MATLAB")
    print("  - Sources: all zero functions matching MATLAB")
    
    # ============================================================================
    # SECTION 3: GEOMETRY HANDLING (NEW GEOMETRY-FIRST APPROACH)
    # ============================================================================
    
    if geometry is not None:
        # Use provided geometry
        print(f"Using provided geometry: '{geometry.name}' with {geometry.num_domains()} domains")
        print(f"  - Connections: {geometry.num_connections()}")
        
        # Validate provided geometry
        if not geometry.validate_geometry(verbose=True):
            print("⚠️  Warning: Provided geometry validation failed")
        else:
            print("✓ Provided geometry validation passed")
            
    else:
        # Create default custom grid geometry with connections
        geometry = build_default_geometry()
        
        # Validate default geometry
        if not geometry.validate_geometry(verbose=True):
            print("⚠️  Warning: Default geometry validation failed")
        else:
            print("✓ Default geometry validation passed")
    
    # ============================================================================
    # SECTION 4: PROBLEM CREATION (Geometry-based, uniform parameters)
    # ============================================================================
    
    print("Creating problem instances from geometry with uniform parameters...")
    
    problems = []
    discretizations = []
    
    # Apply uniform parameters to all domains based on geometry
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        print(f"  Creating problem for domain {domain_id}: {domain_info.name}")
        
        # Create problem for this domain with uniform MATLAB-compatible parameters
        problem = Problem(
            neq=neq,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            parameters=parameters,  # Same parameters for all domains
            problem_type="organ_on_chip",  # 4-equation OoC system
            name=f"{problem_name}_{domain_info.name}"
        )
        
        # Set uniform initial conditions for all domains (matches MATLAB problem.u0{i})
        problem.set_initial_condition(0, lambda s, t=0: 0.0 * constant_function(s))
        problem.set_initial_condition(1, lambda s, t=0: 0.0 * constant_function(s))
        problem.set_initial_condition(2, lambda s, t=0: 0.0 * constant_function(s))
        problem.set_initial_condition(3, lambda s, t=0: 0.0 * constant_function(s))
        
        # Set uniform source terms for all domains (matches MATLAB force{i})
        problem.set_force(0, lambda s, t: 0.0 * constant_function(s))
        problem.set_force(1, lambda s, t: 0.0 * constant_function(s))
        problem.set_force(2, lambda s, t: 0.0 * constant_function(s))
        problem.set_force(3, lambda s, t: 0.0 * constant_function(s))
        
        # Set uniform nonlinear coupling for all domains (matches MATLAB lambda)
        problem.set_chemotaxis(lambda_func, dlambda_func)
        
        # Set 2D coordinates for visualization from geometry
        problem.set_extrema(domain_info.extrema_start, domain_info.extrema_end)
        
        problems.append(problem)
        
        # Create discretization for this domain with uniform mesh density
        discretization = Discretization(
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            n_elements=20  # Uniform mesh density for all domains
        )

        # Set uniform stabilization parameters for all domains
        discretization.set_tau([0.5, 0.5, 0.5, 0.5])
        discretizations.append(discretization)
    
    # Set specific initial conditions for some domains (from original implementation)
    if len(problems) > 0:
        problems[0].set_initial_condition(2, lambda s, t=0: constant_function(s))
    if len(problems) > 3:
        problems[3].set_initial_condition(0, lambda s, t=0: constant_function(s))
    
    print(f"✓ Created {len(problems)} problem instances with uniform parameters")
    
    # ============================================================================
    # SECTION 5: CONSTRAINT SETUP (NEW GEOMETRY-DRIVEN APPROACH)
    # ============================================================================
    
    constraint_manager = setup_constraints_from_geometry(geometry, problems, neq)
    
    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)
    
    # ============================================================================
    # SECTION 6: GLOBAL DISCRETIZATION AND FINALIZATION
    # ============================================================================
    
    print("Finalizing global discretization...")
    
    global_discretization = GlobalDiscretization(
        spatial_discretizations=discretizations,
    )
    
    global_discretization.set_time_parameters(dt, T)
    print(f"✓ Global discretization created:")
    print(f"  Time: dt={dt}, T={T}, steps={int(T/dt)}")
    print(f"  Domains: {len(discretizations)}")
    print(f"  Constraints: {constraint_manager.n_constraints}")
    print(f"  Multipliers: {constraint_manager.n_multipliers}")
    
    # ============================================================================
    # SECTION 7: VALIDATION AND RETURN
    # ============================================================================
    
    print("Validating problem setup...")
    
    # Validate each problem
    for i, problem in enumerate(problems):
        if not problem.validate_problem(verbose=False):
            print(f"⚠️  Warning: Problem {i} validation failed")
        else:
            print(f"✓ Problem {i} validated")
    
    print(f"\n🎉 OoC problem '{problem_name}' ready!")
    print(f"📊 PROBLEM SUMMARY:")
    print(f"   - Geometry: {geometry.name} with {geometry.num_domains()} domains")
    print(f"   - Physics: Uniform OrganOnChip parameters across all domains") 
    print(f"   - {neq} equations per domain ({neq * len(problems)} total equations)")
    print(f"   - {constraint_manager.n_constraints} constraints")
    print(f"   - Total DOFs: {sum(p.neq * (d.n_elements + 1) for p, d in zip(problems, global_discretization.spatial_discretizations))}")
    
    return problems, global_discretization, constraint_manager, problem_name


def main():
    """Test the grid problem."""
    print("="*60)
    print("REDUCED OOC GRID PROBLEM")
    print("="*60)
    print("Custom grid geometry with MATLAB TestProblem.m physics")
    print()
    
    try:
        problems, global_disc, constraints, name = create_global_framework()
        print(f"\n✅ Grid problem creation successful!")
        print(f"   Problem: {name}")
        print(f"   Domains: {len(problems)}")
        print(f"   Equations per domain: {problems[0].neq if problems else 0}")
        print(f"   Total DOFs: {sum(p.neq * (d.n_elements + 1) for p, d in zip(problems, global_disc.spatial_discretizations))}")
        
        # Additional geometry analysis
        print(f"\n📊 GEOMETRY ANALYSIS:")
        
        # Count domain types
        vertical_count = sum(1 for p in problems if 'vertical' in p.name)
        horizontal_count = len(problems) - vertical_count
        lower_count = sum(1 for p in problems if 'lower' in p.name)
        upper_count = sum(1 for p in problems if 'upper' in p.name)
        
        print(f"   - Vertical domains: {vertical_count}")
        print(f"   - Horizontal domains: {horizontal_count}")
        print(f"     - Lower connectors: {lower_count}")
        print(f"     - Upper connectors: {upper_count}")
        
        # Domain name mapping
        print(f"\n📋 DOMAIN INDEX MAPPING:")
        for i, problem in enumerate(problems):
            domain_type = "VERTICAL" if 'vertical' in problem.name else "HORIZONTAL"
            print(f"   Domain {i:2d}: {domain_type:10s} - {problem.name}")
        
    except Exception as e:
        print(f"\n❌ Grid problem creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
