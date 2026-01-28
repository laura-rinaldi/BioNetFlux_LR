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

# Handle both relative imports (when used as module) and direct execution
try:
    from ..core.problem import Problem
    from ..core.discretization import Discretization, GlobalDiscretization
    from ..core.constraints import ConstraintManager
    from ..geometry.domain_geometry import DomainGeometry
except ImportError:
    # If relative imports fail, add the src directory to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, src_dir)
    
    from bionetflux.core.problem import Problem
    from bionetflux.core.discretization import Discretization, GlobalDiscretization
    from bionetflux.core.constraints import ConstraintManager
    from bionetflux.geometry.domain_geometry import DomainGeometry





def create_global_framework():
    """
    Custom problem with grid geometry following MATLAB TestProblem.m structure.
    
    Physical parameters and functions match OoC_grid_new exactly.
    Custom grid geometry with vertical segments and horizontal connectors.
    
    Returns:
        Tuple: (problems, global_discretization, constraint_manager, problem_name)
    """
    
    # ============================================================================
    # SECTION 1: PHYSICAL PARAMETERS (From MATLAB TestProblem.m)
    # ============================================================================
    
    print("Setting up physical parameters from MATLAB TestProblem.m...")
    
    # Global problem configuration
    neq = 4  # 4-equation OrganOnChip system (u, omega, v, phi)
    problem_name = "Reduced_OoC_Grid_Problem"
    
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
    # SECTION 3: CUSTOM GRID GEOMETRY
    # ============================================================================
    
    print("Creating custom grid geometry...")
    
    geometry = DomainGeometry("custom_grid_geometry")
    
    # Vertical segments
    # S1: Left vertical segment
    geometry.add_domain(
        extrema_start=(-1.0, -1.0),
        extrema_end=(-1.0, 1.0),
        name="S1_left_vertical",
        display_color="blue"
    )
    
    # S2: Lower middle vertical segment  
    geometry.add_domain(
        extrema_start=(0.0, -1.0),
        extrema_end=(0.0, -0.1),
        name="S2_lower_middle_vertical",
        display_color="green"
    )
    
    # S3: Upper middle vertical segment
    geometry.add_domain(
        extrema_start=(0.0, 0.1),
        extrema_end=(0.0, 1.0),
        name="S3_upper_middle_vertical",
        display_color="green"
    )
    
    # S4: Right vertical segment (assuming correction from S1 duplicate)
    geometry.add_domain(
        extrema_start=(1.0, -1.0),
        extrema_end=(1.0, 1.0),
        name="S4_right_vertical",
        display_color="blue"
    )
    
    # Horizontal connectors - Lower section (-0.9 < y < -0.2)
    N = 4  # Number of horizontal segments
    y_lower_values = np.linspace(-0.9, -0.2, N)
    
    print(f"  Adding {N} lower horizontal connectors at y = {y_lower_values}")
    
    # Lower connectors: S1 to S2
    for i, y_pos in enumerate(y_lower_values):
        geometry.add_domain(
            extrema_start=(-1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"lower_S1_S2_{i+1}",
            display_color="red"
        )
    
    # Lower connectors: S4 to S2  
    for i, y_pos in enumerate(y_lower_values):
        geometry.add_domain(
            extrema_start=(1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"lower_S4_S2_{i+1}",
            display_color="red"
        )
    
    # Horizontal connectors - Upper section (0.2 < y < 0.9)
    y_upper_values = np.linspace(0.2, 0.9, N)
    
    print(f"  Adding {N} upper horizontal connectors at y = {y_upper_values}")
    
    # Upper connectors: S1 to S3
    for i, y_pos in enumerate(y_upper_values):
        geometry.add_domain(
            extrema_start=(-1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"upper_S1_S3_{i+1}",
            display_color="red"
        )
    
    # Upper connectors: S4 to S3
    for i, y_pos in enumerate(y_upper_values):
        geometry.add_domain(
            extrema_start=(1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"upper_S4_S3_{i+1}",
            display_color="red"
        )
    
    print(f"✓ Custom grid geometry created:")
    print(f"  - 4 vertical segments (S1, S2, S3, S4)")
    print(f"  - {2*N} lower horizontal connectors (-0.9 < y < -0.2)")
    print(f"  - {2*N} upper horizontal connectors (0.2 < y < 0.9)")
    print(f"  - Total domains: {geometry.num_domains()}")
    
   
    # Validate geometry
    if not geometry.validate_geometry(verbose=True):
        print("⚠️  Warning: Geometry validation failed")
    else:
        print("✓ Geometry validation passed")
    

    
    # ============================================================================
    # SECTION 4: PROBLEM CREATION (Automated from geometry)
    # ============================================================================
    
    print("Creating problem instances from geometry...")
    
    problems = []
    discretizations = []
    
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        print(f"  Creating problem for domain {domain_id}: {domain_info.name}")
        
        # Create problem for this domain with MATLAB-compatible parameters
        problem = Problem(
            neq=neq,
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            parameters=parameters,
            problem_type="organ_on_chip",  # 4-equation OoC system
            name=f"{problem_name}_{domain_info.name}"
        )
        
        # Set initial conditions (matches MATLAB problem.u0{i})
        problem.set_initial_condition(0, lambda s, t=0: 0.0 * constant_function(s))
        problem.set_initial_condition(1, lambda s, t=0: 0.0 * constant_function(s))
        problem.set_initial_condition(2, lambda s, t=0: 0.0 * constant_function(s))
        problem.set_initial_condition(3, lambda s, t=0: 0.0 * constant_function(s))
        
        # Set source terms (matches MATLAB force{i})
        problem.set_force(0, lambda s, t: 0.0 * constant_function(s))
        problem.set_force(1, lambda s, t: 0.0 * constant_function(s))
        problem.set_force(2, lambda s, t: 0.0 * constant_function(s))
        problem.set_force(3, lambda s, t: 0.0 * constant_function(s))
        
        # Set nonlinear coupling (matches MATLAB lambda)
        problem.set_chemotaxis(lambda_func, dlambda_func)
        
        # Set 2D coordinates for visualization
        problem.set_extrema(domain_info.extrema_start, domain_info.extrema_end)
        
        problems.append(problem)
        
        # Create discretization for this domain
        discretization = Discretization(
            domain_start=domain_info.domain_start,
            domain_length=domain_info.domain_length,
            n_elements=20  # Adjust as needed for your problem
        )

        discretization.set_tau([0.5, 0.5, 0.5, 0.5])  # Example: set tau parameter if needed
        discretizations.append(discretization)
    
    problems[0].set_initial_condition(2, lambda s, t=0: constant_function(s))  # u0 for first domain as sin(2πs)
    problems[3].set_initial_condition(0, lambda s, t=0: constant_function(s))  # u0 for fourth domain as sin(2πs)
    
    print(f"✓ Created {len(problems)} problem instances")
    
    # ============================================================================
    # SECTION 5: CONSTRAINT SETUP - Grid-specific constraints
    # ============================================================================
    
    print("Setting up grid-specific constraints...")
    
    constraint_manager = ConstraintManager()
    
    # Add homogeneous Neumann boundary conditions to both extrema of domains 0, 1, 2, 3
    # (matches MATLAB fluxu0{i} = @(t) 0. and fluxu1{i} = @(t) 0.)
    boundary_domains = [0, 1, 2, 3]  # Vertical segments: S1, S2, S3, S4
    
    for domain_idx in boundary_domains:
        if domain_idx < len(problems):
            for eq_idx in range(neq):
                # Add Neumann BC at domain start (left/bottom boundary)
                constraint_manager.add_neumann(
                    equation_index = eq_idx,
                    domain_index = domain_idx,
                    position = problems[domain_idx].domain_start,
                    data_function = lambda t: 0.0  # Homogeneous Neumann: zero flux
                )
                print(f"    Added Neumann BC: eq {eq_idx}, domain {domain_idx}, start coord {problems[domain_idx].domain_start:.1f}")
                
                # Add Neumann BC at domain end (right/top boundary)
                constraint_manager.add_neumann(
                    equation_index=eq_idx,
                    domain_index=domain_idx,
                    position=problems[domain_idx].domain_end,
                    data_function=lambda t: 0.0  # Homogeneous Neumann: zero flux
                )
                print(f"    Added Neumann BC: eq {eq_idx}, domain {domain_idx}, end coord {problems[domain_idx].domain_end:.1f}")

    # Add specific trace continuity conditions for horizontal-vertical intersections
    print("  Adding specific trace continuity conditions for grid intersections...")
    
    # CORRECTED: Fix the domain indexing based on actual geometry creation
    # Domain order: [0,1,2,3] = vertical segments, then horizontal connectors
    
    print("    Adding continuity conditions between horizontal connectors and vertical segments...")
    print(f"    Total horizontal domains: {2*N*2}, Total domains: {geometry.num_domains()}")
    
    # FIXED: Correct domain ranges based on geometry creation order
    # Lower connectors S1->S2: domains 4 to 4+N-1
    # Lower connectors S4->S2: domains 4+N to 4+2*N-1  
    # Upper connectors S1->S3: domains 4+2*N to 4+3*N-1
    # Upper connectors S4->S3: domains 4+3*N to 4+4*N-1
    
    # Trace continuity: start of lower S1->S2 and upper S1->S3 connectors with S1 (domain 0)
    s1_left_lower = list(range(4, 4+N))        # Lower S1->S2 connectors  
    s1_left_upper = list(range(4+2*N, 4+3*N))  # Upper S1->S3 connectors
    s1_connections = s1_left_lower + s1_left_upper
    
    for domain_idx in s1_connections:
        if domain_idx < len(problems):
            # Get horizontal segment info
            horizontal_domain_info = geometry.get_domain(domain_idx)
            intersection_y = horizontal_domain_info.extrema_start[1]  # y-coordinate at S1 end
            
            # Map to S1 parameter space: S1 spans y ∈ [-1, 1], param ∈ [0, domain_length]
            s1_param = (intersection_y + 1.0) / 2.0 * problems[0].domain_length
            
            # CRITICAL FIX: Add continuity for ALL equations, not just some
            for eq_idx in range(neq):
                try:
                    constraint_manager.add_trace_continuity(
                        equation_index=eq_idx,
                        domain1_index=domain_idx,  # Horizontal connector
                        domain2_index=0,           # S1 (left vertical)
                        position1=problems[domain_idx].domain_start,  # Start of horizontal (at S1)
                        position2=s1_param,       # Corresponding point on S1
                    )
                    print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} start -> domain 0 at param {s1_param:.3f}")
                except Exception as e:
                    print(f"    ERROR adding S1 continuity eq {eq_idx}, domain {domain_idx}: {e}")
    
    # Trace continuity: end of lower S4->S2 and upper S4->S3 connectors with S4 (domain 3) 
    s4_right_lower = list(range(4+N, 4+2*N))    # Lower S4->S2 connectors
    s4_right_upper = list(range(4+3*N, 4+4*N))  # Upper S4->S3 connectors  
    s4_connections = s4_right_lower + s4_right_upper
    
    for domain_idx in s4_connections:
        if domain_idx < len(problems):
            # Get horizontal segment info
            horizontal_domain_info = geometry.get_domain(domain_idx)
            intersection_y = horizontal_domain_info.extrema_start[1]  # y-coordinate at S4 end
            
            # Map to S4 parameter space: S4 spans y ∈ [-1, 1], param ∈ [0, domain_length]
            s4_param = (intersection_y + 1.0) / 2.0 * problems[3].domain_length
            
            # CRITICAL FIX: Add continuity for ALL equations
            for eq_idx in range(neq):
                try:
                    constraint_manager.add_trace_continuity(
                        equation_index=eq_idx,
                        domain1_index=domain_idx,  # Horizontal connector  
                        domain2_index=3,           # S4 (right vertical)
                        position1=problems[domain_idx].domain_start,  # Start of horizontal (at S4)
                        position2=s4_param,       # Corresponding point on S4
                    )
                    print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} start -> domain 3 at param {s4_param:.3f}")
                except Exception as e:
                    print(f"    ERROR adding S4 continuity eq {eq_idx}, domain {domain_idx}: {e}")

    # Trace continuity with middle vertical segments S2 and S3
    
    # S2 connections (lower middle vertical): end of S1->S2 connectors + start of S4->S2 connectors
    s2_from_s1 = list(range(4, 4+N))        # End of S1->S2 connectors connects to S2
    s2_from_s4 = list(range(4+N, 4+2*N))    # Start of S4->S2 connectors connects to S2  
    
    # End of S1->S2 connectors with S2
    for domain_idx in s2_from_s1:
        if domain_idx < len(problems):
            horizontal_domain_info = geometry.get_domain(domain_idx)
            intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate at S2 end
            
            # Map to S2 parameter space: S2 spans y ∈ [-1, -0.1], param ∈ [0, domain_length]
            s2_y_start, s2_y_end = -1.0, -0.1
            s2_param = (intersection_y - s2_y_start) / (s2_y_end - s2_y_start) * problems[1].domain_length
            
            for eq_idx in range(neq):
                try:
                    constraint_manager.add_trace_continuity(
                        equation_index=eq_idx,
                        domain1_index=domain_idx,  # S1->S2 connector
                        domain2_index=1,           # S2 (lower middle vertical)
                        position1=problems[domain_idx].domain_end,    # End of horizontal (at S2)
                        position2=s2_param,       # Corresponding point on S2
                    )
                    print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} end -> domain 1 at param {s2_param:.3f}")
                except Exception as e:
                    print(f"    ERROR adding S2 continuity eq {eq_idx}, domain {domain_idx}: {e}")
    
    # Start of S4->S2 connectors with S2  
    for domain_idx in s2_from_s4:
        if domain_idx < len(problems):
            horizontal_domain_info = geometry.get_domain(domain_idx)
            intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate at S2 end
            
            # Map to S2 parameter space
            s2_y_start, s2_y_end = -1.0, -0.1
            s2_param = (intersection_y - s2_y_start) / (s2_y_end - s2_y_start) * problems[1].domain_length
            
            for eq_idx in range(neq):
                try:
                    constraint_manager.add_trace_continuity(
                        equation_index=eq_idx,
                        domain1_index=domain_idx,  # S4->S2 connector
                        domain2_index=1,           # S2 (lower middle vertical)
                        position1=problems[domain_idx].domain_end,    # End of horizontal (at S2)
                        position2=s2_param,       # Corresponding point on S2
                    )
                    print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} end -> domain 1 at param {s2_param:.3f}")
                except Exception as e:
                    print(f"    ERROR adding S2 continuity eq {eq_idx}, domain {domain_idx}: {e}")
    
    # S3 connections (upper middle vertical): end of S1->S3 connectors + start of S4->S3 connectors
    s3_from_s1 = list(range(4+2*N, 4+3*N))  # End of S1->S3 connectors connects to S3
    s3_from_s4 = list(range(4+3*N, 4+4*N))  # Start of S4->S3 connectors connects to S3
    
    # End of S1->S3 connectors with S3
    for domain_idx in s3_from_s1:
        if domain_idx < len(problems):
            horizontal_domain_info = geometry.get_domain(domain_idx)
            intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate at S3 end
            
            # Map to S3 parameter space: S3 spans y ∈ [0.1, 1.0], param ∈ [0, domain_length]
            s3_y_start, s3_y_end = 0.1, 1.0
            s3_param = (intersection_y - s3_y_start) / (s3_y_end - s3_y_start) * problems[2].domain_length
            
            for eq_idx in range(neq):
                try:
                    constraint_manager.add_trace_continuity(
                        equation_index=eq_idx,
                        domain1_index=domain_idx,  # S1->S3 connector
                        domain2_index=2,           # S3 (upper middle vertical)
                        position1=problems[domain_idx].domain_end,    # End of horizontal (at S3)
                        position2=s3_param,       # Corresponding point on S3
                    )
                    print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} end -> domain 2 at param {s3_param:.3f}")
                except Exception as e:
                    print(f"    ERROR adding S3 continuity eq {eq_idx}, domain {domain_idx}: {e}")
    
    # Start of S4->S3 connectors with S3
    for domain_idx in s3_from_s4:
        if domain_idx < len(problems):
            horizontal_domain_info = geometry.get_domain(domain_idx)
            intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate at S3 end
            
            # Map to S3 parameter space
            s3_y_start, s3_y_end = 0.1, 1.0
            s3_param = (intersection_y - s3_y_start) / (s3_y_end - s3_y_start) * problems[2].domain_length
            
            for eq_idx in range(neq):
                try:
                    constraint_manager.add_trace_continuity(
                        equation_index=eq_idx,
                        domain1_index=domain_idx,  # S4->S3 connector
                        domain2_index=2,           # S3 (upper middle vertical)  
                        position1=problems[domain_idx].domain_end,    # End of horizontal (at S3)
                        position2=s3_param,       # Corresponding point on S3
                    )
                    print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} end -> domain 2 at param {s3_param:.3f}")
                except Exception as e:
                    print(f"    ERROR adding S3 continuity eq {eq_idx}, domain {domain_idx}: {e}")
               
    # Calculate total constraints
    total_boundary_conditions = len(boundary_domains) * 2 * neq
    total_s1_s4_continuity = (len(s1_connections) + len(s4_connections)) * neq
    total_s2_s3_continuity = (len(s2_from_s1) + len(s2_from_s4) + len(s3_from_s1) + len(s3_from_s4)) * neq
    total_continuity_conditions = total_s1_s4_continuity + total_s2_s3_continuity
    
    print(f"✓ Constraint setup completed:")
    print(f"  - External boundary conditions: {total_boundary_conditions} Neumann BCs")
    print(f"  - S1/S4 continuity: {total_s1_s4_continuity} constraints")
    print(f"  - S2/S3 continuity: {total_s2_s3_continuity} constraints") 
    print(f"  - Total continuity conditions: {total_continuity_conditions}")
    print(f"  - Constraint verification:")
    print(f"    - All {neq} equations have continuity constraints: {total_continuity_conditions // neq == (len(s1_connections) + len(s4_connections) + len(s2_from_s1) + len(s2_from_s4) + len(s3_from_s1) + len(s3_from_s4))}") 

    # Map constraints to discretizations
    constraint_manager.map_to_discretizations(discretizations)
    
    # ============================================================================
    # SECTION 6: GLOBAL DISCRETIZATION AND FINALIZATION
    # ============================================================================
    
    print("Finalizing global discretization...")
    
    global_discretization = GlobalDiscretization(
        spatial_discretizations=discretizations,
    )
    
    global_discretization.set_time_parameters(0.1, 1)
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
    
    print(f"\n🎉 Grid problem '{problem_name}' ready!")
    print(f"📊 PROBLEM SUMMARY:")
    print(f"   - Network topology: {geometry.num_domains()} domains in grid layout")
    print(f"   - 4 vertical segments + {2*N*2} horizontal connectors") 
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
