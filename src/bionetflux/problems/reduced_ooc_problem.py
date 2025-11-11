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

# Import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    PLOTTING_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Matplotlib not available, geometry plotting disabled")
    PLOTTING_AVAILABLE = False


def plot_geometry_with_indices(geometry, save_filename=None, show_plot=True):
    """
    Plot the geometry with domain indices labeled on each segment.
    
    Args:
        geometry: DomainGeometry instance
        save_filename: Optional filename to save the plot
        show_plot: Whether to display the plot interactively
    """
    if not PLOTTING_AVAILABLE:
        print("⚠️  Cannot plot geometry - matplotlib not available")
        return
    
    print("Creating geometry plot with domain indices...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors for different domain types
    vertical_color = 'blue'
    horizontal_lower_color = 'red'
    horizontal_upper_color = 'green'
    
    # Plot each domain
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        
        # Extract coordinates
        x_start, y_start = domain_info.extrema_start
        x_end, y_end = domain_info.extrema_end
        
        # Determine domain type and color
        if 'vertical' in domain_info.name:
            color = vertical_color
            linewidth = 3
        elif 'lower' in domain_info.name:
            color = horizontal_lower_color
            linewidth = 2
        elif 'upper' in domain_info.name:
            color = horizontal_upper_color
            linewidth = 2
        else:
            color = 'black'
            linewidth = 1
        
        # Plot the segment
        ax.plot([x_start, x_end], [y_start, y_end], 
               color=color, linewidth=linewidth, alpha=0.7)
        
        # Add domain index label at the midpoint
        mid_x = (x_start + x_end) / 2
        mid_y = (y_start + y_end) / 2
        
        # Offset text slightly to avoid overlap with line
        if abs(x_end - x_start) > abs(y_end - y_start):  # Horizontal segment
            text_offset_x = 0.0
            text_offset_y = 0.05
        else:  # Vertical segment
            text_offset_x = 0.05
            text_offset_y = 0.0
        
        ax.text(mid_x + text_offset_x, mid_y + text_offset_y, 
               str(domain_id), 
               fontsize=10, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add grid points
    all_points = set()
    for domain_id in range(geometry.num_domains()):
        domain_info = geometry.get_domain(domain_id)
        all_points.add(domain_info.extrema_start)
        all_points.add(domain_info.extrema_end)
    
    # Plot grid points
    for point in all_points:
        ax.plot(point[0], point[1], 'ko', markersize=6, alpha=0.6)
    
    # Formatting
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title(f'Grid Geometry: {geometry.name}\n{geometry.num_domains()} domains', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=vertical_color, linewidth=3, label='Vertical segments'),
        plt.Line2D([0], [0], color=horizontal_lower_color, linewidth=2, label='Lower connectors'),
        plt.Line2D([0], [0], color=horizontal_upper_color, linewidth=2, label='Upper connectors'),
        plt.Line2D([0], [0], marker='o', color='black', linewidth=0, 
                  markersize=6, label='Grid points')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add geometry summary as text
    bounding_box = geometry.get_bounding_box()
    summary_text = (f"Domains: {geometry.num_domains()}\n"
                   f"X range: [{bounding_box['x_min']:.1f}, {bounding_box['x_max']:.1f}]\n"
                   f"Y range: [{bounding_box['y_min']:.1f}, {bounding_box['y_max']:.1f}]")
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if requested
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Geometry plot saved as: {save_filename}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


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
        name="S1_left_vertical"
    )
    
    # S2: Lower middle vertical segment  
    geometry.add_domain(
        extrema_start=(0.0, -1.0),
        extrema_end=(0.0, -0.1),
        name="S2_lower_middle_vertical"
    )
    
    # S3: Upper middle vertical segment
    geometry.add_domain(
        extrema_start=(0.0, 0.1),
        extrema_end=(0.0, 1.0),
        name="S3_upper_middle_vertical"
    )
    
    # S4: Right vertical segment (assuming correction from S1 duplicate)
    geometry.add_domain(
        extrema_start=(1.0, -1.0),
        extrema_end=(1.0, 1.0),
        name="S4_right_vertical"
    )
    
    # Horizontal connectors - Lower section (-0.9 < y < -0.2)
    N = 6  # Number of horizontal segments
    y_lower_values = np.linspace(-0.9, -0.2, N)
    
    print(f"  Adding {N} lower horizontal connectors at y = {y_lower_values}")
    
    # Lower connectors: S1 to S2
    for i, y_pos in enumerate(y_lower_values):
        geometry.add_domain(
            extrema_start=(-1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"lower_S1_S2_{i+1}"
        )
    
    # Lower connectors: S4 to S2  
    for i, y_pos in enumerate(y_lower_values):
        geometry.add_domain(
            extrema_start=(1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"lower_S4_S2_{i+1}"
        )
    
    # Horizontal connectors - Upper section (0.2 < y < 0.9)
    y_upper_values = np.linspace(0.2, 0.9, N)
    
    print(f"  Adding {N} upper horizontal connectors at y = {y_upper_values}")
    
    # Upper connectors: S1 to S3
    for i, y_pos in enumerate(y_upper_values):
        geometry.add_domain(
            extrema_start=(-1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"upper_S1_S3_{i+1}"
        )
    
    # Upper connectors: S4 to S3
    for i, y_pos in enumerate(y_upper_values):
        geometry.add_domain(
            extrema_start=(1.0, y_pos),
            extrema_end=(0.0, y_pos),
            name=f"upper_S4_S3_{i+1}"
        )
    
    print(f"✓ Custom grid geometry created:")
    print(f"  - 4 vertical segments (S1, S2, S3, S4)")
    print(f"  - {2*N} lower horizontal connectors (-0.9 < y < -0.2)")
    print(f"  - {2*N} upper horizontal connectors (0.2 < y < 0.9)")
    print(f"  - Total domains: {geometry.num_domains()}")
    
    # Plot geometry with domain indices
    try:
        plot_geometry_with_indices(
            geometry, 
            save_filename="reduced_ooc_grid_geometry.png",
            show_plot=True
        )
        print("✓ Geometry plot created with domain indices")
    except Exception as e:
        print(f"⚠️  Warning: Could not create geometry plot: {e}")
    
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
        problem.set_initial_condition(0, lambda s: 0.0 * constant_function(s))
        problem.set_initial_condition(1, lambda s: 0.0 * constant_function(s))
        problem.set_initial_condition(2, lambda s: 0.0 * constant_function(s))
        problem.set_initial_condition(3, lambda s: 0.0 * constant_function(s))
        
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
            n_elements=10  # Adjust as needed for your problem
        )

        discretization.set_tau([0.5, 0.5, 0.5, 0.5])  # Example: set tau parameter if needed
        discretizations.append(discretization)
    
    problems[0].set_initial_condition(2, constant_function)  # u0 for first domain as sin(2πs)
    problems[3].set_initial_condition(0, constant_function)  # u0 for fourth domain as sin(2πs)
    
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
    
    # Trace continuity: start of domains 4-8 and 14-18 with corresponding point of domain 0 (S1)
    
    print("    Adding continuity conditions between horizontal connectors and vertical segments...")
    print(f"    Total horizontal domains: {2*N}, Total domains: {geometry.num_domains()}")
    s1_connections = list(range(4, 4+N)) + list(range(4+2*N, 4+3*N))  # domains 4-8 and 14-18
    
    for domain_idx in s1_connections:
        if domain_idx < len(problems):
            for eq_idx in range(neq):
                # Find the corresponding y-coordinate on domain 0 (S1)
                # Domain 0 is vertical from (-1,-1) to (-1,1), so we need to map the intersection point
                horizontal_domain_info = geometry.get_domain(domain_idx)
                intersection_y = horizontal_domain_info.extrema_start[1]  # y-coordinate of horizontal segment
                
                # Map y-coordinate to parameter space on domain 0 (S1)
                # S1 goes from y=-1 (param=0) to y=1 (param=domain_length)
                s1_param = (intersection_y + 1.0) / 2.0 * problems[0].domain_length
                
                constraint_manager.add_trace_continuity(
                    equation_index=eq_idx,
                    domain1_index=domain_idx,
                    domain2_index=0,  # S1 domain
                    position1=problems[domain_idx].domain_start,  # Start of horizontal segment
                    position2=s1_param,  # Corresponding point on S1
                )
                print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} start -> domain 0 at param {s1_param:.3f}")
    
    # Trace continuity: end of domains 9-13 and 19-23 with corresponding point of domain 3 (S4)
    s4_connections = list(range(4+N, 4+2*N)) + list(range(4+3*N, 4+4*N))  # domains 9-13 and 19-23
    
    for domain_idx in s4_connections:
        if domain_idx < len(problems):
            for eq_idx in range(neq):
                # Find the corresponding y-coordinate on domain 3 (S4)
                horizontal_domain_info = geometry.get_domain(domain_idx)
                intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate of horizontal segment
                
                # Map y-coordinate to parameter space on domain 3 (S4)
                # S4 goes from y=-1 (param=0) to y=1 (param=domain_length)
                s4_param = (intersection_y + 1.0) / 2.0 * problems[3].domain_length
                
                constraint_manager.add_trace_continuity(
                    equation_index=eq_idx,
                    domain1_index=domain_idx,
                    domain2_index=3,  # S4 domain
                    position1=problems[domain_idx].domain_end,  # End of horizontal segment
                    position2=s4_param,  # Corresponding point on S4
                )
                print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} end -> domain 3 at param {s4_param:.3f}")

    # NEW: Additional trace continuity conditions for middle vertical segments
    
    # Trace continuity: end of domains 4-8 with corresponding point of domain 1 (S2)
    s2_lower_connections = list(range(4, 4+N))  # domains 4-8 (lower connectors from S1)
    
    for domain_idx in s2_lower_connections:
        if domain_idx < len(problems):
            for eq_idx in range(neq):
                # Find the corresponding y-coordinate on domain 1 (S2)
                horizontal_domain_info = geometry.get_domain(domain_idx)
                intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate of horizontal segment
                
                # Map y-coordinate to parameter space on domain 1 (S2)
                # S2 goes from y=-1 (param=0) to y=-0.1 (param=domain_length)
                # Parameter mapping: param = (y - y_start) / (y_end - y_start) * domain_length
                s2_y_start = -1.0
                s2_y_end = -0.1
                s2_param = (intersection_y - s2_y_start) / (s2_y_end - s2_y_start) * problems[1].domain_length
                
                constraint_manager.add_trace_continuity(
                    equation_index=eq_idx,
                    domain1_index=domain_idx,
                    domain2_index=1,  # S2 domain
                    position1=problems[domain_idx].domain_end,  # End of horizontal segment
                    position2=s2_param,  # Corresponding point on S2
                )
                print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} end -> domain 1 at param {s2_param:.3f}")
    
    # Trace continuity: end of domains 14-18 with corresponding point of domain 2 (S3)
    s3_upper_connections = list(range(4+2*N, 4+3*N))  # domains 14-18 (upper connectors from S1)
    
    for domain_idx in s3_upper_connections:
        if domain_idx < len(problems):
            for eq_idx in range(neq):
                # Find the corresponding y-coordinate on domain 2 (S3)
                horizontal_domain_info = geometry.get_domain(domain_idx)
                intersection_y = horizontal_domain_info.extrema_end[1]  # y-coordinate of horizontal segment
                
                # Map y-coordinate to parameter space on domain 2 (S3)
                # S3 goes from y=0.1 (param=0) to y=1.0 (param=domain_length)
                s3_y_start = 0.1
                s3_y_end = 1.0
                s3_param = (intersection_y - s3_y_start) / (s3_y_end - s3_y_start) * problems[2].domain_length
                
                constraint_manager.add_trace_continuity(
                    equation_index=eq_idx,
                    domain1_index=domain_idx,
                    domain2_index=2,  # S3 domain
                    position1=problems[domain_idx].domain_end,  # End of horizontal segment
                    position2=s3_param,  # Corresponding point on S3
                )
                print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} end -> domain 2 at param {s3_param:.3f}")
    
    # Trace continuity: start of domains 9-13 with corresponding point of domain 1 (S2)
    s2_lower_right_connections = list(range(4+N, 4+2*N))  # domains 9-13 (lower connectors from S4)
    
    for domain_idx in s2_lower_right_connections:
        if domain_idx < len(problems):
            for eq_idx in range(neq):
                # Find the corresponding y-coordinate on domain 1 (S2)
                horizontal_domain_info = geometry.get_domain(domain_idx)
                intersection_y = horizontal_domain_info.extrema_start[1]  # y-coordinate of horizontal segment
                
                # Map y-coordinate to parameter space on domain 1 (S2)
                # S2 goes from y=-1 (param=0) to y=-0.1 (param=domain_length)
                s2_y_start = -1.0
                s2_y_end = -0.1
                s2_param = (intersection_y - s2_y_start) / (s2_y_end - s2_y_start) * problems[1].domain_length
                
                constraint_manager.add_trace_continuity(
                    equation_index=eq_idx,
                    domain1_index=domain_idx,
                    domain2_index=1,  # S2 domain
                    position1=problems[domain_idx].domain_start,  # Start of horizontal segment
                    position2=s2_param,  # Corresponding point on S2
                )
                print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} start -> domain 1 at param {s2_param:.3f}")
    
    # Trace continuity: start of domains 19-23 with corresponding point of domain 2 (S3)
    s3_upper_right_connections = list(range(4+3*N, 4+4*N))  # domains 19-23 (upper connectors from S4)
    
    for domain_idx in s3_upper_right_connections:
        if domain_idx < len(problems):
            for eq_idx in range(neq):
                # Find the corresponding y-coordinate on domain 2 (S3)
                horizontal_domain_info = geometry.get_domain(domain_idx)
                intersection_y = horizontal_domain_info.extrema_start[1]  # y-coordinate of horizontal segment
                
                # Map y-coordinate to parameter space on domain 2 (S3)
                # S3 goes from y=0.1 (param=0) to y=1.0 (param=domain_length)
                s3_y_start = 0.1
                s3_y_end = 1.0
                s3_param = (intersection_y - s3_y_start) / (s3_y_end - s3_y_start) * problems[2].domain_length
                
                constraint_manager.add_trace_continuity(
                    equation_index=eq_idx,
                    domain1_index=domain_idx,
                    domain2_index=2,  # S3 domain
                    position1=problems[domain_idx].domain_start,  # Start of horizontal segment
                    position2=s3_param,  # Corresponding point on S3
                )
                print(f"    Added continuity: eq {eq_idx}, domain {domain_idx} start -> domain 2 at param {s3_param:.3f}")
               
    total_boundary_conditions = len(boundary_domains) * 2 * neq  # 2 boundaries per domain, neq equations each
    total_continuity_conditions = (len(s1_connections) + len(s4_connections) + 
                                  len(s2_lower_connections) + len(s3_upper_connections) + 
                                  len(s2_lower_right_connections) + len(s3_upper_right_connections)) * neq
    
    print(f"✓ Constraint setup completed:")
    print(f"  - External boundary conditions: {total_boundary_conditions} Neumann BCs")
    print(f"  - S1 continuity connections: {len(s1_connections)} domains × {neq} equations = {len(s1_connections) * neq} constraints")
    print(f"  - S4 continuity connections: {len(s4_connections)} domains × {neq} equations = {len(s4_connections) * neq} constraints")
    print(f"  - S2 continuity connections: {len(s2_lower_connections) + len(s2_lower_right_connections)} domains × {neq} equations = {(len(s2_lower_connections) + len(s2_lower_right_connections)) * neq} constraints")
    print(f"  - S3 continuity connections: {len(s3_upper_connections) + len(s3_upper_right_connections)} domains × {neq} equations = {(len(s3_upper_connections) + len(s3_upper_right_connections)) * neq} constraints")
    print(f"  - Total continuity conditions: {total_continuity_conditions}")

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
