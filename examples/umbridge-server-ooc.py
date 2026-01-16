import umbridge
import sys
import os
import numpy as np
import time as ts
import matplotlib.pyplot as plt
plt.close('all')
# sys.path.append("../../src/limit-equilibrium")
# from base_classes import SoilProperties,SoilState,UniformQuadrature,Options,np,plt
# from circularSlipSurface import circularSlipSurface
# from bishop import bishop
# from gle import spencer,morgerstern_price
# from gridOfCircles import GridOptions, gridComputation,computeEtaMinForSurface
# from gridSimplexComputation import simplexComputation


class ooc_sol(umbridge.Model):

    def __init__(self):
        super().__init__("forward")

    def get_input_sizes(self, config):
        return [2]

    def get_output_sizes(self, config):
        return [3]

    def __call__(self, parameters, config):
            filename = parameters[0][0]#"bionetflux.problems.OoC_grid_new"

            y = np.array(parameters[0][1])

            # Add the python_port directory to path for absolute imports
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

            from setup_solver import quick_setup
            from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

            

            print("="*60)
            print("BIONETFLUX REAL INITIALIZATION TEST")
            print("="*60)
            print("Testing initialization with test_problem2 for MATLAB comparison")

            # =============================================================================
            # STEP 1: Initialize the solver setup
            # =============================================================================
            print("\nStep 1: Initializing solver setup...")
            setup = quick_setup(filename, y, validate=True)

            print(" Setup initialized and validated")

            # Get problem information
            info = setup.get_problem_info()
            print(f"  Problem: {info['problem_name']}")
            print(f"  Parameters: {info['params']}")
            print(f"  Domains: {info['num_domains']}")
            print(f"  Total elements: {info['total_elements']}")
            print(f"  Total trace DOFs: {info['total_trace_dofs']}")

            # Check if constraints attribute exists before accessing it
            if hasattr(setup, 'constraints') and setup.constraints is not None:
                print(f"  Constraints: {info['num_constraints']}")
            else:
                print(f"  Constraints: Not available (attribute missing)")

            print(f"  Time discretization: dt={info['time_discretization']['dt']}, T={info['time_discretization']['T']}")


            # =============================================================================
            # STEP 2: Create initial conditions
            # =============================================================================
            print("\nStep 2: Creating initial conditions...")
            trace_solutions, multipliers = setup.create_initial_conditions()

            discretization_nodes_all_domains = []
            print("✓ Initial trace solutions created:")
            for i, trace in enumerate(trace_solutions):
                print(f"  Domain {i+1}: shape {trace.shape}, range [{np.min(trace):.6e}, {np.max(trace):.6e}]")
                
                # Debug: Print solution values for each equation
                discretization = setup.global_discretization.spatial_discretizations[i]
                n_nodes = len(discretization.nodes)
                discretization_nodes_all_domains.append(discretization.nodes)
                
                for eq_idx in range(setup.problems[0].neq):
                    eq_start = eq_idx * n_nodes
                    eq_end = eq_start + n_nodes
                    eq_values = trace[eq_start:eq_end]
                    eq_name = plotter.equation_names[eq_idx] if 'plotter' in locals() else f'Eq{eq_idx}'
                    print(f"    {eq_name}: range [{np.min(eq_values):.6f}, {np.max(eq_values):.6f}]")
                    if eq_idx == 1:  # omega should be sinusoidal
                        print(f"    {eq_name} values (first 10): {eq_values[:10]}")

            # Initialize the lean matplotlib plotter
            print("\nInitializing LeanMatplotlibPlotter...")

            plotter = LeanMatplotlibPlotter(
                problems=setup.problems,
                discretizations=setup.global_discretization.spatial_discretizations,
                equation_names=None,  # Will auto-detect based on problem type
                figsize=(12, 8),
                output_dir="outputs/plots"  # Set directory for saving figures
            )

            # Plot initial trace solutions

            print("Plotting initial trace solutions...")

            # 2D curve visualization (all equations together)
            print("Creating 2D curve visualization...")
            curves_2d_fig = plotter.plot_2d_curves(
                trace_solutions=trace_solutions,
                title="Initial Solutions - 2D Curves",
                show_bounding_box=True,
                show_mesh_points=True,
                save_filename="bionetflux_initial_2d_curves.png"
            )

            # Flat 3D visualization for each equation
            for eq_idx in range(setup.problems[0].neq):
                flat_3d_fig = plotter.plot_flat_3d(
                    trace_solutions=trace_solutions,
                    equation_idx=eq_idx,
                    title=f"Initial {plotter.equation_names[eq_idx]} Solution - Flat 3D",
                    segment_width=0.1,
                    save_filename=f"bionetflux_initial_{plotter.equation_names[eq_idx]}_flat3d.png",
                    view_angle=(30, 45)
                )
                
                # Bird's eye view visualization
                birdview_fig = plotter.plot_birdview(
                    trace_solutions=trace_solutions,
                    equation_idx=eq_idx,
                    segment_width=0.15,
                    save_filename=f"bionetflux_initial_{plotter.equation_names[eq_idx]}_birdview.png",
                    show_colorbar=True,
                    time=0.0
                )


            # =============================================================================
            # STEP 3: Create global solution vector
            # =============================================================================
            print("\nStep 3: Assembling global solution vector...")
            global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
            print(f"✓ Global solution vector: shape {global_solution.shape}")
            print(f"  Range: [{np.min(global_solution):.6e}, {np.max(global_solution):.6e}]")

            # Test round-trip extraction
            extracted_traces, extracted_multipliers = setup.extract_domain_solutions(global_solution)
            print("✓ Round-trip extraction verified")
            for i, (orig, ext) in enumerate(zip(trace_solutions, extracted_traces)):
                if np.allclose(orig, ext, rtol=1e-14):
                    print(f"  Domain {i+1} trace extraction matches original")
                else:
                    print(f"  ✗ Domain {i+1} trace extraction does NOT match original")
                    


            # =============================================================================
            # STEP 4.0: Initialize bulk data U(t=0.0)
            # =============================================================================
            print("\nStep 4: Creating bulk data and forcing terms...")
            bulk_manager = setup.bulk_data_manager
            bulk_solutions = []

            bulk_guess = bulk_manager.initialize_all_bulk_data(problems=setup.problems,
                                                            discretizations=setup.global_discretization.spatial_discretizations,
                                                            time=0.0)

            for i, bulk in enumerate(bulk_guess):
                print(f"  Domain {i+1} bulk guess: shape {bulk.data.shape}, range [{np.min(bulk.data):.6e}, {np.max(bulk.data):.6e}]")
                
                

            print("\nStep 5: Initializing global assembler...")
            global_assembler = setup.global_assembler
            time = 0.0

            # =============================================================================
            # STEP 6.5: Time Evolution Loop
            # =============================================================================
            print("\nStep 6.5: Starting time evolution...")

            # Get time parameters
            dt = setup.global_discretization.dt
            T = info['time_discretization']['T']

            print(f"    Time evolution parameters:")
            print(f"    Time step dt: {dt}")
            print(f"    Final time T: {T}")
            print(f"    Number of time steps: {int(T/dt)}")

            # Initialize time evolution variables
            time_step = 1
            max_time_steps = int(T/dt) + 1  # Safety limit
            solution_history = [global_solution.copy()]  # Store solution history
            time_history = [0.0]  # Store time history

            current_time = 0.0

            # Newton method parameters
            max_newton_iterations = 20  # Limit to 20 iterations for debugging
            newton_tolerance = 1e-10
            newton_solution = global_solution.copy()  # Start with initial guess

            # Initialize multiplier section with constraint data at current time
            if hasattr(setup, 'constraint_manager') and setup.constraint_manager is not None:
                n_trace_dofs = setup.global_assembler.total_trace_dofs
                n_multipliers = setup.constraint_manager.n_multipliers
                if n_multipliers > 0:
                    # Get constraint data at current time and initialize multipliers
                    constraint_data = setup.constraint_manager.get_multiplier_data(current_time)
                    newton_solution[n_trace_dofs:] = constraint_data

            print(f"  Newton method parameters:")
            print(f"    Max iterations: {max_newton_iterations}")
            print(f"    Tolerance: {newton_tolerance:.1e}")
            # Note: residual variable not defined in this scope, would need to use final_residual if available

            sol_all_times = []
            I_all_times = []
            M_all_times = []
            # Time evolution loop
            while current_time+dt <= T and time_step <= max_time_steps:
                print(f"\n--- Time Step {time_step}: t = {current_time+dt:.6f} ---")


                current_time += dt
                time_step += 1

                # Compute source terms at current time
                source_terms = bulk_manager.compute_source_terms(
                    problems=setup.problems,
                    discretizations=setup.global_discretization.spatial_discretizations,
                    time=current_time
                )
                
                
                
                
                
                # Assemble right-hand side for static condensation
                right_hand_side = []  # For clarity in this step
                for i, (bulk_sol, source, static_cond) in enumerate(zip(bulk_guess, source_terms, setup.static_condensations)):
                    rhs = static_cond.assemble_forcing_term(previous_bulk_solution=bulk_sol.data,
                                                            external_force=source.data)
                    right_hand_side.append(rhs)
                    
                print("  ✓ Right-hand side assembled for static condensation")
                

            
                # Newton iteration loop
                newton_converged = False
                
                
                
                
                for newton_iter in range(max_newton_iterations):
                    
                    # Compute residual and Jacobian at current solution
                    current_residual, current_jacobian = global_assembler.assemble_residual_and_jacobian(
                        global_solution=newton_solution,
                        forcing_terms=right_hand_side,
                        static_condensations=setup.static_condensations,
                        time=current_time
                    )

                    
                    # Check convergence
                    residual_norm = np.linalg.norm(current_residual)
                    
                    if residual_norm < newton_tolerance:
                        print(f"  ✓ Newton method converged in {newton_iter + 1} iterations")
                        newton_converged = True
                        break
                    
                    # Check for singular Jacobian
                    jacobian_cond = np.linalg.cond(current_jacobian)
                    if jacobian_cond > 1e12:
                        print(f"  ⚠ Warning: Jacobian poorly conditioned (cond = {jacobian_cond:.2e})")
                    
                    # Solve linear system: J * delta_x = -F
                    try:
                        delta_x = np.linalg.solve(current_jacobian, -current_residual)
                    except np.linalg.LinAlgError as e:
                        print(f"  ✗ Newton method failed: Linear system singular ({e})")
                        break

                    # Update solution: x_{k+1} = x_k + delta_x
                    newton_solution = newton_solution + delta_x
                    
            
                    
                if not newton_converged:
                    print(f"  ✗ Newton method did not converge after {max_newton_iterations} iterations")
                    print(f"    Final residual norm: {np.linalg.norm(current_residual):.6e}")
                else:
                    # Final verification
                    final_residual, final_jacobian = global_assembler.assemble_residual_and_jacobian(
                        global_solution=newton_solution,
                        forcing_terms=right_hand_side,
                        static_condensations=setup.static_condensations,
                        time=current_time
                    )
                    final_residual_norm = np.linalg.norm(final_residual)
                    print(f"  ✓ Final verification: residual norm = {final_residual_norm:.6e}")
                
                # Update variables for subsequent steps
                global_solution = newton_solution
                

                # Update bulk solutions by static condensation for next time step
                bulk_sol = global_assembler.bulk_by_static_condensation(
                    global_solution=newton_solution,
                    forcing_terms=right_hand_side,
                    static_condensations=setup.static_condensations,
                    time=current_time
                )

                # Update bulk_guess with new bulk solution data for next time step
                # bulk_sol contains the actual bulk solution arrays, not BulkData objects
                for i, new_bulk_data in enumerate(bulk_sol):
                    # new_bulk_data should be a numpy array with the correct shape
                
                    # Extract only the first 2*neq rows (bulk solution part)
                    # neq = setup.problems[i].neq
                    # bulk_data_only = new_bulk_data[:2*neq, :]
                    
                    # Directly set the data array (bypass BulkData.set_data validation)
                    bulk_guess[i].data = new_bulk_data.copy()
                
                print(f"✓ Bulk solutions updated for next time step")
                for i, bulk in enumerate(bulk_guess):
                    print(f"  Domain {i+1} updated bulk: shape {bulk.data.shape}, range [{np.min(bulk.data):.6e}, {np.max(bulk.data):.6e}]")

                print(f"✓ Newton solver completed")
                print(f"  Solution range: [{np.min(global_solution):.6e}, {np.max(global_solution):.6e}]")

                tr =  np.hstack(extracted_traces)
                sol_all_times.append(tr) 

                print(f"Compute QoI")
                # Compute mesh size (vector of spacings)
                h =  np.diff(np.hstack(discretization_nodes_all_domains))[0]
                # Composite trapezoidal rule:
                # sum over h[i] * (sol[i] + sol[i+1]) / 2
                
                I = np.sum(h * (tr[:-1] + tr[1:]) / 2)
                I_all_times.append(I)

                x_tile = np.tile(np.hstack(discretization_nodes_all_domains), 4)


                # Barycenter numerator (Simpson-like rule)
                fa = x_tile[:-1] * tr[:-1]
                fb = x_tile[1:]  * tr[1:]
                fc = (x_tile[:-1] + x_tile[1:]) * (tr[:-1] + tr[1:]) / 4

                numerator = np.sum(h * (fa + fb + 4 * fc) / 6)
                M = numerator/I
                M_all_times.append(M)

            # sol_all_times = np.array(sol_all_times) # diventa array NumPy 
            # I_all_times = np.array(I_all_times)
            # M_all_times = np.array(M_all_times)
            # print("  Time evolution completed.")
            # # Note: residual variable not defined in this scope, would need to use final_residual if available
            # return [[I]]  #sol_all_times, I_all_times, M_all_times]]
            sol_all_times = np.array(sol_all_times).tolist() 
            I_all_times = np.array(I_all_times).tolist() 
            M_all_times = np.array(M_all_times).tolist() 
            return [[sol_all_times, I_all_times, M_all_times]]
        

    def supports_evaluate(self):
        return True


model = ooc_sol()
umbridge.serve_models([model], 4242)
