# first, run the server as: python3 umbridge-server-ooc.py

import umbridge
import sys
import os
import numpy as np
import time as ts
import matplotlib.pyplot as plt
plt.close('all')



class ooc_sol(umbridge.Model):

    def __init__(self):
        super().__init__("forward")

    def get_input_sizes(self, config):
        return [9]

    def get_output_sizes(self, config):
        return [1]#[5]

    def __call__(self, parameters, config):
                config_file = "../../config/ooc_parameters.toml"

                
                physical_vec = [parameters[0][0], parameters[0][1],parameters[0][2],parameters[0][3],parameters[0][4],parameters[0][5],parameters[0][6],parameters[0][7],parameters[0][8]]
                

                # Add the python_port directory to path for absolute imports
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..","src"))

                
                from setup_solver import quick_setup, SolverSetup
                from bionetflux.time_integration import TimeStepper
                from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
                from bionetflux.geometry.domain_geometry import build_grid_geometry
                from bionetflux.utils.mesh_mapping import create_physical_mesh_dict, parametric_to_physical_mesh
                import numpy as np
                import matplotlib.pyplot as plt
                import time
                from typing import Optional
                import tomllib
                import toml

                

                if config_file:
                    print(f"Using configuration file: {config_file}")
                    with open(config_file, "rb") as f: 
                        config = tomllib.load(f)
                    # --- Override tramite vettore passato alla funzione ---
                    if physical_vec is not None: 
                        print("Override parameters") 
                        # Ordine dei parametri nel vettore 
                        mapping = [ ("viscosity", "nu"), ("viscosity", "mu"), ("viscosity", "epsilon"), ("viscosity", "sigma"), ("reaction", "a"), ("reaction", "c"), ("coupling", "b"), ("coupling", "d"), ("coupling", "chi"), ] 
                        if len(physical_vec) != len(mapping): 
                            raise ValueError( f"The vector must have len = {len(mapping)}, " f"but given {len(physical_vec)}." ) 
                            # Applica override 
                        for (section, key), value in zip(mapping, physical_vec): 
                            config["physical_parameters"][section][key] = value
                    # --- Debug: stampa parametri finali --- 
                    print("\nFinal parameters:") 
                    for section, params in config["physical_parameters"].items(): 
                        print(f" [{section}]") 
                        for k, v in params.items(): 
                            print(f" {k} = {v}") 
                            print()
                    new_config_file = "config_modified.toml"
                    with open(new_config_file, "w") as f: 
                        toml.dump(config, f) 
                        print(f"Creato nuovo file TOML modificato: {new_config_file}")

                        
                else:
                    print("Using default parameters")
                print()
                
                
                # ============================================================================
                # STEP 1: SOLVER SETUP (Enhanced with config file support and error handling)
                # ============================================================================
                               
                geometry = build_grid_geometry(N=2)
                
                try:
                    # Use quick_setup with both geometry and config file support
                    setup = quick_setup(
                        problem_module="bionetflux.problems.ooc_problem",
                        validate=True,
                        config_file=new_config_file,  # Pass config file
                        geometry=geometry         # Pass geometry
                    )
                except ValueError as e:
                    # Handle configuration compatibility errors gracefully
                    if "not compatible with" in str(e) or "problem type" in str(e):
                        print(f"\n❌ Configuration Error:")
                        
                        return None, None, None, None
                    else:
                        # Re-raise other ValueError types
                        raise
                except Exception as e:
                    # Handle other setup errors
                    print(f"\n❌ Setup Error: {e}")
                    
                    return None, None, None, None

                # Get problem information
                info = setup.get_problem_info()
                print(f"✓ Problem loaded: {info['problem_name']}")
                print(f"  Domains: {info['num_domains']}")
                print(f"  Total DOFs: {info['total_trace_dofs'] + info['num_constraints']}")
                print(f"  Time discretization: dt={info['time_discretization']['dt']}, T={info['time_discretization']['T']}")
                
                # ============================================================================
                # STEP 2: TIME STEPPER INITIALIZATION 
                # ============================================================================
                
                print("\nStep 2: Initializing time stepper...")
                
                # Create time stepper with Newton solver configuration
                time_stepper = TimeStepper(setup, verbose=True)
                
                # Initialize solution at t=0 (replaces Steps 3-4 and lines 226-233 from original)
                current_solution, current_bulk_data = time_stepper.initialize_solution()
                
                print("✓ Time stepper initialized")
                print(f"✓ Initial solution: shape {current_solution.shape}")
                print(f"✓ Initial bulk data: {len(current_bulk_data)} domains")
                
                setup.compute_geometry_from_problems()
                
                               
                # ============================================================================
                # STEP 4: TIME EVOLUTION
                # ============================================================================
                
                
                # Time evolution parameters
                current_time = 0.0
                dt = setup.global_discretization.dt
                T = min(0.5, setup.global_discretization.T)  # Limit runtime for demo
                max_time_steps = int(T / dt) + 1
                
                # Solution history for analysis
                solution_history = [current_solution.copy()]
                time_history = [current_time]
                
                print(f"Time evolution: t ∈ [0, {T}], dt = {dt}")
                print(f"Maximum time steps: {max_time_steps}")
                print()
                
                # TIME EVOLUTION LOOP - SIMPLIFIED TO ONE LINE PER TIME STEP!
                time_step = 0
                
                
                sol_all_times = []
                I_all_times_phi = []
                I_all_times_w = []
                M_all_times_u = []
                M_all_times_v = []
                while current_time + dt <= T and time_step < max_time_steps:
                    time_step += 1
                    print(f"\n--- Time Step {time_step}: t = {current_time:.6f} → {current_time + dt:.6f} ---")
                    
                    # SINGLE CALL REPLACES ~50 LINES OF COMPLEX NEWTON ITERATION CODE!
                    result = time_stepper.advance_time_step(
                        current_solution=current_solution,
                        current_bulk_data=current_bulk_data,
                        current_time=current_time,
                        dt=dt
                    )

                    ################################################
                    #QOI
                    ################################################
                    extracted_traces_n, extracted_multipliers_n = setup.extract_domain_solutions(current_solution)
                    tr =  np.hstack(extracted_traces_n)
            
                    #singole soluzioni
                    # ['u', 'ω', 'v', 'φ']
                    all_nodes=[]
                    all_nodes_param=[]
                    for domain_idx in range(info['num_domains']):
                        setup.compute_geometry_from_problems()
                
                        domain_info = setup.geometry.domains[domain_idx]
                        dicretization = setup.global_discretization.spatial_discretizations[domain_idx]
                        all_nodes_param.append(setup.global_discretization.spatial_discretizations[domain_idx].nodes) #parametric coord.
                        all_nodes.append(parametric_to_physical_mesh(domain_info, dicretization)[1]) #physical coord.
                        
                    
                    p_number= len(np.hstack( all_nodes_param))
                    tr_u = tr[ 0:p_number]
                    tr_w=  tr[p_number: 2*p_number]
                    tr_v= tr[2*p_number : 3*p_number]
                    tr_phi= tr[3*p_number:]
                    sol_all_times.append(tr) 

                    # Compute mesh size (vector of spacings)

                    time.sleep(5)
                    h =  np.diff(np.hstack( all_nodes_param))
                    
                    # Composite trapezoidal rule:
                    # sum over h[i] * (sol[i] + sol[i+1]) / 2
                    
                    I_phi = np.sum(h * (tr_phi[:-1] + tr_phi[1:]) / 2)
                    I_all_times_phi.append(I_phi)

                    I_w = np.sum(h * (tr_w[:-1] + tr_w[1:]) / 2)
                    I_all_times_w.append(I_w)

                    I_u = np.sum(h * (tr_u[:-1] + tr_u[1:]) / 2)
                    I_v = np.sum(h * (tr_v[:-1] + tr_v[1:]) / 2)



                    x_tile = np.hstack( all_nodes_param)

                    #np.tile(np.hstack(setup.global_discretization.spatial_discretizations), 4)


                    # Barycenter numerator (Simpson-like rule)
                    fa_u = x_tile[:-1] * tr_u[:-1]
                    fb_u = x_tile[1:]  * tr_u[1:]
                    fc_u = (x_tile[:-1] + x_tile[1:]) * (tr_u[:-1] + tr_u[1:]) / 4

                    numerator_u = np.sum(h * (fa_u + fb_u + 4 * fc_u) / 6)
                    M_u = numerator_u/I_u
                    M_all_times_u.append(M_u)

                    fa_v = x_tile[:-1] * tr_v[:-1]
                    fb_v = x_tile[1:]  * tr_v[1:]
                    fc_v = (x_tile[:-1] + x_tile[1:]) * (tr_v[:-1] + tr_v[1:]) / 4

                    numerator_v = np.sum(h * (fa_v + fb_v + 4 * fc_v) / 6)
                    M_v = numerator_v/I_v
                    M_all_times_v.append(M_v)

                
                    
                    # Handle result
                    if result.converged:

                        
                        # Update state for next iteration
                        current_time += dt
                        current_solution = result.updated_solution
                        current_bulk_data = result.updated_bulk_data
                        
                        # Store history
                        solution_history.append(current_solution.copy())
                        time_history.append(current_time)
                        
                    else:
                        print(f"  ✗ Time step failed!")
                        break
                sol_all_times = np.array(sol_all_times) # diventa array NumPy 

                #singole soluzioni
                # ['u', 'ω', 'v', 'φ']
                sol_u = sol_all_times[:, 0:p_number]
                sol_w=  sol_all_times[:,p_number: 2*p_number]
                sol_v= sol_all_times[:, 2*p_number : 3*p_number]
                sol_phi= sol_all_times[:, 3*p_number:]

                sol_all_times = np.array(sol_all_times).tolist() 
                I_all_times_phi = np.array(I_all_times_phi).tolist() 
                I_all_times_w = np.array(I_all_times_w).tolist() 
                M_all_times_u = np.array(M_all_times_u).tolist() 
                M_all_times_v = np.array(M_all_times_v).tolist() 
                

                
                # ============================================================================
                # STEP 5: FINAL RESULTS AND VISUALIZATION
                # ============================================================================
                
                
                successful_steps = len(solution_history) - 1  # Subtract initial condition
                
                # Extract final solutions
                final_traces, final_multipliers = setup.extract_domain_solutions(current_solution)
                
                for i, trace in enumerate(final_traces):
                    trace_norm = np.linalg.norm(trace)
                
                if len(final_multipliers) > 0:
                    multiplier_norm = np.linalg.norm(final_multipliers)
                
                
                return [[M_all_times_v[-1]]] #[[sol_all_times, I_all_times_phi, I_all_times_w, M_all_times_u, M_all_times_v]]
            
        

    def supports_evaluate(self):
        return True


model = ooc_sol()
umbridge.serve_models([model], 4242)
