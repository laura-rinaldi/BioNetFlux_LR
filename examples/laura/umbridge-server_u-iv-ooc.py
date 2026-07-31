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
        return [13]

    def get_output_sizes(self, config):
        return [1]#[5]

    def __call__(self, parameters, config):
                config_file = "../../config/ooc_parameters.toml"

                
                physical_vec = [float(parameters[0][0]), float(parameters[0][1]),float(parameters[0][2]),float(parameters[0][3]),float(parameters[0][4]),float(parameters[0][5]),float(parameters[0][6]),
                                float(parameters[0][7]),float(parameters[0][8]), float(parameters[0][9]),float(parameters[0][10]),float(parameters[0][11]),float(parameters[0][12])]
                

                # Add the python_port directory to path for absolute imports
                
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

                
                from setup_solver import quick_setup, SolverSetup
                from bionetflux.time_integration import TimeStepper
                from bionetflux.time_integration.picard_solver import PicardSolver
                from bionetflux.time_integration.time_stepper import AdaptiveTimeStepper
                from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
                from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry, build_grid_geometry, create_maze_geometry

                #from bionetflux.geometry.domain_geometry import build_grid_geometry
                from bionetflux.utils.mesh_mapping import create_physical_mesh_dict, parametric_to_physical_mesh
                from bionetflux.problems.ooc_config_manager import OoCConfigManager
                import bionetflux.geometry.domain_geometry as _geom_module
                import numpy as np
                import matplotlib.pyplot as plt
                import shutil
                import time
                from datetime import datetime
                from typing import Optional, List, Dict, Any
                import tomli as tomllib 
                import toml

                data_folder = 20260731

                if config_file:
                        print(f"Using configuration file: {config_file}")
                        with open(config_file, "rb") as f: 
                            config = tomllib.load(f)
                        # --- Override tramite vettore passato alla funzione ---
                        if physical_vec is not None: 
                            print("Override parameters") 
                            # Ordine dei parametri nel vettore 
                            mapping = [ ("viscosity", "nu"), ("viscosity", "mu"), ("viscosity", "epsilon"), ("viscosity", "sigma"), ("reaction", "a"), ("coupling", "b"),("reaction", "c"),  ("coupling", "d"), ("chemotaxis", "k1"), ("chemotaxis", "k2"),  ("tumor_suppression", "m1"), ("tumor_suppression", "m2") , ("tumor_suppression", "m3") ] 
                            if len(physical_vec) != len(mapping): 
                                raise ValueError( f"The vector must have len = {len(mapping)}, " f"but given {len(physical_vec)}." ) 
                                # Applica override 
                            for (section, key), value in zip(mapping, physical_vec): 
                                config["physical_parameters"][section][key] = value
                
                        # if disc_dt is not None: 
                        #     print("Override discretization parameters") 
                        #     # Ordine dei parametri nel vettore 
                        #     mapping = [ ("dt"), ] 
                        #     if len(disc_dt) != len(mapping): 
                        #         raise ValueError( f"The vector must have len = {len(mapping)}, " f"but given {len(disc_dt)}." ) 
                        #         # Applica override 
                        #     for value in (disc_dt): 
                        #         config["time_parameters"]["dt"]= value
                        # if disc_dx is not None: 
                        #     print("Override discretization parameters") 
                        #     # Ordine dei parametri nel vettore 
                        #     mapping = [ ("n_elements"),] 
                        #     if len(disc_dx) != len(mapping): 
                        #         raise ValueError( f"The vector must have len = {len(mapping)}, " f"but given {len(disc_dx)}." ) 
                        #         # Applica override 
                        #     for value in (disc_dx): 
                        #         config["discretization"]["n_elements"] = value
                        # --- Debug: stampa parametri finali --- 
                        print("\nFinal parameters:") 
                        for section, params in config["physical_parameters"].items(): 
                            print(f" [{section}]") 
                            for k, v in params.items(): 
                                print(f" {k} = {v}") 
                                print()
                        new_config_file = f"../../examples/laura/outputs/plots/{data_folder}/config_modified.toml"
                        with open(new_config_file, "w") as f: 
                            toml.dump(config, f) 
                            print(f"Creato nuovo file TOML modificato: {new_config_file}")
                
                            
                else:
                        print("Using default parameters")
                print()
                    
                    
                        
                # ============================================================================
                # STEP 1: SOLVER SETUP (Enhanced with config file support and error handling)
                # ============================================================================
                geom_dir = os.path.dirname(_geom_module.__file__)
                geometry = create_maze_geometry(
                    data_dir=os.path.join(geom_dir, "maze_2_data"),
                    length=100.0,
                )                   
                
                try:
                    # Use quick_setup with both geometry and config file support
                    setup = quick_setup(
                        problem_module="bionetflux.problems.ooc_problem_upwind",
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
                
                # Create Picard solver and pass it to the time stepper
                picard_solver = PicardSolver(tolerance=1.e-7, max_iterations=50, verbose=False)
                time_stepper = TimeStepper(setup, picard_solver=picard_solver, verbose=True)
                
                
                # Initialize solution at t=0 (replaces Steps 3-4 and lines 226-233 from original)
                current_solution, current_bulk_data = time_stepper.initialize_solution()
                
                print("✓ Time stepper initialized")
                print(f"✓ Initial solution: shape {current_solution.shape}")
                print(f"✓ Initial bulk data: {len(current_bulk_data)} domains")
                
                # setup.compute_geometry_from_problems()
            
                
                
                # ============================================================================
                # STEP 3: VISUALIZATION SETUP (Same as original)
                # ============================================================================
                print("\nStep 3: Setting up visualization...")
                
                # Initialize plotter
                plotter = LeanMatplotlibPlotter(
                    problems=setup.problems,
                    discretizations=setup.global_discretization.spatial_discretizations,
                    equation_names=None,  # Auto-detect
                    figsize=(15, 10)
                )
                
                
                
                print(f"✓ Plotter initialized for {plotter.ndom} domains, {plotter.neq} equations")
                print(f"✓ Equation names: {plotter.equation_names}")
                
                # Plot geometry
                print("\nPlotting geometry...")
                
                setup.compute_geometry_from_problems()
                # plotter.plot_geometry_with_indices(geometry=setup.geometry,
                #                                save_filename=f"examples/laura/outputs/birdview/geometry_with_indices.png")
                # print("✓ Geometry plot created")
                                
                # ============================================================================
                # STEP 4: TIME EVOLUTION
                # ============================================================================
            
                    
                # Time evolution parameters
                current_time = 0.0
                dt = setup.global_discretization.dt
                T = setup.global_discretization.T# min(0.5, setup.global_discretization.T)# Limit runtime for demo
                max_time_steps = int(T / dt) + 1
                
                # Solution history for analysis
                solution_history = [current_solution.copy()]
                time_history = [current_time]
                
                print(f"Time evolution: t ∈ [0, {T}], dt = {dt}")
                print(f"Maximum time steps: {max_time_steps}")
                print()
                
                # TIME EVOLUTION LOOP - SIMPLIFIED TO ONE LINE PER TIME STEP!
                time_step = 0
                
                sol_u = []
                sol_all_times = []
                I_all_times_phi = []
                I_all_times_omega = []
                I_all_times_u = []
                I_all_times_v = []
                I_all_times_u_sum = []
                    
                while current_time + dt <= T + 1e-12 and time_step <= max_time_steps:
                    time_step += 1
                    
                    print(f"\n--- Time Step {time_step}: t = {current_time:.6f} → {current_time + dt:.6f} ---")
                    print(f"dt = {dt}" )
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
                    
                    
                    #solutions
                    # ['u', 'v', 'φ','ω',]
                    all_nodes=[]
                    all_nodes_param=[]
                    for domain_idx in range(info['num_domains']):
                        setup.compute_geometry_from_problems()
                
                        domain_info = setup.geometry.domains[domain_idx]
                        
                        dicretization = setup.global_discretization.spatial_discretizations[domain_idx]
                        all_nodes_param.append(setup.global_discretization.spatial_discretizations[domain_idx].nodes) #parametric coord.
                        all_nodes.append(parametric_to_physical_mesh(domain_info, dicretization)[1]) #physical coord.
                
            
                    
                    p_number= len(np.hstack( all_nodes_param))
                
                    
                    
            
                    I_phi=[]
                    I_omega =[]
                    I_u=[]
                    I_v =[] 
                    
                    for i in range(info['num_domains']):
                        setup.compute_geometry_from_problems()
                
                        h=setup.geometry.domains[i].domain_length/config["discretization"]['n_elements'] 
                        # Compute mesh size (vector of spacings)
                        # print('trace', extracted_traces_n[i])
                        # print('shape', np.shape(extracted_traces_n[i] ))
                        # time.sleep(5)
                        mh = int(len(extracted_traces_n[i])/4)
                        # print('mh',mh)
                        # print('h', h)
                        # time.sleep(10)
                        tr_u = extracted_traces_n[i][0:mh]
                        # if time_step % 10 == 0 :
                        #     plt.figure()
                        #     plt.plot( tr_u, label=f"trace_u_domain_{i}")
                        #     plt.savefig(f"./outputs/plots1/plot_tru_{i}_{time_step}.png" , bbox_inches="tight")
            
                        tr_omega = extracted_traces_n[i][mh:2*mh]
                        tr_v = extracted_traces_n[i][2*mh:3*mh]
                        tr_phi = extracted_traces_n[i][3*mh:4*mh]
                        
                        # flux_u = extracted_multipliers_n[i][0:mh]
                        # flux_omega = extracted_multipliers_n[i][mh:2*mh]
                        # flux_v = extracted_multipliers_n[i][2*mh:3*mh]
                        # flux_phi = extracted_multipliers_n[i][3*mh:4*mh]
                        
            
                        bulk_data_i = current_bulk_data[i]
            
                        
                        if hasattr(bulk_data_i, "get_data"):
                            bulk_array = bulk_data_i.get_data()
                        else:
                            bulk_array = np.asarray(bulk_data_i)
                        # print('bulk_data_i', np.shape(bulk_array))
                        bulk_u = bulk_array[0:2, :]
                        bulk_omega = bulk_array[2:4, :]   
                        bulk_v = bulk_array[4:6, :]      
                        bulk_phi = bulk_array[6:8, :]  
                        
            
                    
                        
                        # Composite trapezoidal rule:
                        # sum over h[i] * (sol[i] + sol[i+1]) / 2
                        
                        
                        i_u = np.sum(h * (bulk_u[:][0,:] + bulk_u[:][1,:]) /2)
                        i_v = np.sum(h * (bulk_v[:][0,:] + bulk_v[:][1,:]) / 2)
                        i_phi =np.sum(h*(bulk_phi[:][0,:] + bulk_phi[:][1,:]) / 2)
                        i_omega = np.sum(h * (bulk_omega[:][0,:] + bulk_omega[:][1,:]) / 2)
            
                        # print('a)',i,bulk_phi[:,:-1] , bulk_phi[1:], bulk_phi,i_phi)
                        I_phi.append(i_phi)
                        I_omega.append(i_omega)
                        I_u.append(i_u)
                        I_v.append(i_v) 
                        # print('b)',i, I_phi)
            
                
                    # if current_time % 100 == 0:
                    I_all_times_phi.append(sum(I_phi))
                    I_all_times_omega.append(sum(I_omega))
                    I_all_times_u.append(I_u)
                    I_all_times_u_sum.append(sum(I_u))
                    I_all_times_v.append(sum(I_v))
                    # print('Iu', sum(I_u))
                    # print('Iphi', I_all_times_phi)
                    
                    # calcolo centro di massa 
                    vettore_massa = np.zeros((plotter.neq,int(info['num_domains'])))
                    vettore_pesi = np.zeros((plotter.neq,int(info['num_domains'])))
            
                    # ['u', 'ω', 'v', 'φ']
                    vettore_massa[0,:]= np.nan
                    vettore_massa[1,:] = np.nan
                    vettore_massa[2,:]= np.nan
                    vettore_massa[3,:] = np.nan
                    # plots over time steps
                    if  (0  and time_step % 10 == 0) or time_step==0:
                        for eq_idx in range(plotter.neq):
                                        plotter.plot_birdview(
                                            extracted_traces_n,
                                            equation_idx=eq_idx,
                                            time=current_time,
                                            save_filename=f"examples/laura/outputs/birdview/{data_folder}_{ut_t_flag}/final_birdview_eq{eq_idx}_t{current_time:.6f}.png"
                                        )
                                        
            
                    bulk_data_extracted = current_bulk_data
                    
                    # Handle result
                    if result.converged:
            
                        
                        # Update state for next iteration
                        current_time += dt
                        current_solution = result.updated_solution
                        current_bulk_data = result.updated_bulk_data
                        
                        # current_flux_solution = result.updated_flux_solution
                        # Store history
                        solution_history.append(current_solution.copy())
                        time_history.append(current_time)
            
                    
                        
                sol_all_times = np.array(solution_history) # diventa array NumPy 
            
                sol_all_times = np.array(sol_all_times) 
                #singole soluzioni
                # ['u', 'ω', 'v', 'φ']
                N= int(np.shape(sol_all_times)[1]/4)
                sol_u = sol_all_times[:, 0:N]
                sol_omega=  sol_all_times[:,N: 2*N]
                sol_v= sol_all_times[:, 2*N : 3*N]
                sol_phi= sol_all_times[:, 3*N:]
            
            
                I_all_times_phi = np.array(I_all_times_phi) 
                I_all_times_omega = np.array(I_all_times_omega)
                
                I_all_times_u = np.array(I_all_times_u).flatten() 
                I_all_times_v = np.array(I_all_times_v)
                I_all_times_u_sum = np.array(I_all_times_u_sum) 
                
                # ============================================================================
                # STEP 5: FINAL RESULTS AND VISUALIZATION
                # ============================================================================
                
                print(f"\n" + "="*50)
                print("TIME EVOLUTION COMPLETED")
                print("="*50)
                
                successful_steps = len(solution_history) - 1  # Subtract initial condition
                print(f"Successful time steps: {successful_steps}/{max_time_steps}")
                print(f"Final time: {current_time:.6f}")
                print(f"Total solution history: {len(solution_history)} time points")
                
                # Extract final solutions
                final_traces, final_multipliers = setup.extract_domain_solutions(current_solution)
                final_bulk_data = current_bulk_data
            
                print(f"\nFinal solution characteristics:")
                for i, trace in enumerate(final_traces):
                    trace_norm = np.linalg.norm(trace)
                    print(f"  Domain {i}: ||trace|| = {trace_norm:.6e}")
                
                if len(final_multipliers) > 0:
                    multiplier_norm = np.linalg.norm(final_multipliers)
                    print(f"  Multipliers: ||λ|| = {multiplier_norm:.6e}")
                
                
                successful_steps = len(solution_history) - 1  # Subtract initial condition
                
                # print(current_flux_solution )#flux è un array N x 2:
                # qL = current_flux_solution[6][5,:]   # valori a sinistra
                # qR = current_flux_solution[6][6,:]  # valori a destra
                # # print("Flux qL shape:", np.shape(current_flux_solution[0][0]))
                # print("Flux qR shape:", np.shape(qR))
                # # costruiamo i vettori per il plot
                # y_plot = np.zeros(2 *np.shape(qR)[0])
            
                # for i in range(np.shape(qR)[0]):
                #     y_plot[2*i : 2*i+2] = [qL[i], qR[i]]
            
                # plt.figure(figsize=(8,4))
                # plt.plot(y_plot, linewidth=1.8)
                # plt.grid(True)
                # plt.xlabel("x")
                # plt.ylabel("flux(x)")
                # plt.title("Flusso P1 per elemento (lineare)")
                # plt.show()
                #  return sol_omega, sol_phi, sol_u, sol_v
                qoi = np.concatenate([I_all_times_omega[:], I_all_times_phi[:],  I_all_times_u[:], I_all_times_v[:]]).tolist() #np.concatenate([sol_u[12:]]).tolist() #np.concatenate([I_all_times_omega[1:-1], I_all_times_phi[1:-1], sol_u[1:-1], I_all_times_v[1:-1]]).tolist()
                print(qoi)
                return [[qoi] ]
            
        

    def supports_evaluate(self):
        return True


model = ooc_sol()
umbridge.serve_models([model], 4242)
