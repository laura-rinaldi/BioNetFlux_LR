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
        return [12]

    def get_output_sizes(self, config):
        return [1]#[5]

    def __call__(self, parameters, config):
                config_file = "../../config/ooc_parameters.toml"

                
                physical_vec = [float(parameters[0][0]), float(parameters[0][1]),float(parameters[0][2]),float(parameters[0][3]),float(parameters[0][4]),float(parameters[0][5]),float(parameters[0][6]),
                                float(parameters[0][7]),float(parameters[0][8]), float(parameters[0][9]),float(parameters[0][10]),float(parameters[0][11])]
                

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
                from typing import Optional, List
                import tomli as tomllib 
                import toml

                

                if config_file:
                    print(f"Using configuration file: {config_file}")
                    with open(config_file, "rb") as f: 
                        config = tomllib.load(f)
                    # --- Override tramite vettore passato alla funzione ---
                    if physical_vec is not None: 
                        print("Override parameters") 
                        # Ordine dei parametri nel vettore 
                        mapping = [ ("viscosity", "nu"), ("viscosity", "mu"), ("viscosity", "epsilon"), ("viscosity", "sigma"), ("reaction", "a"), ("coupling", "b"),("reaction", "c"),  ("coupling", "d"), ("chemotaxis", "k1"), ("chemotaxis", "k2"),  ("tumor_suppression", "m1"), ("tumor_suppression", "m2") ] 
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
                data_folder = 20260407               
                geometry = build_grid_geometry(N=2, length=500.0)
                
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
                def plot_birdview_bulk(
                    bulk_data_list,
                    setup,
                    equation_idx,
                    time,
                    coord=None,
                    sizepoint=None,
                    save_filename=None,
                    plotter=None
                ):
                    """
                    Plot a birdview-like representation of bulk solution values
                    on 2D xy plane using the actual geometry from build_grid_geometry.
                    """
                    fig, ax = plt.subplots(figsize=(12, 10))
                    spatial_discs = setup.global_discretization.spatial_discretizations
                    num_domains = len(setup.geometry.domains)

                    if len(bulk_data_list) != num_domains:
                        raise ValueError(
                            f"bulk_data_list deve contenere un elemento per dominio, "
                            f"trovati {len(bulk_data_list)} != {num_domains}"
                        )

                    # Get global solution bounds for normalization
                    vmin, vmax = np.inf, -np.inf
                    colormap = 'viridis'
                    
                    for bulk_data in bulk_data_list:
                        if hasattr(bulk_data, "get_data"):
                            bulk_array = bulk_data.get_data()
                        else:
                            bulk_array = np.asarray(bulk_data)
                        base = equation_idx * 2
                        vals = bulk_array[base:base+2, :]
                        vmin = min(vmin, np.min(vals))
                        vmax = max(vmax, np.max(vals))
                    
                    if vmax <= vmin:
                        vmin -= 1
                        vmax += 1
                    
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)

                    # Plot each domain in 2D using actual geometry extrema
                    for domain_idx, (domain_info, disc, bulk_data) in enumerate(
                        zip(setup.geometry.domains, spatial_discs, bulk_data_list)
                    ):
                        # Get parametric coordinates from discretization
                        param_coords = disc.nodes

                        if hasattr(bulk_data, "get_data"):
                            bulk_array = bulk_data.get_data()
                        else:
                            bulk_array = np.asarray(bulk_data)

                        n_elems = bulk_array.shape[1]
                        base = equation_idx * 2
                        
                        # Extract bulk values (average per element)
                        elem_values = (bulk_array[base + 0, :] + bulk_array[base + 1, :]) / 2.0

                        # Map parametric coordinates to 2D using domain extrema
                        extrema_start = domain_info.extrema_start
                        extrema_end = domain_info.extrema_end
                        
                        # Normalize parameter coordinates to [0, 1]
                        param_min, param_max = domain_info.domain_start, domain_info.domain_start + domain_info.domain_length
                        t = (param_coords - param_min) / (param_max - param_min)
                        
                        # Linear interpolation between extrema
                        x_coords = extrema_start[0] + t * (extrema_end[0] - extrema_start[0])
                        y_coords = extrema_start[1] + t * (extrema_end[1] - extrema_start[1])

                        # Plot segments with color based on solution value
                        for i in range(len(elem_values)):
                            color_val = elem_values[i]
                            color = plt.colormaps[colormap](norm(color_val))
                            
                            ax.plot(
                                [x_coords[i], x_coords[i + 1]],
                                [y_coords[i], y_coords[i + 1]],
                                color=color, linewidth=8, alpha=0.8, solid_capstyle='round'
                            )

                        # QoI markers
                        if (coord is not None and sizepoint is not None 
                            and domain_idx < len(coord) and domain_idx < len(sizepoint)
                            and not np.isnan(coord[domain_idx]) and sizepoint[domain_idx] > 0):
                            
                            # Use domain center for QoI marker
                            center_x = (extrema_start[0] + extrema_end[0]) / 2
                            center_y = (extrema_start[1] + extrema_end[1]) / 2
                            
                            ax.scatter(center_x, center_y, s=sizepoint[domain_idx], zorder=10, color='red')

                    # Add colorbar
                    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
                    eq_name = plotter.equation_names[equation_idx] if plotter is not None else f'equation {equation_idx}'
                    cbar.set_label(f'Bulk variable {eq_name}', fontsize=12)

                    ax.set_xlabel('x', fontsize=12)
                    ax.set_ylabel('y', fontsize=12)
                    ax.set_aspect('equal', adjustable='box')
                    
                    # Set limits based on geometry
                    all_x = []
                    all_y = []
                    for domain in setup.geometry.domains:
                        all_x.extend([domain.extrema_start[0], domain.extrema_end[0]])
                        all_y.extend([domain.extrema_start[1], domain.extrema_end[1]])
                    
                    x_margin = (max(all_x) - min(all_x)) * 0.1
                    y_margin = (max(all_y) - min(all_y)) * 0.1
                    
                    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
                    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
                    
                    ax.set_title(f'Bulk birdview (2D) - {plotter.equation_names[equation_idx]} at t = {time:.6f}', fontsize=13)
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()

                    if save_filename:
                        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
                        fig.savefig(save_filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                
               
                    
                
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
               #                                save_filename="geometry_with_indices.png")
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
                   # print('etr=',extracted_traces_n)
                   # time.sleep(30)
                    #tr =  np.hstack(extracted_traces_n)
                    
                    #singole soluzioni
                    # ['u', 'v', 'φ','ω',]
                    all_nodes=[]
                    all_nodes_param=[]
                    for domain_idx in range(info['num_domains']):
                        setup.compute_geometry_from_problems()
                
                        domain_info = setup.geometry.domains[domain_idx]
                        dicretization = setup.global_discretization.spatial_discretizations[domain_idx]
                        all_nodes_param.append(setup.global_discretization.spatial_discretizations[domain_idx].nodes) #parametric coord.
                        all_nodes.append(parametric_to_physical_mesh(domain_info, dicretization)[1]) #physical coord.
              

                    x_tile_tot= np.hstack( all_nodes_param)
                    #print('nodes', len(x_tile),x_tile[0:51] ,x_tile[52:103], x_tile[104:156])
                    
                    p_number= len(np.hstack( all_nodes_param))
                    #print('p_number=', p_number, len(tr))
                    
                    nh = int(p_number/int(info['num_domains']))

                    I_phi=[]
                    I_omega =[]
                    I_u=[]
                    I_v =[] 
                    u_weights =[] 
                    v_weights =[] 
                    for i in range(info['num_domains']):
                        h=config["discretization"]['h'] 
                        # Compute mesh size (vector of spacings)
                        # np.diff(np.hstack( all_nodes_param))[nh*i:nh*(i+1)-1]
                        x_tile = x_tile_tot[nh*i:nh*(i+1)]
                        
                        
                        tr_u = extracted_traces_n[i][0:nh]
                        tr_omega = extracted_traces_n[i][nh:2*nh]
                        tr_v = extracted_traces_n[i][2*nh:3*nh]
                        tr_phi = extracted_traces_n[i][3*nh:4*nh]
       
                       # print(i, tr_u, tr_v, tr_omega, tr_phi)
                       # time.sleep(30)
                        
                        sol_u.append(tr_u[int(len(x_tile)/2)] )

                    
                        
                        # Composite trapezoidal rule:
                        # sum over h[i] * (sol[i] + sol[i+1]) / 2
                        
                        I_phi.append(np.sum(h * (tr_phi[:-1] + tr_phi[1:]) / 2))
                        I_omega.append(np.sum(h * (tr_omega[:-1] + tr_omega[1:]) / 2))

                    
                        

                        i_u = np.sum(h * (tr_u[:-1] + tr_u[1:]) / 2)
                        I_u.append(i_u)
                        i_v = np.sum(h * (tr_v[:-1] + tr_v[1:]) / 2)
                        I_v.append(i_v)


                    #sol_all_times.append(tr) 
                    #phi_weights =  I_phi/sum(I_phi) 
                    #w_weights =  I_omega/sum(I_omega) 
                # print('w',  I_u, sum(I_u))

                    u_weights =  I_u/sum(I_u) 
                    v_weights =  I_v/sum(I_v)  

                    #print('p%=',phi_weights)
                    I_all_times_phi.append(sum(I_phi))
                    I_all_times_omega.append(sum(I_omega))
                    I_all_times_u.append(sum(I_u))
                    I_all_times_v.append(sum(I_v))

                    # calcolo centro di massa 
                    vettore_massa = np.zeros((plotter.neq,int(info['num_domains'])))
                    vettore_pesi = np.zeros((plotter.neq,int(info['num_domains'])))

                    # ['u', 'ω', 'v', 'φ']
                    vettore_massa[0,:]= np.nan
                    vettore_massa[1,:] = np.nan
                    vettore_massa[2,:]= np.nan
                    vettore_massa[3,:] = np.nan


                  #  vettore_pesi[0,:]= 100*u_weights
                # vettore_pesi[1,:] = np.nan
                  #  vettore_pesi[2,:]= 100*v_weights
                # vettore_pesi[3,:] = np.nan
                #  print('t',current_time ,extracted_traces_n[i] , tr_u)
            
                    # plots over time steps

                    bulk_data_extracted = current_bulk_data

                    if current_time %10 ==0:
                        for eq_idx in range(plotter.neq):
                            plotter.plot_birdview(
                                extracted_traces_n,
                                equation_idx=eq_idx,
                                time=current_time,
                                coord = vettore_massa[eq_idx,:],
                                sizepoint = 20*vettore_pesi[eq_idx,:],
                                save_filename=f"outputs/birdview/{data_folder}/final_birdview_eq{eq_idx}_t{current_time:.6f}.png"
                            ) 
                
                            plot_birdview_bulk(
                                bulk_data_extracted,
                                setup,
                                equation_idx=eq_idx,
                                time=current_time,
                                coord=vettore_massa[eq_idx, :],
                                sizepoint=20 * vettore_pesi[eq_idx, :],
                                save_filename=f"outputs/birdview/{data_folder}/final_bulk_birdview_eq{eq_idx}_t{current_time:.6f}.png",
                                plotter=plotter
                            )
                    # Handle result
                    if result.converged:

                        
                        # Update state for next iteration
                        current_time += dt
                        current_solution = result.updated_solution
                        current_bulk_data = result.updated_bulk_data
                        
                        # Store history
                        solution_history.append(current_solution.copy())
                        time_history.append(current_time)
                      
                sol_all_times = np.array(sol_all_times) # diventa array NumPy 

                #singole soluzioni
                # ['u', 'ω', 'v', 'φ']
                #sol_u = sol_all_times[:, 0:p_number]
                #sol_omega=  sol_all_times[:,p_number: 2*p_number]
                #sol_v= sol_all_times[:, 2*p_number : 3*p_number]
                #sol_phi= sol_all_times[:, 3*p_number:]

                sol_all_times = np.array(sol_all_times) 
                I_all_times_phi = np.array(I_all_times_phi) 
                I_all_times_omega = np.array(I_all_times_omega)
                
                I_all_times_u = np.array(I_all_times_u) 
                I_all_times_v = np.array(I_all_times_v)
                
                #print("dim", np.shape(sol_u))
                #print("dim", np.shape(I_all_times_v ))
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
                print(f"Final traces: {final_traces}, Final bulk data: {final_bulk_data}")
                bulk_data_extracted = final_bulk_data
                for eq_idx in range(plotter.neq):
                    plot_birdview_bulk(
                        bulk_data_extracted,
                        setup,
                        equation_idx=eq_idx,
                        time=current_time,
                        coord=vettore_massa[eq_idx, :],
                        sizepoint=20 * vettore_pesi[eq_idx, :],
                        save_filename=f"outputs/birdview/{data_folder}/final_bulk_birdview_eq{eq_idx}_t{current_time:.6f}.png",
                        plotter=plotter
                    )
                
    
    
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
               # print("dim",(I_all_times_omega[1] ), (I_all_times_phi[1] ), (sol_u[0:12] ),(I_all_times_v[1] ))
               # print("dim",(I_all_times_v[1] ))
                qoi = np.concatenate([I_all_times_omega[:], I_all_times_phi[:], sol_u[:], I_all_times_v[:]]).tolist() #np.concatenate([sol_u[12:]]).tolist() #np.concatenate([I_all_times_omega[1:-1], I_all_times_phi[1:-1], sol_u[1:-1], I_all_times_v[1:-1]]).tolist()

                return [[qoi] ]
            
        

    def supports_evaluate(self):
        return True


model = ooc_sol()
umbridge.serve_models([model], 4242)
