# exec: python3 ./examples/test_maze_upwind.py  config/ooc_maze4_parameters.toml
"""
Evolution Example using new Time Stepper Module

This example demonstrates the same functionality as evolution+plotting_example.py
but using the new TimeStepper module for cleaner, more maintainable code.

The time advancement logic is replaced with a single TimeStepper class that
encapsulates all the Newton iteration and bulk data management.
"""

import random
import sys
import os
# Add the python_port directory to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))



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

# def classify_domains_left_right(
#     setup: "SolverSetup",
#     length: float,
# ) -> List[Dict]:
#     """Classify each domain as left, right, midline, or straddling the midline.

#     Only domains whose y-extent lies within [0, 6*length] are included.
#     The midline is at x = 3*length.

#     For straddling (horizontal-crossing) domains the classification is
#     refined element-by-element so that the trapezoidal integral can be
#     split exactly at the midline.

#     Args:
#         setup: The SolverSetup object (provides problems and discretizations).
#         length: The scaling length used when building the maze geometry.

#     Returns:
#         A list (one entry per domain) of dicts with keys:
#             included  – bool, whether the domain contributes to the integral.
#             side      – 'left' | 'right' | 'midline' | 'straddle' | None.
#             elements  – For 'straddle' domains only: a list (per element) of
#                         dicts {'side': 'left'|'right'|'split',
#                                'frac_left': float} where *frac_left* is the
#                         fraction of the element length on the left of the
#                         midline (relevant only when side == 'split').
#     """
#     midline_x = 3.0 * length
#     y_min = 0.0
#     y_max = 6.0 * length
#     tol = 1e-10

#     n_domains = len(setup.problems)
#     classification: List[Dict] = []

#     for d in range(n_domains):
#         prob = setup.problems[d]
#         (x0, y0) = prob.extrema[0]
#         (x1, y1) = prob.extrema[1]

#         # ---- y-range filter ----
#         seg_y_min = min(y0, y1)
#         seg_y_max = max(y0, y1)
#         if seg_y_min < y_min - tol or seg_y_max > y_max + tol:
#             classification.append({"included": False, "side": None, "elements": None})
#             continue

#         # ---- x classification ----
#         seg_x_min = min(x0, x1)
#         seg_x_max = max(x0, x1)

#         if abs(seg_x_min - midline_x) < tol and abs(seg_x_max - midline_x) < tol:
#             # Vertical segment sitting exactly on the midline
#             classification.append({"included": True, "side": "midline", "elements": None})
#         elif seg_x_max < midline_x - tol:
#             classification.append({"included": True, "side": "left", "elements": None})
#         elif seg_x_min > midline_x + tol:
#             classification.append({"included": True, "side": "right", "elements": None})
#         else:
#             # Horizontal segment crossing the midline – classify per element
#             disc = setup.global_discretization.spatial_discretizations[d]
#             dom_len = prob.domain_length
#             elem_info: List[Dict] = []
#             for k in range(disc.n_elements):
#                 s_left = disc.nodes[k]
#                 s_right = disc.nodes[k + 1]
#                 t_left = s_left / dom_len
#                 t_right = s_right / dom_len
#                 ex_left = x0 + t_left * (x1 - x0)
#                 ex_right = x0 + t_right * (x1 - x0)

#                 elem_x_min = min(ex_left, ex_right)
#                 elem_x_max = max(ex_left, ex_right)

#                 if elem_x_max <= midline_x + tol and elem_x_min < midline_x - tol:
#                     elem_info.append({"side": "left", "frac_left": 1.0})
#                 elif elem_x_min >= midline_x - tol and elem_x_max > midline_x + tol:
#                     elem_info.append({"side": "right", "frac_left": 0.0})
#                 else:
#                     # Element truly straddles the midline
#                     h_phys = abs(ex_right - ex_left)
#                     frac = abs(midline_x - min(ex_left, ex_right)) / h_phys
#                     elem_info.append({"side": "split", "frac_left": frac})

#             classification.append({"included": True, "side": "straddle", "elements": elem_info})

#     return classification


# def compute_left_right_mass(
#     bulk_data_list: list,
#     classification: List[Dict],
#     setup: "SolverSetup",
# ) -> List[float, float]:
#     """Compute the integral of u (equation 0) over the left and right halves.

#     Uses the trapezoidal rule element-by-element, which is exact for
#     piecewise linears:
#         integral_k = (h / 2) * (u_left + u_right)

#     For domains classified as 'midline' the integral is split 50/50.
#     For straddling domains with 'split' elements, u is interpolated at
#     the midline and each sub-interval is integrated separately.

#     Args:
#         bulk_data_list: List of BulkData objects, one per domain.
#         classification: Output of :func:`classify_domains_left_right`.
#         setup: The SolverSetup object.

#     Returns:
#         (mass_left, mass_right) – integrals of u over each half.
#     """
#     mass_left = 0.0
#     mass_right = 0.0

#     for d, info in enumerate(classification):
#         if not info["included"]:
#             continue

#         bulk = bulk_data_list[d]
#         disc = setup.global_discretization.spatial_discretizations[d]
#         h = disc.element_length  # parameter-space element length = physical length
#         n_elem = disc.n_elements

#         side = info["side"]

#         if side in ("left", "right"):
#             # Whole domain belongs to one side
#             domain_mass = 0.0
#             for k in range(n_elem):
#                 u_L = bulk.data[0, k]
#                 u_R = bulk.data[1, k]
#                 domain_mass += 0.5 * h * (u_L + u_R)
#             if side == "left":
#                 mass_left += domain_mass
#             else:
#                 mass_right += domain_mass

#         elif side == "midline":
#             # Split 50 / 50
#             domain_mass = 0.0
#             for k in range(n_elem):
#                 u_L = bulk.data[0, k]
#                 u_R = bulk.data[1, k]
#                 domain_mass += 0.5 * h * (u_L + u_R)
#             mass_left += 0.5 * domain_mass
#             mass_right += 0.5 * domain_mass

#         elif side == "straddle":
#             elem_info = info["elements"]
#             for k in range(n_elem):
#                 u_L = bulk.data[0, k]
#                 u_R = bulk.data[1, k]
#                 e = elem_info[k]
#                 if e["side"] == "left":
#                     mass_left += 0.5 * h * (u_L + u_R)
#                 elif e["side"] == "right":
#                     mass_right += 0.5 * h * (u_L + u_R)
#                 else:
#                     # Element straddles – interpolate u at the midline
#                     frac = e["frac_left"]  # fraction of h on the left
#                     h_left = frac * h
#                     h_right = (1.0 - frac) * h
#                     u_mid = u_L + frac * (u_R - u_L)
#                     mass_left += 0.5 * h_left * (u_L + u_mid)
#                     mass_right += 0.5 * h_right * (u_mid + u_R)

#     return mass_left, mass_right


def run_evolution_with_time_stepper(
    config_file: Optional[str] = None ,
    physical_vec: Optional[List[float]] = None,
    disc_dt: Optional[List[float]] = None,
    disc_dx: Optional[List[float]] = None):
    """
    Main function demonstrating time evolution with the new TimeStepper module.
    
    Args:
        config_file: Optional TOML configuration file path
    """
    print("="*80)
    print("EVOLUTION EXAMPLE WITH TIME STEPPER")
    print("="*80)
    print("Time evolution using the new TimeStepper module")

    data_folder = 20260703

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

        if disc_dt is not None: 
            print("Override discretization parameters") 
            # Ordine dei parametri nel vettore 
            mapping = [ ("dt"), ] 
            if len(disc_dt) != len(mapping): 
                raise ValueError( f"The vector must have len = {len(mapping)}, " f"but given {len(disc_dt)}." ) 
                # Applica override 
            for value in (disc_dt): 
                config["time_parameters"]["dt"]= value
        if disc_dx is not None: 
            print("Override discretization parameters") 
            # Ordine dei parametri nel vettore 
            mapping = [ ("n_elements"),] 
            if len(disc_dx) != len(mapping): 
                raise ValueError( f"The vector must have len = {len(mapping)}, " f"but given {len(disc_dx)}." ) 
                # Applica override 
            for value in (disc_dx): 
                config["discretization"]["n_elements"] = value
        # --- Debug: stampa parametri finali --- 
        print("\nFinal parameters:") 
        for section, params in config["physical_parameters"].items(): 
            print(f" [{section}]") 
            for k, v in params.items(): 
                print(f" {k} = {v}") 
                print()
        new_config_file = f"examples/laura/outputs/plots/{data_folder}/config_modified.toml"
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
    return I_all_times_omega[:], I_all_times_phi[:], I_all_times_u[:], I_all_times_v[:] 












if __name__ == "__main__":
    """Main execution with multiple demonstrations and config file support."""
    
    # Check for config file argument
    # config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"❌ Error: Configuration file '{config_file}' not found")
            print(f"💡 Please check the file path and try again")
            sys.exit(1)
        print(f"A) Using configuration file: {config_file}")
    else:
        # Default to ooc_parameters.toml if no argument provided
        config_file = "config/ooc_parameters.toml"
        if os.path.exists(config_file):
            print(f"B) Using default configuration file: {config_file}")
        else:
            print(f"Default config file '{config_file}' not found, using defaults")
            config_file = None


    
    try:
        # Main evolution example with config file
        times= np.linspace(0,12000,200)
        data_folder = 20260727
        plot_flag = True


       ########################################################################
       # INIZIO CONTROLLO CASO TRATTATO E NON TRATTATO -VALORI NOMINALI 
       ######################################################################## 

        # ut_t_flag = 'ut'
        # for k in [3.9e-1]: # [-2,-4,-6,2,4]: 
        #    #ut 
        #     physical_vecn_t = [ 200.,  700.,   700.,   56.,     0,      0,    1.e-4,   0,    3.9e-1 , 5.e-6,  1.9e-11, 1.e-4, 1.e-5]
        #    #t 
        #    # physical_vecn_t = [ 200.,  700.,   700.,   5.6,    5.e-4,  1.e-6,  5.e-4,  1.e-6,  3.9e-1 , 5.e-6, 1.9e-11, 1.e-4,0.]
           
        #     Iomegan_t ,Iphin_t, Iun_t, Ivn_t =run_evolution_with_time_stepper(config_file, physical_vecn_t)
                
            

       ########################################################################
       # FINE CONTROLLO CASO TRATTATO E NON TRATTATO -VALORI NOMINALI 
       ######################################################################## 

       ########################################################################
       # INIZIO ANALISI QUANTITATIVA 
       ########################################################################  
        # if 1:
        #     plot_flag = True
        #     ut_t_flag = 'ut'
        #     #untreated           # nu,  mu,  epsilon, sigma,    a,       b,      c,     d,      k1,      k2,      m1,    m2,    m3),
        #     physical_vecn_ut = [ 200.,  700.,   700.,   56.,     0,      0,    1.e-4,   0,    3.9e-1 , 5.e-6,  1.9e-11, 1.e-4, 1.e-5]
        #     Iomegan_ut ,Iphin_ut, Iun_ut, Ivn_ut =run_evolution_with_time_stepper(config_file, physical_vecn_ut)
     
        #     ranges_ut=[ [160., 240.],  #nu=50
        #             [560., 840.], #mu=150
        #             [560., 840.], #epsilon=90
        #             [45.0, 67.0] , #sigma=50
        #             [0., 0.], #a=5e-4
        #             [0., 0.], #b=1e-9
        #             [0.8e-4, 1.2e-4], #c=5e-4
        #             [0., 0.], #d=1e-6
        #             [3.1e-1, 4.7e-1] , #k1=3.9e-1
        #             [4.e-6, 6.e-6], #k2=5e-12
        #             [1.5e-11, 2.3e-11], #m1=1.9e-11
        #             [0.8e-4, 1.2e-4], #m2=1e-4
        #             [0.8e-5, 1.2e-5] ] #m3=5e-5
                

        #     #treated            # nu,  mu,  epsilon, sigma,        a,       b,      c,     d,      k1,      k2,      m1,    m2,    m3),
        #     ut_t_flag = 't'
        #     physical_vecn_t = [ 200.,  700.,   700.,   5.6,    5.e-4,  1.e-6,  5.e-4,  1.e-6,  3.9e-1 , 5.e-6, 1.9e-11, 1.e-4,0.]
        #     Iomegan_t ,Iphin_t, Iun_t, Ivn_t =run_evolution_with_time_stepper(config_file, physical_vecn_t)
            
        #     ranges_t=[ [160., 240.],  #nu=50
        #             [560., 840.], #mu=150
        #             [560., 840.], #epsilon=90
        #             [4.5, 6.7] , #sigma=0.5
        #             [0.8e-4, 1.2e-4], #a=5e-4
        #             [4e-6, 6e-6], #b=1e-5
        #             [0.8e-4, 1.2e-4], #c=5e-4
        #             [4e-6, 6e-6], #d=1e-6
        #             [3.1e-1, 4.7e-1] , #k1=7e-9
        #             [4.e-6, 6.e-6], #k2=5e-12
        #             [1.5e-11, 2.3e-11], #m1=1.9e-11
        #             [0.8e-4, 1.2e-4], #m2=1e-4
        #             [0.,0.] ] #m3=0
                
        #     PMm=['nu', 'mu' ,'epsilon','sigma','a','b','c','d','k1','k2','S','eta','kv']
        #     lab=["min", "max"] 
        #     combinazioni_t = []
        #     for i in range(len(PMm)-1):
                
        #         for j in range(2): 
                    
        #             combo = physical_vecn_t.copy()
        #             combo[i] = float(ranges_t[i][j])
        #             combinazioni_t.append(combo)

        #     combinazioni_ut = []
        #     for i in range(len(PMm)):
                
        #         for j in range(2): 
                    
        #             combo = physical_vecn_ut.copy()
        #             combo[i] = float(ranges_ut[i][j])
        #             combinazioni_ut.append(combo)


        #     for i in range(len(PMm)):  
        #         print('i', i, PMm[i], len(PMm))
        #         time.sleep(10)
        #         plot_flag = False 
        #         print(i, np.shape(combinazioni_ut) , combinazioni_ut[2*i][:]) #, combinazioni_ut[2*i+1][:])
    

        #         Iomegam_t ,Iphim_t, Ium_t, Ivm_t =run_evolution_with_time_stepper(config_file, combinazioni_t[2*i])
        #         IomegaM_t ,IphiM_t, IuM_t, IvM_t =run_evolution_with_time_stepper(config_file, combinazioni_t[2*i+1])
        #         Iomegam_ut ,Iphim_ut, Ium_ut, Ivm_ut =run_evolution_with_time_stepper(config_file, combinazioni_ut[2*i])
        #         IomegaM_ut ,IphiM_ut, IuM_ut, IvM_ut =run_evolution_with_time_stepper(config_file, combinazioni_ut[2*i+1])
                


        #         plt.figure() 
        #         plt.plot( times, Iun_t[::17],label="nominal_treated")
        #         plt.plot( times, Ium_t[::17], label="min_treated")
        #         plt.plot( times, IuM_t[::17], label="max_treated")
        #         plt.plot( times, Iun_ut[::17],label="nominal_untreated")
        #         plt.plot( times, Ium_ut[::17], label="min_untreated")
        #         plt.plot( times, IuM_ut[::17], label="max_untreated")
        #         plt.title(f"{PMm[i]}", fontsize=16)
        #         plt.xlabel("time (s)", fontsize=16)
        #         plt.ylabel("Iu", fontsize=16)
        #         plt.xticks(fontsize=14)
        #         plt.yticks(fontsize=14)
        #         plt.legend(fontsize=14)
        #         plt.grid(True) 
        #         # Salvataggio del grafico 
        #         plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s0_{PMm[i]}.png" , bbox_inches="tight")


        #         plt.figure() 
        #         plt.plot( times, Iun_t[1::17] ,label="nominal_treated")
        #         plt.plot( times, Ium_t[1::17] , label="min_treated")
        #         plt.plot( times, IuM_t[1::17] , label="max_treated")
        #         plt.plot( times, Iun_ut[1::17] ,label="nominal_untreated")
        #         plt.plot( times, Ium_ut[1::17] , label="min_untreated")
        #         plt.plot( times, IuM_ut[1::17] , label="max_untreated")
        #         plt.title(f"{PMm[i]}", fontsize=16)
        #         plt.xlabel("time (s)", fontsize=16)
        #         plt.ylabel("Iu", fontsize=16)
        #         plt.xticks(fontsize=14)
        #         plt.yticks(fontsize=14)
        #         plt.legend(fontsize=14)
        #         plt.grid(True) 
        #         # Salvataggio del grafico 
        #         plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s1_{PMm[i]}.png" , bbox_inches="tight")


        #         plt.figure() 
        #         plt.plot( times, Iun_t[13::17] ,label="nominal_treated")
        #         plt.plot( times, Ium_t[13::17] , label="min_treated")
        #         plt.plot( times, IuM_t[13::17] , label="max_treated")
        #         plt.plot( times, Iun_ut[13::17] ,label="nominal_untreated")
        #         plt.plot( times, Ium_ut[13::17] , label="min_untreated")
        #         plt.plot( times, IuM_ut[13::17] , label="max_untreated")
        #         plt.title(f"{PMm[i]}", fontsize=16)
        #         plt.xlabel("time (s)", fontsize=16)
        #         plt.ylabel("Iu", fontsize=16)
        #         plt.xticks(fontsize=14)
        #         plt.yticks(fontsize=14)
        #         plt.legend(fontsize=14)
        #         plt.grid(True) 
        #         # Salvataggio del grafico 
        #         plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s13_{PMm[i]}.png" , bbox_inches="tight")


        #         plt.figure() 
        #         plt.plot( times, Iun_t[14::17],label="nominal_treated")
        #         plt.plot( times, Ium_t[14::17] , label="min_treated")
        #         plt.plot( times, IuM_t[14::17] , label="max_treated")
        #         plt.plot( times, Iun_ut[14::17],label="nominal_untreated")
        #         plt.plot( times, Ium_ut[14::17] , label="min_untreated")
        #         plt.plot( times, IuM_ut[14::17] , label="max_untreated")
        #         plt.title(f"{PMm[i]}", fontsize=16)
        #         plt.xlabel("time (s)", fontsize=16)
        #         plt.ylabel("Iu", fontsize=16)
        #         plt.xticks(fontsize=14)
        #         plt.yticks(fontsize=14)
        #         plt.legend(fontsize=14)
        #         plt.grid(True) 
        #         # Salvataggio del grafico 
        #         plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s14_{PMm[i]}.png" , bbox_inches="tight")

        #         plt.figure()
        #         plt.plot( times, Ivn_t[:],label="nominal_treated")
        #         plt.plot( times,  Ivm_t[:], label="min_treated")
        #         plt.plot( times,  IvM_t[:], label="max_treated")
        #         plt.plot( times, Ivn_ut[:],label="nominal_untreated")
        #         plt.plot( times,  Ivm_ut[:], label="min_untreated")
        #         plt.plot( times,  IvM_ut[:], label="max_untreated")
        #         plt.title(f"{PMm[i]}", fontsize=16)
        #         plt.xlabel("time (s)", fontsize=16)
        #         plt.ylabel("Iv", fontsize=16)
        #         plt.xticks(fontsize=14)
        #         plt.yticks(fontsize=14)
        #         plt.legend(fontsize=14)
        #         plt.grid(True) 
        #         # Salvataggio del grafico 
        #         plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_v_{PMm[i]}.png" , bbox_inches="tight")

        #         plt.figure()
        #         plt.plot( times, Iphin_t[:] ,label="nominal_treated")
        #         plt.plot( times,  Iphim_t[:], label="min_treated")
        #         plt.plot( times,  IphiM_t[:], label="max_treated")
        #         plt.plot( times, Iphin_ut[:] ,label="nominal_untreated")
        #         plt.plot( times,  Iphim_ut[:], label="min_untreated")
        #         plt.plot( times,  IphiM_ut[:], label="max_untreated")
        #         plt.title(f"{PMm[i]}", fontsize=16)
        #         plt.xlabel("time (s)", fontsize=16)
        #         plt.ylabel("Iphi", fontsize=16)
        #         plt.xticks(fontsize=14)
        #         plt.yticks(fontsize=14)
        #         plt.legend(fontsize=14)
        #         plt.grid(True) 
        #         # Salvataggio del grafico 
        #         plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_phi_{PMm[i]}.png" , bbox_inches="tight")

        #         plt.figure()
        #         plt.plot( times, Iomegan_t[:],label="nominal_treated")
        #         plt.plot( times, Iomegam_t[:], label="min_treated")
        #         plt.plot( times, IomegaM_t[:], label="max_treated")
        #         plt.plot( times, Iomegan_ut[:],label="nominal_untreated")
        #         plt.plot( times, Iomegam_ut[:], label="min_untreated")
        #         plt.plot( times, IomegaM_ut[:], label="max_untreated")
        #         plt.title(f"{PMm[i]}", fontsize=16)
        #         plt.xlabel("time (s)", fontsize=16)
        #         plt.ylabel("Iomega", fontsize=16)
        #         plt.xticks(fontsize=14)
        #         plt.yticks(fontsize=14)
        #         plt.legend(fontsize=14)
        #         plt.grid(True) 
        #         # Salvataggio del grafico 
        #         plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_omega_{PMm[i]}.png" , bbox_inches="tight")
 
       ########################################################################
       # FINE ANALISI QUANTITATIVA 
       ######################################################################## 


       ########################################################################
       # INIZIO SCELTA DT E DX
       ######################################################################## 
    
       # physical_vecn_t = [ 200.,  700.,   700.,   0.56,    5.e-4,  1.e-6,  5.e-4,  1.e-6,  3.9e-1 , 5.e-6, 1.9e-11, 1.e-4,0.]
        physical_vecn_t = [ 160.,  700.,   700.,   0.56,    5.e-4,  1.e-6,  5.e-4,  1.e-6,  4.7e-1 , 5.e-6, 1.9e-11, 1.e-4,0.]
       #  physical_vecn_t = [ 240.,  700.,   700.,   0.56,    5.e-4,  0.8e-6,  5.e-4,  1.e-6,  4.7e-1 , 5.e-6, 1.9e-11, 1.e-4,0.]
       #  physical_vecn_t = [ 200.,  700.,   700.,   0.56,    5.e-4,  0.8e-6,  6.e-4,  1.e-6,  3.1e-1 , 5.e-6, 1.9e-11, 1.e-4,0.]
           
       # Iomegan_t ,Iphin_t, Iun_t, Ivn_t =run_evolution_with_time_stepper(config_file, physical_vecn_t)

        ranges_t=[ [160., 240.],  #nu=50
                    [560., 840.], #mu=150
                    [560., 840.], #epsilon=90
                    [0.45, 0.67] , #sigma=0.5
                    [0.8e-4, 1.2e-4], #a=5e-4
                    [4e-6, 6e-6], #b=1e-5
                    [0.8e-4, 1.2e-4], #c=5e-4
                    [4e-6, 6e-6], #d=1e-6
                    [3.1e-1, 4.7e-1] , #k1=7e-9
                    [4.e-6, 6.e-6], #k2=5e-12
                    [1.5e-11, 2.3e-11], #m1=1.9e-11
                    [0.8e-4, 1.2e-4], #m2=1e-4
                    [0.,0.] ] #m3=0
                
        PMm=['nu', 'mu' ,'epsilon','sigma','a','b','c','d','k1','k2','S','eta','kv']
        
        combinazioni_t = []
        for i in range(len(PMm)-1):
                
                for j in range(2): 
                    
                    combo = physical_vecn_t.copy()
                    combo[i] = float(ranges_t[i][j])
                    combinazioni_t.append(combo)


        possible_dt=[10,30,60,120]
        possible_dx=[5,10, 20, 30]
        

        # Main evolution example with config file
        times= np.linspace(0,1200,10)
        data_folder = 20260727
        plot_flag = True
        ut_t_flag = 't'

       # Iomegan_t ,Iphin_t, Iun_t, Ivn_t =run_evolution_with_time_stepper(config_file, physical_vecn_t)
        Delta_i =[]
        Delta_j =[]   
        grid_ij =[]   
        Delta_ij = []    
        #Scelta del dt
       # for i in [random.randint(0, np.shape(combinazioni_t)[0]-1) for _ in range(1)]:

        physical_vec = physical_vecn_t
        i=1
        j=1
        Iomegan_t0 ,Iphin_t0, Iun_t0, Ivn_t0 =run_evolution_with_time_stepper(config_file, physical_vecn_t,[480./(2**(i-1))],[5*2**(j-1)])
        print('Iun_t0', np.shape(Iun_t0))
        import matplotlib.pyplot as plt

       # plt.ion()   # modalità interattiva

        # --- FIGURA 1: Iu ---
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)

        # --- FIGURA 2: Iphi ---
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)    
        while i <= 9 or j <= 4:
                grid_ij.append([i,j])    
                print('grid_ij:', grid_ij)   
                time.sleep(5)                                                
                Iomegan_ti ,Iphin_ti, Iun_ti, Ivn_ti =run_evolution_with_time_stepper(config_file, physical_vecn_t, [480./(2**(i))], [5*2**(j-1)])
                Iomegan_tj ,Iphin_tj, Iun_tj, Ivn_tj =run_evolution_with_time_stepper(config_file, physical_vecn_t, [480./(2**(i-1))], [5*2**(j)])
                  # -----------------------------
                # FIGURA 1: plot Iu
                # -----------------------------
                
                ax1.plot(np.linspace(0,12000,int(12000/(480./2**(i-1)))), Iun_t0[0::17],label=f"dt={480./(2**(i-1))},dx={5*2**(j-1)}")
                ax1.plot(np.linspace(0,12000,int(12000/(480./2**(i-1)))), Iun_ti[0::17][0::2],label=f"dt={480./(2**i)},dx={5*2**(j-1)}")
                ax1.plot(np.linspace(0,12000,int(12000/(480./2**(i-1)))), Iun_tj[0::17],label=f"dt={480./(2**(i-1))},dx={5*2**j}")

                
                ax1.set_xlabel("time (s)", fontsize=16)
                ax1.set_ylabel("Iu", fontsize=16)
                ax1.grid(True)
                ax1.legend(fontsize=12)

             
    
               

                # -----------------------------
                # FIGURA 2: plot Iphi_t0
                # -----------------------------
               
                ax2.plot(np.linspace(0,12000,int(12000/(480./2**(i-1)))), Iphin_t0, label=f"dt={480./(2**(i-1))},dx={5*2**(j-1)}")
                ax2.plot(np.linspace(0,12000,int(12000/(480./2**(i-1)))), Iphin_ti[::2],label=f"dt={480./(2**(i))},dx={5*2**(j-1)}")
                ax2.plot(np.linspace(0,12000,int(12000/(480./2**(i-1)))), Iphin_tj, label=f"dt={480./(2**(i-1))},dx={5*2**j}")

                ax2.set_xlabel("time (s)", fontsize=16)
                ax2.set_ylabel("Iphi", fontsize=16)
                ax2.grid(True)
                ax2.legend(fontsize=12) 
               
                fig1.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_Iu_live.png",
                    bbox_inches="tight")
                fig2.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_Iphi_live.png",  bbox_inches="tight")
                        
               # Iu_L2t = np.abs(Iun_t0-Iun_ti[::2])
                Delta_i = np.linalg.norm(Iun_t0[::17]-Iun_ti[::17][::2]  , ord=2)/np.linalg.norm(Iun_t0[::17]  , ord=2) + np.linalg.norm(Iphin_t0-Iphin_ti[::2]  , ord=2)/np.linalg.norm(Iphin_t0  , ord=2) 

              #  Iu_L2x = np.abs(Iun_t0-Iun_tj)
                Delta_j = np.linalg.norm(Iun_t0[::17]-Iun_tj[::17] , ord=2)/np.linalg.norm(Iun_t0[::17]  , ord=2) + np.linalg.norm(Iphin_t0-Iphin_tj  , ord=2)/np.linalg.norm(Iphin_t0  , ord=2) 

                print('errore:', "Delta_i", Delta_i, "Delta_j", Delta_j)
                time.sleep(5)

                if Delta_i < Delta_j:
                    j+=1
                    Iun_t0 = Iun_tj
                    Iphin_t0 = Iphin_tj
                    Delta_ij.append([Delta_i,Delta_j])  
                else:
                    i+=1
                    Iun_t0 = Iun_ti
                    Iphin_t0 = Iphin_ti
                    Delta_ij.append([Delta_i,Delta_j]) 
        fig1.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_Iu_live.png",
             bbox_inches="tight")
        fig2.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_Iphi_live.png",  
             bbox_inches="tight")   
 
       ################################################
        np.save(f"examples/laura/outputs/plots/{data_folder}/grid_ij.npy", grid_ij)
        np.save(f"examples/laura/outputs/plots/{data_folder}/Delta_ij.npy", Delta_ij)


           
   # print(f"\n🎉 All demonstrations completed successfully!")
                
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Example failed with unexpected error:")
        print(f"   {type(e).__name__}: {e}")
        print(f"\n🔧 Debug information:")
        import traceback
        traceback.print_exc()
        sys.exit(1)