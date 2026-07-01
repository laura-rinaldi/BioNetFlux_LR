# exec: python3 ./examples/test_upwind.py  config/ooc_parameters.toml
"""
Evolution Example using new Time Stepper Module

This example demonstrates the same functionality as evolution+plotting_example.py
but using the new TimeStepper module for cleaner, more maintainable code.

The time advancement logic is replaced with a single TimeStepper class that
encapsulates all the Newton iteration and bulk data management.
"""

import sys
import os
# Add the python_port directory to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))




from setup_solver import quick_setup, SolverSetup
from bionetflux.time_integration import TimeStepper
from bionetflux.time_integration.picard_solver import PicardSolver
from bionetflux.time_integration.time_stepper import AdaptiveTimeStepper
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter
#from bionetflux.geometry.domain_geometry import build_arc_sequence_geometry, build_grid_geometry, create_maze_geometry


from bionetflux.geometry.domain_geometry import build_grid_geometry
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

    data_folder = 20260619

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
                    
    geometry = build_grid_geometry(N=2, length=500.0)
    
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
    #                                save_filename=f"outputs/birdview/{data_folder}/geometry_with_indices.png")
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
        if  (0  and time_step % 10 == 0) or time_step==1:
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
    
    I_all_times_u = np.array(I_all_times_u).flatten() 
    I_all_times_v = np.array(I_all_times_v)
                
    
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
        data_folder = 20260619
        plot_flag = True


       ########################################################################
       # INIZIO CONTROLLO CASO TRATTATO E NON TRATTATO -VALORI NOMINALI 
       ######################################################################## 

        # ut_t_flag = 't'
        # for k in [3.9e-1]: # [-2,-4,-6,2,4]: 
        #    #ut 
        #     physical_vecn_t = [ 200.,  700.,   700.,   56.,     0,      0,    1.e-4,   0,    3.9e-1 , 5.e-6,  1.9e-11, 1.e-4, 1.e-5]
        #  #t 
        #     physical_vecn_t = [ 200.,  700.,   700.,   5.6,    5.e-4,  1.e-6,  5.e-4,  1.e-6,  3.9e-1 , 5.e-6, 1.9e-11, 1.e-4,0.]
           
        #     Iomegan_t ,Iphin_t, Iun_t, Ivn_t =run_evolution_with_time_stepper(config_file, physical_vecn_t)
                
        #     plt.figure() 
        #     plt.plot( times,  Iun_t[::12],label=f"{k}")
        #     plt.title(f"chi(phi)=k1/(k2+ phi)^2", fontsize=16)
        #     plt.xlabel("time (h)", fontsize=16)
        #     plt.ylabel("Iu", fontsize=16)
        #     plt.xticks(fontsize=14)
        #     plt.yticks(fontsize=14)
        #     plt.legend(fontsize=14)
        #     plt.grid(True) 
        #     # Salvataggio del grafico 
        #     plt.savefig(f"examples/laura/outputs/plots3/plot_I_u0_t_chi.png" , bbox_inches="tight")


        #     plt.figure() 
        #     plt.plot( times,  Iphin_t[:],label=f"{k}")
        #     plt.title(f"chi(phi)=k1/(k2+ phi)^2", fontsize=16)
        #     plt.xlabel("time (h)", fontsize=16)
        #     plt.ylabel("Iphi", fontsize=16)
        #     plt.xticks(fontsize=14)
        #     plt.yticks(fontsize=14)
        #     plt.legend(fontsize=14)
        #     plt.grid(True) 
        #     # Salvataggio del grafico 
        #     plt.savefig(f"examples/laura/outputs/plots3/plot_I_phi_t_chi.png" , bbox_inches="tight")

       ########################################################################
       # FINE CONTROLLO CASO TRATTATO E NON TRATTATO -VALORI NOMINALI 
       ######################################################################## 

       ########################################################################
       # INIZIO ANALISI QUANTITATIVA 
       ########################################################################  
        if 1:
            plot_flag = True
            ut_t_flag = 'ut'
            #untreated           # nu,  mu,  epsilon, sigma,    a,       b,      c,     d,      k1,      k2,      m1,    m2,    m3),
            physical_vecn_ut = [ 200.,  700.,   700.,   56.,     0,      0,    1.e-4,   0,    3.9e-1 , 5.e-6,  1.9e-11, 1.e-4, 1.e-5]
            Iomegan_ut ,Iphin_ut, Iun_ut, Ivn_ut =run_evolution_with_time_stepper(config_file, physical_vecn_ut)
     
            ranges_ut=[ [160., 240.],  #nu=50
                    [560., 840.], #mu=150
                    [560., 840.], #epsilon=90
                    [45.0, 67.0] , #sigma=50
                    [0., 0.], #a=5e-4
                    [0., 0.], #b=1e-9
                    [0.8e-4, 1.2e-4], #c=5e-4
                    [0., 0.], #d=1e-6
                    [3.1e-1, 4.7e-1] , #k1=3.9e-1
                    [4.e-6, 6.e-6], #k2=5e-12
                    [1.5e-11, 2.3e-11], #m1=1.9e-11
                    [0.8e-4, 1.2e-4], #m2=1e-4
                    [0.8e-5, 1.2e-5] ] #m3=5e-5
                

            #treated            # nu,  mu,  epsilon, sigma,        a,       b,      c,     d,      k1,      k2,      m1,    m2,    m3),
            ut_t_flag = 't'
            physical_vecn_t = [ 200.,  700.,   700.,   5.6,    5.e-4,  1.e-6,  5.e-4,  1.e-6,  3.9e-1 , 5.e-6, 1.9e-11, 1.e-4,0.]
            Iomegan_t ,Iphin_t, Iun_t, Ivn_t =run_evolution_with_time_stepper(config_file, physical_vecn_t)
            
            ranges_t=[ [160., 240.],  #nu=50
                    [560., 840.], #mu=150
                    [560., 840.], #epsilon=90
                    [4.5, 6.7] , #sigma=0.5
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
            lab=["min", "max"] 
            combinazioni_t = []
            for i in range(len(PMm)-1):
                
                for j in range(2): 
                    
                    combo = physical_vecn_t.copy()
                    combo[i] = float(ranges_t[i][j])
                    combinazioni_t.append(combo)

            combinazioni_ut = []
            for i in range(len(PMm)):
                
                for j in range(2): 
                    
                    combo = physical_vecn_ut.copy()
                    combo[i] = float(ranges_ut[i][j])
                    combinazioni_ut.append(combo)



                # plt.figure() 
                # plt.plot( times, Iun_t[::12],label="nominal_treated")
                # plt.plot( times, Iun_ut[::12],label="nominal_untreated")
                # plt.title(f"{PMm[i]}", fontsize=16)
                # plt.xlabel("time (s)", fontsize=16)
                # plt.ylabel("Iu", fontsize=16)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                # plt.legend(fontsize=14)
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s0_{PMm[i]}_n.png" , bbox_inches="tight")


                # plt.figure() 
                # plt.plot( times, Iun_t[3::12] ,label="nominal_treated")
                # plt.title(f"{PMm[i]}", fontsize=16)
                # plt.xlabel("time (s)", fontsize=16)
                # plt.ylabel("Iu", fontsize=16)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                # plt.legend(fontsize=14)
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s3_{PMm[i]}_tn.png" , bbox_inches="tight")


                # plt.figure() 
                # plt.plot( times, Iun_ut[3::12] ,label="nominal_untreated")
                # plt.title(f"{PMm[i]}", fontsize=16)
                # plt.xlabel("time (s)", fontsize=16)
                # plt.ylabel("Iu", fontsize=16)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                # plt.legend(fontsize=14)
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s3_{PMm[i]}_utn.png" , bbox_inches="tight")

                # plt.figure() 
                # plt.plot(  Iun_t[3::12][-50:]  ,label="nominal_treated")
                # plt.plot( Iun_ut[3::12][-50:] ,label="nominal_untreated")
                # plt.title(f"{PMm[i]}", fontsize=16)
                # plt.xlabel("time (s)", fontsize=16)
                # plt.ylabel("Iu", fontsize=16)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                # plt.legend(fontsize=14)
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s3_{PMm[i]}_zoom_n.png" , bbox_inches="tight")

                # plt.figure() 
                # plt.plot( times, Iun_t[4::12] ,label="nominal_treated")
                # plt.plot( times, Iun_ut[4::12] ,label="nominal_untreated")
                # plt.title(f"{PMm[i]}", fontsize=16)
                # plt.xlabel("time (s)", fontsize=16)
                # plt.ylabel("Iu", fontsize=16)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                # plt.legend(fontsize=14)
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s4_{PMm[i]}_n.png" , bbox_inches="tight")


                # plt.figure() 
                # plt.plot( times, Iun_t[9::12],label="nominal_treated")
                # plt.plot( times, Iun_ut[9::12],label="nominal_untreated")
                # plt.title(f"{PMm[i]}", fontsize=16)
                # plt.xlabel("time (s)", fontsize=16)
                # plt.ylabel("Iu", fontsize=16)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                # plt.legend(fontsize=14)
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s9_{PMm[i]}_n.png" , bbox_inches="tight")

                # plt.figure()
                # plt.plot( times, Ivn_t[:],label="nominal_treated")
                # plt.plot( times, Ivn_ut[:],label="nominal_untreated")
                # plt.title(f"{PMm[i]}", fontsize=16)
                # plt.xlabel("time (s)", fontsize=16)
                # plt.ylabel("Iv", fontsize=16)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                # plt.legend(fontsize=14)
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_v_{PMm[i]}_n.png" , bbox_inches="tight")

                # plt.figure()
                # plt.plot( times, Iphin_t[:] ,label="nominal_treated")
                # plt.plot( times, Iphin_ut[:] ,label="nominal_untreated")
                # plt.title(f"{PMm[i]}", fontsize=16)
                # plt.xlabel("time (s)", fontsize=16)
                # plt.ylabel("Iphi", fontsize=16)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                # plt.legend(fontsize=14)
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_phi_{PMm[i]}_n.png" , bbox_inches="tight")

                # plt.figure()
                # plt.plot( times, Iomegan_t[:],label="nominal_treated")
                # plt.plot( times, Iomegan_ut[:],label="nominal_untreated")
                # plt.title(f"{PMm[i]}", fontsize=16)
                # plt.xlabel("time (s)", fontsize=16)
                # plt.ylabel("Iomega", fontsize=16)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                # plt.legend(fontsize=14)
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_omega_{PMm[i]}_n.png" , bbox_inches="tight")

            for i in range(len(PMm)):  
                print('i', i, PMm[i], len(PMm))
                time.sleep(10)
                plot_flag = False 
                print(i, np.shape(combinazioni_ut) , combinazioni_ut[2*i][:]) #, combinazioni_ut[2*i+1][:])
    

                Iomegam_t ,Iphim_t, Ium_t, Ivm_t =run_evolution_with_time_stepper(config_file, combinazioni_t[2*i])
                IomegaM_t ,IphiM_t, IuM_t, IvM_t =run_evolution_with_time_stepper(config_file, combinazioni_t[2*i+1])
                Iomegam_ut ,Iphim_ut, Ium_ut, Ivm_ut =run_evolution_with_time_stepper(config_file, combinazioni_ut[2*i])
                IomegaM_ut ,IphiM_ut, IuM_ut, IvM_ut =run_evolution_with_time_stepper(config_file, combinazioni_ut[2*i+1])
                


                plt.figure() 
                plt.plot( times, Iun_t[::12],label="nominal_treated")
                plt.plot( times, Ium_t[::12], label="min_treated")
                plt.plot( times, IuM_t[::12], label="max_treated")
                plt.plot( times, Iun_ut[::12],label="nominal_untreated")
                plt.plot( times, Ium_ut[::12], label="min_untreated")
                plt.plot( times, IuM_ut[::12], label="max_untreated")
                plt.title(f"{PMm[i]}", fontsize=16)
                plt.xlabel("time (s)", fontsize=16)
                plt.ylabel("Iu", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(fontsize=14)
                plt.grid(True) 
                # Salvataggio del grafico 
                plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s0_{PMm[i]}.png" , bbox_inches="tight")


                plt.figure() 
                plt.plot( times, Iun_t[3::12] ,label="nominal_treated")
                plt.plot( times, Ium_t[3::12] , label="min_treated")
                plt.plot( times, IuM_t[3::12] , label="max_treated")
                plt.plot( times, Iun_ut[3::12] ,label="nominal_untreated")
                plt.plot( times, Ium_ut[3::12] , label="min_untreated")
                plt.plot( times, IuM_ut[3::12] , label="max_untreated")
                plt.title(f"{PMm[i]}", fontsize=16)
                plt.xlabel("time (s)", fontsize=16)
                plt.ylabel("Iu", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(fontsize=14)
                plt.grid(True) 
                # Salvataggio del grafico 
                plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s3_{PMm[i]}.png" , bbox_inches="tight")


                plt.figure() 
                plt.plot( times, Iun_t[4::12] ,label="nominal_treated")
                plt.plot( times, Ium_t[4::12] , label="min_treated")
                plt.plot( times, IuM_t[4::12] , label="max_treated")
                plt.plot( times, Iun_ut[4::12] ,label="nominal_untreated")
                plt.plot( times, Ium_ut[4::12] , label="min_untreated")
                plt.plot( times, IuM_ut[4::12] , label="max_untreated")
                plt.title(f"{PMm[i]}", fontsize=16)
                plt.xlabel("time (s)", fontsize=16)
                plt.ylabel("Iu", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(fontsize=14)
                plt.grid(True) 
                # Salvataggio del grafico 
                plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s4_{PMm[i]}.png" , bbox_inches="tight")


                plt.figure() 
                plt.plot( times, Iun_t[9::12],label="nominal_treated")
                plt.plot( times, Ium_t[9::12] , label="min_treated")
                plt.plot( times, IuM_t[9::12] , label="max_treated")
                plt.plot( times, Iun_ut[9::12],label="nominal_untreated")
                plt.plot( times, Ium_ut[9::12] , label="min_untreated")
                plt.plot( times, IuM_ut[9::12] , label="max_untreated")
                plt.title(f"{PMm[i]}", fontsize=16)
                plt.xlabel("time (s)", fontsize=16)
                plt.ylabel("Iu", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(fontsize=14)
                plt.grid(True) 
                # Salvataggio del grafico 
                plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_u_s9_{PMm[i]}.png" , bbox_inches="tight")

                plt.figure()
                plt.plot( times, Ivn_t[:],label="nominal_treated")
                plt.plot( times,  Ivm_t[:], label="min_treated")
                plt.plot( times,  IvM_t[:], label="max_treated")
                plt.plot( times, Ivn_ut[:],label="nominal_untreated")
                plt.plot( times,  Ivm_ut[:], label="min_untreated")
                plt.plot( times,  IvM_ut[:], label="max_untreated")
                plt.title(f"{PMm[i]}", fontsize=16)
                plt.xlabel("time (s)", fontsize=16)
                plt.ylabel("Iv", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(fontsize=14)
                plt.grid(True) 
                # Salvataggio del grafico 
                plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_v_{PMm[i]}.png" , bbox_inches="tight")

                plt.figure()
                plt.plot( times, Iphin_t[:] ,label="nominal_treated")
                plt.plot( times,  Iphim_t[:], label="min_treated")
                plt.plot( times,  IphiM_t[:], label="max_treated")
                plt.plot( times, Iphin_ut[:] ,label="nominal_untreated")
                plt.plot( times,  Iphim_ut[:], label="min_untreated")
                plt.plot( times,  IphiM_ut[:], label="max_untreated")
                plt.title(f"{PMm[i]}", fontsize=16)
                plt.xlabel("time (s)", fontsize=16)
                plt.ylabel("Iphi", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(fontsize=14)
                plt.grid(True) 
                # Salvataggio del grafico 
                plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_phi_{PMm[i]}.png" , bbox_inches="tight")

                plt.figure()
                plt.plot( times, Iomegan_t[:],label="nominal_treated")
                plt.plot( times, Iomegam_t[:], label="min_treated")
                plt.plot( times, IomegaM_t[:], label="max_treated")
                plt.plot( times, Iomegan_ut[:],label="nominal_untreated")
                plt.plot( times, Iomegam_ut[:], label="min_untreated")
                plt.plot( times, IomegaM_ut[:], label="max_untreated")
                plt.title(f"{PMm[i]}", fontsize=16)
                plt.xlabel("time (s)", fontsize=16)
                plt.ylabel("Iomega", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(fontsize=14)
                plt.grid(True) 
                # Salvataggio del grafico 
                plt.savefig(f"examples/laura/outputs/plots/{data_folder}/plot_I_omega_{PMm[i]}.png" , bbox_inches="tight")

   
       ########################################################################
       # FINE ANALISI QUANTITATIVA 
       ######################################################################## 


       ########################################################################
       # INIZIO SCELTA DT E DX
       ######################################################################## 

        possible_dt=[0.1, 1.]
        possible_dx=[10, 20]
        
        # #Scelta del dt
        # for i in range(5):
        #         physical_vec = combinazioni[i]
        #         print(physical_vec)                                                           
        #         sol_true, I1_true, I2_true, M1_true, M2_true =run_evolution_with_time_stepper(config_file, physical_vec, [0.1], [20])
        #         err_M1_i=[]
        #         err_M2_i =[]
        #         for j  in possible_dt:
        #             sol, I1, I2, M1, M2 =run_evolution_with_time_stepper(config_file, physical_vec, [j], [20])
                    
        #             err_M1_i.append(max(abs(M1_true[1::int(j/0.1)] -M1[1:])/M1_true[1::int(j/0.1)]))
        #             err_M2_i.append(max(abs(M2_true[1::int(j/0.1)]-M2[1:])/M2_true[1::int(j/0.1)]))
                   

                # plt.figure(1) 
                # plt.plot( [1, 10], err_M1_i[::-1] ,'o-')
                # plt.xlabel("time steps (#)")
                # plt.ylabel("relative error (%)")
                # plt.legend()
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(r"./outputs/plots/plot_err_M1_dt.png" , bbox_inches="tight")
                
                # plt.figure(2)
                # plt.plot([1, 10], err_M2_i[::-1], 'o-')
                # plt.xlabel("time steps (#)")
                # plt.ylabel("relative error (%)")
                # plt.legend()
                # plt.grid(True) 
                # # Salvataggio del grafico 
                # plt.savefig(r"./outputs/plots/plot_err_M2_dt.png" , bbox_inches="tight")

        #Scelta del dx
        # for i in range(5):
        #         physical_vec = combinazioni[i]
        #         print(physical_vec)                                                           
        #         sol_true, I1_true, I2_true, M1_true, M2_true =run_evolution_with_time_stepper(config_file, physical_vec, [0.1], [20])
        #         err_M1_i=[]
        #         err_M2_i =[]
        #         for j  in possible_dx:
        #             sol, I1, I2, M1, M2 =run_evolution_with_time_stepper(config_file, physical_vec, [0.1], [j])
                    
        #             err_M1_i.append(max(abs(M1_true[1:] -M1[1:])/M1_true[1:]))
        #             err_M2_i.append(max(abs(M2_true[1:]-M2[1:])/M2_true[1:]))
                   

        #         plt.figure(3) 
        #         plt.plot( possible_dx, err_M1_i[:] ,'o-')
        #         plt.xlabel("space elements (#)")
        #         plt.ylabel("relative error (%)")
        #         plt.legend()
        #         plt.grid(True) 
        #         # Salvataggio del grafico 
        #         plt.savefig(r"./outputs/plots/plot_err_M1_dx.png" , bbox_inches="tight")
                
        #         plt.figure(4)
        #         plt.plot(possible_dx, err_M2_i[:], 'o-')
        #         plt.xlabel("space elements (#)")
        #         plt.ylabel("relative error (%)")
        #         plt.legend()
        #         plt.grid(True) 
        #         # Salvataggio del grafico 
        #         plt.savefig(r"./outputs/plots/plot_err_M2_dx.png" , bbox_inches="tight")


           
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