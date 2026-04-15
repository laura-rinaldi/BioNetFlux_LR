# exec: pytho3 ./funzione_evol.py ../../config/ooc_parameters.toml
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


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



def run_evolution_with_time_stepper(config_file: Optional[str] = None ,
    physical_vec: Optional[List[float]] = None):
    """
    Main function demonstrating time evolution with the new TimeStepper module.
    
    Args:
        config_file: Optional TOML configuration file path
    """
    print("="*80)
    print("EVOLUTION EXAMPLE WITH TIME STEPPER")
    print("="*80)
    print("Time evolution using the new TimeStepper module")

    data_folder = 20260414 

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
        # --- Debug: stampa parametri finali --- 
        print("\nFinal parameters:") 
        for section, params in config["physical_parameters"].items(): 
            print(f" [{section}]") 
            for k, v in params.items(): 
                print(f" {k} = {v}") 
                print()
        new_config_file = f"outputs/plots/{data_folder}/config_modified.toml"
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
    while current_time + dt <= T and time_step < max_time_steps:
        
        
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
        
        
        #ssolutions
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
        
        p_number= len(np.hstack( all_nodes_param))
        
        nh = int(p_number/int(info['num_domains']))

        I_phi=[]
        I_omega =[]
        I_u=[]
        I_v =[] 
        for i in range(info['num_domains']):
            h=config["discretization"]['h'] 
            # Compute mesh size (vector of spacings)
            x_tile = x_tile_tot[nh*i:nh*(i+1)]
            
            
            tr_u = extracted_traces_n[i][0:nh]
            tr_omega = extracted_traces_n[i][nh:2*nh]
            tr_v = extracted_traces_n[i][2*nh:3*nh]
            tr_phi = extracted_traces_n[i][3*nh:4*nh]


            # Usa:
            bulk_data_i = current_bulk_data[i]
            if hasattr(bulk_data_i, "get_data"):
                bulk_array = bulk_data_i.get_data()
            else:
                bulk_array = np.asarray(bulk_data_i)

            # average bulks (media sinistra-destra per elemento)
            bulk_a_u = (bulk_array[0, :] + bulk_array[1, :]) / 2      # equazione u
            bulk_a_omega = (bulk_array[2, :] + bulk_array[3, :]) / 2  # equazione ω  
            bulk_a_v = (bulk_array[4, :] + bulk_array[5, :]) / 2      # equazione v
            bulk_a_phi = (bulk_array[6, :] + bulk_array[7, :]) / 2    # equazione φ

            bulk_u = bulk_array[0:2, :].flatten() 
            bulk_omega = bulk_array[2:4, :]   
            bulk_v = bulk_array[4:6, :]      
            bulk_phi = bulk_array[6:8, :]  

        

            tr_u = bulk_u
            tr_omega = bulk_omega
            tr_v = bulk_v
            tr_phi = bulk_phi
            
            sol_u.append(tr_u[int(len(x_tile)/2)] )

        
            
            # Composite trapezoidal rule:
            # sum over h[i] * (sol[i] + sol[i+1]) / 2
            len_segment = h*len(tr_u)
            
            i_u = np.sum(h * (tr_u[:-1] + tr_u[1:]) / 2)/len_segment
            i_v = np.sum(h * (tr_v[:-1] + tr_v[1:]) / 2)/len_segment
            i_phi = np.sum(h * (tr_phi[:-1] + tr_phi[1:]) / 2)/ len_segment
            i_omega = np.sum(h * (tr_omega[:-1] + tr_omega[1:]) / 2)/ len_segment

            I_phi.append(i_phi)
            I_omega.append(i_omega)
            I_u.append(i_u)
            I_v.append(i_v) 

    
        # if current_time % 100 == 0:
        I_all_times_phi.append(sum(I_phi))
        I_all_times_omega.append(sum(I_omega))
        I_all_times_u.append(I_u)
        I_all_times_v.append(sum(I_v))

        
        # calcolo centro di massa 
        vettore_massa = np.zeros((plotter.neq,int(info['num_domains'])))
        vettore_pesi = np.zeros((plotter.neq,int(info['num_domains'])))

        # ['u', 'ω', 'v', 'φ']
        vettore_massa[0,:]= np.nan
        vettore_massa[1,:] = np.nan
        vettore_massa[2,:]= np.nan
        vettore_massa[3,:] = np.nan
        # plots over time steps

        bulk_data_extracted = current_bulk_data
       
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
                        # nu,  mu,  epsilon, sigma,    a,       b,      c,     d,      k1,      k2,      m1,    m2,    m3),
        physical_vecn = [ 50.,  150.,   90.,      50. , 5.e-4,  1.e-9,   5.e-4,  1.e-6,  7.e-9 , 5.e-12, 1.9e-11, 1.e-4, 5.e-5]
    
                
        #result = run_evolution_with_time_stepper(config_file, physical_vec)
        times= np.linspace(0,72000,80)
        ranges=[ [40.,60.],  #nu=50
                 [120., 170.], #mu=150
                 [72., 108.], #epsilon=90
                 [40.0, 60.0] , #sigma=50
                 [4.e-4, 6.e-4], #a=5e-4
                 [1.e-9, 1.e-9], #b=1e-9
                 [4.e-4, 6.e-4], #c=5e-4
                 [0.8e-6, 1.2e-6], #d=1e-6
                 [5.6e-9, 8.e-9] , #k1=7e-9
                 [4.e-12, 6.e-12], #k2=5e-12
                 [1.5e-11, 2.3e-11], #m1=1.9e-11
                 [0.8e-4, 1.2e-4], #m2=1e-4
                 [4.e-5, 6.e-5] ] #m3=5e-5
            

        
        PMm=['nu', 'mu' ,'epsilon','sigma','a','b','c','d','k1','k2','S','eta','kv']
        lab=["min", "max"]
        def genera_combinazioni(): 
            combinazioni = np.zeros((2, len(PMm)))
            for i in range(len(PMm)):
                for j in ranges[i]: 
                    combo = physical_vecn.copy()
                    combo[i] = j
                    combinazioni[:, i] = combo
                return combinazioni 
        combinazioni = genera_combinazioni() 
        
        Iomegan ,Iphin, Iun, Ivn =run_evolution_with_time_stepper(config_file, physical_vecn)
        for i in range(len(PMm)):   
            Iomegam ,Iphim, Ium, Ivm =run_evolution_with_time_stepper(config_file, combinazioni[0,i])
            IomegaM ,IphiM, IuM, IvM =run_evolution_with_time_stepper(config_file, combinazioni[1,i])
            data_folder = 20260415 


            plt.figure() 
            plt.plot( times, Iun[3::12],label="nominal")
            plt.plot( times, Ium[3::12], label="min")
            plt.plot( times, IuM[3::12], label="max")
            plt.title("QoI: Iu")
            plt.xlabel("time (s)")
            plt.ylabel("I_u")
            plt.legend()
            plt.grid(True) 
            # Salvataggio del grafico 
            plt.savefig(f"./outputs/plots/{data_folder}/plot_I_u_s3_{PMm[i]}.png" , bbox_inches="tight")


            plt.figure() 
            plt.plot( times, Iun[2::12] ,label="nominal")
            plt.plot( times, Ium[2::12] , label="min")
            plt.plot( times, IuM[2::12] , label="max")
            plt.title("QoI: Iu")
            plt.xlabel("time (s)")
            plt.ylabel("I_u")
            plt.legend()
            plt.grid(True) 
            # Salvataggio del grafico 
            plt.savefig(f"./outputs/plots/{data_folder}/plot_I_u_s2_{PMm[i]}.png" , bbox_inches="tight")


            plt.figure() 
            plt.plot( times, Iun[6::12] ,label="nominal")
            plt.plot( times, Ium[6::12] , label="min")
            plt.plot( times, IuM[6::12] , label="max")
            plt.title("QoI: Iu")
            plt.xlabel("time (s)")
            plt.ylabel("I_u")
            plt.legend()
            plt.grid(True) 
            # Salvataggio del grafico 
            plt.savefig(f"./outputs/plots/{data_folder}/plot_I_u_s6_{PMm[i]}.png" , bbox_inches="tight")


            plt.figure() 
            plt.plot( times, Iun[9::12],label="nominal")
            plt.plot( times, Ium[9::12] , label="min")
            plt.plot( times, IuM[9::12] , label="max")
            plt.title("QoI: Iu")
            plt.xlabel("time (s)")
            plt.ylabel("I_u")
            plt.legend()
            plt.grid(True) 
            # Salvataggio del grafico 
            plt.savefig(f"./outputs/plots/{data_folder}/plot_I_u_s9_{PMm[i]}.png" , bbox_inches="tight")

            plt.figure()
            plt.plot( times, Ivn[:],label="nominal")
            plt.plot( times,  Ivm[:], label="min")
            plt.plot( times,  IvM[:], label="max")
            plt.title("QoI: Iv")
            plt.xlabel("time (s)")
            plt.ylabel("I_v")
            plt.legend()
            plt.grid(True) 
            # Salvataggio del grafico 
            plt.savefig(f"./outputs/plots/{data_folder}/plot_I_v_{PMm[i]}.png" , bbox_inches="tight")

            plt.figure()
            plt.plot( times, Iphin[:] ,label="nominal")
            plt.plot( times,  Iphim[:], label="min")
            plt.plot( times,  IphiM[:], label="max")
            plt.title("QoI: Iphi")
            plt.xlabel("time (s)")
            plt.ylabel("I_phi")
            plt.legend()
            plt.grid(True) 
            # Salvataggio del grafico 
            plt.savefig(f"./outputs/plots/{data_folder}/plot_I_phi_{PMm[i]}.png" , bbox_inches="tight")

            plt.figure()
            plt.plot( times, Iomegan[:],label="nominal")
            plt.plot( times, Iomegam[:], label="min")
            plt.plot( times, IomegaM[:], label="max")
            plt.title("QoI: Iomega")
            plt.xlabel("time (s)")
            plt.ylabel("I_omega")
            plt.legend()
            plt.grid(True) 
            # Salvataggio del grafico 
            plt.savefig(f"./outputs/plots/{data_folder}/plot_I_omega_{PMm[i]}.png" , bbox_inches="tight")

    
        print(f"\n🎉 All demonstrations completed successfully!")
        
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