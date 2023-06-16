import time
import numpy as np

import os
import argparse
import textwrap

from kicked_boson.quantum.pauli_3body import Pauli3Body
from kicked_boson.quantum.system import GenericSystemData
from kicked_boson.functions import golden_ratio

if __name__ == '__main__':
    description = textwrap.dedent('''\
         Pauli all-to-all 3-body interaction:
            N           : number of sites
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-N', type=int, default=5)
    parser.add_argument('-num_ensembles', type=int, default=10)
    parser.add_argument('-root_folder', type=str, default='./')
    parser.add_argument('-save_plots', type=int, default=0)
    parser.add_argument('-show_plots', type=int, default=1)
    parser.add_argument('-save_data', type=int, default=0)
    args = parser.parse_args()
    print(vars(args))

    Ti = 0.01
    Tf = 100
    Nt = 1000

    folder = f'{args.root_folder}/data/N{args.N}_Nsamp{args.num_ensembles}/'
    
    if args.save_plots or args.save_data:
        os.makedirs(f'{folder}', exist_ok=True)
    if args.save_plots:
        print(f"Saving plots to folder '{folder}'")

    start = time.time()
    model = Pauli3Body(N=args.N,
                       num_ensembles=args.num_ensembles,
                       folder=folder,
                      )
    end = time.time()
    print("Unitary construction took", end-start)
    start = time.time()
    model.set_spectral_functions(Ti=Ti, Tf=Tf, Nt=Nt)
    end = time.time()
    print("Spectral function construction took", end-start)
        
    # This is to calculate the frame potential for the ensemble
    # rather than the Haar averaged frame potential. It is expensive
    start = time.time()
    model.set_unitary_evolve(num_ensembles=10)
    #model.set_unitary_evolve(num_ensembles=None)
    #model.set_unitary_evolve()
    end = time.time()
    print("Time evolve unitaries took", end-start)
        
    # For non_haar frame_potential
    start = time.time()
    model.set_unitary_fidelity()
    end = time.time()
    print("Unitaries fidelity took", end-start)
    
    start = time.time()
    model.set_fractal_dimension()
    end = time.time()
    print("Fractal dimension took", end-start)
    
    start = time.time()
    psi = np.zeros(model._d)
    psi[int(model._d/2)] = 1
    model.set_survival_probability_amplitude(psi)
    end = time.time()
    print("Survival probability took", end-start)
    
    start = time.time()
    model.set_fractal_dimension_state()
    end = time.time()
    print("Fractal dimension state took", end-start)
            
    #start = time.time()
    #model.unfold_energies(save=args.save_plots, show=args.show_plots, plot=True)
    #end = time.time()
    #print("Unfolding energies took", end-start)
    
    # Plots
    if args.save_plots or args.show_plots:
        start = time.time()
        model.plot_eigenenergies(save=args.save_plots, show=args.show_plots)
        model.plot_ratios(save=args.save_plots, show=args.show_plots, scale_width=0.5)
        model.plot_fractal_dimension(save=args.save_plots, show=args.show_plots)
        model.plot_spectral_functions(save=args.save_plots, show=args.show_plots)
        model.plot_frame_potential(save=args.save_plots, show=args.show_plots, window=0, non_haar=True)
        model.plot_loschmidt_echo(save=args.save_plots, show=args.show_plots)
        model.plot_fractal_dimension_state(save=args.save_plots, show=args.show_plots)
        model.plot_survival_probability(psi, save=args.save_plots, show=args.show_plots)
        end = time.time()
        print("Plotting took", end-start)
        
    if args.save_data:
        r_avg, r_err, r_I = model.average_level_ratios()
        eta_ratios = model.eta_ratios()
        p_pois, p_goe, p_gue = model.chi_distance()
        
        filename = folder + "/data.h5"
        print(f"Saving data to '{filename}'")
        modeldata = GenericSystemData(filename=filename,
                                      r_avg=r_avg,
                                      r_err=r_err,
                                      eta_ratios=eta_ratios,
                                      p_pois=p_pois,
                                      p_goe=p_goe,
                                      p_gue=p_gue,
                                     )
        modeldata.save()