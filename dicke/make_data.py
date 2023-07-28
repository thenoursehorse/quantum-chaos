import time
import numpy as np

import os
import argparse
import textwrap

from quantum_chaos.quantum.dicke import Dicke

if __name__ == '__main__':
    description = textwrap.dedent('''\
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-N', type=int, default=32) #16
    parser.add_argument('-Nc', type=int, default=320) #320
    parser.add_argument('-kappa', type=float, default=1.2)
    parser.add_argument('-lambda0', type=float, default=0.5)
    parser.add_argument('-cutoff_upper', type=float, default=4)
    parser.add_argument('-cutoff_lower', type=float, default=0.4)
    parser.add_argument('-root_folder', type=str, default='./')
    parser.add_argument('-save_plots', type=int, default=0)
    parser.add_argument('-show_plots', type=int, default=1)
    parser.add_argument('-save_data', type=int, default=0)
    args = parser.parse_args()
    print(vars(args))
    
    Ti = 0.01
    Tf = 1000
    Nt = 1000
    
    folder = f'{args.root_folder}/data/N{args.N}_Nc{args.Nc}_kappa{args.kappa}_lambda{args.lambda0}/'
    
    if args.save_plots or args.save_data:
        os.makedirs(f'{folder}', exist_ok=True)
    if args.save_plots:
        print(f"Saving plots to folder '{folder}'")

    start = time.time()
    model = Dicke(N=args.N,
                  Nc=args.Nc,
                  kappa=args.kappa,
                  lambda0=args.lambda0,
                  cutoff_upper=args.cutoff_upper,
                  cutoff_lower=args.cutoff_lower,
                  folder=folder) 
    end = time.time()
    print("Hamiltonian construction took", end-start)
    
    start = time.time()
    model.unfold_energies(save=args.save_plots, show=args.show_plots, plot=True)
    end = time.time()
    print("Unfolding energies took", end-start)
    
    start = time.time()
    model.set_spectral_functions(Ti=Ti, Tf=Tf, Nt=Nt)
    end = time.time()
    print("Spectral function construction took", end-start)
        
    #start = time.time()
    #model.set_unitary_evolve(Ti=Ti, Tf=Tf, Nt=Nt)
    #end = time.time()
    #print("Time evolve unitaries took", end-start)
    
    start = time.time()
    model.set_fractal_dimension(goe=False, gue=False)
    end = time.time()
    print("Fractal dimension took", end-start)
    
    #start = time.time()
    #psi = np.zeros(model._dim)
    #psi[int(model._dim/2)] = 1
    #model.set_survival_probability_amplitude(psi)
    #end = time.time()
    #print("Survival probability took", end-start)
    
    #start = time.time()
    #model.set_fractal_dimension_state()
    #end = time.time()
    #print("Fractal dimension state took", end-start)
        
    #start = time.time()
    #success_all, elements_all_t = model.check_submatrix()
    #print("Gaussian after:", np.where(success_all > 0)[0])
    #end = time.time()
    #print("Submatrix took", end-start)
        
    # Plots
    if args.save_plots or args.show_plots:
        start = time.time()
        model.plot_eigenenergies(save=args.save_plots, show=args.show_plots)
        model.plot_unfolded_eigenenergies(save=args.save_plots, show=args.show_plots)
        model.plot_ratios(save=args.save_plots, show=args.show_plots)
        model.plot_fractal_dimension(save=args.save_plots, show=args.show_plots)
        model.plot_spectral_functions(save=args.save_plots, show=args.show_plots)
        model.plot_loschmidt_echo(save=args.save_plots, show=args.show_plots)
        model.plot_frame_potential(save=args.save_plots, show=args.show_plots, window=0, non_haar=False)
        #model.plot_fractal_dimension_state(save=args.save_plots, show=args.show_plots)
        #model.plot_survival_probability(psi, save=args.save_plots, show=args.show_plots)
        end = time.time()
        print("Plotting took", end-start)