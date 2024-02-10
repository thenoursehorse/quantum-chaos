import time
import numpy as np

import os
import argparse

from quantum_chaos.quantum.fibonacci_bosons import FibonacciBosons
from quantum_chaos.quantum.system import GenericSystemData

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', type=int, default=10)
    parser.add_argument('-num_ensembles', type=int, default=1)
    
    parser.add_argument('-thetaOmega', type=float, default=7.4)
    parser.add_argument('-Omega', type=float, default=np.pi/4.0)
    
    parser.add_argument('-root_folder', type=str, default='./')
    parser.add_argument('-scale_width', type=float, default=1)
    parser.add_argument('-save_plots', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-show_plots', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-save_data', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    print(vars(args))
    
    time_arr = np.linspace(1, 400, 10000)
        
    ##########################################
    ## Make parameters for Floquet operator ##
    ##########################################
            
    theta = args.thetaOmega / (16.0 * args.Omega)
    
    folder = f'{args.root_folder}/data/M{args.M}_Nsamp{args.num_ensembles}/thetaOmega{args.thetaOmega:.2f}'
    if args.save_plots or args.save_data:
        os.makedirs(f'{folder}', exist_ok=True)
    if args.save_plots:
        print(f"Saving plots to folder '{folder}'")

    start = time.time()
    model = FibonacciBosons(M=args.M,
                            num_ensembles=args.num_ensembles,
                            theta=theta,
                            Omega=args.Omega,
                            folder=folder,
    )
    end = time.time()
    print("Unitary construction took", end-start)
    
    #start = time.time()
    #model.unfold_energies(save=args.save_plots, show=args.show_plots, plot=True)
    #end = time.time()
    #print("Unfolding energies took", end-start)
        
    start = time.time()
    model.set_spectral_functions(time=time_arr, unfold=False)
    end = time.time()
    print("Spectral function construction took", end-start)
            
    # Plots
    if args.save_plots or args.show_plots:
        start = time.time()
        #model.plot_eigenenergies(save=args.save_plots, show=args.show_plots, ylabel=r'$\xi_{\alpha}$', xlabel=r'$\alpha$')
        #model.plot_unfolded_eigenenergies(save=args.save_plots, show=args.show_plots, ylabel=r'$\bar{\xi}_{\alpha}$', xlabel=r'$\alpha$')
        model.plot_ratios(save=args.save_plots, show=args.show_plots, scale_width=args.scale_width)
        #model.plot_spacings(save=args.save_plots, show=args.show_plots)
        model.plot_spectral_functions(save=args.save_plots, show=args.show_plots, scale_width=args.scale_width)
        model.plot_loschmidt_echo(save=args.save_plots, show=args.show_plots)
        
        #if args.only_spectral:
        #    model.plot_frame_potential(save=args.save_plots, show=args.show_plots, window=0, non_haar=False, scale_width=args.scale_width)
        #print("Plotting took", end-start)

    if args.save_data:
        r_avg, r_err, r_I = model.average_level_ratios()
        eta_ratios = model.eta_ratios()
        
        filename = folder + "/data.h5"
        print(f"Saving data to '{filename}'")
        modeldata = GenericSystemData(filename=filename,
                                      r_avg=r_avg,
                                      r_err=r_err,
                                      eta_ratios=eta_ratios,
                                      )
        modeldata.save()