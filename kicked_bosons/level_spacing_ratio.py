import time
import numpy as np

import os
import argparse
import textwrap

from quantum_chaos.quantum.kicked_bosons import KickedBosons
from quantum_chaos.quantum.system import GenericSystemData
from quantum_chaos.functions import golden_ratio

if __name__ == '__main__':
    description = textwrap.dedent('''\
         Kicked bosons chain:
            N           : number of sites
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-M', type=int, default=300)
    parser.add_argument('-num_ensembles', type=int, default=100)
    
    parser.add_argument('-thetaOmega', type=float, default=7.4)
    parser.add_argument('-thetaOmega_disorder', type=float, default=0.0)
    
    parser.add_argument('-Omega', type=float, default=np.pi/4.0)
    parser.add_argument('-WOmega', type=float, default=2)
    
    parser.add_argument('-etaOmega', type=float, default=0)
    #parser.add_argument('-etaOmega', type=float, default=golden_ratio())
    parser.add_argument('-etaOmega_disorder', type=float, default=0)
    
    parser.add_argument('-T', type=float, default=1)
    parser.add_argument('-num_modes', type=int, default=2)
    parser.add_argument('-excitations', type=int, default=1)
    parser.add_argument('-periodic', type=int, default=0)
    
    parser.add_argument('-use_qutip', type=int, default=0)
    
    parser.add_argument('-scale_width', type=float, default=1)
    parser.add_argument('-root_folder', type=str, default='./')
    parser.add_argument('-save_plots', type=int, default=0)
    parser.add_argument('-show_plots', type=int, default=1)
    parser.add_argument('-save_data', type=int, default=0)
    args = parser.parse_args()
    print(vars(args))
        
    ##########################################
    ## Make parameters for Floquet operator ##
    ##########################################
            
    theta = args.thetaOmega / (16.0 * args.Omega)
    theta_disorder = args.thetaOmega_disorder / (16.0 * args.Omega)
    phi_disorder = args.WOmega / (16.0 * args.Omega)
    eta = args.etaOmega / (16.0 * args.Omega)
    eta_disorder = args.etaOmega_disorder / (16.0 * args.Omega)
    
    folder = f'{args.root_folder}/data/M{args.M}_Nsamp{args.num_ensembles}/thetaOmega{args.thetaOmega:.2f}_WOmega{args.WOmega:.2f}'
    if args.save_plots or args.save_data:
        os.makedirs(f'{folder}', exist_ok=True)
    if args.save_plots:
        print(f"Saving plots to folder '{folder}'")

    start = time.time()
    model = KickedBosons(M=args.M,
                         num_ensembles=args.num_ensembles,
                         theta=theta,
                         Omega=args.Omega,
                         eta=eta,
                         theta_disorder=theta_disorder,
                         phi_disorder=phi_disorder,
                         eta_disorder=eta_disorder,
                         T=args.T,
                         num_modes=args.num_modes,
                         excitations=args.excitations,
                         periodic=args.periodic,
                         folder=folder,
                         use_qutip=args.use_qutip) 
    end = time.time()
    print("Unitary construction took", end-start)
    
    start = time.time()
    model.unfold_energies(save=args.save_plots, show=args.show_plots, plot=True)
    end = time.time()
    print("Unfolding energies took", end-start)
        
    if args.save_plots or args.show_plots:
        model.plot_eigenenergies(save=args.save_plots, show=args.show_plots, ylabel=r'$\xi_{\alpha}$', xlabel=r'$\alpha$')
        model.plot_unfolded_eigenenergies(save=args.save_plots, show=args.show_plots, ylabel=r'$\bar{\xi}_{\alpha}$', xlabel=r'$\alpha$')
        model.plot_ratios(save=args.save_plots, show=args.show_plots, scale_width=args.scale_width)
        model.plot_spacings(save=args.save_plots, show=args.show_plots)
        
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