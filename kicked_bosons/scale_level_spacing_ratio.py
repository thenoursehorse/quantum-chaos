import time
import numpy as np

from copy import deepcopy

import os
import argparse
import textwrap

from quantum_chaos.plotter import Plotter

if __name__ == '__main__':
    description = textwrap.dedent('''\
         Kicked bosons chain:
            N           : number of sites
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-M_min', type=int, default=50)
    parser.add_argument('-M_max', type=int, default=450)
    parser.add_argument('-M_step', type=int, default=50)
    parser.add_argument('-num_ensembles', type=int, default=100)
    parser.add_argument('-root_folder', type=str, default='./')
    
    parser.add_argument('-thetaOmega', type=float, default=7.4)
    parser.add_argument('-thetaOmega_disorder', type=float, default=0.0)
    parser.add_argument('-Omega', type=float, default=np.pi/4.0)
    parser.add_argument('-WOmega', type=float, default=2)
    parser.add_argument('-etaOmega', type=float, default=0)
    parser.add_argument('-etaOmega_disorder', type=float, default=0)
    
    parser.add_argument('-save_root', type=str, default='./')
    parser.add_argument('-save_plots', type=int, default=0)
    parser.add_argument('-show_plots', type=int, default=1)
    
    parser.add_argument('-from_data', type=int, default=1)
    args = parser.parse_args()
    print(vars(args))

    M_arr = np.arange(args.M_min, args.M_max + args.M_step/2.0, args.M_step, dtype=int)
    r_avg_arr = np.empty(shape=(M_arr.size))
    r_err_arr = np.empty(shape=(M_arr.size))

    if args.from_data: 
        from quantum_chaos.quantum.system import GenericSystemData
        
        for i, M in enumerate(M_arr):
            folder = f'{args.root_folder}/data/M{M}_Nsamp{args.num_ensembles}/thetaOmega{args.thetaOmega:.2f}_WOmega{args.WOmega:.2f}'
            filename = folder + "/data.h5"
            modeldata = GenericSystemData(filename=filename)
            modeldata.load()
            r_avg_arr[i] = modeldata.r_avg
            r_err_arr[i] = modeldata.r_err
    
    else:
        from quantum_chaos.quantum.kicked_bosons import KickedBosons
        
        # Get parameters for model
        theta = args.thetaOmega / (16.0 * args.Omega)
        theta_disorder = args.thetaOmega_disorder / (16.0 * args.Omega)
        phi_disorder = args.WOmega / (16.0 * args.Omega)
        eta = args.etaOmega / (16.0 * args.Omega)
        eta_disorder = args.etaOmega_disorder / (16.0 * args.Omega)
        
        for i, M in enumerate(M_arr):
            start = time.time()
            model = KickedBosons(M=M,
                                 num_ensembles=args.num_ensembles,
                                 theta=theta,
                                 Omega=args.Omega,
                                 eta=eta,
                                 theta_disorder=theta_disorder,
                                 phi_disorder=phi_disorder,
                                 eta_disorder=eta_disorder,
                                 calc_eigenvectors=False,
            )
            r_avg_arr[i], r_err_arr[i], _ = model.average_level_ratios()
            end = time.time()
            print(f"Unitary construction for M = {M} took", end-start)
    
    plot = Plotter(N_figs=1,
                   save_root=args.save_root,
                   save_filename=f'r_avg_scale_thetaOmega{args.thetaOmega:.2f}_WOmega{args.WOmega:.2f}.pdf',
                   show=args.show_plots, save=args.save_plots,
                   use_tics=True)
    fig = plot.fig
    axis = plot.axis
            
    goe = 4-2*np.sqrt(3)
    poiss = 2*np.log(2)-1
    
    plot.line(x=1/M_arr, y=r_avg_arr, color='black', marker='o', ax_idx=0)
    plot.line(x=[0, 1/M_arr[0]], y=[goe, goe], color='black', linestyle='--', ax_idx=0)
    axis[0].text(1/M_arr[0], goe, 'GOE', fontsize=8, va='center', ha='right', backgroundcolor='w')
    plot.line(x=[0, 1/M_arr[0]], y=[poiss, poiss], color='black', linestyle='--', ax_idx=0)
    axis[0].text(1/M_arr[0], poiss, 'Poisson', fontsize=8, va='center', ha='right', backgroundcolor='w')
    axis[0].text(1/M_arr[0], 0.1, r'$16W\Omega$ = {}'.format(args.WOmega), fontsize=8, va='center', ha='right', backgroundcolor='w')
    axis[0].text(1/M_arr[0], 0.2, r'$16\theta\Omega$ = {}'.format(args.thetaOmega), fontsize=8, va='center', ha='right', backgroundcolor='w')
    axis[0].set_xlim(xmin=0)
    axis[0].set_ylim(ymin=0)
    axis[0].set_xlabel(r'$1/M$')
    axis[0].set_ylabel(r'$\langle r \rangle$')

#    plot2 = Plotter(N_figs=1,
#                    save_root=args.save_root,
#                    save_filename=f'r_avg_scale_log_thetaOmega{args.thetaOmega:.2f}_WOmega{args.WOmega:.2f}',
#                    show=args.show_plots, save=args.save_plots,
#                    use_tics=True)
#    fig2 = plot2.fig
#    axis2 = plot2.axis
#    plot2.set_log(axis='x')
#            
#    goe = 4-2*np.sqrt(3)
#    poiss = 2*np.log(2)-1
#    
#    plot2.line(x=1/M_arr, y=r_avg_arr, color='black', marker='o', ax_idx=0)
#    plot2.line(x=[0, 1/M_arr[0]], y=[goe, goe], color='black', linestyle='--', ax_idx=0)
#    axis2[0].text(1/M_arr[0], goe, 'GOE', fontsize=8, va='center', ha='right', backgroundcolor='w')
#    plot2.line(x=[0, 1/M_arr[0]], y=[poiss, poiss], color='black', linestyle='--', ax_idx=0)
#    axis2[0].text(1/M_arr[0], poiss, 'Poisson', fontsize=8, va='center', ha='right', backgroundcolor='w')
#    axis2[0].text(1/M_arr[0], 0.1, r'$16W\Omega$ = {}'.format(args.WOmega), fontsize=8, va='center', ha='right', backgroundcolor='w')
#    axis2[0].text(1/M_arr[0], 0.2, r'$16\theta\Omega$ = {}'.format(args.thetaOmega), fontsize=8, va='center', ha='right', backgroundcolor='w')
#    axis2[0].set_xlim(xmin=0)
#    axis2[0].set_ylim(ymin=0)
#    axis2[0].set_xlabel(r'log$(1/M)$')
#    axis2[0].set_ylabel(r'$\langle r \rangle$')
