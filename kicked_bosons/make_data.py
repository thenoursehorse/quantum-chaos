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
    parser.add_argument('-num_ensembles', type=int, default=10)
    
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
    
    parser.add_argument('-only_spectral', type=int, default=0)
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

    #time_arr = [1,2,3,4,5,6,7,8,9,10]
    #time_arr += [i for i in range(11,101)]
    #time_arr += [i for i in range(110,1010,10)]
    #time_arr += [i for i in range(1100,10100,100)]
    #time_arr = [1,2,3,4,5,6,7,8,9,10,
    #            20,30,40,50,60,70,80,90,100,
    #            200,300,400,500,600,700,800,900,1000,
    #            2000,3000,4000,5000,6000,7000,7000,8000,9000,10000]
    time_arr = np.arange(1,1000+0.5,1)
    #time_arr = np.arange(1,300+0.5,1)
    #time_arr = None
        
    t_idx = -1 #299
    ix = 0
    iy = 0 #199
    stride = 5 # 30
    vmax_abs = 0.3 # 0.2
    m = -1

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
        
    start = time.time()
    model.set_spectral_functions(time=time_arr, unfold=False)
    end = time.time()
    print("Spectral function construction took", end-start)
            
    if not args.only_spectral:
        # This is to calculate the frame potential for the ensemble
        # rather than the Haar averaged frame potential. It is expensive
        # because it evolves every unitary.
        start = time.time()
        #model.set_unitary_evolve(num_ensembles=args.num_ensembles, time=time_arr)
        model.set_unitary_evolve_floquet(num_ensembles=args.num_ensembles, time=time_arr)
        end = time.time()
        print("Time evolve unitaries took", end-start)
            
        # For non_haar frame_potential
        start = time.time()
        model.set_unitary_fidelity(num_ensembles=50)
        #model.set_unitary_fidelity()
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

        start = time.time()
        success_all, elements_all_t = model.check_submatrix(ix=ix, iy=iy, stride=stride)
        #print("Gaussian after (prop):", np.where(prop > 0.5)[0])
        print("Gaussian after:", np.where(success_all > 0)[0])
        #print("Gaussian after (success_all_scaled):", np.where(success_all_scaled > 0)[0])
        #print("Gaussian after (pvalue_combined):", np.where(pvalue_combined > 0.05)[0])
        #print("Gaussian after (pvalue_bootstrap):", np.where(pvalue_bootstrap > 0.95)[0])
        end = time.time()
        print("Submatrix took", end-start)
                    
    # Plots
    if args.save_plots or args.show_plots:
        start = time.time()
        model.plot_eigenenergies(save=args.save_plots, show=args.show_plots, ylabel=r'$\xi_{\alpha}$', xlabel=r'$\alpha$')
        model.plot_unfolded_eigenenergies(save=args.save_plots, show=args.show_plots, ylabel=r'$\bar{\xi}_{\alpha}$', xlabel=r'$\alpha$')
        model.plot_ratios(save=args.save_plots, show=args.show_plots, scale_width=args.scale_width)
        model.plot_spacings(save=args.save_plots, show=args.show_plots)
        model.plot_spectral_functions(save=args.save_plots, show=args.show_plots, scale_width=args.scale_width)
        model.plot_loschmidt_echo(save=args.save_plots, show=args.show_plots)
        if args.only_spectral:
            model.plot_frame_potential(save=args.save_plots, show=args.show_plots, window=0, non_haar=False, scale_width=args.scale_width)
        
        if not args.only_spectral:
            model.plot_frame_potential(save=args.save_plots, show=args.show_plots, window=0, non_haar=True, scale_width=args.scale_width)
            model.plot_fractal_dimension(save=args.save_plots, show=args.show_plots)
            model.plot_fractal_dimension_state(save=args.save_plots, show=args.show_plots)
            model.plot_survival_probability(psi, save=args.save_plots, show=args.show_plots)
            
            model.plot_matrix(save=args.save_plots, show=args.show_plots, scale_width=args.scale_width, t_idx=t_idx, vmin=-0.2, vmax=0.2, vmax_abs=vmax_abs)
            model.plot_submatrix(save=args.save_plots, show=args.show_plots, scale_width=args.scale_width, t_idx=t_idx, ix=ix, iy=iy, stride=stride, vmin=-0.05, vmax=0.05, vmax_abs=vmax_abs)
            #model.plot_submatrix_probability(save=args.save_plots, show=args.show_plots, t_idx=t_idx, ix=ix, iy=iy, stride=stride, vec_scaled=elements_all_scaled_t[t_idx], m=m)
            model.plot_submatrix_probability(save=args.save_plots, show=args.show_plots, t_idx=t_idx, ix=ix, iy=iy, stride=stride, vec_scaled=elements_all_t[t_idx], m=m, scale_width=args.scale_width)#, bins=25)
            #model.plot_submatrix_probability(save=args.save_plots, show=args.show_plots, t_idx=t_idx, ix=ix, iy=iy, stride=stride, m=m, scale_width=args.scale_width)
            model.plot_qq(elements_all_t[t_idx], save=args.save_plots, show=args.show_plots, scale_width=args.scale_width)
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