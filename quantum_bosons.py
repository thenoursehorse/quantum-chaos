import time
import numpy as np

import os
import argparse
import textwrap

from kicked_boson.quantum.kicked_rotor import BosonChain
from kicked_boson.quantum.system import GenericSystemData
from kicked_boson.functions import golden_ratio

if __name__ == '__main__':
    description = textwrap.dedent('''\
         Kicked bosons chain:
            N           : number of sites
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-N', type=int, default=300)
    parser.add_argument('-num_ensembles', type=int, default=10)
    
    parser.add_argument('-J', type=float, default=1)
    #parser.add_argument('-J', type=float, default=8.9/(4.0*np.pi))
    
    parser.add_argument('-KChi', type=float, default=7)
    
    parser.add_argument('-Omega', type=float, default=np.pi/4.0)
    #parser.add_argument('-Omega', type=float, default=0.25)
    
    parser.add_argument('-eta', type=float, default=0)
    #parser.add_argument('-eta', type=float, default=golden_ratio())
    
    parser.add_argument('-theta_noise', type=float, default=0.0)
    #parser.add_argument('-phi_noise', type=float, default=0.05)
    parser.add_argument('-phi_noise', type=float, default=0.0125)
    parser.add_argument('-eta_noise', type=float, default=0)
    
    parser.add_argument('-num_modes', type=int, default=2)
    parser.add_argument('-excitations', type=int, default=1)
    parser.add_argument('-periodic', type=int, default=0)
    
    parser.add_argument('-use_qutip', type=int, default=0)
    
    parser.add_argument('-root_folder', type=str, default='./')
    parser.add_argument('-save_plots', type=int, default=0)
    parser.add_argument('-show_plots', type=int, default=1)
    parser.add_argument('-save_data', type=int, default=0)
    args = parser.parse_args()
    print(vars(args))

    Omega = args.Omega
    
    if np.abs(args.KChi) > 1e-8:
        J = args.KChi / (16.0 * Omega)
        # phi_noise = input * pi / Omega = input * pi / (pi/4) = 4 * input -> input = 0.05/4 = 0.0125
        phi_noise = args.phi_noise * np.pi / Omega # pi because kicked_rotor already multiplies by pi
        folder = f'{args.root_folder}/data/N{args.N}_Nsamp{args.num_ensembles}/KChi{args.KChi:.2f}'
    else:
        J = args.J
        phi_noise = args.phi_noise
        folder = f'{args.root_folder}/data/N{args.N}_Nsamp{args.num_ensembles}/J{args.J:.2f}_Omega{args.Omega:.2f}'
    
    if args.save_plots or args.save_data:
        os.makedirs(f'{folder}', exist_ok=True)
    if args.save_plots:
        print(f"Saving plots to folder '{folder}'")

    start = time.time()
    bosons = BosonChain(N=args.N,
                        num_ensembles=args.num_ensembles,
                        J=J,
                        Omega=Omega,
                        eta=args.eta,
                        theta_noise=args.theta_noise,
                        phi_noise=phi_noise,
                        eta_noise=args.eta_noise,
                        num_modes=args.num_modes,
                        excitations=args.excitations,
                        periodic=args.periodic,
                        folder=folder,
                        use_qutip=args.use_qutip) 
    end = time.time()
    print("Unitary construction took", end-start)
        
    start = time.time()
    bosons.set_spectral_functions(window=0)
    end = time.time()
    print("Spectral function construction took", end-start)
        
    #start = time.time()
    #bosons.set_unitary_evolve(Ti=0.1, Tf=1e3, Nt=10, num_ensembles=2)
    ##bosons.set_unitary_evolve()
    #end = time.time()
    #print("Time evolve unitaries took", end-start)
        
    ## For frame_potential2, needs estimate=True
    #start = time.time()
    #bosons.set_unitary_fidelity()
    #end = time.time()
    #print("Unitaries fidelity took", end-start)
    
    start = time.time()
    bosons.set_fractal_dimension()
    end = time.time()
    print("Fractal dimension took", end-start)
    
    start = time.time()
    psi = np.zeros(bosons._d)
    psi[int(bosons._d/2)] = 1
    bosons.set_survival_probability_amplitude(psi)
    end = time.time()
    print("Survival probability took", end-start)
    
    start = time.time()
    bosons.set_fractal_dimension_state()
    end = time.time()
    print("Fractal dimension state took", end-start)
            
    start = time.time()
    bosons.unfold_energies(save=args.save_plots, show=args.show_plots, plot=True)
    print("Unfolding energies took", end-start)
    
    # Plots
    if args.save_plots or args.show_plots:
        bosons.plot_eigenenergies(save=args.save_plots, show=args.show_plots)
        bosons.plot_ratios(save=args.save_plots, show=args.show_plots)
        bosons.plot_spacings(save=args.save_plots, show=args.show_plots)
        bosons.plot_fractal_dimension(save=args.save_plots, show=args.show_plots)
        bosons.plot_frame_potential(save=args.save_plots, show=args.show_plots, window=0, estimate=False)
        bosons.plot_spectral_functions(save=args.save_plots, show=args.show_plots)
        bosons.plot_loschmidt_echo(save=args.save_plots, show=args.show_plots)
        bosons.plot_fractal_dimension_state(save=args.save_plots, show=args.show_plots)
        bosons.plot_survival_probability(psi, save=args.save_plots, show=args.show_plots)
        
    # Some averages
    r_avg, r_err = bosons.average_level_ratios()
    eta_ratios = bosons.eta_ratios()
    p_pois, p_goe, p_gue = bosons.chi_distance()

    if args.save_data:
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

        # TODO
        # fix phi (Omega) to pi/4
        # fix theta (J) to Kchi/(16phi)
        # Now vary Kchi
        # Now vary J.