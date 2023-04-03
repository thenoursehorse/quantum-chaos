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
    parser.add_argument('-num_ensembles', type=int, default=100)
    parser.add_argument('-J', type=float, default=1)
    #parser.add_argument('-J', type=float, default=8.9/(4.0*np.pi))
    #parser.add_argument('-Omega', type=float, default=0.25*np.pi)
    parser.add_argument('-Omega', type=float, default=0.25)
    parser.add_argument('-eta', type=float, default=0)
    #parser.add_argument('-eta', type=float, default=golden_ratio())
    parser.add_argument('-theta_noise', type=float, default=0.0)
    parser.add_argument('-phi_noise', type=float, default=0.05)
    parser.add_argument('-eta_noise', type=float, default=0)
    parser.add_argument('-num_modes', type=int, default=2)
    parser.add_argument('-excitations', type=int, default=1)
    parser.add_argument('-periodic', type=int, default=0)
    parser.add_argument('-root_folder', type=str, default='./')
    parser.add_argument('-save', type=int, default=0)
    parser.add_argument('-show', type=int, default=1)
    args = parser.parse_args()
    print(vars(args))

    folder = f'{args.root_folder}/figs/N{args.N}_Nsamp{args.num_ensembles}/J{args.J:.2f}_Omega{args.Omega:.2f}'
    if args.save:
        os.makedirs(f'{folder}', exist_ok=True)

    start = time.time()
    bosons = BosonChain(N=args.N,
                        num_ensembles=args.num_ensembles,
                        J=args.J,
                        Omega=args.Omega,
                        eta=args.eta,
                        theta_noise=args.theta_noise,
                        phi_noise=args.phi_noise,
                        eta_noise=args.eta_noise,
                        num_modes=args.num_modes,
                        excitations=args.excitations,
                        periodic=args.periodic,
                        folder=folder) 
    end = time.time()
    print("Unitary construction took", end-start)
        
    start = time.time()
    bosons.set_spectral_functions(window=0)
    end = time.time()
    print("Spectral function construction took", end-start)
        
    start = time.time()
    bosons.set_unitary_evolve(num_ensembles=10)
    end = time.time()
    print("Time evolve unitaries took", end-start)
        
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
    bosons.unfold_energies(save=args.save, show=args.show, plot=True)
    print("Unfolding energies took", end-start)

    # Plots
    bosons.plot_eigenenergies(save=args.save, show=args.show)
    bosons.plot_ratios(save=args.save, show=args.show)
    bosons.plot_fractal_dimension(save=args.save, show=args.show)
    bosons.plot_frame_potential(save=args.save, show=args.show, window=0, estimate=False)
    bosons.plot_spectral_functions(save=args.save, show=args.show)
    bosons.plot_loschmidt_echo(save=args.save, show=args.show)
    bosons.plot_spacings(save=args.save, show=args.show)
        
    # Some averages
    r_avg, r_err = bosons.average_level_ratios()
    eta_ratios = bosons.eta_ratios()
    p_pois, p_goe, p_gue = bosons.chi_distance()

    print(f'eta = {eta_ratios}')

    #modeldata = SystemData(filename=filename_elm,
    #                    tlist=tlist,
    #    nndata.save()    