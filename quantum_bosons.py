import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import os
import argparse
import textwrap

from kicked_boson.quantum.system import BosonChain
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
    parser.add_argument('-J', type=float, default=8.9/(4.0*np.pi))
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
    args = parser.parse_args()
    print(vars(args))

    def run(J, save=False, show=True):
        Omega = args.Omega
        folder = f'{args.root_folder}/figs/N{args.N}_Nsamp{args.num_ensembles}/J{J:.2f}_Omega{Omega:.2f}'
        if save:
            os.makedirs(f'{folder}', exist_ok=True)

        start = time.time()
        bosons = BosonChain(N=args.N,
                            num_ensembles=args.num_ensembles,
                            J=J,
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
        bosons.plot_eigenenergies(save=save, show=show)
        bosons.plot_ratios(save=save, show=show)
        bosons.plot_frame_potential(save=save, show=show, window=0)
        bosons.plot_spectral_functions(save=save, show=show)
        bosons.plot_loschmidt_echo(save=save, show=show)
        bosons.unfold_energies(save=save, show=show, plot=True)
        bosons.plot_spacings(save=save, show=show)
        end = time.time()
        print("Plotting took", end-start)

        start = time.time()
        bosons.set_unitary_time(num_ensembles=2)
        bosons.plot_frame_potential2(save=save, show=show, window=2)
        end = time.time()
        print("Frame potential 2 took", end-start)
        
        return bosons


    # FIXME
    # check other measures used in those two papers (circuit paper and the debye model paper)
    
    bosons = run(args.J)
    r_avg, r_err = bosons.average_level_ratios()
    p_pois, p_goe, p_gue = bosons.chi_distance()

    # NOTE from https://doi.org/10.1088/1742-5468/acb52d
    # A popular measure of information mixing is the k -design state that cannot be distinguished from
    # the Haar random state when considering averages of polynomials of degree not higher than k.