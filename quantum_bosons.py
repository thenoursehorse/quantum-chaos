import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import argparse
import textwrap

import qutip as qt

from kicked_boson.quantum.system import BosonChain

golden_ratio = (1 + 5 ** 0.5) / 2

def linear_extrapolate_zero(x, y):
    def f(x, a, c):
        return a*x + c
    popt, pcov = curve_fit(f, x, y)
    c = popt[-1]
    c_err = np.sqrt(pcov[-1,-1])
    return c, c_err


if __name__ == '__main__':
    description = textwrap.dedent('''\
         Kicked bosons chain:
            N           : number of sites
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-N', type=int, default=300)
    parser.add_argument('-num_ensembles', type=int, default=1)
    parser.add_argument('-J', type=float, default=8.9/(4.0*np.pi))
    parser.add_argument('-eta', type=float, default=golden_ratio)
    parser.add_argument('-Omega', type=float, default=0.25*np.pi)
    parser.add_argument('-theta_noise', type=float, default=0.0)
    parser.add_argument('-phi_noise', type=float, default=0.05)
    parser.add_argument('-eta_noise', type=float, default=0)
    parser.add_argument('-num_modes', type=int, default=2)
    parser.add_argument('-excitations', type=int, default=1)
    parser.add_argument('-periodic', type=int, default=0)
    parser.add_argument('-plot', type=int, default=1)
    args = parser.parse_args()
    print(vars(args))

    # FIXME do a theta/phi varying plot and it's deviation from a GUE level statistics
    # and Poisson level statistics.

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
                        periodic=args.periodic) 

    end = time.time()
    print("Unitary construction took", end-start)
    
    #bosons.unfold_energies(plot=True)
    #bosons.plot_eigenenergies()
    
    #bosons.plot_spacings()
    bosons.plot_ratios()
    
    bosons.set_form_factor()
    bosons.plot_form_factor()
    
    bosons.plot_frame_potential()
    bosons.plot_loschmidt_echo()

    print(bosons.chi_distance())