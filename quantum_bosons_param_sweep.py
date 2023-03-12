import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import argparse
import textwrap

import seaborn as sns

import os

from kicked_boson.quantum.system import BosonChain
from kicked_boson.functions import ecdf, chi_distance

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
    parser.add_argument('-num_ensembles', type=int, default=100)
    parser.add_argument('-eta', type=float, default=golden_ratio)
    parser.add_argument('-theta_noise', type=float, default=0.0)
    parser.add_argument('-phi_noise', type=float, default=0.05)
    parser.add_argument('-eta_noise', type=float, default=0)
    parser.add_argument('-num_modes', type=int, default=2)
    parser.add_argument('-excitations', type=int, default=1)
    parser.add_argument('-periodic', type=int, default=0)
    parser.add_argument('-plot', type=int, default=1)
    args = parser.parse_args()
    print(vars(args))

    def run(J, Omega):
        folder = f'figs/N{args.N}_J{J:.2f}_Omega{Omega:.2f}'
        os.makedirs(f'{folder}', exist_ok=True)

        bosons = BosonChain(N=args.N,
                            num_ensembles=args.num_ensembles,
                            J=J,
                            Omega=Omega,
                            eta=args.eta,
                            theta_noise=args.theta_noise,
                            phi_noise=args.phi_noise,
                            eta_noise=args.eta_noise,
                            num_modes=args.num_modes,
                            excitations=args.excitations,
                            periodic=args.periodic,
                            folder=folder) 

        bosons.set_form_factor()
        bosons.plot_ratios(save=True, show=False)
        bosons.plot_form_factor(save=True, show=False)
        bosons.plot_frame_potential(save=True, show=False)
        bosons.plot_loschmidt_echo(save=True, show=False)
        return bosons.chi_distance()
    
    Ji = 0
    Jf = 5
    dJ = 0.1
    Omega = 1
    J_arr = np.arange(Ji, Jf+0.01, dJ)
    p_pois = np.empty(len(J_arr))
    p_goe = np.empty(len(J_arr))
    p_gue = np.empty(len(J_arr))
    
    for i,J in enumerate(J_arr):
        p_pois[i], p_goe[i], p_gue[i] = run(J=J, Omega=Omega)
    
    data = {'J': J_arr,
            'p_pois': p_pois,
            'p_goe': p_goe,
            'p_gue': p_gue,
            'cutoff': 0.05*np.ones(len(J_arr)),
            }

    sns.set_theme()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    fig, ax = plt.subplots()
    fig.set_size_inches(3.386,2.54)
    sns.despine()
    
    sns.lineplot(data=data, x='J', y='p_pois', label='Poisson', ax=ax)
    sns.lineplot(data=data, x='J', y='p_goe', label='GOE', ax=ax)
    sns.lineplot(data=data, x='J', y='p_gue', label='GUE', ax=ax)
    sns.lineplot(data=data, x='J', y='cutoff', ax=ax)
    ax.set_xlabel(r'$J/ \Omega$')
    ax.set_ylabel(r'$p$-value')
    ax.set_xlim(xmin=Ji, xmax=Jf)
    ax.set_ylim(ymin=0, ymax=1)
    fig.savefig(f'figs/p-values.pdf', bbox_inches="tight")