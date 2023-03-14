import time
import numpy as np
import matplotlib.pyplot as plt

import argparse
import textwrap

import seaborn as sns

import os
import psutil
import ray
from ray.util.multiprocessing import Pool
#from multiprocessing import Pool

from kicked_boson.quantum.system import BosonChain

golden_ratio = (1 + 5 ** 0.5) / 2

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
parser.add_argument('-root_folder', type=str, default='./')
parser.add_argument('-Ji', type=float, default=0)
parser.add_argument('-Jf', type=float, default=5)
parser.add_argument('-dJ', type=float, default=0.1)
parser.add_argument('-num_cpus', type=int, default=1)
args = parser.parse_args()

#@ray.remote
def run(J):
    Omega = 1
    folder = f'{args.root_folder}/figs/N{args.N}_Nsamp{args.num_ensembles}/J{J:.2f}_Omega{Omega:.2f}'
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
    
if __name__ == '__main__':
    print(vars(args))

    os.makedirs(f'{args.root_folder}/figs/', exist_ok=True)

    if args.num_cpus == 0:
        num_cpus = psutil.cpu_count(logical=False)
    else:
        num_cpus = args.num_cpus

    ray.init(num_cpus=num_cpus)

    J_arr = np.arange(args.Ji, args.Jf+0.01, args.dJ)
    p_pois = np.empty(len(J_arr))
    p_goe = np.empty(len(J_arr))
    p_gue = np.empty(len(J_arr))

    print(f'Spreading {len(J_arr)} parameters over {num_cpus} cpus.')
    
    with Pool(num_cpus) as pool: 
        for i,result in enumerate(pool.imap_unordered(run, J_arr)):
            p_pois[i], p_goe[i], p_gue[i] = result

    
    
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
    ax.set_xlim(xmin=args.Ji, xmax=args.Jf)
    ax.set_ylim(ymin=0, ymax=1)
    fig.savefig(f'{args.root_folder}/figs/N{args.N}_Nsamp{args.num_ensembles}/p-values.pdf', bbox_inches="tight")