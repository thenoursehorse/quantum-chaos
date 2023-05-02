import time
import numpy as np

import os
import argparse
import textwrap

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kicked_boson.quantum.system import GenericSystemData

if __name__ == '__main__':
    description = textwrap.dedent('''\
         Kicked bosons chain:
            N           : number of sites
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-N', type=int, default=300)
    parser.add_argument('-num_ensembles', type=int, default=10)
    parser.add_argument('-root_folder', type=str, default='./')
    parser.add_argument('-KChi_min', type=float, default=0)
    parser.add_argument('-KChi_max', type=float, default=50)
    parser.add_argument('-dKChi', type=float, default=0.5)
    parser.add_argument('-eps_min', type=float, default=0.1)
    parser.add_argument('-eps_max', type=float, default=0.9)
    parser.add_argument('-deps', type=float, default=0.01)
    parser.add_argument('-sigma', type=float, default=5)
    parser.add_argument('-use_filter', type=int, default=1)
    parser.add_argument('-save_plots', type=int, default=0)
    parser.add_argument('-show_plots', type=int, default=1)
    args = parser.parse_args()
    print(vars(args))
    
    sigma = args.sigma
    use_filter = args.use_filter
    show = args.show_plots
    save = args.save_plots

    KChi_arr = np.arange(args.KChi_min, args.KChi_max+args.dKChi/2.0, args.dKChi)
    eps_arr = np.arange(args.eps_min, args.eps_max+args.deps/2.0, args.deps)
    
    r_avg_arr = np.empty(shape=(KChi_arr.size, eps_arr.size))
    r_err_arr = np.empty(shape=(KChi_arr.size, eps_arr.size))
    
    # Load data
    start = time.time()
    for i,KChi in enumerate(KChi_arr):
        for j,eps in enumerate(eps_arr):
            folder = f'{args.root_folder}/data/N{args.N}_Nsamp{args.num_ensembles}/KChi{KChi:.2f}_disorder{eps:.2f}'
            filename = folder + "/data.h5"
            modeldata = GenericSystemData(filename=filename)
            modeldata.load()
            r_avg_arr[i,j] = modeldata.r_avg
            r_err_arr[i,j] = modeldata.r_err
    end = time.time()
    print("Loading data took", end-start)
    
    sns.set_theme()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1, layout="constrained")
    fig.set_size_inches(3.386,2*2.54)
    sns.despine(ax=ax[0], top=True, right=True, left=True, bottom=True)
    sns.despine(ax=ax[1])

    poiss = 2*np.log(2)-1
    goe = 4-2*np.sqrt(3)
    gue = 2*np.sqrt(3)/np.pi - 1.0/2.0
    gse = (32.0/15.0) * np.sqrt(3)/np.pi - 1.0/2.0

    if use_filter:
        c = ax[0].pcolormesh(KChi_arr, eps_arr, gaussian_filter(r_avg_arr.T,sigma), vmin=poiss, vmax=goe)
    else:
        c = ax[0].pcolormesh(KChi_arr, eps_arr, r_avg_arr.T, vmin=poiss, vmax=goe)
    
    ct = ax[0].contour(KChi_arr, eps_arr, gaussian_filter(r_avg_arr.T,sigma), levels=[goe,gue,gse], colors=('k',))
    fmt = {}
    strs = ['GOE', 'GUE', 'GSE']
    for l,s in zip(ct.levels,strs):
        fmt[l] = s
    plt.clabel(ct, fmt=fmt, colors='k')
    
    ax[0].tick_params(axis='both', which='both',length=0)
    ax[0].set_xlim(xmin=KChi_arr[0], xmax=KChi_arr[-1])
    ax[0].set_yticks(np.linspace(KChi_arr[0], KChi_arr[1], 6))
    ax[0].set_ylim(ymin=eps_arr[0], ymax=eps_arr[-1])
    ax[0].set_yticks(np.linspace(eps_arr[0], eps_arr[-1], 5))
    ax[0].set_xlabel(r'$\bar{G}$')
    ax[0].set_ylabel(r'$\bar{\epsilon}$')

    ticks = np.linspace(poiss, goe, 6)
    cbar = fig.colorbar(c, ax=ax[0], ticks=ticks, pad=0.01, format=lambda x, _: f"{x:.2f}")
    cbar.outline.set_visible(False)
    cbar.ax.set_title(r'$\langle r \rangle$')
    cbar.ax.tick_params(axis='both', which='both',length=0)

    eps_val = 0.2
    idx = np.where(np.abs(eps_arr - eps_val) < 1e-10)[0][0]
    ax[1].plot(KChi_arr, r_avg_arr[:,idx])
    ax[1].fill_between(KChi_arr,
                       r_avg_arr[:,idx]-r_err_arr[:,idx], 
                       r_avg_arr[:,idx]+r_err_arr[:,idx], 
                       alpha=0.1,
                       )
    
    transx = ax[1].get_xaxis_transform()
    transy = ax[1].get_yaxis_transform()
    
    ax[1].axhline(y=poiss, linestyle='--', color='black')
    ax[1].text(0.5, poiss, 'Poisson', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    
    ax[1].axhline(y=goe, linestyle='--', color='black')
    ax[1].text(0.4, goe, 'GOE', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    
    ax[1].axhline(y=gue, linestyle='--', color='black')
    ax[1].text(0.4, gue, 'GUE', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    
    ax[1].axhline(y=gse, linestyle='--', color='black')
    ax[1].text(0.8, gse, 'GSE', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    
    ax[1].text(0.5, (goe+poiss)/2.0, r'$\bar{\epsilon} = eps_val$'.replace('eps_val', str(eps_val)), transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    
    ax[1].set_xlim(xmin=KChi_arr[0], xmax=KChi_arr[-1])
    ax[1].set_xlabel(r'$\bar{G}$')
    ax[1].set_ylabel(r'$\langle r \rangle$')
    #ax[1].set_title(r'$\bar{\epsilon} = eps_val$'.replace('eps_val', str(eps_val)))

    if save:
        filename = f'{args.root_folder}/data/N{args.N}_Nsamp{args.num_ensembles}/r_avg.pdf'
        fig.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()

    # Places to mark (but GUE or GSE never fully follows the r spectrum !)
    # GOE
    # (0.2, 8)
    # (0.4, 40)