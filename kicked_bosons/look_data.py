import time
import numpy as np

import os
import argparse
import textwrap

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from quantum_chaos.quantum.system import GenericSystemData
from quantum_chaos.plotter import Plotter

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
    
    plot = Plotter(N_figs=2,
                   save_root=f'{args.root_folder}/data/N{args.N}_Nsamp{args.num_ensembles}/', 
                   save_filename='r_avg.pdf', 
                   show=args.show_plots, save=args.save_plots,
                   use_tics=True)
    fig = plot.fig
    axis = plot.axis
    
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
            
    poiss = 2*np.log(2)-1
    goe = 4-2*np.sqrt(3)
    gue = 2*np.sqrt(3)/np.pi - 1.0/2.0
    gse = (32.0/15.0) * np.sqrt(3)/np.pi - 1.0/2.0

    if use_filter:
        c = plot.colormesh(x=KChi_arr, y=eps_arr, z=gaussian_filter(r_avg_arr,sigma), vmin=poiss, vmax=goe, ax_idx=0)
    else:
        c = plot.colormesh(x=KChi_arr, y=eps_arr, z=r_avg_arr, vmin=poiss, vmax=goe, ax_idx=0)
    
    ct = axis[0].contour(KChi_arr, eps_arr, gaussian_filter(r_avg_arr.T,sigma), levels=[goe,gue,gse], colors=('k',))
    fmt = {}
    strs = ['GOE', 'GUE', 'GSE']
    for l,s in zip(ct.levels,strs):
        fmt[l] = s
    plt.clabel(ct, fmt=fmt, colors='k')
        
    axis[0].set_xlim(xmin=KChi_arr[0], xmax=KChi_arr[-1])
    axis[0].set_yticks(np.linspace(KChi_arr[0], KChi_arr[1], 6))
    axis[0].set_ylim(ymin=eps_arr[0], ymax=eps_arr[-1])
    axis[0].set_yticks(np.linspace(eps_arr[0], eps_arr[-1], 5))
    axis[0].set_xlabel(r'$\bar{G}$')
    axis[0].set_ylabel(r'$\bar{\epsilon}$')

    ticks = np.linspace(poiss, goe, 6)
    cbar = fig.colorbar(c, ax=axis[0], ticks=ticks, pad=0.01, format=lambda x, _: f"{x:.2f}")
    cbar.outline.set_visible(False)
    cbar.ax.set_title(r'$\langle r \rangle$')
    cbar.ax.tick_params(axis='both', which='both',length=0)
    
    # Places to mark (but GUE or GSE never fully follows the r spectrum !)
    plot.line(x=7.2, y=0.2, color='black', marker='o', ax_idx=0) # GOE
    plot.line(x=20, y=0.34, color='black', marker='^', ax_idx=0) # GUE
    plot.line(x=22, y=0.21, color='black', marker='v', ax_idx=0) # GSE
    plot.line(x=45, y=0.8, color='black', marker='*', ax_idx=0) # GOE
    
    eps_val = 0.2
    idx = np.where(np.abs(eps_arr - eps_val) < 1e-10)[0][0]
    plot.line(x=KChi_arr, y=r_avg_arr[:,idx], ax_idx=1)
    plot.fill_betweeny(x=KChi_arr, y_lower=r_avg_arr[:,idx]-r_err_arr[:,idx], y_upper=r_avg_arr[:,idx]+r_err_arr[:,idx], ax_idx=1)
    
    transy = axis[1].get_yaxis_transform()
    
    axis[1].axhline(y=poiss, linestyle='--', color='black')
    axis[1].text(0.5, poiss, 'Poisson', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    
    axis[1].axhline(y=goe, linestyle='--', color='black')
    axis[1].text(0.4, goe, 'GOE', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    
    axis[1].axhline(y=gue, linestyle='--', color='black')
    axis[1].text(0.4, gue, 'GUE', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    
    axis[1].axhline(y=gse, linestyle='--', color='black')
    axis[1].text(0.8, gse, 'GSE', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    
    axis[1].text(0.5, (goe+poiss)/2.0, r'$\bar{\epsilon} = eps_val$'.replace('eps_val', str(eps_val)), transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    
    axis[1].set_xlim(xmin=KChi_arr[0], xmax=KChi_arr[-1])
    axis[1].set_xlabel(r'$\bar{G}$')
    axis[1].set_ylabel(r'$\langle r \rangle$')
    #axis[1].set_title(r'$\bar{\epsilon} = eps_val$'.replace('eps_val', str(eps_val)))