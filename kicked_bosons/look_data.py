import time
import numpy as np

import os
import argparse
import textwrap

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from quantum-chaos.quantum.system import GenericSystemData
from quantum-chaos.plotter import Plotter

if __name__ == '__main__':
    description = textwrap.dedent('''\
         Kicked bosons chain:
            N           : number of sites
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-M', type=int, default=300)
    parser.add_argument('-num_ensembles', type=int, default=100)
    parser.add_argument('-root_folder', type=str, default='./')
    
    parser.add_argument('-thetaOmega_min', type=float, default=0)
    parser.add_argument('-thetaOmega_max', type=float, default=20)
    parser.add_argument('-dthetaOmega', type=float, default=0.2)
    
    parser.add_argument('-WOmega_min', type=float, default=1)
    parser.add_argument('-WOmega_max', type=float, default=9)
    parser.add_argument('-dWOmega', type=float, default=0.1)
    
    parser.add_argument('-sigma', type=float, default=5)
    parser.add_argument('-use_filter', type=int, default=1)
    
    parser.add_argument('-save_plots', type=int, default=0)
    parser.add_argument('-show_plots', type=int, default=1)
    args = parser.parse_args()
    print(vars(args))
    
    plot = Plotter(N_figs=1,
                   save_root=f'{args.root_folder}/data/M{args.M}_Nsamp{args.num_ensembles}/', 
                   save_filename='r_avg.pdf', 
                   show=args.show_plots, save=args.save_plots,
                   use_tics=True)
    fig = plot.fig
    axis = plot.axis
    
    sigma = args.sigma
    use_filter = args.use_filter
    show = args.show_plots
    save = args.save_plots

    thetaOmega_arr = np.arange(args.thetaOmega_min, args.thetaOmega_max+args.dthetaOmega/2.0, args.dthetaOmega)
    WOmega_arr = np.arange(args.WOmega_min, args.WOmega_max+args.dWOmega/2.0, args.dWOmega)
    
    r_avg_arr = np.empty(shape=(thetaOmega_arr.size, WOmega_arr.size))
    r_err_arr = np.empty(shape=(thetaOmega_arr.size, WOmega_arr.size))
    
    # Load data
    start = time.time()
    for i,thetaOmega in enumerate(thetaOmega_arr):
        for j,WOmega in enumerate(WOmega_arr):
            folder = f'{args.root_folder}/data/M{args.M}_Nsamp{args.num_ensembles}/thetaOmega{thetaOmega:.2f}_WOmega{WOmega:.2f}'
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
        c = plot.colormesh(x=thetaOmega_arr, y=WOmega_arr, z=gaussian_filter(r_avg_arr,sigma), vmin=poiss, vmax=goe, ax_idx=0)
    else:
        c = plot.colormesh(x=thetaOmega_arr, y=WOmega_arr, z=r_avg_arr, vmin=poiss, vmax=goe, ax_idx=0)
    
    #ct = axis[0].contour(thetaOmega_arr, WOmega_arr, gaussian_filter(r_avg_arr.T,sigma), levels=[goe,gue,gse], colors=('k',))
    ct = axis[0].contour(thetaOmega_arr, WOmega_arr, gaussian_filter(r_avg_arr.T,sigma), levels=[goe], colors=('k',))
    fmt = {}
    #strs = ['GOE', 'GUE', 'GSE']
    strs = ['GOE']
    for l,s in zip(ct.levels,strs):
        fmt[l] = s
    plt.clabel(ct, fmt=fmt, colors='k')
        
    axis[0].set_xlim(xmin=thetaOmega_arr[0], xmax=thetaOmega_arr[-1])
    axis[0].set_xticks(np.linspace(thetaOmega_arr[0], thetaOmega_arr[-1], 6))
    axis[0].set_ylim(ymin=WOmega_arr[0], ymax=WOmega_arr[-1])
    axis[0].set_yticks(np.linspace(WOmega_arr[0], WOmega_arr[-1], 5))
    axis[0].tick_params(axis=u'both', which=u'both',length=0)
    axis[0].set_xlabel(r'$16 \theta \Phi$')
    axis[0].set_ylabel(r'$16 W \Phi$')

    ticks = np.linspace(poiss, goe, 6)
    cbar = fig.colorbar(c, ax=axis[0], ticks=ticks, pad=0.01, format=lambda x, _: f"{x:.2f}")
    cbar.outline.set_visible(False)
    cbar.ax.set_title(r'$\langle r \rangle$')
    cbar.ax.tick_params(axis='both', which='both',length=0)
    
    # Places to mark (but GUE or GSE never fully follows the r spectrum !)
    # FIXME
    plot.line(x=7.4, y=2, color='black', marker='o', ax_idx=0) # GOE
    axis[0].plot(7.4, 7, color='deeppink', marker='^', clip_on=False, zorder=100) # POISS
    plot.line(x=18, y=3, color='black', marker='v', ax_idx=0) # deep into GOE
    plot.line(x=7.4, y=3.5, color='black', marker='*', ax_idx=0) # crossover
    
    #WOmega_val = 2
    #idx = np.where(np.abs(WOmega_arr - WOmega_val) < 1e-10)[0][0]
    #plot.line(x=thetaOmega_arr, y=r_avg_arr[:,idx], ax_idx=1)
    #plot.fill_betweeny(x=thetaOmega_arr, y_lower=r_avg_arr[:,idx]-r_err_arr[:,idx], y_upper=r_avg_arr[:,idx]+r_err_arr[:,idx], ax_idx=1)
    #
    #transy = axis[1].get_yaxis_transform()
    #
    #axis[1].axhline(y=poiss, linestyle='--', color='black')
    #axis[1].text(0.5, poiss, 'Poisson', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    #
    #axis[1].axhline(y=goe, linestyle='--', color='black')
    #axis[1].text(0.4, goe, 'GOE', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    #
    #axis[1].axhline(y=gue, linestyle='--', color='black')
    #axis[1].text(0.4, gue, 'GUE', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    #
    #axis[1].axhline(y=gse, linestyle='--', color='black')
    #axis[1].text(0.8, gse, 'GSE', transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    #
    ##axis[1].text(0.5, (goe+poiss)/2.0, r'$\bar{\epsilon} = eps_val$'.replace('eps_val', str(eps_val)), transform=transy, va="center", ha="center", bbox = dict(ec='1',fc='1'))
    #
    #axis[1].set_xlim(xmin=thetaOmega_arr[0], xmax=thetaOmega_arr[-1])
    #axis[1].set_xlabel(r'$\theta/(16\Omega)$')
    #axis[1].set_ylabel(r'$\langle r \rangle$')
    ##axis[1].set_title(r'$\bar{\epsilon} = eps_val$'.replace('eps_val', str(eps_val)))