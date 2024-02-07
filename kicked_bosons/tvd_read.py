import argparse
import numpy as np
import scipy

import matplotlib.pyplot as plt

from quantum_chaos.stats import mase, mae
from quantum_chaos.plotter import Plotter

# tables can be installed via conda install pytables
import pandas as pd

def transform_x(x, fnc_type='log'):
    match fnc_type:
        case 'None':
            return x
        case 'log':
            return np.log(x)

def inv_transform_x(x, fnc_type='log'):
    match fnc_type:
        case 'None':
            return x
        case 'log':
            return np.exp(x)

def transform_y(y, fnc_type='log'):
    match fnc_type:
        case 'None':
            return y
        case 'log':
            return np.log(y)

def inv_transform_y(y, fnc_type='log'):
    match fnc_type:
        case 'None':
            return y
        case 'log':
            return np.exp(y)

def get_time_str(timee):                
    if timee == 'heisenberg':
        return 'heis'
    elif float(timee) > 1e6:
        return 'inf'
    else:
        return f'{float(args.time):.2f}'

def get_h5_key(sample_type, timee, thetaOmega, WOmega):
    key = sample_type + "_tvd"
    if sample_type == 'kicked-boson':
        key = key + "_t_" + get_time_str(args.time) + f'_thetaOmega{args.thetaOmega:.2f}WOmega{args.WOmega:.2f}'
    return key

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sample_type', type=str, default='normal')
    
    parser.add_argument('-thetaOmega', type=float, default=7.4)
    parser.add_argument('-WOmega', type=float, default=2)
    parser.add_argument('-time', type=str, default='1e12')
    
    parser.add_argument('-root_folder', type=str, default='./')
    parser.add_argument('-show_plots', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    print(vars(args))

    def get_tvd_from_file(filename, key, sample_type, M, plot, ax_idx=0, N=2, timee=None, show_plots=False):
        fnc_type_x = 'log'
        fnc_type_y = 'log'
        
        def func(x, a, b, c):
            '''
            Asumptotic regression

            For x -> inf, y = c
            '''
            return (a - c) * np.exp(-x/b) + c
        
        with pd.HDFStore(filename, 'r') as f:
            df = f.get(key)

        df2 = df.loc[(df['M'] == M) & (df['N'] == N)]
        #data = np.sort(df2[['Nsamp', 'tvd']].values, axis=0)
        data = df2[['Nsamp', 'tvd']].values
        ind = np.lexsort((data[:,1],data[:,0]))
        data = data[ind]
        idx = np.where(data[:,0] > 0)
        num_ensembles = transform_x(data[:,0][idx], fnc_type_x)
        tvd_data = transform_y(data[:,1][idx], fnc_type_y)
                    
        popt, pcov = scipy.optimize.curve_fit(func, num_ensembles, tvd_data, maxfev=100000, bounds=([-np.inf, -np.inf, transform_y(0, fnc_type_y)], [np.inf, np.inf, transform_y(1, fnc_type_y)]))
        tvd = inv_transform_y(popt[-1], fnc_type_y)
        tvd_err = np.sqrt(np.diag(pcov))[-1]
        tvd_fit = func(num_ensembles, *popt)
        tvd_mae = mae(inv_transform_y(tvd_data, fnc_type_y), inv_transform_y(tvd_fit, fnc_type_y))
        tvd_mase = mase(inv_transform_y(tvd_data, fnc_type_y), inv_transform_y(tvd_fit, fnc_type_y))
        
        print("transform_x :", fnc_type_x, "transform_y :", fnc_type_y)
        print("Parameters for fit function (a - c) * np.exp(-x/b) + c :", popt)
        print("tvd :", tvd)
        print("tvd_mae :", tvd_mae)
        print("tvd_mase :", tvd_mase)
            
        if show_plots:            
            x = np.linspace(min(inv_transform_x(num_ensembles, fnc_type_x)), max(inv_transform_x(num_ensembles, fnc_type_x)), 1000, endpoint=True)
            y = inv_transform_y(func(transform_x(x, fnc_type_x), *popt), fnc_type_y)
            
            plot.scatter(inv_transform_x(num_ensembles, fnc_type_x), inv_transform_y(tvd_data, fnc_type_y), ax_idx=ax_idx, alpha=1)
            plot.line(x, y, ax_idx=ax_idx)
            #plot.fill_betweeny(x, y-tvd_mae, y+tvd_mae, ax_idx=0, color=None)
            plot.axis[ax_idx].set_xlim(xmin=0)
            #ax.set_ylim(ymin=0, ymax=1)
            plot.axis[ax_idx].set_xlabel(r'$|\mathcal{E}|$')
            match sample_type:
                case 'normal':
                    P = r'\mathcal{MN}'
                case 'haar':
                    P = r'\mathrm{Haar}(M=' + str(M) + ')'
                case 'kicked-boson':
                    P = r'\mathrm{KB}(M=' + str(M) + ')'
            plot.axis[ax_idx].set_ylabel(r'$TV[{},{}={}]$'.format(r'\mathcal{MN}', r'\mathcal{P}', P))
        return tvd
    
    filename = args.root_folder + "/tvd.h5"
    key = get_h5_key(args.sample_type, args.time, args.thetaOmega, args.WOmega)
    print(f"Opening file '{filename}' with key '{key}'")
    with pd.HDFStore(filename, 'r') as f:
        df = f.get(key)
    print('Unique (M,N) combinations:')
    print(df.groupby(['M','N']).size())
    M_N_pairs = np.delete(df.groupby(['M','N']).size().reset_index().values, 2, axis=1)
    
    plot = Plotter(N_figs=M_N_pairs.shape[0]+1,
                   use_tics=True)

    tvd_arr = np.empty(M_N_pairs.shape[0])
    for i, (M, N) in enumerate(M_N_pairs):
        tvd_arr[i] = get_tvd_from_file(filename=filename, key=key, sample_type=args.sample_type, M=M, plot=plot, ax_idx=i, N=N, timee=args.time, show_plots=args.show_plots)
    plot.linepoints(M_N_pairs[:,0], tvd_arr, ax_idx=-1)
    plot.axis[-1].set_xlim(xmin=0)
    plot.axis[-1].set_ylim(ymin=0, ymax=1)
    plt.show()