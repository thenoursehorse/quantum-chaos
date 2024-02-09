import argparse
import numpy as np
import scipy

import matplotlib.pyplot as plt

from quantum_chaos.stats import mase, mae
from quantum_chaos.plotter import Plotter

# tables can be installed via conda install pytables
import pandas as pd

def transform_x(x, fnc_type='None'):
    match fnc_type:
        case 'None':
            return x
        case 'log':
            return np.log(x)
        case 'log10':
            return np.log10(x)
        case 'log2':
            return np.log2(x)
        case 'inv_log':
            return 1 / np.log(x)
            #return np.log(1/x)

def inv_transform_x(x, fnc_type='None'):
    match fnc_type:
        case 'None':
            return x
        case 'log':
            return np.exp(x)
        case 'log10':
            return 10**x
        case 'log2':
            return 2**x
        case 'inv_log':
            return np.exp(1/x)
            #return 1/np.exp(x)

def transform_y(y, fnc_type='None'):
    match fnc_type:
        case 'None':
            return y
        case 'log':
            return np.log(y)
        case 'log10':
            return np.log10(y)
        case 'log2':
            return np.log2(y)

def inv_transform_y(y, fnc_type='None'):
    match fnc_type:
        case 'None':
            return y
        case 'log':
            return np.exp(y)
        case 'log10':
            return 10**y
        case 'log2':
            return 2**y

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

def func_lin(x, a, b):
    return a * x + b

def func_pow(x, a, b, c):
    return a*x**b + c
        
def func_asymptotic(x, a, b, c):
    '''
    Asymptotic regression

    For x -> inf, y = c
    '''
    return (a - c) * np.exp(-x/b) + c
    #return (a - c) * 2**(-x/b) + c
    #return (a - c) * 10**(-x/b) + c
    #return c * x ** a  / (x ** a + b)
    #return c / (1 + np.exp(-b*(x-a)))

#def func(x, a, b, c, d):
#    return d + (a - d)/(1 + (x/c)**b)

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
        fnc_type_x = 'inv_log'
        fnc_type_y = 'log'
                
        with pd.HDFStore(filename, 'r') as f:
            df = f.get(key)

        df2 = df.loc[(df['M'] == M) & (df['N'] == N)]
        data = df2[['Nsamp', 'tvd']].values
        ind = np.lexsort((data[:,1],data[:,0]))
        data = data[ind]
        num_ensembles = data[:,0]
        tvd_data = data[:,1]
        
        idx = np.where(num_ensembles > 10000)
        x = transform_x(num_ensembles[idx], fnc_type_x)
        y = transform_y(tvd_data[idx], fnc_type_y)

        func = func_lin
        #func = func_pow
        #func = func_asymptotic
                    
        popt, pcov = scipy.optimize.curve_fit(func, x, y, maxfev=100000)
        # For asymptotic regression
        #popt, pcov = scipy.optimize.curve_fit(func, num_ensembles, tvd_data, maxfev=100000, bounds=([-np.inf, -np.inf, transform_y(0, fnc_type_y)], [np.inf, np.inf, transform_y(1, fnc_type_y)]))
        popt_err = np.sqrt(np.diag(pcov))
        #tvd = inv_transform_y(popt[-1], fnc_type_y)
        #tvd = inv_transform_y(func(transform_x(1e15, fnc_type_x), *popt), fnc_type_y)
        #tvd_err = inv_transform_y(popt_err[-1])
        # For func_lin and fnc_type_x = 'inv_log'
        tvd = np.exp(popt[-1])
        #tvd_err = np.exp(popt_err[-1])
        tvd_fit = func(x, *popt)
        tvd_mae = mae(inv_transform_y(y, fnc_type_y), inv_transform_y(tvd_fit, fnc_type_y))
        #tvd_mase = mase(inv_transform_y(y, fnc_type_y), inv_transform_y(tvd_fit, fnc_type_y))
        
        print("transform_x :", fnc_type_x, "transform_y :", fnc_type_y)
        print("Parameters for fit function a * x + b :", popt, "+-", popt_err)
        print("tvd :", tvd)
        #print("tvd_err :", tvd_err)
        print("tvd_mae :", tvd_mae)
        #print("tvd_mase :", tvd_mase)
            
        if show_plots:            
            match sample_type:
                case 'normal':
                    P = r'\mathcal{MN}'
                case 'haar':
                    P = r'\mathrm{Haar}(M=' + str(M) + ')'
                case 'kicked-boson':
                    P = r'\mathrm{KB}(M=' + str(M) + ')'
            
            x = transform_x(np.linspace(min(num_ensembles), max(num_ensembles), 1000, endpoint=True), fnc_type_x)
            y = func(x, *popt)
            plot.scatter(transform_x(num_ensembles, fnc_type_x), transform_y(tvd_data, fnc_type_y), ax_idx=ax_idx, alpha=1, clip_on=False)
            plot.line(x, y, ax_idx=ax_idx)
            plot.axis[ax_idx].set_xlabel(r'$1 / \log(|\mathcal{E}|)$')
            plot.axis[ax_idx].set_ylabel(r'$\log[ TV[{},{}={}] ]$'.format(r'\mathcal{MN}', r'\mathcal{P}', P))
            
            #x = np.linspace(min(num_ensembles), max(num_ensembles), 1000, endpoint=True)
            #y = inv_transform_y(func(transform_x(x, fnc_type_x), *popt), fnc_type_y)
            #plot.scatter(num_ensembles, tvd_data, ax_idx=ax_idx, alpha=1, clip_on=False)
            #plot.line(x, y, ax_idx=ax_idx)
            #plot.axis[ax_idx].set_xlim(xmin=0)
            #plot.axis[ax_idx].set_xlabel(r'$|\mathcal{E}|$')
            #plot.axis[ax_idx].set_ylabel(r'$TV[{},{}={}]$'.format(r'\mathcal{MN}', r'\mathcal{P}', P))
        return tvd
    
    filename = args.root_folder + "/tvd.h5"
    key = get_h5_key(args.sample_type, args.time, args.thetaOmega, args.WOmega)
    print(f"Opening file '{filename}' with key '{key}'")
    with pd.HDFStore(filename, 'r') as f:
        df = f.get(key)
    print('Unique (M,N) combinations:')
    print(df.groupby(['M','N']).size())
    M_N_pairs = np.delete(df.groupby(['M','N']).size().reset_index().values, 2, axis=1)
    M_arr = M_N_pairs[:,0]
    
    if args.show_plots:
        if args.sample_type == 'kicked-boson':
            save_filename = f'tvd_{args.sample_type}' '_t_' + get_time_str(args.time) + f'_theta{args.thetaOmega:.2f}W{args.WOmega:.2f}.pdf' 
        else:
            save_filename = f'tvd_{args.sample_type}.pdf'
        plot = Plotter(N_figs=M_arr.shape[0]+1,
                       save=True,
                       use_tics=True,
                       save_filename = save_filename,
                       save_root='./data/',
        )

    tvd_arr = np.empty(M_arr.shape[0])
    #for i, (M, N) in enumerate(M_N_pairs):
    for i, M in enumerate(M_arr):
        tvd_arr[i] = get_tvd_from_file(filename=filename, key=key, sample_type=args.sample_type, M=M, plot=plot, ax_idx=i, N=2, timee=args.time, show_plots=args.show_plots)
    
    fnc_type_x = 'log'
    fnc_type_y = 'log'
    idx = np.where(M_arr > 3)
    func = func_asymptotic
    popt, pcov = scipy.optimize.curve_fit(func, transform_x(M_arr[idx], fnc_type_x), transform_y(tvd_arr[idx], fnc_type_y), maxfev=100000)
    #popt, pcov = scipy.optimize.curve_fit(func, transform_x(M_arr[idx], fnc_type_x), transform_y(tvd_arr[idx], fnc_type_y), maxfev=100000, bounds=([-np.inf, -np.inf, transform_y(0, fnc_type_y)], [np.inf, np.inf, transform_y(1, fnc_type_y)]))

    #theta_W_pairs=("7.4 7" "7.4 3.5" "7.4 2" "18 3" "20 0.5")
    
    if args.show_plots:
        x = np.linspace(min(M_arr[idx]), max(M_arr[idx]), 1000, endpoint=True)
        y = inv_transform_y(func(transform_x(x, fnc_type_x), *popt), fnc_type_y)
        plot.line(x, y, ax_idx=-1)
        plot.scatter(M_arr, tvd_arr, ax_idx=-1, alpha=1, clip_on=False)
        plot.axis[-1].set_xlim(xmin=0)
        plot.axis[-1].set_ylim(ymin=0, ymax=1)
        match args.sample_type:
            case 'normal':
                P = r'\mathcal{MN}'
            case 'haar':
                P = r'\mathrm{Haar}'
            case 'kicked-boson':
                P = r'\mathrm{KB}'
        plot.axis[-1].set_xlabel(r'$M$')
        plot.axis[-1].set_ylabel(r'$TV[{},{}={}]$'.format(r'\mathcal{MN}', r'\mathcal{P}=', P))
        plt.show()