import time
import numpy as np
import scipy
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from quantum_chaos.plotter import SeabornFig2Grid

from KDEpy import FFTKDE
from scipy.stats import unitary_group
from randomgen import ExtendedGenerator

from quantum_chaos.stats import vec, mase, round_to_n, mle_MNorm, transform_standard_MNorm
from quantum_chaos.stats import get_x_grid, fft_density, tvd_integral_fft

import pandas as pd
import h5py

# Transformations of probability density function to possibly fix bounds 
def transformation(x):
    return x
    #return np.log(x)

def inv_transformation(x):
    return x
    #return np.exp(x)

def det_jacobian(x):
    return np.ones(x.shape[0])
    #return 1.0 / np.prod(x, axis=-1)
                
def get_time_str(timee):                
    if timee == 'heisenberg':
        return 't_heis'
    elif float(timee) > 1e6:
        return 't_inf'
    else:
        return f't_{float(args.time):.2f}'

def h5_wait(h5file, wait=3, max_wait=30):
    
    waited = 0

    while True:
        try:
            h5f = h5py.File(h5file,'r')
            break
                
        except FileNotFoundError:
            print('\nError: HDF5 File not found\n')
            return False
        
        except OSError:   
            if waited < max_wait:
                print(f'Warning: HDF5 File locked, sleeping {wait} seconds...')
                time.sleep(wait) 
                waited += wait  
            else:
                print(f'\nWaited too long = {waited} secs, exiting...\n')
                return False

    h5f.close()
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', type=int, default=30)
    parser.add_argument('-N', type=int, default=2)
    parser.add_argument('-num_ensembles', type=int, default=100)
    parser.add_argument('-num_repeats', type=int, default=1)
    parser.add_argument('-sample_type', type=str, default='normal')
    
    parser.add_argument('-thetaOmega', type=float, default=7.4)
    parser.add_argument('-WOmega', type=float, default=2)
    parser.add_argument('-time', type=str, default='1e12')
    
    parser.add_argument('-root_folder', type=str, default='./data/')
    parser.add_argument('-random_ensemble_size', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-show_plots', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-save_data', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    print(vars(args))

    ix = 0
    iy = 0
    
    for i in range(args.num_repeats):
        print(f"Iteration {i+1}/{args.num_repeats}")
        
        if args.random_ensemble_size:
            num_ensembles = np.random.default_rng().integers(low=500, high=args.num_ensembles, endpoint=True)
            print(f'num_ensembles from uniform integer distribution: {num_ensembles}')
        else:
            num_ensembles = args.num_ensembles

        # Probability distribution to test against
        eg = ExtendedGenerator()
        cnorm = eg.multivariate_complex_normal(loc=[0] * (args.N*args.N), 
                                               gamma=np.eye(args.N*args.N), 
                                               size=num_ensembles).reshape(num_ensembles, args.N, args.N)

        # Sample of distribution testing
        match args.sample_type:
            case 'normal':
                print('Calculating for a complex matrix normal distribution')
                samples = eg.multivariate_complex_normal(loc=[0] * (args.N*args.N), 
                                                         gamma=np.eye(args.N*args.N), 
                                                         size=num_ensembles).reshape(num_ensembles, args.N, args.N)
            
            case 'haar':
                print('Calculating for a truncated Haar distribution')
                U = unitary_group.rvs(args.M, size=num_ensembles)
                samples = U[:, ix:ix+args.N, iy:iy+args.N] * np.sqrt(args.M)
            
            case 'random':
                print('Calculating for a random distribution')
                samples = eg.random(( num_ensembles, args.N, args.N ))
            
            case 'kicked-boson':
                print('Calculating for a kicked boson distribution')
                
                folder = args.root_folder + "/" + get_time_str(args.time) + "/"
                filename = folder + f'/thetaOmega{args.thetaOmega:.2f}WOmega{args.WOmega:.2f}_Utrunc.h5'
                
                rng = np.random.default_rng()
                if os.path.isfile(filename):
                    h5stat = h5_wait(filename)
                    if h5stat is False:
                        sys.exit('Error: HDF5 File not available')
                
                with h5py.File(filename, 'r') as f:
                    key = f'M{args.M}N{args.N}'
                    g = f[key]
                    
                    if g['real'].shape != g['imag'].shape:
                        raise ValueError('Real and imaginary parts of unitary must be the same dimensions !')
                    if g['real'].shape[-1] != g['real'].shape[-2]:
                        raise ValueError(f"Truncated unitary must be square but got {g['real'].shape[-2]} times {g['real'].shape[-1]} !")
                    if g['real'].shape[-1] != args.N:
                        raise ValueError(f"Truncated unitary must be square with dim={args.N} but is dim={g['real'].shape[-1]} !")

                    # Slow access, low memory
                    #idx = np.sort( rng.choice(g['real'].shape[0], size=num_ensembles, replace=False, shuffle=False) )
                    #samples = g['real'][idx] + 1j*g['imag'][idx]

                    # Fast access, larger memory, and resamples
                    samples = rng.choice(g['real'][...] + 1j*g['imag'][...], size=num_ensembles, replace=True)
            case _:
                raise ValueError('Not a valid sample type selection.')

        # Find maximum liklihood estimates (MLE) of the matrix normal parameters
        mu, Sigma_s, Sigma_c, sigma_squared = mle_MNorm(samples)
        mu_cnorm, Sigma_s_cnorm, Sigma_c_cnorm, sigma_squared_cnorm = mle_MNorm(cnorm)

        # Transform data to standard normal estimate
        samples_transformed = transform_standard_MNorm(samples, mu, Sigma_s, Sigma_c)
        mu_transformed, Sigma_s_transformed, Sigma_c_transformed, sigma_squared_transformed = mle_MNorm(samples_transformed)
        cnorm_transformed = transform_standard_MNorm(cnorm, mu_cnorm, Sigma_s_cnorm, Sigma_c_cnorm)

        # SVD (squared) of samples (complex Wishart distribution for matrix normal)
        samples_S, _ = np.linalg.eigh(samples_transformed @ np.swapaxes(samples_transformed.conj(), axis1=-1, axis2=-2))
        cnorm_S, _ = np.linalg.eigh(cnorm_transformed @ np.swapaxes(cnorm_transformed.conj(), axis1=-1, axis2=-2))

        # Find kernel density estimate 
        start = time.time()
        # bw = 'scott' ? 'silvermann' ?
        kde_fft = FFTKDE( kernel='gaussian', bw=np.power(samples_S.shape[0], -0.2) ).fit(transformation(samples_S))
        kde_fft_cnorm = FFTKDE( kernel='gaussian', bw=np.power(cnorm_S.shape[0], -0.2) ).fit(transformation(cnorm_S))
        end = time.time()
        print("Calculate kde_fft took", end-start)
        
        # Calculate tvd on increasingly larger equispaced grid
        #npoints = np.array([2**n for n in range(10,11)])
        #npoints = np.array([2**n for n in range(10,12)])
        #npoints = np.array([2**n for n in range(10,13)])
        npoints = np.array([2**n for n in range(8,14)])
        tvd_fft = np.zeros(len(npoints))
        start = time.time()
        for i, n in enumerate(npoints):
            x_mesh, x, x_linear, dx = get_x_grid(samples=[cnorm_S, samples_S], npoints=n)
            density = fft_density(kde=kde_fft, x=x, fnc=transformation, fnc_det_jacobian=det_jacobian, shape=(n,n))
            density_cnorm = fft_density(kde=kde_fft_cnorm, x=x, fnc=transformation, fnc_det_jacobian=det_jacobian, shape=(n,n))
            #tvd_fft[i] = tvd_integral_fft(density1=density, density2=density_cnorm, dx=dx)
            tvd_fft[i] = tvd_integral_fft(density1=density, density2=density_cnorm, x_linear=x_linear)
        end = time.time()
        print("Calculate tvd_fft integral took", end-start)
        print(f"tvd_fft = {tvd_fft}")

        def func(x, a, b, c):
            return a*x**b + c
        popt, pcov = scipy.optimize.curve_fit(func, 1/npoints, tvd_fft)
        tvd = popt[-1]
        tvd_err = np.sqrt(np.diag(pcov))[-1]
        tvd_fit = func(1/npoints, *popt)
        tvd_mase = mase(tvd_fft, tvd_fit)

        if args.show_plots:
            npoints_x = np.linspace(0, max(1/npoints))
            fig, ax = plt.subplots()
            ax.plot(1/npoints, tvd_fft, 'o-')
            ax.plot(npoints_x, func(npoints_x, *popt), 'g--')
            ax.set_xlim(xmin=0)
            ax.set_xlabel(r'1/grid size')
            ax.set_ylabel(r'$TV(\mathcal{MN},\mathcal{P})$')
            plt.show()
        
        if args.save_data:
            columns = ['M', 'N', 'Nsamp', 'tvd', 'tvd_err', 'MASE']
            data = [[args.M, args.N, num_ensembles, tvd, tvd_err, tvd_mase]]
            
            filename = args.root_folder + "/tvd.h5"
                
            if os.path.isfile(filename):
                h5stat = h5_wait(filename)
                if h5stat is False:
                    sys.exit('Error: HDF5 File not available')

            print(f"Saving data to '{filename}'")
            with pd.HDFStore(filename, 'a') as f:
                
                key = args.sample_type + "_tvd"
                if args.sample_type == 'kicked-boson':
                    key = key + "_t_" + get_time_str(args.time)
               
                if key in f:
                    df = pd.DataFrame(data=data, columns=columns, index=[f.get_storer(key).nrows])
                    f.append(key=key, value=df)
                else:
                    df = pd.DataFrame(data=data, columns=columns)
                    f.put(key=key, value=df, format='t')