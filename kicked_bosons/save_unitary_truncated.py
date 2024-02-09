import time
import numpy as np

import os
import sys
import argparse

import h5py

from quantum_chaos.quantum.kicked_bosons import KickedBosons

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

def get_truncated_unitary(M, N, num_ensembles, theta, theta_disorder, Omega, phi_disorder, eta, eta_disorder, t=1e12, ix=0, iy=0):
    start = time.time()
    model = KickedBosons(M=M,
                         num_ensembles=num_ensembles,
                         theta=theta,
                         Omega=Omega,
                         eta=eta,
                         theta_disorder=theta_disorder,
                         phi_disorder=phi_disorder,
                         eta_disorder=eta_disorder
    )
    end = time.time()
    print("Unitary construction took", end-start)
    
    if t == 'heisenberg':
        t = model.heisenberg_time()
    else:
        t = float(t)
         
    start = time.time()
    model.set_unitary_evolve(num_ensembles=args.num_ensembles, time=np.array([t]))
    end = time.time()
    print("Time evolve unitaries took", end-start)
    
    # Take truncated part
    return model._Ut[-1][:, ix:ix+args.N, iy:iy+args.N]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', type=int, default=300)
    parser.add_argument('-N', type=int, default=2)
    parser.add_argument('-num_ensembles', type=int, default=10)
    parser.add_argument('-num_repeats', type=int, default=1)
    
    parser.add_argument('-thetaOmega', type=float, default=7.4)
    parser.add_argument('-thetaOmega_disorder', type=float, default=0.0)
    
    parser.add_argument('-Omega', type=float, default=np.pi/4.0)
    parser.add_argument('-WOmega', type=float, default=2)

    parser.add_argument('-i', type=int, default=0)
    parser.add_argument('-j', type=int, default=0)
    
    parser.add_argument('-etaOmega', type=float, default=0)
    parser.add_argument('-etaOmega_disorder', type=float, default=0)
    
    parser.add_argument('-time', type=str, default='1e12')
    
    parser.add_argument('-root_folder', type=str, default='./data/')
    parser.add_argument('-save_data', action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()
    print(vars(args))
    
    ##########################################
    ## Make parameters for Floquet operator ##
    ##########################################
    theta = args.thetaOmega / (16.0 * args.Omega)
    theta_disorder = args.thetaOmega_disorder / (16.0 * args.Omega)
    phi_disorder = args.WOmega / (16.0 * args.Omega)
    eta = args.etaOmega / (16.0 * args.Omega)
    eta_disorder = args.etaOmega_disorder / (16.0 * args.Omega)

    for i in range(args.num_repeats):
        print(f"Iteration {i+1}/{args.num_repeats}")

        U_trunc = get_truncated_unitary(args.M, args.N, args.num_ensembles, theta, theta_disorder, args.Omega, phi_disorder, eta, eta_disorder, args.time, args.i, args.j)
                            
        if args.save_data:
            if args.time == 'heisenberg':
                folder = args.root_folder + f'/t_heis'
            elif float(args.time) > 1e6:
                folder = args.root_folder + f'/t_inf'
            else:
                folder = args.root_folder + f'/t_{float(args.time):.2f}'
            os.makedirs(f'{folder}', exist_ok=True)
            filename = folder + f'/thetaOmega{args.thetaOmega:.2f}WOmega{args.WOmega:.2f}_Utrunc.h5'

            # Check that file is not in use
            if os.path.isfile(filename):
                h5stat = h5_wait(filename)
                if h5stat is False:
                    sys.exit('Error: HDF5 File not available')
            
            # Write real and imaginary parts separately
            with h5py.File(filename, 'a') as f:
                key = f'M{args.M}N{args.N}'
                g = f.require_group(key)

                key = 'real' 
                if key not in g:
                    d = g.create_dataset(key, data=U_trunc.real, maxshape=(None, args.N, args.N), dtype='float64')
                else:
                    d = g[key]
                    size = d.shape[0] + U_trunc.shape[0]
                    d.resize(size, axis=0)
                    d[-U_trunc.shape[0]:] = U_trunc.real
                
                key = 'imag' 
                if key not in g:
                    d = g.create_dataset(key, data=U_trunc.imag, maxshape=(None, args.N, args.N), dtype='float64')
                else:
                    # Append to existing dataset
                    d = g[key]
                    size = d.shape[0] + U_trunc.shape[0]
                    d.resize(size, axis=0)
                    d[-U_trunc.shape[0]:] = U_trunc.imag


# try to fibonacci stuff!