import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import argparse
import textwrap

import qutip as qt

from kicked_boson.quantum.ensemble import RandomBosonChainEnsemble, TimeBosonChainEnsemble, HaarEnsemble

def linear_extrapolate_zero(x, y):
    def f(x, a, c):
        return a*x + c
    popt, pcov = curve_fit(f, x, y)
    c = popt[-1]
    c_err = np.sqrt(pcov[-1,-1])
    return c, c_err


if __name__ == '__main__':
    description = textwrap.dedent('''\
         Ising chain:
            N           : number of sites

            J           : Nearest-neighbor coupling
                          (~0.7 pi chaos, ~0.1 pi regular)
            
            Omega       : trapping potential strength
                          (default 1)

            eta         : time-reversal phase
                          (default 0, no TRS breaking)

            theta_noise : Noise on hopping rotation angle
                          (default 0.05)
            
            phi_noise   : Noise on phase gate angle
                          (default 0.05)
            
            eta_noise   : Noise on TRS breaking angle
                          (default 0.05)
                          
             
            num_modes   : number of bosonic modes per site 
                          (default empty, occupied)
             
            excitations : maximum number of particles in chain
                          (default single particle sector)
             
            periodic    : whether chain has periodic boundaries
                          (default open boundaries)
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-N', type=int, default=5)
    parser.add_argument('-J', type=float, default=1)
    parser.add_argument('-eta', type=float, default=0)
    parser.add_argument('-Omega', type=float, default=2)
    parser.add_argument('-theta_noise', type=float, default=0.05)
    parser.add_argument('-phi_noise', type=float, default=0.05)
    parser.add_argument('-eta_noise', type=float, default=0.05)
    parser.add_argument('-num_modes', type=int, default=2)
    parser.add_argument('-excitations', type=int, default=1)
    parser.add_argument('-periodic', type=int, default=0)
    parser.add_argument('-random', type=int, default=1)
    parser.add_argument('-plot', type=int, default=1)
    args = parser.parse_args()
    print(vars(args))

    # FIXME do a theta/phi varying plot and it's deviation from a GUE level statistics
    # and Poisson level statistics.

    #ensemble_size = [d**(2*k)] # recommended at least d**(2*k)
    #ensemble_sizes = np.array([400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000])
    #ensemble_sizes = np.array([400, 500, 600, 700, 800, 900, 1000])
    ensemble_sizes = np.array([400, 500, 600])
    num_ensembles = len(ensemble_sizes)
    
    d = args.N
    bosons = []
    haars = []
    start = time.time()
    for num_ensembles in ensemble_sizes:
        if args.random > 0:
            bosons.append( RandomBosonChainEnsemble(num_ensembles=num_ensembles, 
                                                    N=args.N,
                                                    J=args.J,
                                                    Omega=args.Omega,
                                                    eta=args.eta,
                                                    theta_noise=args.theta_noise,
                                                    phi_noise=args.phi_noise,
                                                    eta_noise=args.eta_noise,
                                                    num_modes=args.num_modes,
                                                    excitations=args.excitations,
                                                    periodic=args.periodic) 
                         )
        else:
            bosons.append( TimeBosonChainEnsemble(num_ensembles=num_ensembles, 
                                                  N=args.N,
                                                  J=args.J,
                                                  Omega=args.Omega,
                                                  eta=args.eta,
                                                  num_modes=args.num_modes,
                                                  excitations=args.excitations,
                                                  periodic=args.periodic) 
                         )

        haars.append( HaarEnsemble(num_ensembles=num_ensembles, d=d) )
    end = time.time()
    print("Ensemble construction took", end-start)

    # Time evolve bosons
    #for E in bosons:
    #    E.evolve(dT=1000)

    k_max = 2
    sum_type = 'einsum_fast'
    start = time.time()
    F_U = np.asarray([ensemble.frame_potential(k_max=k_max, sum_type=sum_type) for ensemble in bosons]).T
    F_H = np.asarray([ensemble.frame_potential(k_max=k_max, sum_type=sum_type) for ensemble in haars]).T
    end = time.time()
    print("Frame potential took", end-start)

    x_list = np.asarray(1/ensemble_sizes)
    x_list_limits = np.append(np.insert(x_list,0,0), 1.1*max(x_list))

    F_U_inf = np.empty(k_max)
    F_U_inf_err = np.empty(k_max)
    F_H_inf = np.empty(k_max)
    F_H_inf_err = np.empty(k_max)
    last_pt = 5
    for k in range(k_max):
        F_U_inf[k], F_U_inf_err[k] = linear_extrapolate_zero(x_list[-last_pt:-1], F_U[k][-last_pt:-1])
        F_H_inf[k], F_H_inf_err[k] = linear_extrapolate_zero(x_list[-last_pt:-1], F_H[k][-last_pt:-1])

    fig, ax = plt.subplots(2, k_max, figsize=(5*k_max, 10))
    for k in range(k_max):
        ax[0,k].plot(x_list, F_U[k], 'o-', label=r'$\mathbb{U}$')
        ax[0,k].errorbar(0, F_U_inf[k], F_U_inf_err[k], label=r'$\mathbb{U}_{\infty}$', fmt='o')
        ax[0,k].plot(x_list_limits, [np.math.factorial(k+1)*d**(k+1) for _ in range(len(x_list_limits))], '--k', label=r'$k!d^2$')
        
        ax[0,k].plot(x_list, F_H[k], 'o-', label='Haar')
        ax[0,k].errorbar(0, F_H_inf[k], F_H_inf_err[k], label=r'Haar$_{\infty}$', fmt='o')
        ax[0,k].plot(x_list_limits, [np.math.factorial(k+1) for _ in range(len(x_list_limits))], '--k', label=r'$k!$')
        
        ax[0,k].set_yscale('log')
        ax[0,k].set_ylabel(r'$F_{\mathcal{E}}$')
        ax[0,k].set_xlabel(r'$1/|\mathcal{E}|$')
        ax[0,k].set_xlim(xmin=0, xmax=max(x_list))
        ax[0,k].set_ylim(ymin=0)
    
        ax[1,k].plot(x_list, np.sqrt(F_U[k] - F_H[k]), 'o-')
        ax[1,k].set_ylabel(r'$\varepsilon_{\mathbb{U}}$')
        ax[1,k].set_xlabel(r'$1/|\mathcal{E}|$')
        ax[1,k].set_xlim(xmin=0)
    ax[0,0].legend()
    plt.show()

    # Initialize a single particle in the chain
    #num_basis_psi0 = [0 for _ in range(N)]
    #num_basis_psi0[int(N/2)] = 1
    ##num_basis_psi0[2] = 1
    #psi0 = qt.enr_fock(dims, excitations, num_basis_psi0)
    #
    ## Real-space expectation over many periods
    #num_periods = 10
    #psi = psi0.copy()
    #position_expectation = np.zeros([num_periods+1, N])
    #position_expectation[0,:] = qt.expect(num_Op, psi)
    #for n in range(1,num_periods+1):
    #    psi = U * psi
    #    position_expectation[n,:] = qt.expect(num_Op, psi)
    #
    #U_ns = U_s**1
    #plt.matshow(U_ns.full().real, cmap=plt.cm.magma)
    ##plt.matshow(U_ns.full().imag, cmap=plt.cm.magma)
    ##plt.matshow(np.abs(U_ns.full()), cmap=plt.cm.magma)
    ##plt.matshow(np.angle(U_ns.full()), cmap=plt.cm.magma)
    #plt.show()

    #plt.imshow(position_expectation[-20:,:], origin='lower', cmap=plt.cm.magma)
    ##plt.imshow(position_expectation[-20:,:], origin='lower', cmap=plt.cm.magma, interpolation='bilinear')
    #plt.show()
    
    # 
    #psi = U**10 * psi0
    #expectation_j = [0 for _ in range(N)]
    #for j in range(N):
    #    expectation_j[j] = psi.dag() * num_Op[j] * psi