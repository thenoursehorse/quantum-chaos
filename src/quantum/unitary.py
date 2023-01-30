import time
import numpy as np
import scipy
from copy import deepcopy
import matplotlib.pyplot as plt

import argparse
import textwrap

import qutip as qt

def get_destroy_operators(dims, excitations=1):
    return qt.enr_destroy(dims, excitations)

def phase_shift_unitary_old(N, phi_list, b_list):
    j = 0
    Op = -1j * phi_list[j] * b_list[j].dag() * b_list[j]
    U = Op.expm()
    for j in range(1, N):
        Op = -1j * phi_list[j] * b_list[j].dag() * b_list[j]
        U *= Op.expm()
    return U

def phase_shift_unitary(N, phi_list, b_list):
    Op = 0
    for j in range(N):
        Op += -1j * phi_list[j] * b_list[j].dag() * b_list[j]
    return Op.expm()

def hopping_unitary(N, theta, eta, b_list, periodic=False):
    Op = 0
    if periodic:
        for j in range(N):
            k = (j+1)%N
            Op += np.exp(1j*eta) * b_list[j].dag() * b_list[k]
            Op += np.exp(-1j*eta) * b_list[k].dag() * b_list[j]
    else:
        for j in range(N-1):
            k = j+1
            Op += np.exp(1j*eta) * b_list[j].dag() * b_list[k]
            Op += np.exp(-1j*eta) * b_list[k].dag() * b_list[j]
    Op *= -1j * theta
    return Op.expm()

# Define trapping potential
def delta_wj(N, j, Omega=1):
    return (4 * Omega / N**2) * (j - N/2.0)**2

def frame_potential_qobj(U_list, k=1):
    F = 0
    for U in U_list:
        for V in U_list:
            F += ((U.dag() * V).tr())**k * ((U * V.dag()).tr())**k
    return F / len(U_list)**2

def frame_potential_hadamard(U_list, k=1):
    F = 0
    for U in U_list:
        for V in U_list:
            F += np.sum(U.conj().T * V.T)**k * np.sum(V.conj().T * U.T)**k
    return F / len(U_list)**2

def frame_potential_einsum(U_list, k=1):
    F = 0
    for U in U_list:
        for V in U_list:
            F += np.einsum('ij,ji->', U.conj().T, V)**k * np.einsum('ij,ji->', U, V.conj().T)**k
    return F / len(U_list)**2

def frame_potential(U_list, k=1, sum_type='einsum'):
    if sum_type == 'einsum':
        U_list_new = [U.full() for U in U_list]
        return frame_potential_einsum(U_list_new, k)
    elif sum_type =='hadamard':
        U_list_new = [U.full() for U in U_list]
        return frame_potential_hadamard(U_list_new, k)
    elif sum_type =='qobj':
        return frame_potential_hadamard(U_list, k)

def scale_frame_potential(U_list, k=1):
    num_ensembles = len(U_list)
    ensemble_sizes = [len(U_list[m]) for m in range(num_ensembles)]
    F = [0 for _ in range(num_ensembles)]
    for m,U in enumerate(U_list):
        F[m] = frame_potential(U, k)
    return F

if __name__ == '__main__':
    description = textwrap.dedent('''\
         Ising chain:
            N           : number of sites

            J           : Nearest-neighbor coupling
                           (~0.5 pi chaos, ~0.1 pi regular)

            eta         : time-reversal phase
                           (default 0, no TRS breaking)

            Omega       : trapping potential strength
                           (default 1)
             
            num_modes   : number of bosonic modes per site 
                           (default empty, occupied)
             
            excitations : maximum number of particles in chain
                           (default single particle sector)
             
            periodic    : whether chain has periodic boundaries
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-N', type=int, default=5)
    parser.add_argument('-J', type=int, default=0.49*np.pi)
    parser.add_argument('-eta', type=int, default=0)
    parser.add_argument('-Omega', type=int, default=1)
    parser.add_argument('-num_modes', type=int, default=2)
    parser.add_argument('-excitations', type=int, default=1)
    parser.add_argument('-periodic', type=int, default=1)
    parser.add_argument('-plot', type=int, default=1)
    args = parser.parse_args()
    print(vars(args))

    N = args.N
    J = args.J
    eta = args.eta
    Omega = args.Omega
    num_modes = args.num_modes
    excitations = args.excitations
    periodic = args.periodic
    dims = [num_modes for _ in range(N)]

    # Define operators
    destroy_Op = get_destroy_operators(dims, excitations)
    num_Op = deepcopy(destroy_Op)
    for j in range(len(num_Op)):
        num_Op[j] = num_Op[j].dag() * num_Op[j]

    # Define parameters of problem
    T = 1 # period
    theta = J*T

    # Phases specifying trapping potential
    phi_list = [delta_wj(N, j, Omega) * T for j in range(N)]

    # Floquet operator at period T
    U_h = hopping_unitary(N=N, theta=theta, eta=eta, b_list=destroy_Op, periodic=periodic)
    U_p = phase_shift_unitary(N=N, phi_list=phi_list, b_list=destroy_Op)
    U = U_h*U_p
    # NOTE for a single-particle restriction, qutip puts index 0 as the vacuum. 
    # Then the next element is site N ... the last element N+1 is site 0

    # Effective Hamiltonian at period T
    H_eff = 1j * (U * U).logm() / (2. * T)
    #H_eff = 1j * U.logm() / T
    e = H_eff.eigenenergies()

    # Initialize a single particle in the chain
    num_basis_psi0 = [0 for _ in range(N)]
    num_basis_psi0[int(N/2)] = 1
    #num_basis_psi0[2] = 1
    psi0 = qt.enr_fock(dims, excitations, num_basis_psi0)

    # Real-space expectation over many periods
    num_periods = 10
    psi = psi0.copy()
    position_expectation = np.zeros([num_periods+1, N])
    position_expectation[0,:] = qt.expect(num_Op, psi)
    for n in range(1,num_periods+1):
        psi = U * psi
        position_expectation[n,:] = qt.expect(num_Op, psi)

    #plt.imshow(position_expectation[-20:,:], origin='lower', cmap=plt.cm.magma)
    ##plt.imshow(position_expectation[-20:,:], origin='lower', cmap=plt.cm.magma, interpolation='bilinear')
    #plt.show()

    # Single particle sector only
    U_s = U[1:,1:]
    U_s = qt.Qobj(U_s, dims=U.dims)
    #U_s = U_s**100
    d = U_s.shape[0]
    id = (U_s.dag() * U_s).tidyup()

    k = 2
    #ensemble_size = [d**(2*k)] # recommended at least d**(2*k)
    ensemble_sizes = np.array([40, 80, 120, 160, 200, 240, 280, 320, 360, 400])
    num_ensembles = len(ensemble_sizes)
    #ensemble_sizes = [100, 200]
    U_list = []
    harr_list = []
    #id_list = []
    for m,ensemble_size in enumerate(ensemble_sizes):
        U_list.append( [U_s for _ in range(ensemble_size)] )
        for n in range(1, ensemble_size):
            U_list[m][n] = U_list[m][n] * U_list[m][n-1]
    
        harr_list.append( [qt.rand_unitary(dimensions=d, distribution='haar') for _ in range(ensemble_size)] )
        #id_list.append( [id for _ in range(ensemble_size)] )

    F_U = np.asarray(scale_frame_potential(U_list=U_list, k=k))
    F_H = np.asarray(scale_frame_potential(U_list=harr_list, k=k))
    #F_I = scale_frame_potential(U_list=id_list, k=k)

    x_list = np.asarray(1/ensemble_sizes)
    fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    #if not isinstance(ax, list):
    #    ax = [ax]
    ax[0].plot(x_list, F_U, 'o-', label='U')
    ax[0].plot(x_list, F_H, 'o-', label='Haar')
    ax[0].plot(np.insert(x_list,0,0), [np.math.factorial(k)*d**k for _ in range(num_ensembles+1)], '--k', label=r'$k!d^2$')
    ax[0].plot(np.insert(x_list,0,0), [np.math.factorial(k) for _ in range(num_ensembles+1)], '--k', label=r'$k!$')
    ax[0].set_ylabel(r'$F$')
    ax[0].set_xlabel('1/ensemble size')
    ax[0].set_xlim(xmin=0)
    ax[0].set_ylim(ymin=0)
    ax[0].legend()
    
    ax[1].plot(x_list, np.sqrt(F_U - F_H), 'o-')
    ax[1].set_ylabel(r'$\varepsilon_U$')
    ax[1].set_xlabel('1/ensemble size')
    ax[1].set_xlim(xmin=0)
    ax[1].set_ylim(ymin=0)
    plt.show()

    
    #print(frame_potential_einsum(U_list, k), np.math.factorial(k) * d**k)
    ##print(frame_potential_einsum(id_list, k), d**(2*k)) # This works fine
    #print(frame_potential_einsum(harr_list, k), np.math.factorial(k))

    #start = time.time()
    #print(frame_potential_einsum(harr_list, k), np.math.factorial(k))
    #end = time.time()
    #print("einsum:", end-start)
    #
    #start = time.time()
    #print(frame_potential_hadamard(harr_list, k), np.math.factorial(k))
    #end = time.time()
    #print("hadamard:", end-start)
    #
    #start = time.time()
    #print(frame_potential_qobj(harr_list, k), np.math.factorial(k))
    #end = time.time()
    #print("qobj:", end-start)

    #U_n = U_list[-1]
    #U_ns = U_n[1:,1:]
    #U_ns = qt.Qobj(U_ns, dims=U.dims)
    #plt.matshow(U_ns.full().real, cmap=plt.cm.magma)
    ##plt.matshow(U_ns.full().imag, cmap=plt.cm.magma)
    ##plt.matshow(np.abs(U_ns.full()), cmap=plt.cm.magma)
    ##plt.matshow(np.angle(U_ns.full()), cmap=plt.cm.magma)
    #plt.show()

    # 
    #psi = U**10 * psi0
    #expectation_j = [0 for _ in range(N)]
    #for j in range(N):
    #    expectation_j[j] = psi.dag() * num_Op[j] * psi