import time
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import qutip as qt

def randnz(shape, norm=1 / np.sqrt(2), seed=None):
    # This function is intended for internal use.
    """
    Returns an array of standard normal complex random variates.
    The Ginibre ensemble corresponds to setting ``norm = 1`` [Mis12]_.

    Parameters
    ----------
    shape : tuple
        Shape of the returned array of random variates.
    norm : float
        Scale of the returned random variates, or 'ginibre' to draw
        from the Ginibre ensemble.
    """
    if seed is not None:
        np.random.seed(seed=seed)
    if norm == 'ginibre':
        norm = 1
    UNITS = np.array([1, 1j])
    return np.sum(np.random.randn(*(shape + (2,))) * UNITS, axis=-1) * norm

def rand_unitary_haar(N=2, dims=None, seed=None):
    """
    Returns a Haar random unitary matrix of dimension
    ``dim``, using the algorithm of [Mez07]_.

    Parameters
    ----------
    N : int
        Dimension of the unitary to be returned.
    dims : list of lists of int, or None
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].

    Returns
    -------
    U : Qobj
        Unitary of dims ``[[dim], [dim]]`` drawn from the Haar
        measure.
    """
    import scipy.linalg as la

    #if dims is not None:
    #    _check_dims(dims, N, N)
    #else:
    #    dims = [[N], [N]]

    # Mez01 STEP 1: Generate an N × N matrix Z of complex standard
    #               normal random variates.
    Z = randnz((N, N), seed=seed)

    # Mez01 STEP 2: Find a QR decomposition Z = Q · R.
    Q, R = la.qr(Z)

    # Mez01 STEP 3: Create a diagonal matrix Lambda by rescaling
    #               the diagonal elements of R.
    Lambda = np.diag(R).copy()
    Lambda /= np.abs(Lambda)

    # Mez01 STEP 4: Note that R' := Λ¯¹ · R has real and
    #               strictly positive elements, such that
    #               Q' = Q · Λ is Haar distributed.
    # NOTE: Λ is a diagonal matrix, represented as a vector
    #       of the diagonal entries. Thus, the matrix dot product
    #       is represented nicely by the NumPy broadcasting of
    #       the *scalar* multiplication. In particular,
    #       Q · Λ = Q_ij Λ_jk = Q_ij δ_jk λ_k = Q_ij λ_j.
    #       As NumPy arrays, Q has shape (N, N) and
    #       Lambda has shape (N, ), such that the broadcasting
    #       represents precisely Q_ij λ_j.
    U = qt.Qobj(Q * Lambda)
    U.dims = dims
    return U

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
    U_list_new = [U.full() for U in U_list]
    F = 0
    for U in U_list_new:
        for V in U_list_new:
            F += np.sum(U.conj().T * V.T)**k * np.sum(V.conj().T * U.T)**k
    return F / len(U_list)**2

def frame_potential_einsum(U_list, k=1):
    U_list_new = [U.full() for U in U_list]
    F = 0
    for U in U_list_new:
        for V in U_list_new:
            F += np.einsum('ij,ji->', U.conj().T, V)**k * np.einsum('ij,ji->', U, V.conj().T)**k
    return F / len(U_list)**2

# Define physical aspects of problem
N = 30
num_modes = 2 # empty, occupied
dims = [num_modes for _ in range(N)]
excitations = 1 # restrict unitary to 1 particle sector
periodic = False

# Define operators
destroy_Op = get_destroy_operators(dims, excitations)
num_Op = deepcopy(destroy_Op)
for j in range(len(num_Op)):
    num_Op[j] = num_Op[j].dag() * num_Op[j]

# Define parameters of problem
T = 1 # period
#J = 0.1 * np.pi # regular
J = 0.49 * np.pi # chaos
theta = J*T
eta = 0 # time reverasl
#eta = 0.49 * np.pi

# Phases specifying trapping potential
Omega = 1 # trapping potential strength
phi_list = [delta_wj(N,j,Omega) * T for j in range(N)]

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
d = U_s.shape[0]
id = (U_s.dag() * U_s).tidyup()

# FIXME this should be done at several ensemble sizes and then
# extrapolated to infinity
k = 1
num_periods = d**(2*k) # recommended at least d**(2*k)
U_list = [U_s for _ in range(num_periods)]
for n in range(1, num_periods):
    U_list[n] = U_list[n] * U_list[n-1]
id_list = [id for _ in range(num_periods)]
harr_list = [rand_unitary_haar(N=N, dims=dims) for _ in range(num_periods)]
print(frame_potential_einsum(U_list, k), np.math.factorial(k) * d**k)
#print(frame_potential_einsum(id_list, k), d**(2*k)) # This works fine
print(frame_potential_qobj(harr_list, k), np.math.factorial(k))

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