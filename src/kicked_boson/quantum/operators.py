
import qutip as qt
import numpy as np

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
        Op += phi_list[j] * b_list[j].dag() * b_list[j]
    Op *= -1j
    return Op.expm()

def hopping_unitary(N, theta_list, eta_list, b_list, periodic=False):
    Op = 0
    if periodic:
        for j in range(N):
            k = (j+1)%N
            Op += theta_list[j] * np.exp(1j*eta_list[j]) * b_list[j].dag() * b_list[k]
            Op += theta_list[j] * np.exp(-1j*eta_list[j]) * b_list[k].dag() * b_list[j]
    else:
        for j in range(N-1):
            k = j+1
            Op += theta_list[j] * np.exp(1j*eta_list[j]) * b_list[j].dag() * b_list[k]
            Op += theta_list[j] * np.exp(-1j*eta_list[j]) * b_list[k].dag() * b_list[j]
    Op *= -1j
    return Op.expm()

# parabolic potential
def delta_wj(N, j, Omega=2, scale_N=True):
    if scale_N:
        return (4.0 * Omega / N**2) * (j - N/2.0)**2
    else:
        return 4.0 * Omega * (j - N/2.0)**2

def delta_wj_fast(x, Omega=2, scale_N=True):
    if scale_N:
        N = len(x)
        #return (4.0 * Omega / N) * x
        return (4.0 * Omega / N**2) * x
    else:
        # NOTE omega = 0.01 = GSE statistics
        return Omega * x

def get_sigma_ops(N, axis):
    import qutip as qt
    si = qt.qeye(2)
   
    if axis == 'x':
        s = qt.sigmax()
    elif axis == 'y':
        s = qt.sigmay()
    elif axis == 'z':
        s = qt.sigmaz()
    elif axis == '+':
        s = qt.sigmap()
    elif axis == '-':
        s = qt.sigmam()
    else:
        raise ValueError('must be x,y,z,+,- for spin operators')

    s_list = []
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = s
        s_list.append(qt.tensor(op_list))
    return s_list
        
def ising_chain(N, sx_list, sz_list, g=0, alpha=np.inf):
    J = 1
    H = 0

    # magnetic field
    for n in range(N):
        H -= g * sx_list[n]

    # interaction terms
    if alpha == np.inf:
        for i in range(N):
            H += J * sz_list[i] * sz_list[(i+1) % N]

    else:
        for i in range(N):
            for j in range(N):
                if i > j:
                    coupling = J / np.power( np.abs(i-j), alpha)
                    H += coupling * sz_list[i] * sz_list[j]
    return H