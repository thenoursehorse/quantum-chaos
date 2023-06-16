import numpy as np
import scipy

from quantum_chaos.quantum.system import GenericSystem

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

class KickedBosons(GenericSystem):
    def __init__(self, N,
                       J=1,
                       Omega=1,
                       eta=0,
                       theta_noise=0,
                       phi_noise=0.05,
                       eta_noise=0,
                       excitations=1,
                       num_modes=2,
                       periodic=False,
                       use_qutip=False,
                       **kwargs):
    
        
        self._N = N
        self._J = J
        self._Omega = Omega
        self._eta = eta
        self._theta_noise = theta_noise
        self._phi_noise = phi_noise
        self._eta_noise = eta_noise
        self._excitations = excitations
        self._num_modes = num_modes
        self._periodic = periodic
        self._use_qutip = use_qutip

        self._dims = [num_modes for _ in range(self._N)]

        self._remove_vac = True
        
        super().__init__(**kwargs)
        self.run()

    def run(self):
        if self._use_qutip:
            self.make_operators()
        self._U = [self.make_unitary() for _ in range(self._num_ensembles)]
        self._d = self._U[0].shape[0]
        self._eigenenergies = []
        self._eigenvectors = []
        self._eigenenergies = np.empty(shape=(self._num_ensembles, self._d))
        self._eigenvectors = np.empty(shape=(self._num_ensembles, self._d, self._d), dtype=np.complex_)
        for m in range(self._num_ensembles):
            self._eigenenergies[m], self._eigenvectors[m] = self.make_eigenenergies(self._U[m])
        self.make_H_eff()

    def make_H_eff(self, U=None):
        if U is None:
            self._H_eff = np.empty(shape=(self._num_ensembles, self._d, self._d), dtype=np.complex_)
            for m in range(self._num_ensembles):
                self._H_eff[m] = self._eigenvectors[m] @ \
                                    np.diag(self._eigenenergies[m]) @ \
                                        self._eigenvectors[m].conj().T
            return self.H_eff
        else:
            # NOTE very expensive to calculate because of the logm
            if self._use_qutip:
                return 1j * U.logm() / self.T
            else:
                return 1j * scipy.linalg.logm(U.todense()) / self.T

    def make_eigenenergies(self, U):
        #eigenenergies = H_eff.eigenenergies() # qutip
        #return eigenenergies
        
        #return U.eigenenergies()
        
        #e, v = np.linalg.eigh(H_eff.full())
        #e, v = np.linalg.eigh(H_eff.todense())
        
        if self._use_qutip: 
            e, v = np.linalg.eig(U.full())
            #e, v_qt = U.eigenstates()
            #v = np.empty(shape=(self.d, self.d), dtype=np.complex_)
            #for j in range(self.d):
            #    v[:,j] = v_qt[j].full().flatten()
        else:
            e, v = np.linalg.eig(U.todense())
        
        ev = -1 * np.angle( e )

        # should quasienergies should only be defined up to multiples of the driving frequency omega = 2*pi*T^{-1}?
        # If so restrict ev to lie within the interval [0,2pi]? Wait but np.angle already restricts it to [-pi,pi]
        # so it is fine
        #ev = ev % (2*np.pi*(1/self.T))

        ind = ev.argsort()
        return ev[ind], v[:,ind] 
    
    def make_operators(self):
        '''
        Make bosonic annihilation operator and the number operator on each site
        restricted to the maximum number of excitations
        '''
        from kicked_boson.quantum.operators import get_destroy_operators
        self._destroy_Op = get_destroy_operators(self._dims, self._excitations)
        self._num_Op = [self._destroy_Op[j].dag() * self._destroy_Op[j] for j in range(self._N)]
    
    def make_unitary(self):
        T = 1
        rng = np.random.default_rng()
        
        # Phases specifying hopping
        #theta_list = [self._J*T*np.pi for _ in range(self._N)]
        theta_list = self._J*T * np.ones(self._N)
        theta_list += self._theta_noise * rng.uniform(-np.pi, np.pi, self._N)
        
        # Phases specifying trapping potential with noise
        #phi_list = np.asarray([delta_wj(self._N, j+1, self._Omega) * T for j in range(self._N)])
        phi_list = delta_wj_fast(np.arange(1, self._N+1), self._Omega)
        phi_list += self._phi_noise * rng.uniform(-np.pi, np.pi, self._N)
        #phi_list += self._phi_noise * rng.normal(0, 1/(self._N), self._N)

        # Time reversal symmetry breaking on hopping
        #eta_list = [self._eta for _ in range(self._N)]
        eta_list = self._eta * np.ones(self._N)
        eta_list += self._eta_noise * rng.uniform(-np.pi, np.pi, self._N)

        if self._use_qutip:
            U_h = hopping_unitary(N=self._N, theta_list=theta_list, eta_list=eta_list, b_list=self._destroy_Op, periodic=self._periodic)
            U_p = phase_shift_unitary(N=self._N, phi_list=phi_list, b_list=self._destroy_Op)
            U = U_h*U_p
            # NOTE for a single-particle restriction, qutip puts index 0 as the vacuum. 
            # Then the next element is site N ... the last element N+1 is site 0

            # Single particle sector only
            if self._remove_vac:
                from qutip import Qobj
                U = Qobj(U[1:,1:], dims=U.dims)

        else:
            DiagBand = phi_list
            H1 = scipy.sparse.csc_array( np.diag(DiagBand) )
            
            # Now TR symmetry breaking
            #OffDiagBand = theta_list[:self._N-1]
            #H2 = scipy.sparse.coo_array( np.diag(OffDiagBand, 1) + np.diag(OffDiagBand.conj(), -1) )
            
            # TR symmetry breaking like Victor does (I don't think it does anything)
            OffDiagBand = theta_list[:self._N-1] * np.exp(1j*eta_list[:self._N-1])
            H2 = scipy.sparse.coo_array( np.diag(OffDiagBand, 1) + np.diag(OffDiagBand.conj(), -1) )
            
            if self._periodic:
                H2 = H2.tolil()
                H2[0,-1] = theta_list[-1] * np.exp(1j*eta_list[-1])
                H2[-1,0] = (theta_list[-1] * np.exp(1j*eta_list[-1])).conj()
            
            # TR symmetry breaking like Haldane model
            #OffOffDiagBand = 1j*eta_list[:self._N-2]
            #H2 += scipy.sparse.coo_array( np.diag(OffOffDiagBand, 2) + np.diag(OffOffDiagBand.conj(), -2) )
            #if self._periodic:
            #    H2 = H2.tolil()
            #    something here to handle it
            
            H2 = H2.tocsc()
            
            U = scipy.sparse.linalg.expm(-1j*H2) @ scipy.sparse.linalg.expm(-1j*H1)

        return U 
    
    @property
    def U(self):
        return self._U
    
    @property
    def H_eff(self):
        return self._H_eff