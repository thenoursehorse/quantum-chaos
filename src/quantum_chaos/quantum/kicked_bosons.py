import numpy as np
import scipy

from quantum_chaos.quantum.system import GenericSystem

def phase_shift_unitary(M, phi_list, b_list):
    '''
    Sets the phase on each site (uses qutip)
    '''
    Op = 0
    for j in range(M):
        Op += phi_list[j] * b_list[j].dag() * b_list[j]
    Op *= -1j
    return Op.expm()

def hopping_unitary(M, theta_list, eta_list, b_list, periodic=False):
    '''
    Sets the multi-port beam splitter (uses qutip)
    '''
    Op = 0
    if periodic:
        for j in range(M):
            k = (j+1)%M
            Op += theta_list[j] * np.exp(1j*eta_list[j]) * b_list[j].dag() * b_list[k]
            Op += theta_list[j] * np.exp(-1j*eta_list[j]) * b_list[k].dag() * b_list[j]
    else:
        for j in range(M-1):
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

def delta_wj_fast(j_arr, Omega=2, scale_N=True):
    N = len(j_arr)
    if scale_N:
        return (4.0 * Omega / N**2) * (j_arr - N/2.0)**2
    else:
        # NOTE omega = 0.01 = GSE statistics
        return Omega * (j_arr - N/2.0)**2

class KickedBosons(GenericSystem):
    def __init__(self, M,
                       theta,
                       Omega,
                       eta,
                       theta_disorder=0,
                       phi_disorder=0,
                       eta_disorder=0,
                       T=1,
                       excitations=1,
                       num_modes=2,
                       periodic=False,
                       use_qutip=False,
                       **kwargs):
    
        
        self._M = M
        self._N = self._M # FIXME
        self._theta = theta
        self._Omega = Omega
        self._eta = eta
        self._theta_disorder = theta_disorder
        self._phi_disorder = phi_disorder
        self._eta_disorder = eta_disorder
        self._T = T
        self._excitations = excitations
        self._num_modes = num_modes
        self._periodic = periodic
        self._use_qutip = use_qutip

        self._dims = [num_modes for _ in range(self._M)]

        self._remove_vac = True
        
        super().__init__(**kwargs)
        self._model = 'Kicked rotor'
        self._rng = np.random.default_rng()
        self.run()

    def run(self):
        if self._use_qutip:
            self.make_operators()
        self._U = [self.make_unitary() for _ in range(self._num_ensembles)]
        self._d = self._U[0].shape[0]
        #self._eigenenergies = []
        #self._eigenvectors = []
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
                return 1j * U.logm() / self._T
            else:
                return 1j * scipy.linalg.logm(U.todense()) / self._T

    def make_eigenenergies(self, U):
        #eigenenergies = H_eff.eigenenergies() # qutip
        #return eigenenergies
        
        #return U.eigenenergies()
        
        #e, v = np.linalg.eigh(H_eff.full())
        #e, v = np.linalg.eigh(H_eff.todense())
        
        if self._use_qutip: 
            e, v = np.linalg.eig(U.full()) # qutip object to dense numpy array 
            #e, v_qt = U.eigenstates()
            #v = np.empty(shape=(self.d, self.d), dtype=np.complex_)
            #for j in range(self.d):
            #    v[:,j] = v_qt[j].full().flatten()
        else:
            e, v = np.linalg.eig(U.todense()) # scipy sparse to dense numpy array
        
        ev = -1 * np.angle( e ) / self._T

        # quasienergies should only be defined up to multiples of the driving frequency omega = 2*pi*T^{-1}?
        # If so restrict ev to lie within the interval [0,2pi]? Wait but np.angle already restricts it to [-pi,pi]
        # so it is fine
        #ev = ev % (2*np.pi*(1/self._T))

        ind = ev.argsort()
        return ev[ind], v[:,ind] 
    
    def make_operators(self):
        '''
        Make bosonic annihilation operator and the number operator on each site
        restricted to the maximum number of excitations
        '''
        from quantum_chaos.quantum.operators import get_destroy_operators
        self._destroy_Op = get_destroy_operators(self._dims, self._excitations)
        self._num_Op = [self._destroy_Op[j].dag() * self._destroy_Op[j] for j in range(self._M)]
    
    def make_unitary(self):
        # Phases specifying hopping
        theta_list = self._theta * np.ones(self._M)
        theta_list += self._theta_disorder * self._rng.uniform(-1, 1, self._M)
            
        # Phases specifying trapping potential with noise
        phi_list = delta_wj_fast(np.arange(1, self._M+1), self._Omega)
        phi_list += self._phi_disorder * self._rng.uniform(-1, 1, self._M)

        # Time reversal symmetry breaking on hopping
        eta_list = self._eta * np.ones(self._M)
        eta_list += self._eta_disorder * self._rng.uniform(-1, 1, self._M)
        
        if self._use_qutip:
            U2 = hopping_unitary(M=self._M, theta_list=theta_list, eta_list=eta_list, b_list=self._destroy_Op, periodic=self._periodic)
            U1 = phase_shift_unitary(M=self._M, phi_list=phi_list, b_list=self._destroy_Op)
            U = U2*U1
            # NOTE for a single-particle restriction, qutip puts index 0 as the vacuum. 
            # Then the next element is site N ... the last element N+1 is site 0

            # Single particle sector only
            if self._remove_vac:
                from qutip import Qobj
                U = Qobj(U[1:,1:], dims=U.dims)

        else:
            # Phase shift angles
            #U1 = scipy.sparse.csc_array( np.diag(phi_list) )
            U1 = scipy.sparse.lil_array( np.diag(phi_list) )
            #U1 = scipy.sparse.coo_array( np.diag(phi_list) )
            
            # Multi-port beam splitter
            
            # with TR symmetry breaking like Victor does (I don't think it does anything)
            OffDiagBand = theta_list[:self._M-1] * np.exp(1j*eta_list[:self._M-1])
            #U2 = scipy.sparse.coo_array( np.diag(OffDiagBand, 1) + np.diag(OffDiagBand.conj(), -1) )
            U2 = scipy.sparse.lil_array( np.diag(OffDiagBand, 1) + np.diag(OffDiagBand.conj(), -1) )
            
            if self._periodic:
                #U2 = U2.tolil()
                U2[0,-1] = theta_list[-1] * np.exp(1j*eta_list[-1])
                U2[-1,0] = (theta_list[-1] * np.exp(1j*eta_list[-1])).conj()
            
            # with TR symmetry breaking like Haldane model
            #OffOffDiagBand = 1j*eta_list[:self._M-2]
            #U2 += scipy.sparse.coo_array( np.diag(OffOffDiagBand, 2) + np.diag(OffOffDiagBand.conj(), -2) )
            #if self._periodic:
            #    U2 = U2.tolil()
            #    something here to handle it
            
            #U2 = U2.tocsc()
            #U2 = U2.tolil()
            
            U = scipy.sparse.linalg.expm(-1j*U2) @ scipy.sparse.linalg.expm(-1j*U1)
            #U = (scipy.sparse.linalg.expm(-1j*U2) @ scipy.sparse.linalg.expm(-1j*U1)).tolil()

        return U

    #def make_unitary(self):
        #import qutip as qt
        # CUE
        #return scipy.sparse.csc_array( qt.random_objects.rand_unitary(self._N).full() )
        # Poisson
        #return scipy.sparse.csc_array( qt.random_objects.rand_unitary(self._N, distribution='exp').full() )
        # Poisson
        #H = scipy.sparse.csc_array( qt.random_objects.rand_herm(self._N).full() )
        #return scipy.sparse.linalg.expm(-1j*H*self._T)
        #H = scipy.sparse.csc_array( scipy.stats.ortho_group.rvs(self._N) )
        #return scipy.sparse.linalg.expm(-1j*H*self._T)
        #H = scipy.sparse.csc_array( scipy.stats.unitary_group.rvs(self._N) )
        #return scipy.sparse.linalg.expm(-1j*H*self._T)

    @property
    def U(self):
        return self._U
    
    @property
    def H_eff(self):
        return self._H_eff