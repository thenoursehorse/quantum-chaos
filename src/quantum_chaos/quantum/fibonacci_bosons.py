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

def hopping_unitary(M, theta_list, b_list, periodic=False):
    '''
    Sets the multi-port beam splitter (uses qutip)
    '''
    Op = 0
    if periodic:
        for j in range(M):
            k = (j+1)%M
            Op += theta_list[j] * b_list[j].dag() * b_list[k]
            Op += theta_list[j] * b_list[k].dag() * b_list[j]
    else:
        for j in range(M-1):
            k = j+1
            Op += theta_list[j] * b_list[j].dag() * b_list[k]
            Op += theta_list[j] * b_list[k].dag() * b_list[j]
    Op *= -1j
    return Op.expm()

def delta_wj_fast(j_arr, Omega=2, scale_N=True):
    N = len(j_arr)
    if scale_N:
        return (4.0 * Omega / N**2) * (j_arr - N/2.0)**2
    else:
        return Omega * (j_arr - N/2.0)**2

def fib_word(n):
    '''
    From https://www.geeksforgeeks.org/fibonacci-word/
    Returns the word as a string
    '''
    Sn_1 = "0"
    Sn = "01"
    tmp = ""
    for i in range(2, n + 1):
        tmp = Sn
        Sn += Sn_1
        Sn_1 = tmp
    return Sn

class FibonacciBosons(GenericSystem):
    def __init__(self, M,
                       theta,
                       Omega,
                       truncation_order=2,
                       T=1,
                       excitations=1,
                       num_modes=2,
                       periodic=False,
                       use_qutip=False,
                       calc_H_eff=False,
                       calc_eigenvectors=True,
                       word_length=15,
                       **kwargs):
    
        if use_qutip:
            raise ValueError('Qutip construction not implemented yet !')
        
        self._M = M
        self._N = self._M # FIXME
        self._theta = theta
        self._Omega = Omega
        self._T = T
        self._excitations = excitations
        self._num_modes = num_modes
        self._periodic = periodic
        self._use_qutip = use_qutip
        self._word_length = word_length

        self._dims = [num_modes for _ in range(self._M)]

        self._remove_vac = True
        
        self._word = fib_word(self._word_length)
        print('Max symbol size in word sequence', len(self._word))
        #print('Fibonacci word', self._word)
        
        super().__init__(**kwargs)
        self._model = 'Fibonacci bosons'
        self._rng = np.random.default_rng()
        self.run(calc_eigenvectors=calc_eigenvectors, calc_H_eff=calc_H_eff)

    def run(self, calc_eigenvectors=True, calc_H_eff=False):
        self._U0U1 = self.make_U0U1()
        self._U = [self.make_unitary() for _ in range(self._num_ensembles)]
        self._d = self._U[0].shape[0]
        
        self._eigenenergies = np.empty(shape=(self._num_ensembles, self._d))
        if calc_eigenvectors:
            self._eigenvectors = np.empty(shape=(self._num_ensembles, self._d, self._d), dtype=complex)
            for m in range(self._num_ensembles):
                self._eigenenergies[m], self._eigenvectors[m] = self.make_eigenenergies(self._U[m])
        else:
            for m in range(self._num_ensembles):
                self._eigenenergies[m] = self.make_eigenenergies(self._U[m], return_vecs=False)
        
        if calc_H_eff:
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
                return 1j * scipy.linalg.logm(U) / self._T

    def make_eigenenergies(self, U, return_vecs=True):
        if self._use_qutip: 
            if return_vecs:
                e, v = np.linalg.eig(U.full()) # qutip object to dense numpy array 
            else:
                e = np.linalg.eigvals(U.full())
        else:
            if return_vecs:
                e, v = np.linalg.eig(U)
            else:
                e = np.linalg.eigvals(U)
        
        ev = -1 * np.angle( e ) / self._T

        # quasienergies should only be defined up to multiples of the driving frequency omega = 2*pi*T^{-1}?
        # If so restrict ev to lie within the interval [0,2pi]? Wait but np.angle already restricts it to [-pi,pi]
        # so it is fine
        #ev = ev % (2*np.pi*(1/self._T))

        ind = ev.argsort()
        
        if return_vecs:
            return ev[ind], v[:,ind]
        else:
            return ev[ind]
    
    def make_operators(self):
        '''
        Make bosonic annihilation operator and the number operator on each site
        restricted to the maximum number of excitations
        '''
        from quantum_chaos.quantum.operators import get_destroy_operators
        self._destroy_Op = get_destroy_operators(self._dims, self._excitations)
        self._num_Op = [self._destroy_Op[j].dag() * self._destroy_Op[j] for j in range(self._M)]
    
    def make_U0U1(self):
        # Phases specifying hopping
        theta_list = self._theta * np.ones(self._M)
            
        # Phases specifying trapping potential with noise
        phi_list = delta_wj_fast(np.arange(1, self._M+1), self._Omega)

        if self._use_qutip:
            U1 = hopping_unitary(M=self._M, theta_list=theta_list, b_list=self._destroy_Op, periodic=self._periodic)
            U0 = phase_shift_unitary(M=self._M, phi_list=phi_list, b_list=self._destroy_Op)
            # NOTE for a single-particle restriction, qutip puts index 0 as the vacuum. 
            # Then the next element is site N ... the last element N+1 is site 0

            # Single particle sector only
            if self._remove_vac:
                from qutip import Qobj
                U0 = Qobj(U0[1:,1:], dims=U0.dims)
                U1 = Qobj(U1[1:,1:], dims=U1.dims)
            
            U0U1 = [U0, U1]
            #U = U1*U0

        else:
            # Phase shift angles
            H0 = scipy.sparse.lil_array( np.diag(phi_list) , dtype=np.longcomplex)
            
            # Multiport beam splitter
            OffDiagBand = theta_list[:self._M-1]
            H1 = scipy.sparse.lil_array( np.diag(OffDiagBand, 1) + np.diag(OffDiagBand.conj(), -1) , dtype=np.longcomplex)
            
            if self._periodic:
                H1[0,-1] = theta_list[-1]
                H1[-1,0] = theta_list[-1].conj()

            U0 = scipy.sparse.linalg.expm(-1j*H0)
            U1 = scipy.sparse.linalg.expm(-1j*H1)
    
            U0U1 = [U0, U1]
            
        return U0U1

    def make_unitary(self):
        #t = self._rng.integers(low=2, high=len(self._word), endpoint=True)
        t = self._rng.integers(low=len(self._word)/2, high=len(self._word), endpoint=True)
        U = scipy.sparse.lil_array(np.eye(self._U0U1[0].shape[0]), dtype=np.longcomplex)
        for i in range(t):
            idx = int(self._word[i])
            U = self._U0U1[idx] @ U
        return U.todense()

    @property
    def U(self):
        return self._U
    
    @property
    def H_eff(self):
        return self._H_eff