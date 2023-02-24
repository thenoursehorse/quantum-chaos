import numpy as np
import qutip as qt

import matplotlib.pyplot as plt

from kicked_boson.quantum.operators import *
    
def frame_potential_qobj(U_list, k_max=1):
    F = np.array([0 for _ in range(k_max)])
    for U in U_list:
        for V in U_list:
            val = (U.dag() * V).tr()
            val *= val.conjugate()
            val = val.real
            for k in range(k_max):
                F[k] += val**(k+1)
    return F / len(U_list)**2

def frame_potential_hadamard(U_list, k_max=1):
    F = np.array([0 for _ in range(k_max)])
    for U in U_list:
        for V in U_list:
            val = np.sum(U.conj().T * V.T) 
            val *= val.conjugate()
            val = val.real
            for k in range(k_max):
                F[k] += val**(k+1)
    return F / len(U_list)**2

def frame_potential_einsum(U_list, k_max=1):
    F = np.array([0 for _ in range(k_max)])
    for U in U_list:
        for V in U_list:
            val = np.einsum('ij,ji->', U.conj().T, V)
            val *= val.conjugate()
            val = val.real
            for k in range(k_max):
                F[k] += val**(k+1)
    return F / len(U_list)**2

def frame_potential_einsum_fast(U_list, k_max=1):
    F = np.array([0 for _ in range(k_max)])
    set_size = U_list.shape[0]
    val = np.einsum('nij,mji->nm', U_list, np.transpose(U_list.conj(), (0,2,1)))
    val *= val.conj()
    for k in range(k_max):
        F[k] = np.sum(np.power(val.real, k+1))
    return F / set_size**2

# FIXME this bound does not work and I am misunderstanding how to implement it
def frame_potential_bound(U_list, k_max=1):
    F = np.array([0 for _ in range(k_max)])
    d = U_list[0].shape[0]
    for U in U_list:
        val = U.tr()
        val *= val.conjugate()
        for k in range(k_max):
            F[k] += val**(2*(k+1)) / d**(2*(k+1))
    return F #/ len(U_list)

class Ensemble(object):
    def __init__(self, num_ensembles):
        self._num_ensembles = num_ensembles
        
        self.T = 1
        self._d = None
        self._U_list = None
        self._H_eff_list = None
        self._eigenenergies_list = None

    def run(self):
        self.make_operators()
        self._U_list = [self.make_unitary() for _ in range(self._num_ensembles)]
        self._d = self._U_list[0].shape[0]
        
        self._H_eff_list = [self.make_H_eff(self._U_list[m]) for m in range(self._num_ensembles)]
        self._eigenenergies_list = [self.make_eigenvalues(self._H_eff_list[m]) for m in range(self._num_ensembles)]

    def make_H_eff(self, U):
        return 1j * U.logm() / self.T

    def make_eigenvalues(self, H_eff):
        return H_eff.eigenenergies()

    def check_generic_spectrum(self):
        return None

    def level_spacings(self):
        return np.diff(self._eigenenergies_list)

    def level_ratios(self):
        s = self.level_spacings()
        s_shift_up = np.roll(s,-1,axis=1)
        return np.minimum(s_shift_up[:,1:], s[:,1:]) / np.maximum(s_shift_up[:,1:], s[:,1:])

    def plot_spacings(self, m=0):
        s = self.level_spacings()[m]
        x = np.linspace(0,max(s))
        poiss = np.exp(-x)
        goe = (np.pi / 2.0) * x * np.exp(-np.pi * x**2 / 4.0)
        gue = (32.0 / np.pi**2) * x**2 * np.exp(-4.0 * x**2 / np.pi)
        plt.plot(x, poiss, label='Poisson')
        plt.plot(x, goe, label='GOE')
        plt.plot(x, gue, label='GUE')
        # FIXME data is normalied diff to the others, so is wrong
        plt.hist(s, bins='sturges', density=True)
        plt.show()

    def plot_ratios(self, m=0):
        r = self.level_ratios()[m]
        x = np.linspace(0,1)
        poiss = (2.0 / ((1 + x)**2))
        goe = (27.0 / 4.0) * (x + x**2) / ((1 + x + x**2)**(5.0/2.0))
        plt.plot(x, poiss, label='Poisson')
        plt.plot(x, goe, label='GOE')
        plt.hist(r, bins='sturges', density=True)
        plt.show()

    def spectral_form_factor(self, order, t=1):
        c = [0 for _ in range(self._num_ensembles)]
        
        if order == 2:
            for m,e in enumerate(self._eigenenergies_list):
                c[m] = np.sum(np.exp(1j*(e[:,None] - e)*t)).real / self._d**2
        
        elif order == 3:
            for m,e in enumerate(self._eigenenergies_list):
                c[m] = 0 / self._d**3
        
        elif order == 4:
            for m,e in enumerate(self._eigenenergies_list):
                val = e[:,None] + e
                val = val[:,None] - e[:,None]
                val = val[:,None] - e[:,None,None]
                c[m] = np.sum(np.exp(1j*val*t)).real / self._d**4

        return np.asarray(c)

    def frame_potential_ff(self, t=1):
        c2 = self.spectral_form_factor(2, t)
        c4 = self.spectral_form_factor(4, t)
        #F = (self.d**2 / (self.d**2 - 1)) * ( np.sum(self.d**2 * c4 - 2 * c2) + 1 )
        F = (self.d**2 / (self.d**2 - 1)) * self.d**2 * c4 - 2 * c2 + 1
        return np.sum(F) / self._num_ensembles
        #F = (self.d**2 / (self.d**2 - 1)) * (self.d**2 * c4[0] - 2 * c2[0] + 1)
        #return F

    def evolve(self, T):
        dT = T-self.T
        self.T = T
        for m in range(self._num_ensembles):
            self._U_list[m] = self._U_list[m]**(dT+1)
        self._H_eff_list = [self.make_H_eff(self._U_list[m]) for m in range(self._num_ensembles)]
        self._eigenenergies_list = [self.make_eigenvalues(self._H_eff_list[m]) for m in range(self._num_ensembles)]

    def frame_potential(self, k_max=1, sum_type='einsum_fast'):
        if sum_type == 'einsum':
            U_list = [U.full() for U in self._U_list]
            return frame_potential_einsum(U_list, k_max)
        if sum_type == 'einsum_fast':
            U_list = np.asarray([U.full() for U in self._U_list])
            return frame_potential_einsum_fast(U_list, k_max)
        elif sum_type == 'hadamard':
            U_list = [U.full() for U in self._U_list]
            return frame_potential_hadamard(U_list, k_max)
        elif sum_type == 'qobj':
            return frame_potential_qobj(self._U_list, k_max)
        elif sum_type == 'bound':
            return frame_potential_bound(self._U_list, k_max)
        else:
            raise ValueError('Frame potential method not recognized !')

    def OTOC(self):
        return None

    def butterfly_velocity(self):
        return None

    def position(self):
        return None

    def level_statistics(self):
        return None

    @property
    def unitaries(self):
        return self._U_list
    
    @property
    def eigenenergies(self):
        return self._eigenenergies_list
    
    @property
    def H_effs(self):
        return self._H_eff_list

    @property
    def d(self):
        return self._d

class TimeIsingChainEnsemble(Ensemble):
    def __init__(self, num_ensembles,
                       N,
                       g=1,
                       alpha=1.51,
                       eta=1,
                       periodic=False):
    
        super().__init__(num_ensembles)
        
        self._N = N
        self._g = g
        self._alpha = alpha
        self._periodic = periodic

        self.U = 1
        self.run()
    
    def make_operators(self):
        self._sx_list = get_sigma_ops(N=self._N, axis='x')
        self._sz_list = get_sigma_ops(N=self._N, axis='z')
    
    def make_unitary(self):
        H = ising_chain(self._N, self._sx_list, self._sz_list, g=self._g, alpha=self._alpha)
        return (-1j * H).expm()

class TimeBosonChainEnsemble(Ensemble):
    def __init__(self, num_ensembles,
                       N,
                       J=1,
                       Omega=1,
                       eta=1,
                       excitations=1,
                       num_modes=2,
                       periodic=False):
    
        super().__init__(num_ensembles)
        
        self._N = N
        self._J = J
        self._Omega = Omega
        self._eta = eta
        self._excitations = excitations
        self._num_modes = num_modes
        self._periodic = periodic

        self._dims = [num_modes for _ in range(self._N)]

        self.U = 1
        self._remove_vac = True
        self.run()
    
    def make_operators(self):
        '''
        Make bosonic annihilation operator and the number operator on each site
        restricted to the maximum number of excitations
        '''
        self._destroy_Op = get_destroy_operators(self._dims, self._excitations)
        self._num_Op = [self._destroy_Op[j].dag() * self._destroy_Op[j] for j in range(self._N)]
    
    def make_unitary(self):
        T = 1
        # Phases specifying hopping
        theta_list = [self._J*T*np.pi for _ in range(self._N)]

        # Phases specifying trapping potential with noise
        phi_list = np.asarray([delta_wj(self._N, j, self._Omega) * T * np.pi for j in range(self._N)])

        # Time reversal symmetry breaking on hopping
        eta_list = [self._eta * np.pi for _ in range(self._N)]

        # Floquet operator at period T
        U_h = hopping_unitary(N=self._N, theta_list=theta_list, eta_list=eta_list, b_list=self._destroy_Op, periodic=self._periodic)
        U_p = phase_shift_unitary(N=self._N, phi_list=phi_list, b_list=self._destroy_Op)
        U = U_h*U_p
        # NOTE for a single-particle restriction, qutip puts index 0 as the vacuum. 
        # Then the next element is site N ... the last element N+1 is site 0

        # Single particle sector only
        if self._remove_vac:
            U = qt.Qobj(U[1:,1:], dims=U.dims)
        
        # Make an ensemble of increasing time
        self.U = U * self.U
        return self.U

class RandomBosonChainEnsemble(Ensemble):
    def __init__(self, num_ensembles,
                       N,
                       J=1,
                       Omega=1,
                       eta=1,
                       theta_noise=1,
                       phi_noise=1,
                       eta_noise=1,
                       excitations=1,
                       num_modes=2,
                       periodic=False):
    
        super().__init__(num_ensembles)
        
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

        self._dims = [num_modes for _ in range(self._N)]

        self._remove_vac = True
        self.run()
    
    def make_operators(self):
        '''
        Make bosonic annihilation operator and the number operator on each site
        restricted to the maximum number of excitations
        '''
        self._destroy_Op = get_destroy_operators(self._dims, self._excitations)
        self._num_Op = [self._destroy_Op[j].dag() * self._destroy_Op[j] for j in range(self._N)]
    
    def make_unitary(self):
        T = 1
        rng = np.random.default_rng()

        # Phases specifying hopping
        theta_list = [self._J*T*np.pi for _ in range(self._N)]
        theta_list += rng.uniform(-self._theta_noise*np.pi, self._theta_noise*np.pi, self._N)

        # Phases specifying trapping potential with noise
        phi_list = np.asarray([delta_wj(self._N, j, self._Omega) * T * np.pi for j in range(self._N)])
        phi_list += rng.uniform(-self._phi_noise*np.pi, self._phi_noise*np.pi, self._N)

        # Time reversal symmetry breaking on hopping
        eta_list = [self._eta * np.pi for _ in range(self._N)]
        eta_list += rng.uniform(-self._eta_noise*np.pi, self._eta_noise*np.pi, self._N)

        # Floquet operator at period T
        U_h = hopping_unitary(N=self._N, theta_list=theta_list, eta_list=eta_list, b_list=self._destroy_Op, periodic=self._periodic)
        U_p = phase_shift_unitary(N=self._N, phi_list=phi_list, b_list=self._destroy_Op)
        U = U_h*U_p
        # NOTE for a single-particle restriction, qutip puts index 0 as the vacuum. 
        # Then the next element is site N ... the last element N+1 is site 0

        # Single particle sector only
        if self._remove_vac:
            U = qt.Qobj(U[1:,1:], dims=U.dims)
        return U

class HaarEnsemble(Ensemble):
    def __init__(self, num_ensembles,
                       d):
    
        super().__init__(num_ensembles)
        self._d = d
        self.run()
    
    def make_operators(self):
        return None
    
    def make_unitary(self):
        return qt.rand_unitary(dimensions=self._d, distribution='haar')