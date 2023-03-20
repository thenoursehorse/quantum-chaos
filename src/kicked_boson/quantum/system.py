import numpy as np
import scipy
#import qutip as qt
from copy import deepcopy

import time

import seaborn as sns
import matplotlib.pyplot as plt

from kicked_boson.quantum.operators import *
from kicked_boson.functions import *

class GenericSystem(object):
    def __init__(self, num_ensembles=1, folder='figs/', T=1):
        self._num_ensembles = num_ensembles
        self._folder = folder
        self.T = T
        self._T0 = deepcopy(self.T)

        self.run()

    def run(self):
        #self.make_operators()
        self._U = [self.make_unitary() for _ in range(self._num_ensembles)]
        self._d = self._U[0].shape[0]
        #self._U0 = deepcopy(self._U) #FIXME
        # FIXME very expensive to calculate because of the logm
        #self._H_eff_list = [self.make_H_eff(self._U[m]) for m in range(self._num_ensembles)]
        self._eigenenergies = np.empty(shape=(self._num_ensembles, self._d))
        self._eigenvectors = np.empty(shape=(self._num_ensembles, self._d, self._d), dtype=np.complex_)
        for m in range(self._num_ensembles):
            self._eigenenergies[m], self._eigenvectors[m] = self.make_eigenenergies(self._U[m])

    #def reset(self):
    #    self._U = deepcopy(self._U0)
    #    self._T = deepcopy(self._T0)
    #    self.run()

    def make_H_eff(self, U):
        #self._H_eff = 1j * U.logm() / self.T
        self._H_eff = 1j * scipy.linalg.logm(U.todense()) / self.T
        return self._H_eff

    def make_eigenenergies(self, U):
        #eigenenergies = H_eff.eigenenergies()
        #return eigenenergies
        
        #return U.eigenenergies()
        
        #e, v = np.linalg.eigh(H_eff)
        #eigenenergies = e
        #return eigenenergies
        
        #ev = np.linalg.eigvals(U.todense())
        #return np.sort(-1 * np.angle( ev ))
        
        e, v = np.linalg.eig(U.todense())
        ev = -1 * np.angle( e )
        ind = ev.argsort()
        return ev[ind], v[:,ind] 

    def truncate_eigenenergies(self):
        e_min = min(self._eigenenergies.flatten())
        # FIXME remove/mask all energies that are less than 10*emin
        return None
        
    def unfold_energies(self, polytype='chebyshev', deg=48, plot=False, save=False, show=True):
        self._energies_unfolded = unfold_energies(self._eigenenergies, 
                                                  polytype=polytype, 
                                                  deg=deg, 
                                                  folder=self._folder, 
                                                  plot=plot, 
                                                  save=save, 
                                                  show=show)

    def level_spacings(self, e=None):
        if e is None:
            e = self._eigenenergies
        return np.diff(e)

    def level_ratios(self):
        s = self.level_spacings()
        s_shift_up = np.roll(s,-1,axis=-1)
        return np.minimum(s_shift_up[:,1:], s[:,1:]) / np.maximum(s_shift_up[:,1:], s[:,1:])

        #def s(es, n):
        #    return es[n+1] - es[n]
        #def r(es,n):
        #    return min(s(es,n),s(es,n+1)) / max(s(es,n),s(es,n+1))
        #num_eigs = len(self._eigenenergies)
        #ratios = np.empty(num_eigs-2)
        #for n in range(num_eigs-2):
        #    ratios[n] = r(np.sort(self._eigenenergies),n)
        #return ratios

    def average_level_ratios(self):
        r = self.level_ratios()
        r_avg = np.mean(r, axis=1)
        r_err = np.std(r_avg)
        r_avg = np.mean(r_avg)
        
        #y, x = np.histogram(r, bins='auto', range=(0,1), density=True)
        #r_avg = np.trapz(x[1::]*y)

        # FROM 10.1103/PhysRevLett.110.084101
        # <r> = 2 ln 2 - 1 = 0.38629 Poiss
        # <r> = 4 - 2 sqrt(3) = 0.53590 GOE 
        # <r> = 2 sqrt(3) / pi - 1/2 = 0.60266 GUE
        # <r> = 32/15 sqrt(3)/pi - 1/2 = 0.67617 GSE
        
        return r_avg, r_err
    
    def set_spectral_functions(self, Ti=0.1, Tf=1e4, Nt=1000, dT=0.1, window=0, Nt_window=2, minimal=False):
        #Nt = int(np.ceil((Tf - Ti + 1) / dT))
        #self._time = np.linspace(Ti, Tf, Nt, endpoint=True)
        self._time = np.logspace(np.log10(Ti), np.log10(Tf), Nt, endpoint=True)
        self._c2 = np.empty([Nt, self._num_ensembles])
            
        for batch_size in range(10,0,-1):
            if self._num_ensembles % batch_size == 0:
                break
        N_batch = int(self._num_ensembles / batch_size)

        if window > 0:
            print("WARNING: This kind of window averaging gives an incorrect too small t->inf limit for the frame potential !")
        
        for m in range(N_batch):
            if window > 0:
                window_t = np.linspace(-window/2, window/2, Nt_window)
                c2 = 0
                for t in range(Nt_window):
                    c2 += spectral_functions(self._eigenenergies[m*batch_size:(m+1)*batch_size], self._d, 2, t=self._time + t)
                self._c2[:, m*batch_size:(m+1)*batch_size] = c2 / Nt_window
            else:
                self._c2[:, m*batch_size:(m+1)*batch_size] = \
                    spectral_functions(self._eigenenergies[m*batch_size:(m+1)*batch_size], self._d, 2, t=self._time)

        self._c4 = (self._c2 * self._d**2)**2 / self._d**4

        self._c2_avg = np.mean(self._c2, axis=-1)
        self._c2_err = np.std(self._c2, axis=-1)
        
        self._c4_avg = np.mean(self._c4, axis=-1)
        self._c4_err = np.std(self._c4, axis=-1)

        if not minimal:
            self._c41 = np.empty([Nt, self._num_ensembles], dtype=np.complex_)
            self._c42 = np.empty([Nt, self._num_ensembles], dtype=np.complex_)
            for m in range(N_batch):
                if window > 0:
                    window_t = np.linspace(-window/2, window/2, Nt_window)
                    c41 = 0
                    c42 = 0
                    for t in range(Nt_window):
                        c41 += spectral_functions(self._eigenenergies[m*batch_size:(m+1)*batch_size], self._d, 41, t=self._time + t)
                        c42 += spectral_functions(self._eigenenergies[m*batch_size:(m+1)*batch_size], self._d, 42, t=self._time + t)
                    self._c41[:, m*batch_size:(m+1)*batch_size] = c41 / Nt_window
                    self._c42[:, m*batch_size:(m+1)*batch_size] = c42 / Nt_window
                else:
                    self._c41[:, m*batch_size:(m+1)*batch_size] = \
                        spectral_functions(self._eigenenergies[m*batch_size:(m+1)*batch_size], self._d, 41, t=self._time)
                    self._c42[:, m*batch_size:(m+1)*batch_size] = \
                        spectral_functions(self._eigenenergies[m*batch_size:(m+1)*batch_size], self._d, 42, t=self._time)
            
            self._c41_avg = np.mean(self._c41, axis=-1)
            self._c41_err = np.std(self._c41, axis=-1)
            self._c42_avg = np.mean(self._c42, axis=-1)
            self._c42_err = np.std(self._c42, axis=-1)

    def set_unitary_evolve(self, Ti=0.1, Tf=1e4, Nt=10, dT=0.1, num_ensembles=2):
        #tr(G+e^{-iHt}G G+e^(iHt)G)

        if num_ensembles is None:
            num_ensembles = self._num_ensembles
 
        self._time2 = np.logspace(np.log10(Ti), np.log10(Tf), Nt, endpoint=True)
        #self._time2 = np.array([1, 10, 100, 1000, 10000])
        #self._time2 = np.linspace(0, 100, 100, endpoint=True)

        exp_e_diag = np.exp(-1j * self._time2[:, None, None] * self._eigenenergies[:num_ensembles])
        exp_e = np.zeros((len(self._time2), num_ensembles, self._d, self._d), dtype=np.complex_)
        for n in range(self._d):
            exp_e[:,:,n,n] = exp_e_diag[:,:,n]
        
        vecs = self._eigenvectors[:num_ensembles]
        
        self._Ut = np.empty((len(self._time2), num_ensembles, self._d, self._d), dtype=np.complex_)
        for t in range(len(self._time2)):
            for m in range(num_ensembles):
                self._Ut[t,m, ...] = vecs[m] @ \
                                        exp_e[t, m, ...] @ \
                                            vecs[m].conj().T

        #self._Ut = np.empty((len(self._time2), num_ensembles, self._d, self._d), dtype=np.complex_)
        #for i,t in enumerate(self._time2):
        #    for m in range(num_ensembles):
        #        self._Ut[i,m,...] = np.linalg.matrix_power(self._U[m].todense(), int(t))

        # This is not faster
        #self._Ut = np.einsum('mij,tmjk,mjl->tmil', vecs, \
        #                                           exp_e, \
        #                                           np.transpose(vecs.conj(), (0,2,1)), \
        #                                           optimize=True)
    
    def set_unitary_fidelity(self):
        if not hasattr(self, '_Ut'):
            self.set_unitary_evolve()

        num_ensembles = self._Ut.shape[1]

        # Taken from the last appendix in 10.1007/JHEP11(2017)048
        # ignore coincident as they scale as 1/num_ensembles for the frame potential
        # and you only need 2 ensembles and time-average to get
        # a good representation of the large ensemble average
        self._unitary_fidelity = np.zeros((len(self._time2), num_ensembles, num_ensembles))
        for t in range(len(self._time2)):
            for i in range(num_ensembles):
                for j in range(num_ensembles):
                    if i != j: # Ignore coincident
                        self._unitary_fidelity[t,i,j] = np.abs(np.trace( self._Ut[t,i,...] @ self._Ut[t,j,...].conj().T ))**2
 
        # This is not really faster than above, and scales worse for larger # of ensembles
        #for i in range(num_ensembles):
        #    for j in range(num_ensembles):
        #        if i != j: # Ignore coincident
        #            tmp = np.trace(Ut[:,i,...] @ np.transpose( Ut[:,j,...].conj(), (0,2,1)), axis1=1, axis2=2)
        #            tmp *= tmp.conj()
        #            self._unitary_fidelity[:,i,j] = tmp.real
        
        # How to ignore coincident here?
        #self._unitary_fidelity = np.einsum('tmij,tnji->tmn', self._Ut, np.transpose(self._Ut.conj(), (0,1,3,2)), optimize=True)
        
        # Lower bound estimate for frame potential
        # If use Nt=same as isospectral estimate, the frame potential will follow
        # the same curve, but the minimum will go WAY below the Haar average!
        #self._unitary_fidelity = np.zeros((len(self._time2), num_ensembles))
        #for t in range(len(self._time2)):
        #    for i in range(num_ensembles):
        #            self._unitary_fidelity[t,i] = np.abs(np.trace( self._Ut[t,i] ))**4 / self._d**2
        
    def frame_potential(self, k=1):
        # NOTE read last paragraph in Sec. 4.3 on pg. 26. It basically says
        # that the frame potential as calculated from the spectral
        # form factors is valid for any ensemble whose measure is
        # unitarily invariant

        if not hasattr(self, '_c4_avg'):
            self.set_spectral_functions()
        if not hasattr(self, '_c2_avg'):
            self.set_spectral_functions()
        if k == 2:
            if not hasattr(self, '_c41_avg'):
                self.set_spectral_functions()
            if not hasattr(self, '_c42_avg'):
                self.set_spectral_functions()
            return frame_potential(self._d, self._c2_avg, self._c4_avg, self._c41_avg, self._c42_avg, k=2)
        else:
            return frame_potential(self._d, self._c2_avg, self._c4_avg)
        
    def frame_potential2(self):
        if not hasattr(self, '_unitary_fidelity'):
            self.set_unitary_fidelity()
        return frame_potential2(self._unitary_fidelity)
    
    def loschmidt_echo(self, kind='2nd'):
        if kind == '2nd':
            if not hasattr(self, '_c4_avg'):
                self.set_spectral_functions()
            return loschmidt_echo(self._d, c4=self._c4_avg, kind=kind)
        elif kind == '1st':
            if not hasattr(self, '_c2_avg'):
                self.set_spectral_functions()
            return loschmidt_echo(self._d, c2=self._c2_avg, kind=kind)
        else:
            raise ValueError('Unrecognized kind !')
    
    def otoc(self, kind='4-point'):
        # NOTE read Eq. 3.9 in 10.1007/JHEP11(2017)048
        # about how this Harr average is similar to just
        # measuring a few A operators (Pauli's)

        # NOTE read below Eq. 3.16, which implies that the
        # Pauli average above is only valid at long
        # time scales and misses the short time local fluctuations
        # that eventually get washed out

        # FIXME calculate this for like <n> or something
        # and see if it matches this Pauli estimate below
        if kind == '4-point':
            if not hasattr(self, '_c4_avg'):
                self.set_spectral_functions()
            return otoc(self._d, c4=self._c4_avg, kind=kind)
        elif kind == '2-point':
            if not hasattr(self, '_c2_avg'):
                self.set_spectral_functions()
            return otoc(self._d, c2=self._c2_avg, kind=kind)
        else:
            raise ValueError('Unreqognized kind !')
    
    def evolve(self, T):
        dT = T-self.T
        self.T = T
        self._U = self._U**(dT+1)
        self._H_eff = self.make_H_eff(self._U)
        self._eigenenergies = self.make_eigenvalues(self._H_eff_list)

    def plot_eigenenergies(self, show=True, save=False):
        plot_eigenenergies(energies=self._eigenenergies, folder=self._folder, show=show, save=save)
    
    def plot_spacings(self, show=True, save=False):
        if not hasattr(self, "_energies_unfolded"):
            self.unfold_energies()
        s = self.level_spacings(self._energies_unfolded)
        plot_spacings(s, folder=self._folder, show=show, save=save)

    def plot_ratios(self, show=True, save=False):
        r = self.level_ratios()
        plot_ratios(r, folder=self._folder, show=show, save=save)
    
    def plot_spectral_functions(self, show=True, save=False):
        if not hasattr(self, '_c2_avg'):
            self.set_spectral_functions()
        if not hasattr(self, '_c4_avg'):
            self.set_spectral_functions()
        plot_spectral_functions(self._time, self._c2_avg, self._c4_avg, folder=self._folder, show=show, save=save)

    def plot_frame_potential(self, window=0, show=True, save=False):
        F1 = self.frame_potential(k=1)
        F2 = self.frame_potential(k=2)
        F1_est, F2_est = self.frame_potential2()
        plot_frame_potential([self._time, F1, F2], \
                             [self._time2, F1_est, F2_est], \
                             window=window, folder=self._folder, show=show, save=save)
    
    def plot_loschmidt_echo(self, show=True, save=False):
        # FIXME add non-isometric twirl operator version to compare
        le1 = self.loschmidt_echo(kind='1st')
        le2 = self.loschmidt_echo(kind='2nd')
        plot_loschmidt_echo(self._time, le1, le2, folder=self._folder, show=show, save=save)
    
    def chi_distance(self, kind='ratios'):
        if kind == 'ratios':
            obs = self.level_ratios()
        elif kind == 'spacing':
            obs = self.level_spacings(self._energies_unfolded)
        xs, ys = ecdf(obs) # FIXME do the test by incorporating the error too?
        return chi_distance(np.mean(xs, axis=0), kind=kind) 

    @property
    def U(self):
        return self._U
    
    @property
    def eigenenergies(self):
        return self._eigenenergies
    
    @property
    def H_eff(self):
        return self._H_eff

    @property
    def d(self):
        return self._d

class BosonChain(GenericSystem):
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

        self._dims = [num_modes for _ in range(self._N)]

        self._remove_vac = True
        
        super().__init__(**kwargs)
    
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
        #theta_list = [self._J*T*np.pi for _ in range(self._N)]
        theta_list = self._J*T * np.ones(self._N)
        theta_list += self._theta_noise * rng.uniform(-np.pi, np.pi, self._N)

        # Phases specifying trapping potential with noise
        #phi_list = np.asarray([delta_wj(self._N, j+1, self._Omega) * T for j in range(self._N)])
        phi_list = delta_wj_fast(np.arange(1, self._N+1), self._Omega)
        phi_list += self._phi_noise * rng.uniform(-np.pi, np.pi, self._N)

        # Time reversal symmetry breaking on hopping
        eta_list = [self._eta for _ in range(self._N)]
        eta_list += self._eta_noise * rng.uniform(-np.pi, np.pi, self._N)

        # Floquet operator at period T
        #U_h = hopping_unitary(N=self._N, theta_list=theta_list, eta_list=eta_list, b_list=self._destroy_Op, periodic=self._periodic)
        #U_p = phase_shift_unitary(N=self._N, phi_list=phi_list, b_list=self._destroy_Op)
        #U = U_h*U_p
        # NOTE for a single-particle restriction, qutip puts index 0 as the vacuum. 
        # Then the next element is site N ... the last element N+1 is site 0

        # Single particle sector only
        #if self._remove_vac:
        #    U = qt.Qobj(U[1:,1:], dims=U.dims)

        DiagBand = phi_list
        H1 = scipy.sparse.csc_array( np.diag(DiagBand) )
        
        #OffDiagBand = (np.asarray(theta_list) * np.exp(1j*np.asarray(eta_list)))[:self._N-1]
        
        OffDiagBand = theta_list[:self._N-1]
        OffOffDiagBand = 1j*eta_list[:self._N-2]

        H2 = scipy.sparse.coo_array( np.diag(OffDiagBand, 1) + np.diag(OffDiagBand.conj(), -1) )
        H2 += scipy.sparse.coo_array( np.diag(OffOffDiagBand, 2) + np.diag(OffOffDiagBand.conj(), -2) )
        H2 = H2.tocsc()
        U = scipy.sparse.linalg.expm(-1j*H2) @ scipy.sparse.linalg.expm(-1j*H1)

        return U 