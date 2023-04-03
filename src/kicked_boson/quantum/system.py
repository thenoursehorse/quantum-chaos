import numpy as np
import scipy
from copy import deepcopy

from kicked_boson.quantum.operators import *
from kicked_boson.functions import *

class GenericSystemData(object):
    def __init__(self, filename, **kwargs):
        self.filename = filename
        for key in kwargs.keys():
            self.__dict__[key] = kwargs[key]
    
    def save(self):
        with h5py.File(self.filename, 'w') as f:
            for key in self.__dict__.keys():
                if key != 'filename':
                    f.create_dataset(key, data=self.__dict__[key])

    def load(self):
        with h5py.File(self.filename, 'r') as f:
            for key in f.keys():
                self.__dict__[key] = np.array(f[key])

class GenericSystem(object):
    def __init__(self, num_ensembles=1, folder='figs/', T=1):
        self._num_ensembles = num_ensembles
        self._folder = folder
        self.T = T
        self._T0 = deepcopy(self.T)

    def truncate_eigenenergies(self):
        #e_min = min(self._eigenenergies.flatten())
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
    
    def eta_ratios(self):
        r = self.level_ratios()

        r_poiss_avg = 2 * np.log(2) - 1
        r_goe_avg = 4 - 2 * np.sqrt(3)
        r_gue_avg = 2 * np.sqrt(3) / np.pi - 1.0/2.0
        r_gse_avg = 32.0/15.0 * np.sqrt(3)/np.pi - 1.0/2.0

        return (np.mean(np.minimum(r, 1/r)) - r_poiss_avg) / (r_goe_avg - r_poiss_avg)

    def fractal_dimension(self, q, dagger=False):
        # FIXME take dagger so shannon entropy summed over eigenstates? How do I label them then?
        if dagger:
            return fractal_dimension(q, np.transpose(self._eigenvectors.conj(), (0,2,1)), sum_axis=1)
        else:
            return fractal_dimension(q, self._eigenvectors, sum_axis=1)
        # FIXME calcualte the fractal dimension for an initial state
        # at the centre of the chain. What happens to it over time?
    
    def set_fractal_dimension(self, num_ensembles=10, q_keep=2):
        if num_ensembles is None:
            num_ensembles = self._num_ensembles
        if num_ensembles > self._num_ensembles:
            num_ensembles = self._num_ensembles

        self._q_arr = np.arange(1,30+0.1)
        self._q_arr = np.insert(self._q_arr, 0, 0.5)
        
        self._dq_avg = np.empty(shape=self._q_arr.shape)
        self._dq_err = np.empty(shape=self._q_arr.shape)
        self._dq_goe_avg = np.empty(shape=self._q_arr.shape)
        self._dq_goe_err = np.empty(shape=self._q_arr.shape)
        self._dq_gue_avg = np.empty(shape=self._q_arr.shape)
        self._dq_gue_err = np.empty(shape=self._q_arr.shape)
    
        # For comparison to goe and gue
        # FIXME add gsu
        matrix_goe = np.empty([num_ensembles, self._d, self._d])
        matrix_gue = np.empty([num_ensembles, self._d, self._d], dtype=np.complex_)
        for m in range(num_ensembles):
            matrix_goe[m,...] = scipy.stats.ortho_group.rvs(self._d)
            matrix_gue[m,...] = scipy.stats.unitary_group.rvs(self._d)
        self._eigenenergies_goe, self._eigenvectors_goe = np.linalg.eigh(matrix_goe)
        self._eigenenergies_gue, self._eigenvectors_gue = np.linalg.eigh(matrix_gue)
    
        for i,q in enumerate(self._q_arr):
            dq = self.fractal_dimension(q)
            self._dq_avg[i] = np.mean(dq)
            self._dq_err[i] = np.std(dq)
    
            dq_goe = fractal_dimension(q, self._eigenvectors_goe, sum_axis=1)
            self._dq_goe_avg[i] = np.mean(dq_goe)
            self._dq_goe_err[i] = np.std(dq_goe)
            
            dq_gue = fractal_dimension(q, self._eigenvectors_gue, sum_axis=1)
            self._dq_gue_avg[i] = np.mean(dq_gue)
            self._dq_gue_err[i] = np.std(dq_gue)

            if q == q_keep:
                self._dq = dq
                self._dq_goe = dq_goe
                self._dq_gue = dq_gue
                self._q = q
    
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
        if num_ensembles is None:
            num_ensembles = self._num_ensembles
        if num_ensembles > self._num_ensembles:
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
        # NOTE read last paragraph in Sec. 4.3 on pg. 26 of 10.1007/JHEP11(2017)048. 
        # It basically says that the frame potential as calculated from the spectral
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
    
    #def evolve(self, T):
    #    dT = T-self.T
    #    self.T = T
    #    self._U = self._U**(dT+1)
    #    self._H_eff = self.make_H_eff(self._U)
    #    self._eigenenergies = self.make_eigenvalues(self._H_eff_list)

    def plot_eigenenergies(self, show=True, save=False):
        plot_eigenenergies(energies=self._eigenenergies, N=self._N, folder=self._folder, show=show, save=save)
    
    def plot_spacings(self, show=True, save=False):
        if not hasattr(self, "_energies_unfolded"):
            self.unfold_energies()
        s = self.level_spacings(self._energies_unfolded)
        plot_spacings(s, folder=self._folder, show=show, save=save)

    # FIXME plot Sigma^2, the variance in the number of eigenvalues, which shows the
    # long-range fluctuations. See Eq. 4 of doi:10.3390/e18100359 
    # FIXME from above paper plot survival probability (Eq. 13)
    def plot_ratios(self, show=True, save=False):
        # NOTE ratios and level spacings are short-range fluctuations
        r = self.level_ratios()
        plot_ratios(r, folder=self._folder, show=show, save=save)
    
    def plot_vector_coefficients(self, show=True, save=False):
        c = np.abs(self._eigenvectors[0])
        plot_vector_coefficients(c, self._d, folder=self._folder, show=show, save=save)
    
    def plot_fractal_dimension(self, show=True, save=False, dagger=False):
        if not hasattr(self, "_dq"):
            self.set_fractal_dimension()
        if dagger:
            x = np.empty(shape=self._eigenenergies.shape)
            for m in range(x.shape[0]):
                x[m] = np.arange(1, self._d+0.1)
            x_goe = np.empty(shape=self._eigenenergies_goe.shape)
            for m in range(x_goe.shape[0]):
                x_goe[m] = np.arange(1, self._d+0.1)
            x_gue = np.empty(shape=self._eigenenergies_gue.shape)
            for m in range(x_gue.shape[0]):
                x_gue[m] = np.arange(1, self._d+0.1)
            plot_fractal_dimension([x, x_goe, x_gue],
                                   [self._dq, self._dq_goe, self._dq_gue],
                                   [self._dq_avg, self._dq_goe_avg, self._dq_gue_avg],
                                   [self._dq_err, self._dq_goe_err, self._dq_gue_err],
                                   self._q_arr,
                                   folder=self._folder, 
                                   show=show, 
                                   save=save)        
        else:
            plot_fractal_dimension([self._eigenenergies/self._N, self._eigenenergies_goe/self._N, self._eigenenergies_gue/self._N],
                                   [self._dq, self._dq_goe, self._dq_gue],
                                   [self._dq_avg, self._dq_goe_avg, self._dq_gue_avg],
                                   [self._dq_err, self._dq_goe_err, self._dq_gue_err],
                                   self._q_arr, 
                                   folder=self._folder, 
                                   show=show, 
                                   save=save)        
    
    def plot_spectral_functions(self, show=True, save=False):
        if not hasattr(self, '_c2_avg'):
            self.set_spectral_functions()
        if not hasattr(self, '_c4_avg'):
            self.set_spectral_functions()
        plot_spectral_functions(time=self._time, 
                                c2=self._c2_avg, 
                                c4=self._c4_avg,
                                d=self._d,
                                #c2_err=self._c2_err, 
                                #c4_err=self._c4_err,
                                folder=self._folder, 
                                show=show, 
                                save=save)

    def plot_frame_potential(self, window=0, estimate=True, show=True, save=False):
        F1 = self.frame_potential(k=1)
        F2 = self.frame_potential(k=2)
        if estimate:
            F1_est, F2_est = self.frame_potential2()
            plot_frame_potential([self._time, F1, F2], \
                                 [self._time2, F1_est, F2_est], \
                                 window=window, folder=self._folder, show=show, save=save)
        else:
            plot_frame_potential([self._time, F1, F2], \
                                 folder=self._folder, show=show, save=save)
    
    def plot_loschmidt_echo(self, show=True, save=False):
        # FIXME add non-isometric twirl operator version to compare
        le1 = self.loschmidt_echo(kind='1st')
        le2 = self.loschmidt_echo(kind='2nd')
        plot_loschmidt_echo(self._time, le1, le2, self._d, folder=self._folder, show=show, save=save)
    
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