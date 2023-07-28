import numpy as np
import scipy

from quantum_chaos.quantum.system import GenericSystem

class Dicke(GenericSystem):
    def __init__(self, N=16,
                       Nc=320,
                       kappa=1.2,
                       lambda0=0.5,
                       cutoff_upper=None,
                       cutoff_lower=None,
                       **kwargs):
        
        self._N = N
        self._Nc = Nc
        self._j = int(self._N / 2)
        self._Nd =  self._N + 1
        self._dim = self._Nc * self._Nd
        self._kappa = kappa
        self._lambda0 = lambda0
        self._cutoff_upper = cutoff_upper
        self._cutoff_lower = cutoff_lower

        super().__init__(**kwargs)
        self._model = 'Dicke'
        self.run()
    
    def run(self):
        self._H = [self.make_H() for _ in range(self._num_ensembles)]
        self._eigenenergies = [] 
        self._eigenvectors = []
        for m in range(self._num_ensembles):
            e, v = self.make_eigenenergies(self._H[m])
            self._eigenenergies.append(e) 
            self._eigenvectors.append(v)
        self._eigenenergies = np.array(self._eigenenergies)
        self._eigenvectors = np.array(self._eigenvectors)
        self._d = self._eigenenergies[0].shape[0]
        print(f'Keeping {self._d} eigenenergies')
        
    def make_eigenenergies(self, H):
        e, v = np.linalg.eigh(H.todense())
        if self._cutoff_upper is not None:
            idx = e <= self._cutoff_upper / self._N
            e = e[idx]
            v = v[:,idx]
        if self._cutoff_lower is not None:
            idx = e >= self._cutoff_lower / self._N
            e = e[idx]
            v = v[:,idx]
        return e, v

    def make_H(self):
        kappa = self._kappa
        lambda0 = self._lambda0
        Nc = self._Nc
        Nd = self._Nd
        N = self._N
        j = self._j

        H = np.zeros( (self._dim,self._dim), dtype=np.complex_ )
        #H = scipy.sparse.lil_array( (self._dim,self._dim), dtype=np.complex_ )
        for n in range(Nc):
            for nn in range(Nc):
                for i in range(Nd):
                    for ii in range(Nd):
                        m = i - j
                        mm = ii - j

                        if n == nn and m == mm:
                            H[nn*Nd+ii, n*Nd+i] = n + m

                        if m == mm:
                            H[nn*Nd+ii, n*Nd+i] = (kappa/N) * m**2

                        if nn == (n-1) and mm == (m+1):
                            H[nn*Nd+ii, n*Nd+i] = (lambda0/np.sqrt(N)) * np.sqrt(n) * np.sqrt(j*(j+1) - m*(m+1))

                        if nn == (n-1) and mm == (m-1):
                            H[nn*Nd+ii, n*Nd+i] = (lambda0/np.sqrt(N)) * np.sqrt(n) * np.sqrt(j*(j+1) - m*(m-1))
                        
                        if nn == (n+1) and mm == (m+1):
                            H[nn*Nd+ii, n*Nd+i] = (lambda0/np.sqrt(N)) * np.sqrt(n+1) * np.sqrt(j*(j+1) - m*(m+1))
                        
                        if nn == (n+1) and mm == (m-1):
                            H[nn*Nd+ii, n*Nd+i] = (lambda0/np.sqrt(N)) * np.sqrt(n+1) * np.sqrt(j*(j+1) - m*(m-1)) 

        #H = H.tocsr()
        return scipy.sparse.csr_array(H)
    
    @property
    def H(self):
        return self._H