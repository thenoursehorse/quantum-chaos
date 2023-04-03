import numpy as np
import scipy

from kicked_boson.quantum.system import GenericSystem

class Dicke(GenericSystem):
    def __init__(self, N=16,
                       Nc=320,
                       kappa=1.2,
                       lambda0=0.5,
                       **kwargs):
        
        self._N = N
        self._Nc = Nc
        self._j = int(self._N / 2)
        self._Nd =  self._N + 1
        self._dim = self._Nc * self._Nd
        self._kappa = kappa
        self._lambda0 = lambda0

        super().__init__(**kwargs)
        self.run()
    
    def run(self):
        self._H = [self.make_H() for _ in range(self._num_ensembles)]
        self._eigenenergies = []
        self._eigenvectors = []
        for m in range(self._num_ensembles):
            e, v = self.make_eigenenergies(self._H[m])
            self._eigenenergies.append(e)
            self._eigenvectors.append(v)
        self._eigenenergies = np.asarray(self._eigenenergies)
        self._eigenvectors = np.asarray(self._eigenvectors)

    def make_eigenenergies(self, H):
        e, v =  np.linalg.eigh(np.asarray(H))
        print("Finished diagonalizing Hamiltonian.")
        idx = e <= 4 / self._N
        e = e[idx]
        v = v[:,idx]
        idx = e >= 0.4 / self._N
        e = e[idx]
        v = v[:,idx]
        self._d = e.size
        return e, v

    def make_H(self):
        kappa = self._kappa
        lambda0 = self._lambda0
        Nc = self._Nc
        Nd = self._Nd
        N = self._N
        j = self._j

        #H = scipy.sparse.lil_array( (self._dim,self._dim), dtype=np.complex_ )
        H = np.zeros( (self._dim,self._dim), dtype=np.complex_ )
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

        #H = H.tocsc()

        print("Finished making Hamiltonian.")
        return H