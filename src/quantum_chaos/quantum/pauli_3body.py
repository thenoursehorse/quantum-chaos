import numpy as np
import scipy

from quantum_chaos.quantum.system import GenericSystem

class Pauli3Body(GenericSystem):
    def __init__(self, N,
                       **kwargs):
    
        self._N = N
        self._d = 2**self._N
        self._dims = [2 for _ in range(self._N)]

        super().__init__(**kwargs)
        self.run()

    def run(self):
        self.make_operators()
        self._H = [self.make_H() for _ in range(self._num_ensembles)]
        self._eigenenergies = []
        self._eigenvectors = []
        self._eigenenergies = np.empty(shape=(self._num_ensembles, self._d))
        self._eigenvectors = np.empty(shape=(self._num_ensembles, self._d, self._d), dtype=np.complex_)
        for m in range(self._num_ensembles):
            self._eigenenergies[m], self._eigenvectors[m] = self.make_eigenenergies(self._H[m])

    def make_eigenenergies(self, H):
        return np.linalg.eigh(H.full())
    
    def make_operators(self):
        from quantum_chaos.quantum.operators import get_sigma_ops
        sx_list = get_sigma_ops(self._N, 'x')
        sy_list = get_sigma_ops(self._N, 'y')
        sz_list = get_sigma_ops(self._N, 'z')
        self._s_list = [sx_list, sy_list, sz_list]

    def make_H(self):
        J = np.random.default_rng().normal(0, 1/(self._N), [self._N,self._N,self._N,3,3,3])
        H = 0
        for i in range(self._N):
            for j in range(self._N):
                for k in range(self._N):
                    for alpha,s_alpha in enumerate(self._s_list):
                        for beta,s_beta in enumerate(self._s_list):
                            for gamma,s_gamma in enumerate(self._s_list):
                                H += J[i,j,k,alpha,beta,gamma] * s_alpha[i] * s_beta[j] * s_gamma[k]
        return H
    
    @property
    def H(self):
        return self._H