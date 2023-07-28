# quantum-chaos

Chaos measures for quantum models such as the kicked-rotor model of bosons.

Installation
---------------

1. Update packaging softare
    ```
    python3 -m pip install --upgrade pip setuptools wheel
    ```

1. (Optional) Create a 
[virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments).

1. Clone source
    ```
    git clone https://github.com/thenoursehorse/quantum-chaos
    ```

1. Install from local (-e allows to develop code without reinstalling)
    ```
    cd quantum-chaos
    python3 -m pip install -e ./
    ```

To uninstall

```
python3 -m pip uninstall quantum-chaos
```

Dependencies
-------------

* numpy
* scipy
* seaborn
* h5py
* (Optional) qutip for creating the systems. Note that the GenericSystem class assumes 
that eigenvectors and eigenvalues are numpy arrays.

Examples
---------------

* Boson chain
    ```
    python3 kicked_bosons/make_data.py -h
    ```

To do
---------------

1. Dicke model and spin models.
1. Fix qutip eigenvectors to numpy arrays.
1. A way to mask/remove eigenvalues.
1. OTOC and butterfly velocity for some specified operators.
1. 2-Renyi entropy of state in bipartite space and compare with isospectral quantity.
1. Partition eigenstates into energy blocks and do a heatmap of some quantities.
1. Variance in number of eigenvalues.
1. Vector coefficient distribution
1. Canabalize all the old ensemble stuff.
1. Add docstrings.
1. Mobility edges of effective Hamiltonian
1. Check how to see if anti-unitary.


References
---------------