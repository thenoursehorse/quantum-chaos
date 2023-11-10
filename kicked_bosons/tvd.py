import time
import numpy as np

import os
import argparse
import textwrap

from quantum_chaos.quantum.kicked_bosons import KickedBosons
from quantum_chaos.quantum.system import GenericSystemData
from quantum_chaos.functions import golden_ratio

def domain_unit_circle(x):
    # integration domain: sum of x^2 <= 1. 
    # For 2d, it's a unit circle; for 3d it's a unit sphere, etc
    # returns True for inside domain, False for outside
    
    return np.power(x,2).sum() <= 1

def domain_hypercube(x):
    return True

def mc_integrate(func, func_domain, a, b, dim, n = 1000):
    # Monte Carlo integration of given function over domain specified by func_domain
    # dim: dimensions of function
    
    # sample x
    x_list = np.random.uniform(a, b, (n, dim))
    
    # determine whether sampled x is in or outside of domain, and calculate its volume
    inside_domain = [func_domain(x) for x in x_list]
    frac_in_domain = sum(inside_domain)/len(inside_domain)
    domain = np.power(b-a, dim) * frac_in_domain
    
    # calculate expected value of func inside domain
    y = func(x_list)
    y_mean = y[inside_domain].sum()/len(y[inside_domain])
    
    # estimated integration
    integ = domain * y_mean
    
    return integ

if __name__ == '__main__':
    description = textwrap.dedent('''\
         Kicked bosons chain:
            N           : number of sites
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-M', type=int, default=300)
    parser.add_argument('-N', type=int, default=2)
    parser.add_argument('-num_ensembles', type=int, default=10)
    
    parser.add_argument('-thetaOmega', type=float, default=7.4)
    parser.add_argument('-thetaOmega_disorder', type=float, default=0.0)
    
    parser.add_argument('-Omega', type=float, default=np.pi/4.0)
    parser.add_argument('-WOmega', type=float, default=2)
    
    parser.add_argument('-etaOmega', type=float, default=0)
    #parser.add_argument('-etaOmega', type=float, default=golden_ratio())
    parser.add_argument('-etaOmega_disorder', type=float, default=0)
    
    parser.add_argument('-T', type=float, default=1)
    parser.add_argument('-num_modes', type=int, default=2)
    parser.add_argument('-excitations', type=int, default=1)
    parser.add_argument('-periodic', type=int, default=0)
    
    parser.add_argument('-use_qutip', type=int, default=0)
    
    parser.add_argument('-scale_width', type=float, default=1)
    parser.add_argument('-root_folder', type=str, default='./')
    parser.add_argument('-save_plots', type=int, default=0)
    parser.add_argument('-show_plots', type=int, default=1)
    parser.add_argument('-save_data', type=int, default=0)
    args = parser.parse_args()
    print(vars(args))
        
    ##########################################
    ## Make parameters for Floquet operator ##
    ##########################################
            
    theta = args.thetaOmega / (16.0 * args.Omega)
    theta_disorder = args.thetaOmega_disorder / (16.0 * args.Omega)
    phi_disorder = args.WOmega / (16.0 * args.Omega)
    eta = args.etaOmega / (16.0 * args.Omega)
    eta_disorder = args.etaOmega_disorder / (16.0 * args.Omega)
    
    folder = f'{args.root_folder}/data/M{args.M}_Nsamp{args.num_ensembles}/thetaOmega{args.thetaOmega:.2f}_WOmega{args.WOmega:.2f}'
    if args.save_plots or args.save_data:
        os.makedirs(f'{folder}', exist_ok=True)
    if args.save_plots:
        print(f"Saving plots to folder '{folder}'")

    #time_arr = np.array([300])
    time_arr = np.array([300])
        
    t_idx = -1 #299
    ix = 0
    iy = 0 #199
    stride = args.N
    vmax_abs = 0.3 # 0.2
    m = -1

    start = time.time()
    model = KickedBosons(M=args.M,
                         num_ensembles=args.num_ensembles,
                         theta=theta,
                         Omega=args.Omega,
                         eta=eta,
                         theta_disorder=theta_disorder,
                         phi_disorder=phi_disorder,
                         eta_disorder=eta_disorder,
                         T=args.T,
                         num_modes=args.num_modes,
                         excitations=args.excitations,
                         periodic=args.periodic,
                         folder=folder,
                         use_qutip=args.use_qutip) 
    end = time.time()
    print("Unitary construction took", end-start)
    
    start = time.time()
    model.set_unitary_evolve(num_ensembles=args.num_ensembles, time=time_arr)
    #model.set_unitary_evolve_floquet(num_ensembles=args.num_ensembles, time=time_arr)
    end = time.time()
    print("Time evolve unitaries took", end-start)

    samples = model._Ut[t_idx][:, ix:ix+stride, iy:iy+stride]
    samples = samples.reshape((samples.shape[0], -1)) # flatten each matrix into a vector
    #H, edges = np.histogramdd(r, bins=10) # bins = 10 # Doesn't work for large number of random variables (like 25)

    # Normalized covariance matrix
    #cov = np.cov(samples.real.T)
    cov = np.corrcoef(samples.real.T)

    # Standardize data?
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    samples_scaled = scaler.fit_transform(samples.real)

    # A normal distribution
    from scipy.stats import multivariate_normal
    #rv_normal = multivariate_normal(mean=[0] * (stride*stride), cov=[1] * (stride*stride), allow_singular=False)
    rv_normal = multivariate_normal(mean=[0] * (stride*stride), cov=cov, allow_singular=False)
    # Get a sample of data (should be same shape as samples)
    samples_normal = rv_normal.rvs(args.num_ensembles)
    # Get the 'empirical' probability at each of the points above using the pdf
    e_pdf_normal = rv_normal.pdf(samples_normal)
    # Check covariance matrix
    cov_normal = np.corrcoef(samples_normal.T)

    # NOTE: in asymptotics, bw: h->0 as samples: n->inf. Rate of convergence has to be chosen carefully,
    # but is usually h \propto n^(-1/5) = n^(-0.2)
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    start = time.time()
    # bw='cv_ml' bw='normal_reference' # bw=[0.2] * (stride*stride) bw = [np.power(args.num_ensembles, -0.2)] * (stride*stride)
    kde1 = KDEMultivariate(data=samples_scaled, var_type='c' * (stride*stride), bw='cv_ml')
    end = time.time()
    print("statsmodels kde took", end-start)

    # Calculate TDE (function is not smooth enough for tvd)
    #from scipy.integrate import nquad
    #start = time.time()
    #def tvd_integral(x1, x2, x3, x4):
    #    return np.abs(rv_normal.pdf([x1, x2, x3, x4]) - kde1.pdf([x1, x2, x3, x4]))
    #tvd = 0.5 * nquad(tvd_integral, [[-1,1]]* (stride*stride), full_output=True)
    #end = time.time()
    #print("Calculate tvd integral took", end-start)

    def tvd_integral(x):
        return np.abs(rv_normal.pdf(x) - kde1.pdf(x))
    start = time.time()
    tvd = 0.5 * mc_integrate(func = tvd_integral, func_domain = domain_hypercube, a = -3, b = 3, dim = stride*stride, n = 500000) #2000000)
    end = time.time()
    print("Calculate tvd integral took", end-start)
    
    from KDEpy import NaiveKDE, TreeKDE, FFTKDE
    start = time.time()
    #kde2 = FFTKDE(kernel='gaussian', bw=0.2).fit(samples_scaled)
    #kde2 = TreeKDE( kernel='gaussian', bw=np.power(args.num_ensembles, -0.2) ).fit(samples_scaled)
    kde2 = NaiveKDE( kernel='gaussian', bw=np.power(args.num_ensembles, -0.2) ).fit(samples_scaled)
    end = time.time()
    print("KDEpy kde took", end-start)
    #x_lin = np.linspace(-1, 1, 5)
    #grid = np.meshgrid(*([x_lin] * (stride*stride)), indexing='ij')
    #x = np.vstack(list(map(np.ravel, grid)) ).T
    #y = kde2.evaluate(x)
    # How to make this correct?
    #x, y = kde2.evaluate() #evaluate(50) #evaluate(x)

    start = time.time()
    def tvd_integral2(x):
        return np.abs(rv_normal.pdf(x) - kde2.evaluate(x))
    tvd2 = 0.5 * mc_integrate(func = tvd_integral2, func_domain = domain_hypercube, a = -3, b = 3, dim = stride*stride, n = 500000) #2000000)
    end = time.time()
    print("Calculate tvd2 integral took", end-start)

    # Test independence of random variables
    from sklearn.feature_selection import mutual_info_regression
    print()
    print("Independence tests:")
    for i in range(stride*stride):
        print(f"i={i}: {mutual_info_regression( samples_scaled, samples_scaled[:,i]) }")
            
    # Plots
    model.plot_matrix(save=args.save_plots, show=args.show_plots, scale_width=args.scale_width, t_idx=t_idx, vmin=-0.2, vmax=0.2, vmax_abs=vmax_abs)
    model.plot_submatrix(save=args.save_plots, show=args.show_plots, scale_width=args.scale_width, t_idx=t_idx, ix=ix, iy=iy, stride=stride, vmin=-0.05, vmax=0.05, vmax_abs=vmax_abs)