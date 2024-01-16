import time
import numpy as np
from scipy.linalg import cholesky
import scipy
import scipy.special

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import argparse
import textwrap

from quantum_chaos.quantum.kicked_bosons import KickedBosons
from quantum_chaos.quantum.system import GenericSystemData
from quantum_chaos.plotter import SeabornFig2Grid

from copy import deepcopy

def vec(A):
    '''
    Mathematical vectorization procedure on a matrix. Assumes the 
    last two dimensions of A are the matrix dimensions.
    '''
    p = A.shape[-2]
    q = A.shape[-1]
    return A.reshape(-1, p*q)
        
def round_to_n(x, n=1, kind='round'):
    import math
    exponent = -int(math.floor(math.log10(abs(x)))) + (n - 1)
    if kind == 'round':
        return math.round(x * 10**exponent) / 10**exponent
    if kind == 'floor':
        return math.floor(x * 10**exponent) / 10**exponent
    if kind == 'ceil':
        return math.ceil(x * 10**exponent) / 10**exponent

def mle_adjust(n, sigma_squared, S):
    eta = S[0,0] / (n*sigma_squared)
    S = S / S[0,0]
    S[1:,1:] = eta * S[1:,1:] + (1 - eta) * np.outer(S[0,1:], S.conj()[1:,0])
    return S

def mle_MNorm(X, iterations=10000, tol=1e-10, unscale=True):
    '''
    Determine the maximum liklihood error (MLE) estimates for the center, mu, 
    and covariance matrices, Sigma_s and Sigma_c, for a (complex) normal 
    matrix-variate sample distribution.
    '''
    n = X.shape[0]
    p = X.shape[1]
    q = X.shape[2]

    # Initialize guessed parameters
    mu = np.zeros([p, q])
    sigma_squared = 0
    Sigma_s = np.eye(p)
    Sigma_c = np.eye(q)
        
    # Self consistently find maximum liklihood estimate (MLE) of parameters
    converged = False
    for t in range(iterations):
        delta = X - mu

        # Mean estimate 
        mu_new = np.sum(X, axis=0)

        Sigma_s_inv = np.linalg.inv(Sigma_s)
        Sigma_c_inv = np.linalg.inv(Sigma_c)

        # Covariance matrix estimates
        Sigma_c_new = np.sum( np.swapaxes(delta.conj(), axis1=-1, axis2=-2) @ Sigma_s_inv @ delta, axis=0)
        Sigma_s_new = np.sum( delta @ Sigma_c_inv @ np.swapaxes(delta.conj(), axis1=-1, axis2=-2), axis=0)
    
        # Overall covariance scaling factor
        sigma_squared_new = np.sum(vec(delta.conj()) * vec( Sigma_s_inv @ Sigma_c_inv @ delta ) )
        
        # Scale matrices
        mu_new = mu_new / n
        sigma_squared_new = sigma_squared_new / (p*q*n)
        Sigma_c_new = mle_adjust(p*n, sigma_squared_new, Sigma_c_new)
        Sigma_s_new = mle_adjust(q*n, sigma_squared_new, Sigma_s_new)
        
        # Check 1-norm tolerance
        norm = 0
        norm += np.linalg.norm((mu_new - mu).ravel(), ord=1) / np.linalg.norm(mu.ravel(), ord=1)
        norm += np.linalg.norm((Sigma_c_new - Sigma_c).ravel(), ord=1) / np.linalg.norm(Sigma_c.ravel(), ord=1)
        norm += np.linalg.norm((Sigma_s_new - Sigma_s).ravel(), ord=1) / np.linalg.norm(Sigma_s.ravel(), ord=1)
        norm += np.sum( np.abs(sigma_squared_new - sigma_squared) ) / np.sum( np.abs(sigma_squared) )
        if norm < tol:
            converged = True

        # Backup guess
        mu = deepcopy(mu_new)
        sigma_squared = deepcopy(sigma_squared_new)
        Sigma_c = deepcopy(Sigma_c_new)
        Sigma_s = deepcopy(Sigma_s_new)

        if converged:
            break
    
    if converged:
        print("Normal matrix-variate parameter estimation converged after", t, "iterations with tolerance", norm)
    else:
        print("Normal matrix-variate parameter estimation NOT converged after", t, "iterations with tolerance", norm)

    if unscale:
        return mu, Sigma_s * np.sqrt(sigma_squared), Sigma_c * np.sqrt(sigma_squared), sigma_squared
    else:
        return mu, Sigma_s, Sigma_c, sigma_squared

def transform_standard_MNorm(X, mu, Sigma_s, Sigma_c):
    A = cholesky(np.linalg.inv(Sigma_s), lower=False)
    B = cholesky(np.linalg.inv(Sigma_c), lower=True)
    C = - A @ mu @ B
    return A @ X @ B + C

def domain_unit_circle(x):
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
    
    parser.add_argument('-kde1', type=int, default=0)
    parser.add_argument('-kde2', type=int, default=0)
    parser.add_argument('-tvd1', type=int, default=0)
    parser.add_argument('-tvd2', type=int, default=0)
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
    
    # Set time to Heisenberg time
    t_heis = model.heisenberg_time()
    #time_arr = np.array([t_heis])
    time_arr = np.array([1e8])
        
    t_idx = -1 #299
    ix = 0
    iy = 0 #199
    stride = args.N
    vmax_abs = 0.3 # 0.2
    m = -1
    
    start = time.time()
    model.set_unitary_evolve(num_ensembles=args.num_ensembles, time=time_arr)
    #model.set_unitary_evolve_floquet(num_ensembles=args.num_ensembles, time=time_arr)
    end = time.time()
    print("Time evolve unitaries took", end-start)

    # Take upper left corner of unitaries
    samples = model._Ut[t_idx][:, ix:ix+stride, iy:iy+stride] * np.sqrt(args.M)

    # Get the standard normal distribution
    # Real case
    #from scipy.stats import multivariate_normal
    #n = 100
    #rvs_norm = multivariate_normal.rvs(cov=np.eye(args.N*args.N), size=n).reshape(n, args.N, args.N)
    
    # Real case should be same as above
    #rng = np.random.default_rng()
    #n = 100
    #rvs_norm = rng.standard_normal(size=(n, args.N, args.N))

    # Complex case 
    from randomgen import ExtendedGenerator
    eg = ExtendedGenerator()
    n = samples.shape[0]
    #n = 1_000_000
    #n = 100_000
    #n = 10_000
    #n = 1_000
    rvs_cnorm = eg.multivariate_complex_normal(loc=[0] * (stride*stride), gamma=np.eye(stride*stride), size=n).reshape(n, stride, stride)

    # FIXME FOR TESTING
    # Check against normal (works, but not very well if do log(x) transform)
    #samples = eg.multivariate_complex_normal(loc=[0] * (stride*stride), gamma=np.eye(stride*stride), size=n).reshape(n, stride, stride)
    # Check against Haar (works, but not very well if do log(x) transform)
    #from scipy.stats import unitary_group
    #M = 30
    #U = unitary_group.rvs(M, size=n)
    #samples = U[:, ix:ix+stride, iy:iy+stride] * np.sqrt(M)

    # Complex case should be same as above
    #rng = np.random.default_rng()
    #n = 100
    #rvs_cnorm = rng.normal(scale=1/np.sqrt(2), size=(n, args.N, args.N)) + 1j * rng.normal(scale=1/np.sqrt(2), size=(n, args.N, args.N))

    # Find maximum liklihood estimates (MLE) of the matrix normal parameter
    # which assumes that matrices are matrix normal
    mu, Sigma_s, Sigma_c, sigma_squared = mle_MNorm(samples)
    mu_cnorm, Sigma_s_cnorm, Sigma_c_cnorm, sigma_squared_cnorm = mle_MNorm(rvs_cnorm)
    
    # Transform samples to standard normal (just assume that the data is normal, even if it is not)
    samples_transformed = transform_standard_MNorm(samples, mu, Sigma_s, Sigma_c)
    mu_transformed, Sigma_s_transformed, Sigma_c_transformed, sigma_squared_transformed = mle_MNorm(samples_transformed)
    rvs_cnorm_transformed = transform_standard_MNorm(rvs_cnorm, mu_cnorm, Sigma_s_cnorm, Sigma_c_cnorm)
    
    # Take SVD of relevant samples
    #samples = samples_transformed
    #samples_S = np.linalg.svd( samples_transformed, compute_uv=False)**2
    samples_S, _ = np.linalg.eigh(samples_transformed @ np.swapaxes(samples_transformed.conj(), axis1=-1, axis2=-2))
    rvs_cnorm_S, _ = np.linalg.eigh(rvs_cnorm_transformed @ np.swapaxes(rvs_cnorm_transformed.conj(), axis1=-1, axis2=-2))
    #H, edges = np.histogramdd(samples_S, bins=10) # bins = 10 # Doesn't work for large number of random variables (like 25)

#    print("START")
#    rng = np.random.default_rng()
#    W_mean = []
#    W_S = []
#    W_S_mean = []
#    n_arr = np.array([100, 1000, 10000, 100000]) #, 1000000, 10000000, 100000000])
#    for i, n in enumerate(n_arr):
#        A = rng.normal(scale=1/np.sqrt(2), size=(n, args.N, args.N)) + 1j * rng.normal(scale=1/np.sqrt(2), size=(n, args.N, args.N))
#        #A = rng.normal(size=(n, args.N, args.N)) # WISHART distribution
#        W = A @ np.swapaxes(A.conj(), axis1=-1, axis2=-2 )
#        W_mean.append( np.mean(W, axis=0) )
#        e, v = np.linalg.eigh(W)
#        W_S.append( e )
#        W_S_mean.append( np.mean(W_S[i], axis=0) )
#    # Below is only real wishart, not complex
#    from scipy.stats import wishart
#    rv_wishart = wishart(df=args.N, scale=np.eye(args.N))
#    rvs_wishart = rv_wishart.rvs(size=100000)
#    rvs_wishart_S, _ = np.linalg.eigh(rvs_wishart)
#    # Below only makes a single matrix instance
#    from skrmt.ensemble.wishart_ensemble import WishartEnsemble
#    wre = WishartEnsemble(beta=2, p=args.N, n=args.N, tridiagonal_form=True)
#    wce = WishartEnsemble(beta=2, p=args.N, n=args.N, tridiagonal_form=True)
#    print("END")

    # Normalized covariance matrix (probably not valid for singular values?)
    #cov = np.cov(samples_S.T)
    #cov = np.corrcoef(samples_S.T)

    # Complex wishart eigenvalue distribution
    def wishart_eig_pdf(x, n=None):
        def mv_gamma(m, n):
            out = np.pi**(m*(m-1)/2.0)
            for i in range(1, m+1):
                out *= scipy.special.gamma(n - i + 1)
            return out

        if n is None:
            n = x.shape[0]
        m = x.shape[-1]
        
        f = 2**(-m*n) * np.pi**(m*(m-1)) / ( mv_gamma(m, n) * mv_gamma(m, m) )
        f *= np.exp(-0.5 * np.sum(x, axis=-1)) * np.prod(x**(n-m), axis=-1)
        for i in range(m):
            for j in range(m):
                if i < j:
                    f *= (x[:,i] - x[:,j])**2
        return f
    
    # Below is definitely wrong and not the pdf of the eigenvalues 
    def normal_svd_pdf(x):
        n = x.shape[-1]
        return (1/np.pi**(n*2)) * np.prod( np.exp(-x**2), axis=-1)
    
    # Calculate TDE (function is not smooth enough for tvd)
    #from scipy.integrate import nquad
    #start = time.time()
    #def tvd_integral(x1, x2, x3, x4):
    #    return np.abs(rv_normal.pdf([x1, x2, x3, x4]) - kde1.pdf([x1, x2, x3, x4]))
    #tvd = 0.5 * nquad(tvd_integral, [[-1,1]]* (stride*stride), full_output=True)
    #end = time.time()
    #print("Calculate tvd integral took", end-start)

    def transformation(x):
        return x
        #return np.log(x)

    def inv_transformation(x):
        return x
        #return np.exp(x)

    def det_jacobian(x):
        return np.ones(x.shape[0])
        #return 1.0 / np.prod(x, axis=-1)

    if args.kde1: 
        # NOTE: in asymptotics, bw: h->0 as samples: n->inf. Rate of convergence has to be chosen carefully,
        # but is usually h \propto n^(-1/5) = n^(-0.2) for univariate
        from statsmodels.nonparametric.kernel_density import KDEMultivariate
        start = time.time()
        # bw='cv_ml' bw='normal_reference' # bw=[0.2] * (stride*stride) bw = [np.power(args.num_ensembles, -0.2)] * (stride*stride)
        kde1 = KDEMultivariate(data=transformation(samples_S), var_type='c' * stride, bw='cv_ml')
        kde1_cnorm = KDEMultivariate(data=transformation(rvs_cnorm_S), var_type='c' * stride, bw='cv_ml')
        end = time.time()
        print("statsmodels kde took", end-start)

    if args.tvd1:
        def tvd1_integral(x):
            x_transformed = transformation(x)
            #return np.abs(rv_normal.pdf(x) - kde1.pdf(x))
            #return np.abs(normal_svd_pdf(x) - kde1.pdf(x))
            #return np.abs(wishart_eig_pdf(x, stride) - kde1.pdf(x))
            #return np.abs(kde1_cnorm.pdf(x) - kde1.pdf(x))
            return np.abs( (kde1_cnorm.pdf(x_transformed) - kde1.pdf(x_transformed))*det_jacobian(x) )
        #n_cycles = np.array([1000000, 2000000, 3000000, 4000000, 5000000])
        n_cycles = np.array([1000000])
        
        tvd1 = np.zeros(len(n_cycles))
        start = time.time()
        for i, n in enumerate(n_cycles):
            tvd1[i] = 0.5 * mc_integrate(func = tvd1_integral, 
                                         func_domain = domain_hypercube, 
                                         a = min( np.min(rvs_cnorm_S), np.min(samples_S) ), 
                                         b = max( np.max(rvs_cnorm_S), np.max(samples_S) ), 
                                         dim = stride, 
                                         n = n)
        end = time.time()
        print("Calculate tvd1 integral took", end-start)
        print(f"tvd1 = {tvd1}")

        from scipy.integrate import nquad
        def bounds(*args):
            integration_number = len(args) + 1
            num_integrals = samples_S.shape[-1]
            counter = num_integrals - integration_number
            bound_lower = min( np.min(rvs_cnorm_S, axis=0)[counter], np.min(samples_S, axis=0)[counter] )
            bound_upper =  max( np.max(rvs_cnorm_S, axis=0)[counter], np.max(samples_S, axis=0)[counter] )
            return [bound_lower, bound_upper]
        def tvd1_integral_nquad(*args):
            x = np.asarray(args)
            x_transformed = transformation(x)
            return np.abs( (kde1_cnorm.pdf(x_transformed) - kde1.pdf(x_transformed))*det_jacobian(x) )
        #start = time.time()
        #tvd1_nquad = nquad(tvd1_integral_nquad, [bounds, bounds], full_output=True)
        #end = time.time()
        #print("Calculate tvd1 integral nquad took", end-start)
        #print(f"tvd1_nquad = {tvd1_nquad}")


    if args.kde2: 
        from KDEpy import NaiveKDE, TreeKDE, FFTKDE
        start = time.time()
        kde2 = NaiveKDE( kernel='gaussian', bw=np.power(samples_S.shape[0], -0.2) ).fit(transformation(samples_S))
        kde2_cnorm = NaiveKDE( kernel='gaussian', bw=np.power(rvs_cnorm_S.shape[0], -0.2) ).fit(transformation(rvs_cnorm_S))
        end = time.time()
        print("KDEpy kde took", end-start)

    if args.tvd2:
        def tvd2_integral(x):
            x_transformed = transformation(x)
            #return np.abs(rv_normal.pdf(x) - kde2.evaluate(x))
            #return np.abs(normal_svd_pdf(x) - kde2.evaluate(x))
            #return np.abs(wishart_eig_pdf(x, stride) - kde2.evaluate(x))
            #return np.abs(kde2_cnorm.evaluate(x) - kde2.evaluate(x))
            return np.abs( (kde2_cnorm.evaluate(x_transformed) - kde2.evaluate(x_transformed))*det_jacobian(x) )
        #n_cycles = np.array([1000000, 2000000, 3000000, 4000000, 5000000])
        n_cycles = np.array([1000000])
        tvd2 = np.zeros(len(n_cycles))
        start = time.time()
        for i, n in enumerate(n_cycles):
            tvd2[i] = 0.5 * mc_integrate(func = tvd2_integral, 
                                         func_domain = domain_hypercube, 
                                         a = min( np.min(rvs_cnorm_S), np.min(samples_S) ), 
                                         b = max( np.max(rvs_cnorm_S), np.max(samples_S) ), 
                                         dim = stride, 
                                         n = n)
        end = time.time()
        print("Calculate tvd2 integral took", end-start)
        print(f"tvd2 = {tvd2}")
        
#        def tvd2_integral_nquad(*args):
#            x = [np.asarray(args)]
#            x_transformed = transformation(x)
#            return np.abs( (kde2_cnorm.evaluate(x_transformed) - kde2.evaluate(x_transformed))*det_jacobian(x) )
#        start = time.time()
#        tvd2_nquad = nquad(tvd2_integral_nquad, [bounds, bounds], full_output=True)
#        end = time.time()
#        print("Calculate tvd2 integral nquad took", end-start)
#        print(f"tvd2_nquad = {tvd2_nquad}")

    from KDEpy import FFTKDE
    start = time.time()
    # bw = 'scott' ? 'silvermann' ?
    kde_fft = FFTKDE( kernel='gaussian', bw=np.power(samples_S.shape[0], -0.2) ).fit(transformation(samples_S))
    kde_fft_cnorm = FFTKDE( kernel='gaussian', bw=np.power(rvs_cnorm_S.shape[0], -0.2) ).fit(transformation(rvs_cnorm_S))
    end = time.time()
    print("Calculate kde_fft took", end-start)

    def get_x_grid(samples, npoints=4096, indexing='ij'):
        num_features = samples[0].shape[-1]
        x_linear = []
        for i in range(num_features):
            bound_lower = np.inf
            bound_upper = -np.inf
            for s in samples:
                bound_lower = min(bound_lower, np.min(s, axis=0)[i])
                bound_upper = max(bound_upper, np.max(s, axis=0)[i])
            bound_lower = round_to_n(bound_lower, n=5, kind='floor')
            bound_upper = round_to_n(bound_upper, n=5, kind='ceil')
            x_linear.append( np.linspace(bound_lower, bound_upper, npoints, endpoint=True) )
        dx = []
        for i in range(num_features):
            dx.append( (x_linear[i][-1] - x_linear[i][0]) / (npoints-1) )
        x_mesh = np.meshgrid(*x_linear, indexing=indexing)
        x = np.reshape(x_mesh, (num_features, -1)).T
        #x = inv_transformation(y)
        return x_mesh, x, x_linear, dx


    def fft_density(kde, x, fnc, fnc_det_jacobian):
        return kde.evaluate(fnc(x)) * fnc_det_jacobian(x)
    
    def tvd_integral_fft(density1, density2, dx):
        integrand = 0.5 * np.abs( density1 - density2 ) * np.prod(dx)
        return np.sum(integrand)
    
    npoints = [2**n for n in range(10,14)]
    #npoints = [2**n for n in range(10,11)]
    tvd_fft = np.zeros(len(npoints))
    start = time.time()
    for i, n in enumerate(npoints):
        x_mesh, x, x_linear, dx = get_x_grid(samples=[rvs_cnorm_S, samples_S], npoints=n)
        density = fft_density(kde=kde_fft, x=x, fnc=transformation, fnc_det_jacobian=det_jacobian)
        density_cnorm = fft_density(kde=kde_fft_cnorm, x=x, fnc=transformation, fnc_det_jacobian=det_jacobian)
        tvd_fft[i] = tvd_integral_fft(density1=density, density2=density_cnorm, dx=dx)
    end = time.time()
    print("Calculate tvd_fft integral took", end-start)
    print(f"tvd_fft = {tvd_fft}")

    # Test independence of random variables
    from sklearn.feature_selection import mutual_info_regression
    print()
    print("Independence tests:")
    for i in range(stride):
        print(f"i={i}: {mutual_info_regression( samples_S, samples_S[:,i]) }")
    print()
            
#    # TODO: 
#    # calculate tvd as a function of num_ensembles and scale to -> inf
#    # calculate tvd for various N's, and scale to -> inf
#    # do a independence test of random variables (Hilbert-Schmidt Independence Criterion (HSIC)? Or mutual information? Distance correlation/distance covariance?)
#    # Theoretical best rate for KDE convergence is O(n^(-4 / (4+d))), n is number of samples, d is dimension
#    #    in Combinatorial Methods in Density Estimation Ch 5, Ch 9, Ch 11, Ch 14. Ch 5 has something about
#    #    invariance of TVD under monotone transformations?
    
    if stride == 2:
        import seaborn as sns
        import pandas as pd
        #df = pd.DataFrame(list(zip(samples_S[:,0], samples_S[:,1])), columns=['s1', 's2'])
        #df_cnorm = pd.DataFrame(list(zip(rvs_cnorm_S[:,0], rvs_cnorm_S[:,1])), columns=['s1_cnorm', 's2_cnorm'])
        x = transformation(samples_S)
        x_cnorm = transformation(rvs_cnorm_S)
        df = pd.DataFrame(list(zip(x[:,0], x[:,1])), columns=['s1', 's2'])
        df_cnorm = pd.DataFrame(list(zip(x_cnorm[:,0], x_cnorm[:,1])), columns=['s1_cnorm', 's2_cnorm'])


        fig = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

        xmin = min(np.min(x[:,0]), np.min(x_cnorm[:,0]))
        xmax = max(np.max(x[:,0]), np.max(x_cnorm[:,0]))
        ymin = min(np.min(x[:,1]), np.min(x_cnorm[:,1]))
        ymax = max(np.max(x[:,1]), np.max(x_cnorm[:,1]))

        g0 = sns.jointplot(data=df, x='s1', y='s2', xlim=[xmin,xmax], ylim=[ymin,ymax])
        g1 = sns.jointplot(data=df, x='s1', y='s2', xlim=[xmin,xmax], ylim=[ymin,ymax], kind='kde', fill=True, cbar=True)
        #sns.kdeplot(data=df, x='s1', y='s2', fill=True)
        
        g2 = sns.jointplot(data=df_cnorm, x='s1_cnorm', y='s2_cnorm', xlim=[xmin,xmax], ylim=[ymin,ymax])
        g3 = sns.jointplot(data=df_cnorm, x='s1_cnorm', y='s2_cnorm', xlim=[xmin,xmax], ylim=[ymin,ymax], kind='kde', fill=True, cbar=True)
        #sns.kdeplot(data=df_cnorm, x='s1_cnorm', y='s2_cnorm', fill=True)

        mg0 = SeabornFig2Grid(g0, fig, gs[0])
        mg1 = SeabornFig2Grid(g1, fig, gs[1])
        mg2 = SeabornFig2Grid(g2, fig, gs[2])
        mg3 = SeabornFig2Grid(g3, fig, gs[3])

        #gs.tight_layout(fig)


        fig2 = plt.figure()
        
        #n = 1000
        #x_mesh, x, x_linear, dx = get_x_grid(samples=[rvs_cnorm_S, samples_S], npoints=n)
        #y = transformation(x)
        #y_mesh = transformation(x_mesh)
        #y_linear = inv_transformation(x_linear)
        
        n = 1000
        y_mesh, y, y_linear, dy = get_x_grid(samples=transformation([rvs_cnorm_S, samples_S]), npoints=n)
        x = inv_transformation(y)
        x_mesh = inv_transformation(y_mesh)
        x_linear = inv_transformation(y_linear)
        
        density = fft_density(kde=kde_fft, x=x, fnc=transformation, fnc_det_jacobian=det_jacobian)
        density_cnorm = fft_density(kde=kde_fft_cnorm, x=x, fnc=transformation, fnc_det_jacobian=det_jacobian)
        density = np.reshape(density, (n, n))
        density_cnorm = np.reshape(density_cnorm, (n, n))
        diff = density - density_cnorm
        
        levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        cs = plt.contourf(*x_mesh, density, levels=levels)
        cbar = fig.colorbar(cs)
        plt.show()

        from scipy.integrate import simps
        check_norm_x = simps([simps(zz_x, x_linear[0]) for zz_x in density], x_linear[1])  # should be 1
        check_norm_y = simps([simps(zz_x, y_linear[0]) for zz_x in density / det_jacobian(x).reshape(density.shape)], y_linear[1])  # should be 1
        tvd_test = 0.5 * simps([simps(zz_x, x_linear[0]) for zz_x in np.abs(diff)], x_linear[1])
        print(f"norm_x = {check_norm_x}")
        print(f"norm_y = {check_norm_y}")
        print(f"tvd_test = {tvd_test}")
        print()

#        x_mesh, x, x_linear, dx = get_x_grid(samples=[np.array([[1e-12,1e-12],[12,12]])], npoints=n)
#        # singular values should be sorted, so only care about about diagonal
#        idx = (x[:,0] >= x[:,1]).T.reshape(x_mesh[0].shape)
#        z = wishart_eig_pdf(np.array( x ), n=2).T.reshape(x_mesh[0].shape)
#        z[idx] = None
#        check_norm_x = simps([simps(zz_x, x_linear[0]) for zz_x in np.where(np.isnan(z), 0, z)], x_linear[1]) 
#        print(f"norm_wishart = {check_norm_x}")
#        cs = plt.contourf(*x_mesh, z)
#        cbar = fig.colorbar(cs)
#        plt.show()

#        from scipy.stats import gaussian_kde
#        levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
#        kde = gaussian_kde(np.log(samples_S.T))
#        xmin = np.log(samples_S[:,0]).min()
#        ymin = np.log(samples_S[:,1]).min()
#        ymax = np.log(samples_S[:,1]).max()
#        xmax = np.log(samples_S[:,0]).max()
#        X, Y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
#        positions = np.vstack([X.ravel(), Y.ravel()])
#        Z = np.reshape(kde(positions).T, X.shape)
#        print(simps([simps(zz_x, X[:,0]) for zz_x in Z], Y[0]))
#        cs = plt.contourf(np.exp(X), np.exp(Y), Z / np.reshape(np.prod(np.exp(positions), axis=0).T, X.shape), levels=levels )
#        cbar = fig.colorbar(cs)
#        plt.show()
#
#        kde = gaussian_kde(np.log(samples_S.T))
#        xmin = samples_S[:,0].min()
#        ymin = samples_S[:,1].min()
#        ymax = samples_S[:,1].max()
#        xmax = samples_S[:,0].max()
#        X, Y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
#        positions = np.vstack([X.ravel(), Y.ravel()])
#        Z = np.reshape(kde(np.log(positions)).T, X.shape) / np.reshape(np.prod(positions, axis=0).T, X.shape)
#        print(simps([simps(zz_x, X[:,0]) for zz_x in Z], Y[0]))
#        cs = plt.contourf(X, Y, Z, levels=levels )
#        cbar = fig.colorbar(cs)
#        plt.show()