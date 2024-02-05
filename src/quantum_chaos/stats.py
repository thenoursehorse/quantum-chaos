
import math
import numpy as np
import scipy
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
    exponent = -int(math.floor(math.log10(abs(x)))) + (n - 1)
    if kind == 'round':
        return math.round(x * 10**exponent) / 10**exponent
    if kind == 'floor':
        return math.floor(x * 10**exponent) / 10**exponent
    if kind == 'ceil':
        return math.ceil(x * 10**exponent) / 10**exponent

def mae(actual, predicted):
    '''
    Mean absolute error
    '''
    return np.mean(np.abs(actual - predicted))
    
def mase(actual, predicted):
    '''
    Mean absolute scaled error
    '''
    forecast_error = np.mean(np.abs(actual - predicted))
    naive_forecast = np.mean(np.abs(np.diff(actual)))
    return forecast_error / naive_forecast

############################
###### Matrix normal #######
############################

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
    A = scipy.linalg.cholesky(np.linalg.inv(Sigma_s), lower=False)
    B = scipy.linalg.cholesky(np.linalg.inv(Sigma_c), lower=True)
    C = - A @ mu @ B
    return A @ X @ B + C

############################
####### MC integrate #######
############################

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

#####################################
##### Probability distributions #####
#####################################

def wishart_eig_pdf(x, n=None):
    '''
    Complex Wishart eigenvalue distribution.

    Parameters
    ----------
    x : numpy.ndarray
        An m by n matrix in the Wishart distribution.
    n : int, optional
        Degrees of freedom. Default is the size of first dimension of x.

    Returns
    -------
    float
        Probability density for the matrix x.
    '''
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
 
def normal_pdf(x):
    '''
    Standard matrix normal probability density.
    '''
    n = x.shape[-1]
    return (1/np.pi**(n*2)) * np.prod( np.exp(-x**2), axis=-1)


###########################################
############### KDE #######################
###########################################

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
    
def fft_density(kde, x, fnc, fnc_det_jacobian, shape=None):
    pdf = kde.evaluate(fnc(x)) * fnc_det_jacobian(x)
    if shape is None:
        return pdf
    else:
        return np.reshape(pdf, shape)
    
def tvd_integral_fft(density1, density2, dx=None, x_linear=None):
    if dx is not None:
        return np.sum( 0.5 * np.abs( density1 - density2 ) * np.prod(dx) )
    if x_linear is not None:
        if len(x_linear) != 2:
            raise ValueError('simpson only implemented for 2D meshes')
        diff = density1 - density2
        return 0.5 * scipy.integrate.simps([scipy.integrate.simps(zz_x, x_linear[0]) for zz_x in np.abs(diff)], x_linear[1])