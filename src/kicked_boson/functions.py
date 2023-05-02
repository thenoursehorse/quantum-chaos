import numpy as np
import scipy
from scipy.stats import ks_1samp as cdftest
import seaborn as sns
import matplotlib.pyplot as plt
#import pandas as pd

# COLORS
colors = {'poiss': sns.color_palette()[0],
          'goe': sns.color_palette()[1],
          'gue': sns.color_palette()[2],
          'gse': sns.color_palette()[3],
          'model': 'black'}

def golden_ratio():
    return (1 + 5 ** 0.5) / 2

# PDF from 10.1103/PhysRevLett.110.084101
def ratio_poiss(x):
    return (2.0 / ((1 + x)**2))

def ratio_goe(x):
    return (27.0 / 4.0) * (x + x**2) / ( 1 + x + x**2 )**(5.0/2.0)

def ratio_gue(x):
    return (81.0 / 2.0) * (np.sqrt(3.0) / np.pi) * (x + x**2)**2 / ( 1 + x + x**2 )**4

def ratio_gse(x):
    return (724.0 / 2.0) * (np.sqrt(3.0) / np.pi) * (x + x**2)**4 / ( 1 + x + x**2 )**7
        
def spacing_poiss(x):
    return np.exp(-x)

def spacing_goe(x):
    return (np.pi / 2.0) * x * np.exp(-np.pi * x**2 / 4.0)

def spacing_gue(x):
    return (32.0 / np.pi**2) * x**2 * np.exp(-4.0 * x**2 / np.pi)

def vectors_goe(x, d):
    return np.sqrt(d/(2*np.pi)) * np.exp(-d*x**2/2)

# CDF integrated from PDF with lower bound 0 using mathematica
def ratio_cpoiss(x):
    return - ( 2 / (1 + x ) ) + 2

def ratio_cgoe(x):
    return ((1 + 2*x) * (-2+x+x**2)) / (2 * (1+x+x**2)**(3.0/2.0)) + 1
        
def ratio_cgue(x): 
    return (np.sqrt(3) * ((3*x * (-2 - 5*x + 5*x**3 + 2*x**4)) / (1 + x + x**2)**3 + \
           4*np.sqrt(3.0) * np.arctan((1 + 2*x)/np.sqrt(3.0))))/(2 * np.pi) - 1

def ratio_cgse(x):
    return (181 * ((3*x * (-4 - 22*x - 72*x**2 - 159*x**3 - 168*x**4 + 168*x**6 + \
           159*x**7 + 72*x**8 + 22*x**9 + 4*x**10)) / (1 + x + x**2)**6 + \
           8*np.sqrt(3) * np.arctan((1 + 2*x) / np.sqrt(3)))) / (243*np.sqrt(3) * np.pi) \
           - (724/729)
        
def spacing_cpoiss(x):
    return - np.exp(-x) + 1

def spacing_cgoe(x):
    return - np.exp(-np.pi * x**2 / 4) + 1

def spacing_cgue(x):
    return (-4 * np.exp(-4*x**2/np.pi) * x + np.pi * scipy.special.erf(2*x/np.sqrt(np.pi)) ) / np.pi

# FIXME add spacing gue

# Characterisitc functions
def g_poiss(t):
    return 1j / (1j+t)

def g_gue(t):
    return (1j/8) * ( 4*t - np.exp(-np.pi*t**2/16) * (np.pi*t**2 - 8) * (scipy.special.erfi(np.sqrt(np.pi)*t/4) - 1) )

def error_interval(A, axis=-1, num_samples=666, confidence=0.90, stat=np.mean, method='percentile'):
    '''
    Something
    '''
    if method == 'percentile':
        return np.percentile(A,[100*(1-confidence)/2,100*(1-(1-confidence)/2)], axis=axis)
    
    elif method == 'bootstrap_empirical':
        # See https://elizavetalebedeva.com/bootstrapping-confidence-intervals-the-basics/
        # https://math.mit.edu/~dav/05.dir/class24-prep-a.pdf
        A_avg = stat(A, axis=axis)
        rng = np.random.default_rng()
        values = np.array([stat(rng.choice(A, size=A.shape[axis], axis=axis, replace=True), axis=axis) for _ in range(num_samples)])
        deltastar = values - A_avg
        CI = np.percentile(deltastar,[100*(1-confidence)/2,100*(1-(1-confidence)/2)], axis=0)
        return [A_avg-CI[0], A_avg-CI[1]]
    
    elif method == 'bootstrap_percentile':
        # See https://towardsdatascience.com/how-to-calculate-confidence-intervals-in-python-a8625a48e62b
        rng = np.random.default_rng()
        values = np.array([stat(rng.choice(A, size=A.shape[axis], axis=axis, replace=True), axis=axis) for _ in range(num_samples)])
        return np.percentile(values,[100*(1-confidence)/2,100*(1-(1-confidence)/2)], axis=0)
    
    elif method == 'normal_CI':
        A_avg = stat(A, axis=axis)
        A_se = np.std(A, axis=axis) / A.shape[axis]
        return [A_avg-1.96*A_se, A_avg+1.96*A_se]
    
    else:
        raise ValueError('Unrecognized error interval !')

def ecdf(x):
    size = x.shape[-1]
    ys = np.arange(1, size+1)/float(size)
    xs = np.sort(x, axis=-1)
    return xs, ys

def integrated_dos(E, energies):
    return np.sum( np.heaviside(E - energies, 0.5) )
    
def chi_distance(xs, kind='ratios'):

    if kind == 'ratios':
        respois = cdftest(xs, ratio_cpoiss)
        resgoe = cdftest(xs, ratio_cgoe)
        resgue = cdftest(xs, ratio_cgue)
        #resgse = cdftest(xs, ratio_cgse)
    elif kind == 'spacing':
        respois = cdftest(xs, spacing_cpoiss)
        resgoe = cdftest(xs, spacing_cgoe)
        resgue = cdftest(xs, spacing_cgue)
        #resgse = cdftest(xs, spacing_cgse)
        
    return respois.pvalue, resgoe.pvalue, resgue.pvalue #, resgse.pvalue

def unfold_energies(energies, polytype='chebyshev', deg=48, folder='./', plot=False, show=True, save=False):
    energies_unfolded = np.empty(shape=energies.shape)
    num_ensembles = energies.shape[0]
        
    idos = np.empty(shape=energies.shape)
    for m in range(num_ensembles):
        for n,E in enumerate(energies[m]):
            idos[m][n] = integrated_dos(E, energies[m])
                
    if plot:
        sns.set_theme()
        sns.set_style("white")
        sns.set_style("ticks")
        sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
        sns.set_context("paper")
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(3.386,6.25)
        sns.despine()
            
    if polytype == 'chebyshev':
        polyfit = np.polynomial.Chebyshev.fit
    elif polytype == 'polynomial':
        polyfit = np.polynomial.Polynomial.fit
    elif polytype == 'hermite':
        polyfit = np.polynomial.Hermite.fit
    elif polytype == 'legendre':
        polyfit = np.polynomial.Legendre.fit
    elif polytype == 'aguerre':
        polyfit = np.polynomial.Laguerre.fit
    else:
        raise ValueError('Unrecognized polyfit polynomial type !')

    for m in range(num_ensembles):
        p = polyfit(energies[m], idos[m], deg=deg)
            
        energies_unfolded[m] = np.sort(p(energies[m]))
        x = np.linspace(energies_unfolded[m][0],
                        energies_unfolded[m][-1],
                        len(energies_unfolded[m]),
                        endpoint=True)
            
        idos_unfolded = np.empty(len(x))
        for n,E in enumerate(x):
            idos_unfolded[n] = integrated_dos(E, energies_unfolded[m])
                
        if plot:
            if m == 0:
                ax[0].plot(energies[m],idos[m], color='black', alpha=0.25, label='Data')
                ax[0].plot(energies[m], p(energies[m]), color='red', alpha=0.25, label='Fit')
            else:
                ax[0].plot(energies[m],idos[m], alpha=0.25, color='black')
                ax[0].plot(energies[m], p(energies[m]), alpha=0.25, color='red')
            ax[1].plot(x, idos_unfolded, alpha=0.25, color='black')
        
    if plot:
        ax[0].set_xlabel(r'$E_n$')
        ax[0].set_ylabel(r'$N(E_n)$')
        ax[0].legend()
        ax[1].set_xlabel(r'$x_n$')
        ax[1].set_ylabel(r'$N(x_n)$')
        if save:
            fig.savefig(f'{folder}/unfolded_energies.pdf', bbox_inches="tight")
        if show:
            plt.show()
        sns.reset_orig()
        plt.close(fig)
    
    return energies_unfolded
    
def fractal_dimension(q, vecs, sum_axis=1):
    # NOTE for |vecs|^2 = 1/d, fractal dimension = 1 for all q (easy to derive)
    sum_coeff = np.abs(vecs)**(2*q)
    d = vecs.shape[-1]
    # the sum is over the coefficients of each eigenvector
    if q == 1: # reduces to shannon entropy
        return (1/np.log(d)) * scipy.stats.entropy(sum_coeff, axis=sum_axis)
        #return -1/np.log(self._d) * np.sum(sum_coeff * np.log(sum_coeff), axis=1)
    else:
        return -1/(q-1) * (1/np.log(d)) * np.log( np.sum(sum_coeff, axis=sum_axis) )
    
def survival_probability_amplitude(psi, time, eigs, vecs, init=False):
    exp_e = np.exp(-1j * time[:, None, None] * eigs)
    if init:
        psi_n = np.abs(vecs @ psi)**2
        return np.sum(psi_n * exp_e, axis=-1)
    else:
        right = (vecs @ psi) * exp_e
        return np.einsum('mka,tma->tmk', np.transpose(vecs.conj(), (0,2,1)), right)
    
def spectral_functions(e, d, order, t):
    '''
    t is a numpy array of time values
    '''

    if order == 2:
        #c = np.sum( np.exp(1j*(e[...,None] - e)*t) ).real / self._d**2
            
        #c = np.sum( np.exp(1j*e*t) )
        c = np.sum( np.exp(1j * t[:, None, None] * e) , axis=-1)
        c *= c.conj()
        c = c.real
        c /= d**2
        
    elif order == 3:
        #val = 2*e[...,None] - e
        #val = val[...,None] - e
        #c = np.sum(np.exp(1j*val*t)) / self._d**3

        #c_l = np.sum( np.exp(2*1j*e*t) )
        #c_r = np.sum( np.exp(-1j*e*t) )
        c_l = np.sum( np.exp(2*1j * t[:, None, None] * e) , axis=-1)
        c_r = np.sum( np.exp(-1j * t[:, None, None] * e) , axis=-1)
        c = c_l * c_r * c_r
        c /= d**3
        
    elif order == 4:
        # This will abslutely wreck my RAM
        #val = e[...,None] + e
        #val = val[...,None] - e
        #val = val[...,None] - e
        #c = np.sum(np.exp(1j*val2*t)).real / self._d**4

        #c = np.sum( np.exp(1j*e*t) )
        c = np.sum( np.exp(1j * t[:, None, None] * e) , axis=-1)
        c *= c.conj()
        c = c.real
        c /= d**2
        c = c**2
        c /= d**2

    elif order == 41: # same as order=3 but with a minus sign
        c_l = np.sum( np.exp(-2*1j * t[:, None, None] * e) , axis=-1)
        c_r = np.sum( np.exp(1j * t[:, None, None] * e) , axis=-1)
        c = c_r * c_r
        c /= d**2
        c *= (c_l / d)
        
    elif order == 42:
        c = np.sum( np.exp(2*1j * t[:, None, None] * e) , axis=-1)
        c *= c.conj()
        c = c.real
        c /= d**2

    return c
    
def frame_potential(d, c2, c4, c41=None, c42=None, k=1):
    # NOTE from https://doi.org/10.1088/1742-5468/acb52d
    # A popular measure of information mixing is the k -design state that cannot be distinguished from
    # the Haar random state when considering averages of polynomials of degree not higher than k.
    if k == 1:
        return (d**2 / (d**2 - 1)) * (d**2 * c4 - 2*c2 + 1)
    elif k == 2:
        #c2 = c2*d**2 # NOTE c2 *= d**2 will modify c2 in system method, yikes
        #c4 = c4*d**4
        #c41 = c41*d**3
        #c42 = c42*d**2
        # NOTE this ends up having zero imag even though c41 is complex
        #return ( (d**4 - 8*d**2 + 6)*c4**2 +4*d**2*(d**2-9)*c4 + 4*(d**6 - 9*d**4 + 4*d*2 + 24)*c2**2 \
        #        - 8*d**2*(d**4 - 11*d**2 + 18)*c2 + 2*(d**4 - 7*d**2 + 12)*c41**2 - 4*d**2*(d**2 - 9)*c42 \
        #            + (d**4 - 8*d**2 + 6)*c42**2 - 8*(d**4 - 8*d**2 + 6)*c2*c4 - 4*d*(d**2 -4)*c4*c41 \
        #                + 16*d*(d**2 - 4)*c2*c41 - 8*(d**2 + 6)*c2*c42 + 2*(d**2 + 6)*c4*c42 \
        #                    -4*d*(d**2 - 4)*c41*c42 + 2*d**4*(d**4 - 12*d**2 + 27) ) \
        #                        / ((d-3)*(d-2)*(d-1)*d**2*(d+1)*(d+2)*(d+3) )

        # FIXME 
        #C = (d-3)*(d-2)*(d-1)*(d+1)*(d+2)*(d+3)
        A = (d**4 - 8*d**2 + 6)*c4**2*d**6 +4*d**2*(d**2-9)*c4*d**2 + 4*(d**6 - 9*d**4 + 4*d*2 + 24)*c2**2*d**2 \
                - 8*d**2*(d**4 - 11*d**2 + 18)*c2 + 2*(d**4 - 7*d**2 + 12)*np.abs(c41)**2*d**4 - 4*d**2*(d**2 - 9)*c42 \
                    + (d**4 - 8*d**2 + 6)*c42**2*d**2 - 8*(d**4 - 8*d**2 + 6)*c2*c4*d**4 - 4*d*(d**2 -4)*c4*c41*d**5 \
                        + 16*d*(d**2 - 4)*c2*c41*d**3 - 8*(d**2 + 6)*c2*c42*d**2 + 2*(d**2 + 6)*c4*c42*d**4 \
                            -4*d*(d**2 - 4)*c41*c42*d**3
        A /= (d-3)
        A /= (d-2)
        A /= (d-1)
        A /= (d+1)
        A /= (d+2)
        A /= (d+3)

        B = 2*d**2*(d**4 - 12*d**2 + 27) / ((d-3)*(d-2)*(d-1)*(d+1)*(d+2)*(d+3) )
        A += B

        return A
                            
    else:
        raise ValueError('Unrecognized order !')

def frame_potential2(unitary_fidelity, num_ensembles=None):
    if num_ensembles is None:
        num_ensembles = unitary_fidelity.shape[-1]

    F1 = np.sum(unitary_fidelity, axis=(1,2)) / num_ensembles**2
    F2 = np.sum(unitary_fidelity**2, axis=(1,2)) / num_ensembles**2

    # For the lower bound estimate
    #F1 = np.mean(unitary_fidelity, axis=1)
    #F2 = np.mean(unitary_fidelity**2, axis=1)
    return F1, F2

def loschmidt_echo(d, c2=None, c4=None, kind='2nd'):
    # FIXME calculate explicitly for some operators
    if kind == '2nd':
        # NOTE only valid for averaging over Pauli operators
        return c4 + 1.0 / d**2
    elif kind == '1st':
        # NOTE it isn't clear to me when this is valid
        return (d*c2 + 1) / (d + 1)
    else:
        raise ValueError('Unrecognized kind !')

def otoc(d, c2=None, c4=None, A=None, B=None, kind='4-point'):
    # FIXME calculate explicitly for some operators
    if kind == '4-point':
        # NOTE only valid for A and B non-overlapping Pauli operators on qubits
        return c4 - 1.0 / d**2
    elif kind == '2-point':
        # NOTE only valid for A being a pure state |Psi>
        # FIXME figure out properly
        return (c2 + 1) / (d + 1)
    elif kind == 'operators':
        return None
    else:
        raise ValueError('Unrecognized kind !')
    
def plot_eigenenergies(energies, N, folder='./', show=True, save=False):
    sns.set_theme()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots()
    fig.set_size_inches(3.386,2.54)
    sns.despine()

    num_ensembles = energies.shape[0]
    
    for m in range(num_ensembles):
        ax.plot(energies[m]/N, '.')
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$E_n$')
    if save:
        fig.savefig(f'{folder}/eigenenergies.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)

def plot_ratios(r, folder='./', show=True, save=False):
    x = np.linspace(0,1)
    xs, ys = ecdf(r)
    xs_avg = np.mean(xs, axis=0)
    xs_I = error_interval(xs, axis=0)

    data_flat = {'r': r.flatten()}
    data = {'x': x,
            'poiss': ratio_poiss(x),
            'goe': ratio_goe(x),
            'gue': ratio_gue(x),
            'gse': ratio_gse(x),
            'cpoiss': ratio_cpoiss(x),
            'cgoe': ratio_cgoe(x),
            'cgue': ratio_cgue(x),
            'cgse': ratio_cgse(x),
            }
                
    sns.set_theme()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,2*2.54)
    sns.despine()
    
    sns.histplot(data=data_flat, 
                 x='r',
                 stat='density',
                 bins='auto',
                 alpha=0.25,
                 label='Model',
                 ax=ax[0],
                 color=colors['model'])
    #sns.kdeplot(data=data,
    #            x='r',
    #            label='Kicked rotor',
    #            cut=0,
    #            bw_adjust=2,
    #            ax=ax[0])
    sns.lineplot(data=data, x='x', y='poiss', label='Poisson', ax=ax[0], color=colors['poiss'])
    sns.lineplot(data=data, x='x', y='goe', label='GOE', ax=ax[0], color=colors['goe'])
    sns.lineplot(data=data, x='x', y='gue', label='GUE', ax=ax[0], color=colors['gue'])
    sns.lineplot(data=data, x='x', y='gse', label='GSE', ax=ax[0], color=colors['gse'])
    ax[0].set_xlabel(r'$r$')
    ax[0].set_ylabel(r'$P(r)$')
    ax[0].set_xlim(xmin=0, xmax=1)
    ax[0].set_ylim(ymin=0, ymax=2)

    ax[1].plot(xs_avg, ys, label='Model', color=colors['model'])
    ax[1].fill_betweenx(ys, xs_I[0], xs_I[1], color=colors['model'], alpha=0.25)
    #sns.ecdfplot(data=data_single, x='r', color='black', alpha=0.25, label='Kicked rotor', ax=ax[1])
    sns.lineplot(data=data, x='x', y='cpoiss', label='Poisson', ax=ax[1], color=colors['poiss'])
    sns.lineplot(data=data, x='x', y='cgoe', label='GOE', ax=ax[1], color=colors['goe'])
    sns.lineplot(data=data, x='x', y='cgue', label='GUE', ax=ax[1], color=colors['gue'])
    sns.lineplot(data=data, x='x', y='cgse', label='GSE', ax=ax[1], color=colors['gse'])
    ax[1].set_xlabel(r'$r$')
    ax[1].set_ylabel(r'$F(r)$')
    ax[1].set_xlim(xmin=0, xmax=1)
    ax[1].set_ylim(ymin=0, ymax=1)

    if save:
        fig.savefig(f'{folder}/ratio.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)

def plot_spacings(s, folder='./', show=True, save=False):
    x = np.linspace(0,5)
    xs, ys = ecdf(s)
    xs_avg = np.mean(xs, axis=0)
    xs_I = error_interval(xs, axis=0)

    data_flat = {'s': s.flatten()}
    data = {'x': x,
            'poiss': spacing_poiss(x),
            'goe': spacing_goe(x),
            'gue': spacing_gue(x),
            'cpoiss': spacing_cpoiss(x),
            'cgoe': spacing_cgoe(x),
            'cgue': spacing_cgue(x),
            }
                
    sns.set_theme()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,2*2.54)
    sns.despine()
    
    sns.histplot(data=data_flat, 
                 x='s',
                 stat='density',
                 bins='auto',
                 alpha=0.25,
                 label='Model',
                 ax=ax[0],
                 color=colors['model'])
    #sns.kdeplot(data=data,
    #            x='r',
    #            label='Kicked rotor',
    #            cut=0,
    #            bw_adjust=2,
    #            ax=ax[0])
    sns.lineplot(data=data, x='x', y='poiss', label='Poisson', ax=ax[0], color=colors['poiss'])
    sns.lineplot(data=data, x='x', y='goe', label='GOE', ax=ax[0], color=colors['goe'])
    sns.lineplot(data=data, x='x', y='gue', label='GUE', ax=ax[0], color=colors['gue'])
    #sns.lineplot(data=data, x='x', y='gse', label='GSE', ax=ax[0], color=colors['gse'])
    ax[0].set_xlabel(r'$s$')
    ax[0].set_ylabel(r'$P(s)$')
    ax[0].set_xlim(xmin=0, xmax=5)
    ax[0].set_ylim(ymin=0, ymax=1)

    ax[1].plot(xs_avg, ys, label='Model', color=colors['model'])
    ax[1].fill_betweenx(ys, xs_I[0], xs_I[1], color=colors['model'], alpha=0.25)
    sns.lineplot(data=data, x='x', y='cpoiss', label='Poisson', ax=ax[1], color=colors['poiss'])
    sns.lineplot(data=data, x='x', y='cgoe', label='GOE', ax=ax[1], color=colors['goe'])
    sns.lineplot(data=data, x='x', y='cgue', label='GUE', ax=ax[1], color=colors['gue'])
    #sns.lineplot(data=data, x='x', y='cgse', label='GSE', ax=ax[1], color=colors['gse'])
    ax[1].set_xlabel(r'$s$')
    ax[1].set_ylabel(r'$F(s)$')
    ax[1].set_xlim(xmin=0, xmax=5)
    ax[1].set_ylim(ymin=0, ymax=1)

    if save:
        fig.savefig(f'{folder}/spacing.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)

def plot_vector_coefficients(c, d, folder='./', show=True, save=False):
    # FIXME this doesn't really work, I use fractal dimension instead
    x = np.linspace(-0.1,0.1)
    xs, ys = ecdf(c)
    xs_avg = np.mean(xs, axis=0)
    xs_I = error_interval(xs, axis=0)

    data_flat = {'c': c.flatten().real}
    data = {'x': x,
            'goe': vectors_goe(x, d),
            }

    sns.set_theme()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,2*2.54)
    sns.despine()
    
    sns.histplot(data=data_flat, 
                 x='c',
                 stat='density',
                 bins='auto',
                 color='black',
                 alpha=0.25,
                 label='Kicked rotor',
                 ax=ax[0])
    sns.lineplot(data=data, x='x', y='goe', label='GOE', ax=ax[0])
    ax[0].set_xlabel(r'$c$')
    ax[0].set_ylabel(r'$P(c)$')
    #ax[0].set_xlim(xmin=0, xmax=1)
    #ax[0].set_ylim(ymin=0, ymax=2)

    #sns.ecdfplot(data=data_single, x='r', color='black', alpha=0.25, label='Kicked rotor', ax=ax[1])
    ax[1].plot(xs_avg, ys, label='Kicked rotor', color='k')
    ax[1].fill_betweenx(ys, xs_I[0], xs_I[1], color='k', alpha=0.25)
    #sns.lineplot(data=data, x='x', y='cgoe', label='GOE', ax=ax[1])
    ax[1].set_xlabel(r'$c$')
    ax[1].set_ylabel(r'$F(c)$')
    #ax[1].set_xlim(xmin=0, xmax=1)
    #ax[1].set_ylim(ymin=0, ymax=1)

    if save:
        fig.savefig(f'{folder}/vectors.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)

def plot_fractal_dimension(energies, dq, dq_avg, dq_I, q_arr, q=2, num_ensembles=1, folder='./', show=True, save=False):
    xmin = np.inf
    xmax = -np.inf
    for i in range(len(energies)):
        xmin_est = np.min(energies[i])
        xmax_est = np.max(energies[i])
        if xmin_est < xmin:
            xmin = xmin_est
        if xmax_est > xmax:
            xmax = xmax_est
    x = np.array([xmin, xmax])

    energies_goe = energies[1]
    energies_gue = energies[2]
    energies = energies[0]
    dq_goe = dq[1]
    dq_gue = dq[2]
    dq = dq[0]

    y = np.mean(dq) * np.ones(shape=x.shape)
    y_goe = np.mean(dq_goe) * np.ones(shape=x.shape)
    y_gue = np.mean(dq_gue) * np.ones(shape=x.shape)

    dq_goe_avg = dq_avg[1]
    dq_gue_avg = dq_avg[2]
    dq_avg = dq_avg[0]
    dq_goe_I = dq_I[1]
    dq_gue_I = dq_I[2]
    dq_I = dq_I[0]
 
    sns.set_theme()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,2*2.54)
    sns.despine()
    
    ax[0].plot(x, y_goe, color=colors['goe'], label='GOE')
    ax[0].plot(x, y_gue, color=colors['gue'], label='GUE')
    ax[0].plot(x, y, color=colors['model'], label='Model')

    ymin = np.inf
    ymax = 0
    for m in range(num_ensembles):
        ax[0].scatter(x=energies_goe[m], y=dq_goe[m], s=5, color=colors['goe'], alpha=0.25)
        ax[0].scatter(x=energies_gue[m], y=dq_gue[m], s=5, color=colors['gue'], alpha=0.25)
        ax[0].scatter(x=energies[m], y=dq[m], s=5, color=colors['model'], alpha=0.25)

        ymin_est = min(np.min(dq_goe[m]), np.min(dq_gue[m]), np.min(dq[m]))
        if ymin_est < ymin:
            ymin = ymin_est
        ymax_est = max(np.max(dq_goe[m]), np.max(dq_gue[m]), np.max(dq[m]))
        if ymax_est > ymax:
            ymax = ymax_est
        
    ax[0].legend()
    ax[0].set_xlabel(r'$E_n/N$')
    ax[0].set_ylabel(f'$D_{q}$')
    ax[0].set_xlim(xmin=x[0], xmax=x[-1])
    if ymin < 0.05:
        ymin = 0.05
    if ymax > 0.95:
        ymax=0.95
    ax[0].set_ylim(ymin=ymin-0.05, ymax=ymax+0.05)

    ax[1].plot(q_arr, dq_avg, 'o-', markersize=2, color=colors['model'], label='Model')
    ax[1].fill_between(q_arr, dq_I[0], dq_I[1], alpha=0.1, color=colors['model'])
    ax[1].plot(q_arr, dq_goe_avg, 'o-', markersize=2, color=colors['goe'], label='GOE')
    ax[1].fill_between(q_arr, dq_goe_I[0], dq_goe_I[1], alpha=0.1, color=colors['goe'])
    ax[1].plot(q_arr, dq_gue_avg, 'o-', markersize=2, color=colors['gue'], label='GUE')
    ax[1].fill_between(q_arr, dq_gue_I[0], dq_gue_I[1], alpha=0.1, color=colors['gue'])
    ax[1].legend()
    ax[1].set_xlabel(r'$q$')
    ax[1].set_ylabel(f'$D_q$')
    ax[1].set_xlim(xmin=0, xmax=q_arr[-1])

    if save:
        fig.savefig(f'{folder}/fractal_dimension_q{q}.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)

def plot_fractal_dimension_state(time, 
                                dq_state_avg, 
                                q_arr, 
                                q=2, 
                                dq_state_I=None, 
                                folder='./', show=True, save=False):
    sns.set_theme()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,6.25)
    sns.despine()

    q_idx = np.where(np.abs(q_arr - q) < 1e-10)[0][0]
    ax[0].plot(time, dq_state_avg[:,q_idx], color=colors['model'], label='Model')
    if dq_state_I is not None:
        ax[0].fill_between(time, 
                           dq_state_I[0][:,q_idx], 
                           dq_state_I[1][:,q_idx], 
                           alpha=0.1, 
                           color=colors['model'])

    #ax[0].legend()
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(f'$D_{q}(\psi)$')
    ax[0].set_xlim(xmin=time[0], xmax=time[-1])
    ax[0].set_ylim(ymin=0, ymax=1)

    c = ax[1].pcolormesh(time, q_arr, dq_state_avg.T, vmin=0, vmax=1, shading='auto')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(f'$q$')
    #fig.colorbar(c, orientation='horizontal', ax=ax[1])
    
    if save:
        fig.savefig(f'{folder}/fractal_dimension_state_q{q}.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)

def plot_survival_probability(time,
                              w_init_avg,
                              w_ti_avg,
                              w_init_I=None,
                              vmax=0.1,
                              folder='./', show=True, save=False):
    sns.set_theme()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,6.25)
    sns.despine()
    
    ax[0].plot(time, w_init_avg, color=colors['model'], label='Model')
    if w_init_I is not None:
        ax[0].fill_between(time,
                           w_init_I[0],
                           w_init_I[1],
                           alpha=0.1, 
                           color=colors['model'])
    ax[0].set_xscale('log')
    ax[0].set_xlim(xmin=time[0], xmax=time[-1])
    ax[0].set_ylim(ymin=0, ymax=1)
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(f'$W(t)$')
    
    d = w_ti_avg.shape[-1]
    i_arr = np.arange(1, d+0.1)
    c = ax[1].pcolormesh(time, i_arr, w_ti_avg.T, vmin=0, vmax=vmax, shading='auto')
    ax[1].set_xscale('log')
    ax[1].set_xlim(xmin=time[0], xmax=time[-1])
    ax[1].set_ylim(ymin=i_arr[0], ymax=i_arr[-1])
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(f'$i$')

    if save:
        fig.savefig(f'{folder}/fractal_survival_probability.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)

def plot_spectral_functions(time, c2, c4, d, c2_I=None, c4_I=None, folder='./', show=True, save=False):
    data = {'time': time, 
            'asymptote2': (1/d)*np.ones(len(time)),
            'asymptote4': ((2*d-1)/d**3)*np.ones(len(time)),
            }
        
    sns.set_theme()
    sns.set_style("white")
    #sns.set_style("ticks")
    #sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,6.25)
    sns.despine()
        
    sns.lineplot(data=data, x='time', y='asymptote2', ax=ax[0], color='black')
    ax[0].plot(time, c2)
    if c2_I is not None:
        ax[0].fill_between(time, c2_I[0], c2_I[1], alpha=0.1, color='black')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$ \tilde{c}_2(t) $')
        
    sns.lineplot(data=data, x='time', y='asymptote4', ax=ax[1], color='black')
    ax[1].plot(time, c4)
    if c4_I is not None:
        ax[1].fill_between(time, c4_I[0], c4_I[1], alpha=0.1, color='black')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$ \tilde{c}_4(t) $')
        
    if save:
        fig.savefig(f'{folder}/spectral_functions.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)
    
def plot_frame_potential(time, 
                         F1,
                         F2,
                         F1_I,
                         F2_I,
                         time_est=None,
                         F1_est=None,
                         F2_est=None,
                         window=0, 
                         folder='./', show=True, save=False):
    
    if window > 0 and F1_est is not None:
        # Time-bin average sliding window:
        Nt = int(len(time_est) - window)
        F1_avg = np.empty(Nt)
        F2_avg = np.empty(Nt)
        for t in range(Nt):
            F1_avg[t] = np.mean(F1_est[t:t+window])
            F2_avg[t] = np.mean(F2_est[t:t+window])

        time_est = time_est[:Nt]
        F1_est = F1_avg
        F2_est = F2_avg

    data = {'time': time, 
            #'upperbound1': d**2*np.ones(len(time)), # d^(2k)
            #'upperbound2': d**4*np.ones(len(time)),
            'harr1': np.ones(len(time)),
            'harr2': 2*np.ones(len(time)),
            'asymptote1': 3*np.ones(len(time)),
            #'asymptote2': 10*np.ones(len(time)),
            'asymptote2': 24*np.ones(len(time)),
            }
        
    sns.set_theme()
    sns.set_style("white")
    #sns.set_style("ticks")
    #sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,6.25)
    sns.despine()
    
    #sns.lineplot(data=data, x='time', y='upperbound1', ax=ax[0])
    sns.lineplot(data=data, x='time', y='harr1', ax=ax[0], color='black')
    sns.lineplot(data=data, x='time', y='asymptote1', ax=ax[0], color='black')
    ax[0].plot(time, F1)
    if F1_I is not None:
        ax[0].fill_between(time, F1_I[0], F1_I[1], alpha=0.1, color='black')
    if F1_est is not None:
        ax[0].plot(time_est, F1_est)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$ \mathcal{F}_{\mathcal{E}_H}^{(1)} (t) $')
    
    #sns.lineplot(data=data, x='time', y='upperbound2', ax=ax[1])
    sns.lineplot(data=data, x='time', y='harr2', ax=ax[1], color='black')
    sns.lineplot(data=data, x='time', y='asymptote2', ax=ax[1], color='black')
    ax[1].plot(time, F2)
    if F2_I is not None:
        ax[1].fill_between(time, F2_I[0], F2_I[1], alpha=0.1, color='black')
    if F2_est is not None:
        ax[1].plot(time_est, F2_est)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$ \mathcal{F}_{\mathcal{E}_H}^{(2)} (t) $')
        
    if save:
        fig.savefig(f'{folder}/frame_potential.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)
    
def plot_loschmidt_echo(time, le1, le2, d, le1_I=None, le2_I=None, folder='./', show=True, save=False):
    # FIXME add non-isometric twirl operator version to compare
    data = {'time': time, 
            'asymptote2': (3/d**2)*np.ones(len(time)),
            'gue2': (1/d**2)*np.ones(len(time)),
            'asymptote1': (2/d)*np.ones(len(time)),
            'gue1': (1/d)*np.ones(len(time)),
            }
        
    sns.set_theme()
    sns.set_style("white")
    #sns.set_style("ticks")
    #sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,6.25)
    sns.despine()
        
    sns.lineplot(data=data, x='time', y='gue1', ax=ax[0], color='black')
    sns.lineplot(data=data, x='time', y='asymptote1', ax=ax[0], color='black')
    ax[0].plot(time, le1)
    if le1_I is not None:
        ax[0].fill_between(time, le1_I[0], le1_I[1], alpha=0.1, color='black')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$\langle \mathcal{L}_1(t) \rangle_G$')
        
    sns.lineplot(data=data, x='time', y='gue2', ax=ax[1], color='black')
    sns.lineplot(data=data, x='time', y='asymptote2', ax=ax[1], color='black')
    ax[1].plot(time, le2)
    if le2_I is not None:
        ax[1].fill_between(time, le2_I[0], le2_I[1], alpha=0.1, color='black')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$\langle \mathcal{L}_2(t) \rangle_G$')
        
    if save:
        fig.savefig(f'{folder}/echo.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)