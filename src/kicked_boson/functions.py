import numpy as np
import scipy
from scipy.stats import ks_1samp as cdftest
import seaborn as sns
import matplotlib.pyplot as plt
#import pandas as pd

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

# CDF integrated from PDF with lower bound 0
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
     
def ecdf(x):
    size = x.shape[-1]
    ys = np.arange(1, size+1)/float(size)
    xs = np.sort(x, axis=-1)
    return xs, ys
    
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
    
def spectral_functions(e, d, order, t):
    '''
    t is a numpy array of time values
    '''

    if order == 2:
        #c = np.sum( np.exp(1j*(e[...,None] - e)*t) ).real / self._d**2
            
        #c = np.sum( np.exp(1j*e*t) )
        c = np.sum( np.exp(1j * t[:, None, None] * e) , axis=-1)
        c *= c.conj()
        c /= d**2
        c = c.real
        
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
        c = c.real**2
        c /= d**4

    elif order == 41: # same as order=3 but with a minus sign
        c_l = np.sum( np.exp(-2*1j * t[:, None, None] * e) , axis=-1)
        c_r = np.sum( np.exp(1j * t[:, None, None] * e) , axis=-1)
        c = c_r * c_r * c_l
        # FIXME scale here
        
    elif order == 42:
        c = np.sum( np.exp(2*1j * t[:, None, None] * e) , axis=-1)
        c *= c.conj()
        # FIXME scale here

    return c
    
def frame_potential(d, c2, c4, c41=None, c42=None, k=1):
    if k == 1:
        return (d**2 / (d**2 - 1)) * (d**2 * c4 - 2*c2 + 1)
    elif k == 2:
        c2 *= d**2
        c4 *= d**4
        # NOTE this ends up having zero imag even though c41 is complex
        return ( (d**4 - 8*d**2 + 6)*c4**2 +4*d**2*(d**2-9)*c4 + 4*(d**6 - 9*d**4 + 4*d*2 + 24)*c2**2 \
                - 8*d**2*(d**4 - 11*d**2 + 18)*c2 + 2*(d**4 - 7*d**2 + 12)*c41**2 - 4*d**2*(d**2 - 9)*c42 \
                    + (d**4 - 8*d**2 + 6)*c42**2 - 8*(d**4 - 8*d**2 + 6)*c2*c4 - 4*d*(d**2 -4)*c4*c41 \
                        + 16*d*(d**2 - 4)*c2*c41 - 8*(d**2 + 6)*c2*c42 + 2*(d**2 + 6)*c4*c42 \
                            -4*d*(d**2 - 4)*c41*c42 + 2*d**4*(d**4 - 12*d**2 +27) ) \
                                / ((d-3)*(d-2)*(d-1)*d**2*(d+1)*(d+2)*(d+3) )
    else:
        raise ValueError('Unrecognized order !')

def loschmidt_echo(d, c2=None, c4=None, kind='2nd'):
    if kind == '2nd':
        return c4 + 1.0 / d**2
    elif kind == '1st':
        return (c2 + 1) / (d + 1)
    else:
        raise ValueError('Unrecognized kind !')

def otoc(d, c2=None, c4=None, kind='4-point'):
    if kind == '4-point':
        return c4 - 1.0 / d**2
    elif kind == '2-point':
        # FIXME figure out properly
        return (c2 + 1) / (d + 1)
    else:
        raise ValueError('Unrecognized kind !')
    
def plot_eigenenergies(energies, folder='./', show=True, save=False):
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
        ax.plot(energies[m], '.')
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$E_n$')
    if save:
        fig.savefig(f'{self._folder}/eigenenergies.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)

def plot_ratios(r, folder='./', show=True, save=False):
    x = np.linspace(0,1)
    xs, ys = ecdf(r)
    xs_avg = np.mean(xs, axis=0)
    xs_err = np.std(xs, axis=0)

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
                 color='black',
                 alpha=0.25,
                 label='Kicked rotor',
                 ax=ax[0])
    #sns.kdeplot(data=data,
    #            x='r',
    #            label='Kicked rotor',
    #            cut=0,
    #            bw_adjust=2,
    #            ax=ax[0])
    sns.lineplot(data=data, x='x', y='poiss', label='Poisson', ax=ax[0])
    sns.lineplot(data=data, x='x', y='goe', label='GOE', ax=ax[0])
    sns.lineplot(data=data, x='x', y='gue', label='GUE', ax=ax[0])
    sns.lineplot(data=data, x='x', y='gse', label='GSE', ax=ax[0])
    ax[0].set_xlabel(r'$r$')
    ax[0].set_ylabel(r'$P(r)$')
    ax[0].set_xlim(xmin=0, xmax=1)
    ax[0].set_ylim(ymin=0, ymax=2)

    #sns.ecdfplot(data=data_single, x='r', color='black', alpha=0.25, label='Kicked rotor', ax=ax[1])
    ax[1].plot(xs_avg, ys, label='Kicked rotor', color='k')
    ax[1].fill_betweenx(ys, xs_avg-xs_err, xs_avg+xs_err, color='k', alpha=0.25)
    sns.lineplot(data=data, x='x', y='cpoiss', label='Poisson', ax=ax[1])
    sns.lineplot(data=data, x='x', y='cgoe', label='GOE', ax=ax[1])
    sns.lineplot(data=data, x='x', y='cgue', label='GUE', ax=ax[1])
    sns.lineplot(data=data, x='x', y='cgse', label='GSE', ax=ax[1])
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
    xs_err = np.std(xs, axis=0)

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
                 color='black',
                 alpha=0.25,
                 label='Kicked rotor',
                 ax=ax[0])
    #sns.kdeplot(data=data,
    #            x='r',
    #            label='Kicked rotor',
    #            cut=0,
    #            bw_adjust=2,
    #            ax=ax[0])
    sns.lineplot(data=data, x='x', y='poiss', label='Poisson', ax=ax[0])
    sns.lineplot(data=data, x='x', y='goe', label='GOE', ax=ax[0])
    sns.lineplot(data=data, x='x', y='gue', label='GUE', ax=ax[0])
    #sns.lineplot(data=data, x='x', y='gse', label='GSE', ax=ax[0])
    ax[0].set_xlabel(r'$s$')
    ax[0].set_ylabel(r'$P(s)$')
    ax[0].set_xlim(xmin=0, xmax=5)
    ax[0].set_ylim(ymin=0, ymax=1)

    ax[1].plot(xs_avg, ys, label='Kicked rotor', color='k')
    ax[1].fill_betweenx(ys, xs_avg-xs_err, xs_avg+xs_err, color='k', alpha=0.25)
    sns.lineplot(data=data, x='x', y='cpoiss', label='Poisson', ax=ax[1])
    sns.lineplot(data=data, x='x', y='cgoe', label='GOE', ax=ax[1])
    sns.lineplot(data=data, x='x', y='cgue', label='GUE', ax=ax[1])
    #sns.lineplot(data=data, x='x', y='cgse', label='GSE', ax=ax[1])
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
    
def plot_spectral_functions(time, c2, c4, c2_err=None, c4_err=None, folder='./', show=True, save=False):
    data = {'time': time, 
            'c2': c2,
            'c4': c4}
        
    sns.set_theme()
    sns.set_style("white")
    #sns.set_style("ticks")
    #sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,6.25)
    sns.despine()
        
    sns.lineplot(data=data, x='time', y='c2', ax=ax[0])
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$ \tilde{c}_2(t) $')
        
    sns.lineplot(data=data, x='time', y='c4', ax=ax[1])
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
    
def plot_frame_potential(time, F1, F2, window=0, folder='./', show=True, save=False):
    
    if window > 0:
        # Time-bin average sliding window:
        Nt = int(len(time) - window)
        F1_avg = np.empty(Nt)
        F2_avg = np.empty(Nt)
        for t in range(Nt):
            F1_avg[t] = np.mean(F1[t:t+window])
            F2_avg[t] = np.mean(F2[t:t+window])

        time = time[:Nt]
        F1 = F1_avg
        F2 = F2_avg

    data = {'time': time, 
            'F1': F1,
            'F2': F2,
            'harr1': np.ones(len(time)),
            'harr2': 2*np.ones(len(time)),
            'asymptote1': 3*np.ones(len(time)),
            'asymptote2': 10*np.ones(len(time)),
            }
        
    sns.set_theme()
    sns.set_style("white")
    #sns.set_style("ticks")
    #sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,6.25)
    sns.despine()
    
    sns.lineplot(data=data, x='time', y='F1', ax=ax[0])
    sns.lineplot(data=data, x='time', y='harr1', ax=ax[0])
    sns.lineplot(data=data, x='time', y='asymptote1', ax=ax[0])
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$ \mathcal{F}_{\mathcal{E}_H}^{(1)} (t) $')
    
    sns.lineplot(data=data, x='time', y='F2', ax=ax[1])
    sns.lineplot(data=data, x='time', y='harr2', ax=ax[1])
    sns.lineplot(data=data, x='time', y='asymptote2', ax=ax[1])
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
    
def plot_loschmidt_echo(time, le1, le2, le1_err=None, le2_err=None, folder='./', show=True, save=False):
    # FIXME add non-isometric twirl operator version to compare
    data = {'time': time, 
            'le1': le1,
            'le2': le2}
        
    sns.set_theme()
    sns.set_style("white")
    #sns.set_style("ticks")
    #sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,6.25)
    sns.despine()
        
    sns.lineplot(data=data, x='time', y='le1', ax=ax[0])
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$\langle \mathcal{L}_1(t) \rangle$')
        
    sns.lineplot(data=data, x='time', y='le2', ax=ax[1])
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$\langle \mathcal{L}_2(t) \rangle$')
        
    if save:
        fig.savefig(f'{folder}/echo.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)

def integrated_dos(E, energies):
    return np.sum( np.heaviside(E - energies, 0.5) )

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

def frame_potential2(Ut):
    #tr(G+e^{-iHt}G G+e^(iHt)G)

    # Taken from the last appendix in 10.1007/JHEP11(2017)048
    # ignore coincident as they scale as 1/num_ensembles
    # and you only need 2 ensembles and time-average to get
    # a good representation of the large ensemble average
    
    Nt = Ut.shape[0]
    num_ensembles = Ut.shape[1]

    F = np.zeros((Nt,num_ensembles,num_ensembles))
    
    # This is not really faster, and scales worse for larger # of ensembles
    #for i in range(num_ensembles):
    #    for j in range(num_ensembles):
    #        if i != j: # Ignore coincident
    #            tmp = np.trace(Ut[:,i,...] @ np.transpose( Ut[:,j,...].conj(), (0,2,1)), axis1=1, axis2=2)
    #            tmp *= tmp.conj()
    #            F[:,i,j] = tmp.real
    
    for t in range(Nt):
        for i in range(num_ensembles):
            for j in range(num_ensembles):
                if i != j: # Ignore coincident
                    tmp = np.trace( Ut[t,i,...] @ Ut[t,j,...].conj().T )
                    tmp *= tmp.conj()
                    F[t,i,j] = tmp.real

    F1 = np.sum(F, axis=(1,2)) / num_ensembles**2
    F2 = np.sum(F**3, axis=(1,2)) / num_ensembles**2

    return F1, F2