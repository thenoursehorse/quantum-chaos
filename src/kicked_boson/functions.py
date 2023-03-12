import numpy as np
import scipy
from scipy.stats import ks_1samp as cdftest
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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


def frame_potential(d, c2, c4):
    return (d**2 / (d**2 - 1)) * (d**2 * c4 - 2*c2 + 1)
    #return (1 / (d**2 - 1)) * ( (c2*d**2)**2 + d**2 - 2*(c2*d**2) )

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

    
def plot_ratios(r, r_err=None, folder='', show=True, save=False):
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
    
def plot_form_factor(time, c2, c4, c2_err=None, c4_err=None, folder='', show=True, save=False):
    data = {'time': time, 
            'c2': c2,
            'c4': c4}
        
    sns.set_theme()
    sns.set_style("white")
    #sns.set_style("ticks")
    #sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(3.386,2*2.54)
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
        fig.savefig(f'{folder}/form_factors.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)
    
def plot_frame_potential(time, F, F_err=None, folder='', show=True, save=False):
    data = {'time': time, 
            'F': F,
            'harr': np.ones(len(time)),
            'asymptote': 3*np.ones(len(time))}
        
    sns.set_theme()
    sns.set_style("white")
    #sns.set_style("ticks")
    #sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context("paper")
    fig, ax = plt.subplots()
    fig.set_size_inches(3.386,2.54)
    sns.despine()
    
    sns.lineplot(data=data, x='time', y='F', ax=ax)
    sns.lineplot(data=data, x='time', y='harr', ax=ax)
    sns.lineplot(data=data, x='time', y='asymptote', ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ \mathcal{F}_{\mathcal{E}_H}^{(1)} (t) $')
        
    if save:
        fig.savefig(f'{folder}/frame_potential.pdf', bbox_inches="tight")
    if show:
        plt.show()
    sns.reset_orig()
    plt.close(fig)
    
def plot_loschmidt_echo(time, le1, le2, le1_err=None, le2_err=None, folder='', show=True, save=False):
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
    fig.set_size_inches(3.386,2*2.54)
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