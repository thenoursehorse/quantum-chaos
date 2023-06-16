#import numpy as np
import seaborn as sns
#from scipy.optimize import curve_fit

#import matplotlib as mpl
import matplotlib.pyplot as plt
#from pylab import cm

class Plotter(object):
    def __init__(self, N_figs=1,
                       column_width=3.4, 
                       full_column_width=7.0,
                       save=False,
                       show=True,
                       save_root='figs/',
                       save_filename = 'my_fig.pdf', 
                       use_tics=False,
                       scale_width=1,
                       scale_fullwidth=1,
                       ):
        self._N_figs = N_figs
        self._column_width = scale_width*column_width
        self._full_column_width = scale_fullwidth*full_column_width
        self._save = save
        self._show = show
        self._save_root = save_root
        self._save_filename = save_filename
        
        #plt.rcParams.update({
        #    #'text.usetex' : True
        #    #'mathtext.fontset' : 'stix',
        #    'mathtext.fontset' : 'cm',
        #    'font.family' : 'STIXGeneral',
        #    'lines.markersize' : 4,
        #})
    
        sns.set_theme()
        sns.set_style("white")
        sns.set_context("paper")
        if use_tics:
            sns.set_style("ticks")
            sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

        self._set_figure(N_figs=self._N_figs)

    def __del__(self):
        self.finish()
    
    def finish(self):
        if self._save:
            self._fig.savefig(f'{self._save_root}/{self._save_filename}')
        if self._show:
            plt.show()
        sns.reset_orig()
        plt.close(self._fig)

    def _set_figure(self, N_figs=None, gap_height_ratio=0.2):
        if N_figs is None:
            N_figs = self._N_figs
        num_gaps = N_figs - 1                           # for vertical stacking
        
        canvas_width = self._column_width * 0.7          # fixed size area we draw on
        gap_width = canvas_width * 0.075                 # For the right part of figure
        x_start_width = canvas_width * 0.25              # for label and axis
        fig_width = x_start_width + canvas_width + gap_width
        
        x_start = x_start_width / fig_width
        x_span = canvas_width / fig_width
        
        canvas_height = canvas_width * 0.7              # fixed size area we draw on
        title_height = canvas_height * 0.3              # for super title
        gap_height = canvas_height * gap_height_ratio   # for axis (and maybe label)
        y_start_height = canvas_height * 0.3            # for label and axis
        fig_height = N_figs*canvas_height + num_gaps*gap_height + y_start_height + title_height
        
        title_span = title_height / fig_height
        y_start = y_start_height / fig_height
        gap_span = gap_height / fig_height
        y_span = canvas_height / fig_height

        fig = plt.figure(figsize=(self._column_width, fig_height))
        # add_axes(left, bottom, width, height) as fractions of fig height and width
        axis = []
        for i in range(N_figs):
            y = y_start + i*y_span + i*gap_span
            axis.append( fig.add_axes([x_start, y, x_span, y_span]) )
        axis = list(reversed(axis))

        for ax in axis:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        
        self._fig = fig
        self._axis = axis
    
    def set_log(self, axis='x', ax_idx=None):
        if ax_idx is None:
            for ax in self._axis:
                if axis == 'x':
                    ax.set_xscale('log')
                elif axis == 'y':
                    ax.set_yscale('log')
                else:
                    raise ValueError('Unrecognized axis !')
        else:
            if axis == 'x':
                self._axis[ax_idx].set_xscale('log')
            elif axis == 'y':
                self._axis[ax_idx].set_yscale('log')
            else:
                raise ValueError('Unrecognized axis !')
    
    def set_loglog(self, ax_idx=None):
        self.set_log(axis='x', ax_idx=ax_idx)
        self.set_log(axis='y', ax_idx=ax_idx)
    
    def line(self, x, y, ax_idx=None, linestyle=None, marker=None, markersize=None, color=None, label=None, alpha=None):
        if ax_idx is None:
            ax_idx = 0
        self._axis[ax_idx].plot(x, y, linestyle=linestyle, marker=marker, markersize=markersize, color=color, label=label, alpha=alpha)
    
    def linepoints(self, x, y, ax_idx=None, linestyle='-', marker='o', markersize=2, color=None, label=None, alpha=None):
        self.line(x=x, y=y, ax_idx=ax_idx, linestyle=linestyle, marker=marker, markersize=markersize, color=color, label=label, alpha=alpha)

    def scatter(self, x, y, ax_idx=None, marker='o', markersize=5, color=None, label=None, alpha=0.25):
        if ax_idx is None:
            ax_idx = 0
        self._axis[ax_idx].scatter(x=x, y=y, marker=marker, s=markersize, color=color, label=label, alpha=alpha)
    
    def fill_betweeny(self, x, y_lower, y_upper, ax_idx=None, color='black', label=None, alpha=0.1):
        if ax_idx is None:
            ax_idx = 0
        self._axis[ax_idx].fill_between(x, y_lower, y_upper, alpha=alpha, color=color, label=label)
    
    def fill_betweenx(self, y, x_lower, x_upper, ax_idx=None, color='black', label=None, alpha=0.25):
        if ax_idx is None:
            ax_idx = 0
        self._axis[ax_idx].fill_betweenx(y, x_lower, x_upper, alpha=alpha, color=color, label=label)
    
    def histplot(self, x, ax_idx=None, color=None, label=None, alpha=0.25):
        if ax_idx is None:
            ax_idx = 0
        data = {'x': x.flatten()}
        sns.histplot(data=data, x='x', stat='density', bins='auto', alpha=alpha, label=label, ax=self._axis[ax_idx], color=color) 
    
    def colormesh(self, x, y, z, ax_idx=None, vmin=0, vmax=1, shading='auto'):
        sns.despine(ax=self._axis[ax_idx], top=True, right=True, left=True, bottom=True)
        if ax_idx is None:
            ax_idx = 0
        return self._axis[ax_idx].pcolormesh(x, y, z.T, vmin=vmin, vmax=vmax, shading=shading)
    
    @property
    def fig(self):
        return self._fig
    
    @property
    def axis(self):
        return self._axis