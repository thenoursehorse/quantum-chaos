import atexit

import numpy as np
import seaborn as sns
#from scipy.optimize import curve_fit

#import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        sns.set_context("paper",font_scale=0.9) # default is 10pt, we want 9pt
        #sns.set_context("paper", rc={'font-size':9})
        if use_tics:
            sns.set_style("ticks")
            sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

        self._set_figure(N_figs=self._N_figs)

        atexit.register(self.__close)

    #def __del__(self):
    #    self.finish()
    
    def __close(self):
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
        self._axis[ax_idx].fill_between(x, y_lower, y_upper, alpha=alpha, color=color, label=label, linewidth=0.0)
    
    def fill_betweenx(self, y, x_lower, x_upper, ax_idx=None, color='black', label=None, alpha=0.25):
        if ax_idx is None:
            ax_idx = 0
        self._axis[ax_idx].fill_betweenx(y, x_lower, x_upper, alpha=alpha, color=color, label=label, linewidth=0.0)
    
    def histplot(self, x, ax_idx=None, color=None, label=None, alpha=0.25, bins='auto'):
        if ax_idx is None:
            ax_idx = 0
        data = {'x': x.flatten()}
        sns.histplot(data=data, x='x', stat='density', bins=bins, alpha=alpha, label=label, ax=self._axis[ax_idx], color=color) 
    
    def colormesh(self, x, y, z, ax_idx=None, vmin=0, vmax=1, shading='auto', rasterized=True, colorbar=None, cmap=None):
        sns.despine(ax=self._axis[ax_idx], top=True, right=True, left=True, bottom=True)
        if ax_idx is None:
            ax_idx = 0
        plot_image = self._axis[ax_idx].pcolormesh(x, y, z.T, vmin=vmin, vmax=vmax, shading=shading, rasterized=rasterized, cmap=cmap)
        if colorbar is not None:
            if isinstance(colorbar, bool):
                cb = self._fig.colorbar(plot_image, orientation='vertical', pad=0.01, ax=self._axis[ax_idx])
            else:
                cb = self._fig.colorbar(plot_image, orientation='horizontal', pad=0.01, ax=self._axis[ax_idx])
            cb.outline.set_visible(False)
            cb.ax.tick_params(width=0)
            return plot_image, cb
        else:
            return plot_image
    
    @property
    def fig(self):
        return self._fig
    
    @property
    def axis(self):
        return self._axis

class SeabornFig2Grid():
    """
    Moves seaborn figures to a new figure.
    Copy and pasted exactly from https://stackoverflow.com/a/47664533 under 
    the CC BY-SA 3.0 license. See https://stackoverflow.com/help/licensing.
    """

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())