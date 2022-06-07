""" Functions to ease plotting with mathplotlib.pyplot, statistcs, pandas and some aditional calculations. """

import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects

# Functions defined:

class CenteredFormatter(matplotlib.ticker.ScalarFormatter):

    """Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "center"."""

    center = 0

    def __call__(self, value, pos=None):
        if value == self.center:
            return ''
        else:
            return matplotlib.ticker.ScalarFormatter.__call__(self,
value, pos)

def splot(x_axis,y_axis, xname=None, yname=None, tname=None, note=None,
          y_max=None, y_min=None, x_max=None, x_min=None, alp=1,
          fig=None,axs=None,coord=None, tl=False, lw=1, ms=1,
          ls='solid', m='o', col='black', y_scale='linear',x_scale='linear'):
    """ Pyplot plottage for any instance. """
    # For simple plottage
    if fig == None:
        fig, axs = plt.subplots(tight_layout=tl)
        if y_axis == None:
            if note != None:
                axs.plot(x_axis,note,alpha=alp,linewidth=lw,markersize=ms)
            elif col != 'black':
                axs.plot(x_axis,color=col,alpha=alp,marker=m,linestyle=ls)
            else:
                axs.plot(x_axis,color=col,alpha=alp)
        else:
            if note != None:
                axs.plot(x_axis,y_axis,note,alpha=alp,linewidth=lw,markersize=ms)
            elif col != None:
                axs.plot(x_axis,y_axis,color=col,alpha=alp,marker=m,linestyle=ls)
            else:
                axs.plot(x_axis,y_axis,color=col,alpha=alp)
        axs.set(xlabel=xname, ylabel=yname, title=tname)
        axs.set_yscale(y_scale)
        axs.set_xscale(x_scale)
        if y_max != None:
            plt.ylim(ymax=y_max)
        if y_min != None:
            plt.ylim(ymin=y_min)
        if x_max != None:
            plt.xlim(xmax=x_max)
        if x_min != None:
            plt.xlim(xmin=x_min)
    
    # For multiplotage
    else:
        if len(coord) == 1:
            if y_axis == None:
                if note != None:
                    axs[coord[0]].plot(x_axis,note,alpha=alp,linewidth=lw,markersize=ms)
                elif col != None:
                    axs[coord[0]].plot(x_axis,color=col,alpha=alp,marker=m,linestyle=ls)
                else:
                    axs[coord[0]].plot(x_axis,color=col,alpha=alp)
            else:
                if note != None:
                    axs[coord[0]].plot(x_axis,y_axis,note,alpha=alp,linewidth=lw,markersize=ms)
                elif col != None:
                    axs[coord[0]].plot(x_axis,y_axis,color=col,alpha=alp,marker=m,linestyle=ls)
                else:
                    axs[coord[0]].plot(x_axis,y_axis,color=col,alpha=alp)
            axs[coord[0]].set(xlabel=xname, ylabel=yname, title=tname)
            axs[coord[0]].set_yscale(y_scale)
            axs[coord[0]].set_xscale(x_scale)
            if y_max != None:
                plt.ylim(ymax=y_max)
            if y_min != None:
                plt.ylim(ymin=y_min)
            if x_max != None:
                plt.xlim(xmax=x_max)
            if x_min != None:
                plt.xlim(xmin=x_min)
        else:
            if y_axis == None:
                if note != None:
                    axs[coord[0]][coord[1]].plot(x_axis,note,alpha=alp,linewidth=lw,markersize=ms)
                elif col != None:
                    axs[coord[0]][coord[1]].plot(x_axis,color=col,alpha=alp,marker=m,linestyle=ls)
                else:
                    axs[coord[0]][coord[1]].plot(x_axis,color=col,alpha=alp)
            else:
                if note != None:
                    axs[coord[0]][coord[1]].plot(x_axis,y_axis,note,alpha=alp,linewidth=lw,markersize=ms)
                elif col != None:
                    axs[coord[0]][coord[1]].plot(x_axis,y_axis,color=col,alpha=alp,marker=m,linestyle=ls)
                else:
                    axs[coord[0]][coord[1]].plot(x_axis,y_axis,color=col,alpha=alp)
            axs[coord[0]][coord[1]].set(xlabel=xname, ylabel=yname, title=tname)
            axs[coord[0]][coord[1]].set_yscale(y_scale)
            axs[coord[0]][coord[1]].set_xscale(x_scale)
            if y_max != None:
                plt.ylim(ymax=y_max)
            if y_min != None:
                plt.ylim(ymin=y_min)
            if x_max != None:
                plt.xlim(xmax=x_max)
            if x_min != None:
                plt.xlim(xmin=x_min)
    return fig, axs

def center_spines(ax=None, centerx=0, centery=0):
    """Centers the axis spines at <centerx, centery> on the axis "ax", and
    places arrows at the end of the axis spines."""

    if ax is None:
        ax = plt.gca()

    # Set the axis's spines to be centered at the given point
    # (Setting all 4 spines so that the tick marks go in both directions)

    ax.spines['left'].set_position(('data', centerx))
    ax.spines['bottom'].set_position(('data', centery))
    ax.spines['right'].set_position(('data', centerx - 1))
    ax.spines['top'].set_position(('data', centery - 1))

    # Draw an arrow at the end of the spines
    # Hide the line (but not ticks) for "extra" spines

    for side in ['right', 'top']:
        ax.spines[side].set_color('none')

    # On both the x and y axes...

    for (axis, center) in zip([ax.xaxis, ax.yaxis], [centerx, centery]):

        # Turn on minor and major gridlines and ticks

        axis.set_ticks_position('both')
        axis.grid(True, 'major', ls='solid', lw=0.5, color='gray')
        axis.grid(True, 'minor', ls='solid', lw=0.1, color='gray')
        axis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        # Hide the ticklabels at <centerx, centery>

        formatter = CenteredFormatter()
        formatter.center = center
        axis.set_major_formatter(formatter)

    # Add offset ticklabels at <centerx, centery> using annotation
    # (Should probably make these update when the plot is redrawn...)

    (xlabel, ylabel) = map(formatter.format_data, [centerx, centery])
    ax.annotate(
        '(%s, %s)' % (xlabel, ylabel),
        (centerx, centery),
        xytext=(-4, -4),
        textcoords='offset points',
        ha='right',
        va='top',
)

def normalizer(n,d):
    """ Normalizes a value for another, considering the divisions by zero. Returns "None" if it goes to infinity """
    if d == 0:
        if n == 0:
            r = 0
        elif n != 0: 
            r = None 
    elif d != 0:
        r = (n/d)
    return r
    
def point_show(result_a,result_b,a_cut=None,b_cut=None):
    higher = [];
    k_list = list(result_a.keys())
    if a_cut == None and b_cut == None:
        a_val = list(result_a.values()); b_val = list(result_b.values()) 
        a_cut = st.median(a_val); b_cut = st.median(b_val)
        for key in k_list:
            if result_a[key] >= a_cut:
                if result_b[key] >= b_cut:
                    higher.append(key) 
    else:
        a_cut = float(a_cut); b_cut = float(b_cut)
        for key in k_list:
            if result_a[key] >= a_cut: 
                if result_b[key] >= b_cut: 
                    higher.append(key)
    return higher
    
""" END OF THE CODE """
