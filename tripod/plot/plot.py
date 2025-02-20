import tripod.constants as c
from tripod.simulation import Simulation
from tripod.utils.read_data import read_data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from scipy.constants import golden
from scipy.interpolate import interp1d
import os
import warnings


def panel(data,
          filename="data", extension="hdf5",
          ia=0, ir=0, it=0,
          show_limits=False, show_St1=False
          ):
    """Simple plotting script for data files or simulation objects.

    Parameters
    ----------
    data : ``dustpy.Simulation`` or string
        Either instance of ``dustpy.Simulation`` or path to data directory to be plotted
    filename : string, optional, default : "data"
    extension : string, optional, default : "hdf5"
        Plotting script is looking for files with pattern ``<data>/<filename>*.<extension>``
    ia : int, optional, default : 0
        Number of size bin along which density distribution is plotted
    ir : int, optional, default : 0
        Number of radial grid index along density distribution is plotted
    it : int, optional, default : 0
        Index of snapshot to be plotted
    show_limits : boolean, optional, default : True
        If True growth limits are plotted
    show_St1 : boolean, optional, default : True
        If True St=1 line is plotted"""

    from tripod import __version__

    data = read_data(data, filename=filename, extension=extension)

    # Helper quantities
    Nt, Nr, Na = data.dust.Sigma_recon.shape

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, Nt-1)
    it = int(it)
    ia = np.maximum(0, ia)
    ia = np.minimum(ia, Na-1)
    ia = int(ia)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, Nr-1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.dust.sigma_recon.max()))
    sg_max = np.ceil(np.log10(data.gas.Sigma.max()))
    Mmax = np.ceil(np.log10(data.gas.M.max()/c.M_sun)) + 1
    levels = np.linspace(sd_max-6, sd_max, 7)

    width = 4.8
    height = width/golden
    fig = plt.figure(figsize=(3*width, 2*height), dpi=150)
    ax00 = fig.add_subplot(231)
    ax01 = fig.add_subplot(232)
    ax02 = fig.add_subplot(233)
    ax10 = fig.add_subplot(234)
    ax11 = fig.add_subplot(235)
    ax11r = ax11.twinx()

    # Density distribution
    plt00 = ax00.contourf(data.grid.r[it, ...]/c.au,
                          data.grid.a_recon[it],
                          np.log10(data.dust.sigma_recon[it, ...].T),
                          levels=levels,
                          cmap="magma",
                          extend="both"
                          )
    if show_St1:
        ax00.contour(data.grid.r[it, ...]/c.au,
                     data.grid.a_recon[it, ...],
                     data.dust.St_recon[it, ...].T,
                     levels=[1.],
                     colors="white",
                     linewidths=2
                     )
    if show_limits:
        ax00.contour(data.grid.r[it, ...]/c.au,
                     data.grid.a_recon[it, ...],
                     (data.dust.St_recon - data.dust.StDr[..., None])[it, ...].T,
                     levels=[0.],
                     colors="C2",
                     linewidths=1
                     )
        ax00.contour(data.grid.r[it, ...]/c.au,
                     data.grid.a_recon[it, ...],
                     (data.dust.St_recon - data.dust.StFr[..., None])[it, ...].T,
                     levels=[0.],
                     colors="C0",
                     linewidths=1
                     )

    ax00.axhline(data.grid.a_recon[it, ia], color="#AAAAAA", lw=1, ls="--")
    ax00.axvline(data.grid.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")

    cbar00 = plt.colorbar(plt00, ax=ax00)
    cbar00.ax.set_ylabel(r"$\sigma_\mathrm{d}$ [g/cm²]")
    cbar00ticklabels = []
    for i in levels:
        cbar00ticklabels.append("$10^{{{:d}}}$".format(int(i)))
    cbar00.ax.set_yticklabels(cbar00ticklabels)
    ax00.set_xlim(data.grid.r[it, 0]/c.au, data.grid.r[it, -1]/c.au)
    ax00.set_xscale("log")
    ax00.set_yscale("log")
    ax00.set_xlabel("Distance from star [AU]")
    ax00.set_ylabel("Particle size [cm]")
    sd_max_loc = np.ceil(np.log10(max(data.dust.Sigma[it,ir,0],data.dust.Sigma[it,ir,1], data.dust.sigma_recon[it,ir,:].max())))
    ax01.loglog(data.grid.a_recon[it, ...],
                data.dust.sigma_recon[it, ir, :], c="C3")
    ax01.set_xlim(data.grid.a_recon[it, 0], data.grid.a_recon[it, -1])
    ax01.axvline(data.dust.a[it, ir,0])
    ax01.axvline(data.dust.a[it, ir,2])
    ax01.axvline(data.dust.a[it, ir,4])
    a_int = (data.dust.a[it, ir,4] * data.dust.s.min[it,ir] )**0.5
    a_min = data.dust.s.min[it,ir]
    a_max = data.dust.a[it, ir,4]
    y_min = 10.**(sd_max_loc-3.)
    ax01.add_patch(patches.Rectangle((a_min, y_min), a_int - a_min, data.dust.Sigma[it,ir,0], color='green', alpha=0.3))
    ax01.add_patch(patches.Rectangle((a_int, y_min), a_max - a_int, data.dust.Sigma[it,ir,1], color='blue', alpha=0.3))
    ax01.text(0.05,0.95,"q = %.2f"%data.dust.q[it,ir],va="top",ha="left", transform=ax01.transAxes)


    ax01.set_ylim(10.**(sd_max_loc-3.), 10.**sd_max_loc)
    ax01.set_xlabel("Particle size [cm]")
    ax01.set_ylabel(r"$\sigma_\mathrm{d}$ [g/cm²]")

    if Nt < 3:
        ax02.set_xticks([0., 1.])
        ax02.set_yticks([0., 1.])
        ax02.text(0.5,
                  0.5,
                  "Not enough data points.",
                  verticalalignment="center",
                  horizontalalignment="center",
                  size="large")
    else:
        ax02.loglog(data.t/c.year, data.gas.M/c.M_sun, c="C0", label="Gas")
        ax02.loglog(data.t/c.year, data.dust.M /
                    c.M_sun, c="C1", label="Dust")
        ax02.axvline(data.t[it]/c.year, c="#AAAAAA", lw=1, ls="--")
        ax02.set_xlim(data.t[1]/c.year, data.t[-1]/c.year)
        ax02.set_ylim(10.**(Mmax-6.), 10.**Mmax)
        ax02.legend()
    ax02.set_xlabel("Time [yrs]")
    ax02.set_ylabel(r"Mass [$M_\odot$]")

    ax10.loglog(data.grid.r[it, ...]/c.au,
                data.dust.sigma_recon[it, :, ia], c="C3")
    ax10.set_xlim(data.grid.r[it, 0]/c.au, data.grid.r[it, -1]/c.au)
    ax10.set_ylim(10.**(sd_max-6.), 10.**sd_max)
    ax10.set_xlabel("Distance from star [au]")
    ax10.set_ylabel(r"$\sigma_\mathrm{d}$ [g/cm²]")

    ax11.loglog(data.grid.r[it, ...]/c.au,
                data.gas.Sigma[it, ...], label="Gas")
    ax11.loglog(data.grid.r[it, ...]/c.au,
                data.dust.Sigma[it, ...].sum(-1), label="Dust")
    ax11.set_xlim(data.grid.r[it, 0]/c.au, data.grid.r[it, -1]/c.au)
    ax11.set_ylim(10.**(sg_max-6), 10.**sg_max)
    ax11.set_xlabel("Distance from star [AU]")
    ax11.set_ylabel(r"$\Sigma$ [g/cm²]")
    ax11.legend()
    ax11r.loglog(data.grid.r[it, ...]/c.au,
                 data.dust.eps[it, ...], color="C7", lw=1)
    ax11r.set_ylim(1.e-5, 1.e1)
    ax11r.set_ylabel("Dust-to-gas ratio")

    fig.set_layout_engine("tight")

    fig.text(0.99, 0.01, "TriPoD v"+__version__, horizontalalignment="right",
             verticalalignment="bottom")

    plt.show()


def ipanel(data,
           filename="data", extension="hdf5",
           ia=0, ir=0, it=0,
           show_limits=False, show_St1=False):
    """Simple interactive plotting script for data files or simulation objects.

    Parameters
    ----------
    data : ``dustpy.Simulation`` or string
        Either instance of ``dustpy.Simulation`` or path to data directory to be plotted
    filename : string, optional, default : "data"
    extension : string, optional, default : "hdf5"
        Plotting script is looking for files with pattern ``<data>/<filename>*.<extension>``
    ia : int, optional, default : 0
        Number of size bin along which density distribution is plotted
    ir : int, optional, default : 0
        Number of radial grid index along density distribution is plotted
    it : int, optional, default : 0
        Index of snapshot to be plotted
    show_limits : boolean, optional, default : True
        If True growth limits are plotted
    show_St1 : boolean, optional, default : True
        If True St=1 line is plotted"""

    from tripod.plot import __version__

    data = read_data(data, filename=filename, extension=extension)

    # Helper quantities
    Nt, Nr, Na = data.dust.Sigma_recon.shape

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, Nt-1)
    it = int(it)
    ia = np.maximum(0, ia)
    ia = np.minimum(ia, Na-1)
    ia = int(ia)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, Nr-1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.dust.sigma_recon.max()))
    sg_max = np.ceil(np.log10(data.gas.Sigma.max()))
    Mmax = np.ceil(np.log10(data.gas.M.max()/c.M_sun)) + 1
    levels = np.linspace(sd_max-6, sd_max, 7)

    width = 4.8
    height = width/golden
    fig = plt.figure(figsize=(3*width, 2*height), dpi=150)
    ax00 = fig.add_subplot(231)
    ax01 = fig.add_subplot(232)
    ax02 = fig.add_subplot(233)
    ax10 = fig.add_subplot(234)
    ax11 = fig.add_subplot(235)
    ax11r = ax11.twinx()

    global plt00
    global plt00Dr
    global plt00Fr
    global plt00St

    # Density distribution
    plt00 = ax00.contourf(data.grid.r[it, ...]/c.au,
                          data.grid.a_recon[it],
                          np.log10(data.dust.sigma_recon[it, ...].T),
                          levels=levels,
                          cmap="magma",
                          extend="both"
                          )
    if show_St1:
        plt00St = ax00.contour(data.grid.r[it, ...]/c.au,
                               data.grid.a_recon[it, ...],
                               data.dust.St_recon[it, ...].T,
                               levels=[1.],
                               colors="white",
                               linewidths=2
                               )
    if show_limits:
        plt00Dr = ax00.contour(data.grid.r[it, ...]/c.au,
                               data.grid.a_recon[it, ...],
                               (data.dust.St_recon - data.dust.StDr[..., None])[it, ...].T,
                               levels=[0.],
                               colors="C2",
                               linewidths=1
                               )
        plt00Fr = ax00.contour(data.grid.r[it, ...]/c.au,
                               data.grid.a_recon[it, ...],
                               (data.dust.St_recon - data.dust.StFr[..., None])[it, ...].T,
                               levels=[0.],
                               colors="C0",
                               linewidths=1
                               )
    plt00hl = ax00.axhline(
        data.grid.a_recon[it, ia], color="#AAAAAA", lw=1, ls="--")
    plt00vl = ax00.axvline(
        data.grid.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")

    cbar00 = plt.colorbar(plt00, ax=ax00)
    cbar00.ax.set_ylabel(r"$\sigma_\mathrm{d}$ [g/cm²]")
    cbar00ticklabels = []
    for i in levels:
        cbar00ticklabels.append("$10^{{{:d}}}$".format(int(i)))
    cbar00.ax.set_yticklabels(cbar00ticklabels)
    ax00.set_xlim(data.grid.r[it, 0]/c.au, data.grid.r[it, -1]/c.au)
    ax00.set_xscale("log")
    ax00.set_yscale("log")
    ax00.set_xlabel("Distance from star [AU]")
    ax00.set_ylabel("Particle size [cm]")

    plt01 = ax01.loglog(
        data.grid.a_recon[it, ...], data.dust.sigma_recon[it, ir, :], c="C3")
    plt01vl = ax01.axvline(
        data.grid.a_recon[it, ia], color="#AAAAAA", lw=1, ls="--")
    ax01.set_xlim(data.grid.a_recon[it, 0], data.grid.a_recon[it, -1])
    ylim1 = np.ceil(np.log10(np.max(data.dust.sigma_recon[it, ir, :])))
    ylim0 = ylim1 - 6.
    ax01.set_ylim(10.**ylim0, 10.**ylim1)
    ax01.set_xlabel("Particle size [cm]")
    ax01.set_ylabel(r"$\sigma_\mathrm{d}$ [g/cm²]")

    if Nt < 3:
        ax02.set_xticks([0., 1.])
        ax02.set_yticks([0., 1.])
        ax02.text(0.5,
                  0.5,
                  "Not enough data points.",
                  verticalalignment="center",
                  horizontalalignment="center",
                  size="large")
    else:
        ax02.loglog(data.t/c.year, data.gas.M/c.M_sun, c="C0", label="Gas")
        ax02.loglog(data.t/c.year, data.dust.M /
                    c.M_sun, c="C1", label="Dust")
        plt02vl = ax02.axvline(data.t[it]/c.year, c="#AAAAAA", lw=1, ls="--")
        ax02.set_xlim(data.t[1]/c.year, data.t[-1]/c.year)
        ax02.set_ylim(10.**(Mmax-6.), 10.**Mmax)
        ax02.legend()
    ax02.set_xlabel("Time [yrs]")
    ax02.set_ylabel(r"Mass [$M_\odot$]")

    plt10 = ax10.loglog(data.grid.r[it, ...]/c.au,
                        data.dust.sigma_recon[it, :, ia], c="C3")
    plt10vl = ax10.axvline(
        data.grid.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")
    ax10.set_xlim(data.grid.r[it, 0]/c.au, data.grid.r[it, -1]/c.au)
    ylim1 = np.ceil(np.log10(np.max(data.dust.sigma_recon[it, :, ia])))
    ylim0 = ylim1 - 6.
    ax10.set_ylim(10.**ylim0, 10.**ylim1)
    ax10.set_xlabel("Distance from star [au]")
    ax10.set_ylabel(r"$\sigma_\mathrm{d}$ [g/cm²]")

    plt11g = ax11.loglog(data.grid.r[it, ...]/c.au,
                         data.gas.Sigma[it, ...], label="Gas")
    plt11d = ax11.loglog(data.grid.r[it, ...]/c.au,
                         data.dust.Sigma[it, ...].sum(-1), label="Dust")
    plt11vl = ax11.axvline(
        data.grid.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")
    ax11.set_xlim(data.grid.r[it, 0]/c.au, data.grid.r[it, -1]/c.au)
    ax11.set_ylim(10.**(sg_max-6), 10.**sg_max)
    ax11.set_xlabel("Distance from star [AU]")
    ax11.set_ylabel(r"$\Sigma$ [g/cm²]")
    ax11.legend()
    plt11d2g = ax11r.loglog(data.grid.r[it, ...]/c.au,
                            data.dust.eps[it, ...], color="C7", lw=1)
    ax11r.set_ylim(1.e-5, 1.e1)
    ax11r.set_ylabel("Dust-to-gas ratio")

    fig.tight_layout()

    width = ax02.get_position().x1 - ax02.get_position().x0
    fig._widgets = []

    if Nt > 2:
        axSliderTime = plt.axes([ax02.get_position().x0 + 0.15 * width,
                                 0.375,
                                 0.75 * width,
                                 0.02], facecolor="lightgoldenrodyellow")
        sliderTime = Slider(axSliderTime, "Time", 0,
                            int(Nt-1), valinit=it, valfmt="%i")
        axSliderTime.set_title("t = {:9.3e} yr".format(data.t[it]/c.year))
        fig._widgets += [sliderTime]

    axSliderSize = plt.axes([ax02.get_position().x0 + 0.15 * width,
                             0.25,
                             0.75 * width,
                             0.02], facecolor="lightgoldenrodyellow")
    sliderSize = Slider(axSliderSize, "Mass", 0,
                        int(Na-1), valinit=ia, valfmt="%i")
    axSliderSize.set_title("a = {:9.3e} cm".format(data.grid.a_recon[it, ia]))
    fig._widgets += [sliderSize]

    axSliderDist = plt.axes([ax02.get_position().x0 + 0.15 * width,
                             0.125,
                             0.75 * width,
                             0.02], facecolor="lightgoldenrodyellow")
    sliderDist = Slider(axSliderDist, "Distance", 0,
                        int(Nr-1), valinit=ir, valfmt="%i")
    axSliderDist.set_title("r = {:9.3e} AU".format(data.grid.r[it, ir]/c.au))
    fig._widgets += [sliderDist]

    def update(val):

        global plt00
        global plt00Dr
        global plt00Fr
        global plt00St

        it = 0
        if Nt > 2:
            it = int(np.floor(sliderTime.val))
            axSliderTime.set_title("t = {:9.3e} yr".format(data.t[it]/c.year))
        ia = int(np.floor(sliderSize.val))
        axSliderSize.set_title(
            "a = {:9.3e} cm".format(data.grid.a_recon[it, ia]))
        ir = int(np.floor(sliderDist.val))
        axSliderDist.set_title(
            "r = {:9.3e} AU".format(data.grid.r[it, ir]/c.au))

        for row in plt00.collections:
            row.remove()
        plt00 = ax00.contourf(data.grid.r[it, ...]/c.au,
                              data.grid.a_recon[it],
                              np.log10(data.dust.sigma_recon[it, ...].T),
                              levels=np.linspace(sd_max-6, sd_max, 7),
                              cmap="magma",
                              extend="both"
                              )
        if show_St1:
            for row in plt00St.collections:
                row.remove()
            plt00St = ax00.contour(data.grid.r[it, ...]/c.au,
                                   data.grid.a_recon[it, ...],
                                   data.dust.St[it, ...].T,
                                   levels=[1.],
                                   colors="white",
                                   linewidths=2
                                   )
        if show_limits:
            for row in plt00Dr.collections:
                row.remove()
            plt00Dr = ax00.contour(data.grid.r[it, ...]/c.au,
                                   data.grid.a_recon[it, ...],
                                   (data.dust.St - data.dust.StDr[..., None])[it, ...].T,
                                   levels=[0.],
                                   colors="C2",
                                   linewidths=1
                                   )
            for row in plt00Fr.collections:
                row.remove()
            plt00Fr = ax00.contour(data.grid.r[it, ...]/c.au,
                                   data.grid.a_recon[it, ...],
                                   (data.dust.St - data.dust.StFr[..., None])[it, ...].T,
                                   levels=[0.],
                                   colors="C0",
                                   linewidths=1
                                   )
        plt00vl.set_xdata([data.grid.r[it, ir]/c.au, data.grid.r[it, ir]/c.au])
        plt00hl.set_ydata([data.grid.a_recon[it, ia],
                          data.grid.a_recon[it, ia]])

        plt01[0].set_xdata(data.grid.a_recon[it, ...])
        plt01[0].set_ydata(data.dust.sigma_recon[it, ir, :])
        ax01.set_xlim(data.grid.a_recon[it, 0], data.grid.a_recon[it, -1])
        ylim1 = np.ceil(np.log10(np.max(data.dust.sigma_recon[it, ir, :])))
        ylim0 = ylim1 - 6.
        ax01.set_ylim(10.**ylim0, 10.**ylim1)
        plt01vl.set_xdata([data.grid.a_recon[it, ia],
                          data.grid.a_recon[it, ia]])
        plt01vl.set_ydata([0., 1.e100])

        if Nt > 2:
            plt02vl.set_xdata([data.t[it]/c.year, data.t[it]/c.year])
        plt10vl.set_xdata([data.grid.r[it, ir]/c.au, data.grid.r[it, ir]/c.au])
        plt11vl.set_xdata([data.grid.r[it, ir]/c.au, data.grid.r[it, ir]/c.au])

        plt10[0].set_xdata(data.grid.r[it, ...]/c.au)
        plt10[0].set_ydata(data.dust.sigma_recon[it, :, ia])
        ax10.set_xlim(data.grid.r[it, 0]/c.au, data.grid.r[it, -1]/c.au)
        ylim1 = np.ceil(np.log10(np.max(data.dust.sigma_recon[it, :, ia])))
        ylim0 = ylim1 - 6.
        ax10.set_ylim(10.**ylim0, 10.**ylim1)
        plt10vl.set_xdata([data.grid.r[it, ir]/c.au, data.grid.r[it, ir]/c.au])
        plt10vl.set_ydata([0., 1.e100])

        plt11g[0].set_xdata(data.grid.r[it, ...]/c.au)
        plt11g[0].set_ydata(data.gas.Sigma[it, ...])
        plt11d[0].set_xdata(data.grid.r[it, ...]/c.au)
        plt11d[0].set_ydata(data.dust.Sigma[it, ...].sum(-1))
        plt11vl.set_xdata([data.grid.r[it, ir]/c.au, data.grid.r[it, ir]])
        plt11vl.set_ydata([0., 1.e100])
        plt11d2g[0].set_xdata(data.grid.r[it, ...]/c.au)
        plt11d2g[0].set_ydata(data.dust.eps[it, ...])

    if Nt > 2:
        sliderTime.on_changed(update)
    sliderSize.on_changed(update)
    sliderDist.on_changed(update)

    fig.text(0.99, 0.01, "TriPoD v"+__version__, horizontalalignment="right",
             verticalalignment="bottom")

    plt.show()
