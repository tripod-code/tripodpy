import twopoppy.constants as c
from twopoppy.simulation import Simulation

from types import SimpleNamespace
import numpy as np
from dustpy.std import dust_f as dp_dust_f
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d
from simframe.io.writers import hdf5writer
import os


def panel(data, filename="data", extension="hdf5", im=0, ir=0, it=0, show_limits=True, show_St1=True):
    """Simple plotting script for data files or simulation objects.

    Parameters
    ----------
    data : ``twopoppy.Simulation`` or string
        Either instance of ``twopoppy.Simulation`` or path to data directory to be plotted
    filename : string, optional, default : "data"
    extension : string, optional, default : "hdf5"
        Plotting script is looking for files with pattern ``<data>/<filename>*.<extension>``
    im : int, optional, default : 0
        Number of mass bin along which density distribution is plotted
    ir : int, optional, default : 0
        Number of radial grid index along density distribution is plotted
    it : int, optional, default : 0
        Index of snapshot to be plotted
    show_limits : boolean, optional, default : True
        If True growth limits are plotted
    show_St1 : boolean, optional, default : True
        If True St=1 line is plotted"""

    from twopoppy.plot import __version__

    data = _readdata(data, filename=filename, extension=extension)

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, data.Nt-1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data.Nmi[it, ...]-1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...]-1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.sigmaDusti.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max()/c.M_sun)) + 1
    levels_num = 7
    levels = np.linspace(sd_max-(levels_num-1), sd_max, levels_num)

    width = 3.5
    fig = plt.figure(figsize=(3.*width, 2.*width/1.618), dpi=150)
    ax00 = fig.add_subplot(231)
    ax01 = fig.add_subplot(232)
    ax02 = fig.add_subplot(233)
    ax10 = fig.add_subplot(234)
    ax11 = fig.add_subplot(235)
    ax11r = ax11.twinx()

    # Density distribution
    plt00 = ax00.contourf(data.r[it, ...]/c.au,
                          data.mic[it, ...],
                          np.log10(data.sigmaDusti[it, ...].T),
                          levels=levels,
                          cmap="magma",
                          extend="both"
                          )
    if show_St1:
        ax00.contour(data.r[it, ...]/c.au,
                     data.mic[it, ...],
                     data.Sti[it, ...].T,
                     levels=[1.],
                     colors="white",
                     linewidths=2
                     )
    if show_limits:
        ax00.contour(data.r[it, ...]/c.au,
                     data.mic[it, ...],
                     (data.Sti - data.StDr[..., None])[it, ...].T,
                     levels=[0.],
                     colors="C2",
                     linewidths=1
                     )
        ax00.contour(data.r[it, ...]/c.au,
                     data.mic[it, ...],
                     (data.Sti - data.StFr[..., None])[it, ...].T,
                     levels=[0.],
                     colors="C0",
                     linewidths=1
                     )

    ax00.axhline(data.mic[it, im], color="#AAAAAA", lw=1, ls="--")
    ax00.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")

    cbar00 = plt.colorbar(plt00, ax=ax00)
    cbar00.ax.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
    cbar00ticklabels = []
    for i in levels:
        cbar00ticklabels.append("$10^{{{:d}}}$".format(int(i)))
    cbar00.ax.set_yticklabels(cbar00ticklabels)
    ax00.set_xscale("log")
    ax00.set_yscale("log")
    ax00.set_xlabel("Distance from star [AU]")
    ax00.set_ylabel("Particle mass [g]")

    ax01.loglog(data.mic[it, ...], data.sigmaDusti[it, ir, :], c="C3")
    ax01.set_xlim(data.mic[it, 0], data.mic[it, -1])
    ax01.set_ylim(10.**(sd_max-6.), 10.**sd_max)
    ax01.set_xlabel("Particle mass [g]")
    ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

    if data.Nt < 3:
        ax02.set_xticks([0., 1.])
        ax02.set_yticks([0., 1.])
        ax02.text(0.5,
                  0.5,
                  "Not enough data points.",
                  verticalalignment="center",
                  horizontalalignment="center",
                  size="large")
    else:
        ax02.loglog(data.t/c.year, data.Mgas/c.M_sun, c="C0", label="Gas")
        ax02.loglog(data.t/c.year, data.Mdust /
                    c.M_sun, c="C1", label="Dust")
        ax02.axvline(data.t[it]/c.year, c="#AAAAAA", lw=1, ls="--")
        ax02.set_xlim(data.t[1]/c.year, data.t[-1]/c.year)
        ax02.set_ylim(10.**(Mmax-6.), 10.**Mmax)
        ax02.legend()
    ax02.set_xlabel("Time [yrs]")
    ax02.set_ylabel("Mass [$M_\odot$]")

    ax10.loglog(data.r[it, ...]/c.au, data.sigmaDusti[it, :, im], c="C3")
    ax10.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
    ax10.set_ylim(10.**(sd_max-6.), 10.**sd_max)
    ax10.set_xlabel("Distance from star [au]")
    ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

    ax11.loglog(data.r[it, ...]/c.au, data.SigmaGas[it, ...], label="Gas")
    ax11.loglog(data.r[it, ...]/c.au,
                data.SigmaDusti[it, ...].sum(-1), label="Dust")
    ax11.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
    ax11.set_ylim(10.**(sg_max-6), 10.**sg_max)
    ax11.set_xlabel("Distance from star [AU]")
    ax11.set_ylabel("$\Sigma$ [g/cm²]")
    ax11.legend()
    ax11r.loglog(data.r[it, ...]/c.au, data.eps[it, ...], color="C7", lw=1)
    ax11r.set_ylim(1.e-5, 1.e1)
    ax11r.set_ylabel("Dust-to-gas ratio")

    fig.tight_layout()

    fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
             verticalalignment="bottom")

    plt.show()


def panel_new(data, filename="data", extension="hdf5", im=0, ir=0, it=0, show_limits=True, show_St1=True):
    """Simple plotting script for data files or simulation objects.

    Parameters
    ----------
    data : ``twopoppy.Simulation`` or string
        Either instance of ``twopoppy.Simulation`` or path to data directory to be plotted
    filename : string, optional, default : "data"
    extension : string, optional, default : "hdf5"
        Plotting script is looking for files with pattern ``<data>/<filename>*.<extension>``
    im : int, optional, default : 0
        Number of mass bin along which density distribution is plotted
    ir : int, optional, default : 0
        Number of radial grid index along density distribution is plotted
    it : int, optional, default : 0
        Index of snapshot to be plotted
    show_limits : boolean, optional, default : True
        If True growth limits are plotted
    show_St1 : boolean, optional, default : True
        If True St=1 line is plotted"""

    from twopoppy.plot import __version__

    data = _readdata(data, filename=filename, extension=extension)

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, data.Nt-1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data.Nmi[it, ...]-1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...]-1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.sigmaDusti.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max()/c.M_sun)) + 1
    levels_num = 7
    levels = np.linspace(sd_max-(levels_num-1), sd_max, levels_num)

    out1, out2, out3, out4, out5 = widgets.Output(), widgets.Output(), widgets.Output(), \
    widgets.Output(), widgets.Output()

    with out1:
        width = 6.
        height = width/golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax00 = fig.add_subplot(111)

        # Density distribution
        plt00 = ax00.contourf(data.r[it, ...]/c.au,
                              data.mic[it, ...],
                              np.log10(data.sigmaDusti[it, ...].T),
                              levels=levels,
                              cmap="magma",
                              extend="both"
                              )
        if show_St1:
            ax00.contour(data.r[it, ...]/c.au,
                        data.mic[it, ...],
                        data.Sti[it, ...].T,
                        levels=[1.],
                        colors="white",
                        linewidths=2
                        )
        if show_limits:
            ax00.contour(data.r[it, ...]/c.au,
                        data.mic[it, ...],
                        (data.Sti - data.StDr[..., None])[it, ...].T,
                        levels=[0.],
                        colors="C2",
                        linewidths=1
                        )
            ax00.contour(data.r[it, ...]/c.au,
                        data.mic[it, ...],
                        (data.Sti - data.StFr[..., None])[it, ...].T,
                        levels=[0.],
                        colors="C0",
                        linewidths=1
                        )

        ax00.axhline(data.mic[it, im], color="#AAAAAA", lw=1, ls="--")
        ax00.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")

        cbar00 = plt.colorbar(plt00, ax=ax00)
        cbar00.ax.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
        cbar00ticklabels = []
        for i in levels:
            cbar00ticklabels.append("$10^{{{:d}}}$".format(int(i)))
        cbar00.ax.set_yticklabels(cbar00ticklabels)
        ax00.set_xscale("log")
        ax00.set_yscale("log")
        ax00.set_xlabel("Distance from star [AU]")
        ax00.set_ylabel("Particle mass [g]")

        fig.tight_layout()
        fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.show()

    with out2:
        width = 6.
        height = width/golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax01 = fig.add_subplot(111)

        ax01.loglog(data.mic[it, ...], data.sigmaDusti[it, ir, :], c="C3")
        ax01.set_xlim(data.mic[it, 0], data.mic[it, -1])
        ax01.set_ylim(10.**(sd_max-6.), 10.**sd_max)
        ax01.set_xlabel("Particle mass [g]")
        ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

        fig.tight_layout()
        fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.show()

    with out3:
        width = 6.
        height = width/golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax02 = fig.add_subplot(111)

        if data.Nt < 3:
            ax02.set_xticks([0., 1.])
            ax02.set_yticks([0., 1.])
            ax02.text(0.5,
                      0.5,
                      "Not enough data points.",
                      verticalalignment="center",
                      horizontalalignment="center",
                      size="large")
        else:
            ax02.loglog(data.t/c.year, data.Mgas/c.M_sun, c="C0", label="Gas")
            ax02.loglog(data.t/c.year, data.Mdust /
                        c.M_sun, c="C1", label="Dust")
            try:
                ax02.loglog(data.t/c.year, data.Mplanet /
                            c.M_sun, c="C2", label="Planetesimals")
            except:
                pass
            ax02.axvline(data.t[it]/c.year, c="#AAAAAA", lw=1, ls="--")
            ax02.set_xlim(data.t[1]/c.year, data.t[-1]/c.year)
            ax02.set_ylim(10.**(Mmax-6.), 10.**Mmax)
            ax02.legend()
        ax02.set_xlabel("Time [yrs]")
        ax02.set_ylabel("Mass [$M_\odot$]")

        fig.tight_layout()
        fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.show()

    with out4:
        width = 6.
        height = width/golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax10 = fig.add_subplot(111)

        ax10.loglog(data.r[it, ...]/c.au, data.sigmaDusti[it, :, im], c="C3")
        ax10.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
        ax10.set_ylim(10.**(sd_max-6.), 10.**sd_max)
        ax10.set_xlabel("Distance from star [au]")
        ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

        fig.tight_layout()
        fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.show()

    with out5:
        width = 6.
        height = width/golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax11 = fig.add_subplot(111)
        ax11r = ax11.twinx()

        ax11.loglog(data.r[it, ...]/c.au, data.SigmaGas[it, ...], label="Gas")
        ax11.loglog(data.r[it, ...]/c.au,
                    data.SigmaDusti[it, ...].sum(-1), label="Dust")
        try:
            ax11.loglog(data.r[it, ...]/c.au,
                        data.SigmaPlanet[it, ...], label="Planetesimals")
        except:
            pass
        ax11.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
        ax11.set_ylim(10.**(sg_max-6), 10.**sg_max)
        ax11.set_xlabel("Distance from star [AU]")
        ax11.set_ylabel("$\Sigma$ [g/cm²]")
        ax11.legend()
        ax11r.loglog(data.r[it, ...]/c.au, data.eps[it, ...], color="C7", lw=1)
        ax11r.set_ylim(1.e-5, 1.e1)
        ax11r.set_ylabel("Dust-to-gas ratio")

        fig.tight_layout()
        fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.show()

    tab = widgets.Tab(children = [out1, out2, out4, out3, out5],
          layout=widgets.Layout(width='100%'))
    tab.set_title(0, 'Dust Surface Density')
    tab.set_title(1, 'Sigma Dust [im]')
    tab.set_title(2, 'Sigma Dust [ir]')
    tab.set_title(3, 'Mass Evolution')
    tab.set_title(4, 'Surface Densities')
    display(tab)


def ipanel(data, filename="data", extension="hdf5", im=0, ir=0, it=0, show_limits=True, show_St1=True):
    """Simple interactive plotting script for data files or simulation objects.

    Parameters
    ----------
    data : ``twopoppy.Simulation`` or string
        Either instance of ``twopoppy.Simulation`` or path to data directory to be plotted
    filename : string, optional, default : "data"
    extension : string, optional, default : "hdf5"
        Plotting script is looking for files with pattern ``<data>/<filename>*.<extension>``
    im : int, optional, default : 0
        Number of mass bin along which density distribution is plotted
    ir : int, optional, default : 0
        Number of radial grid index along density distribution is plotted
    it : int, optional, default : 0
        Index of snapshot to be plotted
    show_limits : boolean, optional, default : True
        If True growth limits are plotted
    show_St1 : boolean, optional, default : True
        If True St=1 line is plotted"""

    from twopoppy.plot import __version__

    data = _readdata(data, filename=filename, extension=extension)

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, data.Nt-1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data.Nmi[it, ...]-1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...]-1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.sigmaDusti.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max()/c.M_sun)) + 1
    levels_num = 7
    levels = np.linspace(sd_max-(levels_num-1), sd_max, levels_num)

    width = 3.5
    fig = plt.figure(figsize=(3.*width, 2.*width/1.618), dpi=150)
    ax00 = fig.add_subplot(231)
    ax01 = fig.add_subplot(232)
    ax02 = fig.add_subplot(233)
    ax10 = fig.add_subplot(234)
    ax11 = fig.add_subplot(235)
    ax11r = ax11.twinx()

    # Density distribution
    plt00 = ax00.contourf(data.r[it, ...]/c.au,
                          data.mic[it, ...],
                          np.log10(data.sigmaDusti[it, ...].T),
                          levels=levels,
                          cmap="magma",
                          extend="both"
                          )
    plt00Collections = plt00.collections[:]
    if show_St1:
        plt00St = ax00.contour(data.r[it, ...]/c.au,
                               data.mic[it, ...],
                               data.Sti[it, ...].T,
                               levels=[1.],
                               colors="white",
                               linewidths=2
                               )
        plt00StCollections = plt00St.collections[:]
    if show_limits:
        plt00Dr = ax00.contour(data.r[it, ...]/c.au,
                               data.mic[it, ...],
                               (data.Sti - data.StDr[..., None])[it, ...].T,
                               levels=[0.],
                               colors="C2",
                               linewidths=1
                               )
        plt00DrCollections = plt00Dr.collections[:]
        plt00Fr = ax00.contour(data.r[it, ...]/c.au,
                               data.mic[it, ...],
                               (data.Sti - data.StFr[..., None])[it, ...].T,
                               levels=[0.],
                               colors="C0",
                               linewidths=1
                               )
        plt00FrCollections = plt00Fr.collections[:]
    plt00hl = ax00.axhline(data.mic[it, im], color="#AAAAAA", lw=1, ls="--")
    plt00vl = ax00.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")

    cbar00 = plt.colorbar(plt00, ax=ax00)
    cbar00.ax.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
    cbar00ticklabels = []
    for i in levels:
        cbar00ticklabels.append("$10^{{{:d}}}$".format(int(i)))
    cbar00.ax.set_yticklabels(cbar00ticklabels)
    ax00.set_xscale("log")
    ax00.set_yscale("log")
    ax00.set_xlabel("Distance from star [AU]")
    ax00.set_ylabel("Particle mass [g]")

    plt01 = ax01.loglog(data.mic[it, ...], data.sigmaDusti[it, ir, :], c="C3")
    plt01vl = ax01.axvline(data.mic[it, im], color="#AAAAAA", lw=1, ls="--")
    ax01.set_xlim(data.mic[it, 0], data.mic[it, -1])
    ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, ir, :])))
    ylim0 = ylim1 - 6.
    ax01.set_ylim(10.**ylim0, 10.**ylim1)
    ax01.set_xlabel("Particle mass [g]")
    ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

    if data.Nt < 3:
        ax02.set_xticks([0., 1.])
        ax02.set_yticks([0., 1.])
        ax02.text(0.5,
                  0.5,
                  "Not enough data points.",
                  verticalalignment="center",
                  horizontalalignment="center",
                  size="large")
    else:
        ax02.loglog(data.t/c.year, data.Mgas/c.M_sun, c="C0", label="Gas")
        ax02.loglog(data.t/c.year, data.Mdust /
                    c.M_sun, c="C1", label="Dust")
        plt02vl = ax02.axvline(data.t[it]/c.year, c="#AAAAAA", lw=1, ls="--")
        ax02.set_xlim(data.t[1]/c.year, data.t[-1]/c.year)
        ax02.set_ylim(10.**(Mmax-6.), 10.**Mmax)
        ax02.legend()
    ax02.set_xlabel("Time [yrs]")
    ax02.set_ylabel("Mass [$M_\odot$]")

    plt10 = ax10.loglog(data.r[it, ...]/c.au,
                        data.sigmaDusti[it, :, im], c="C3")
    plt10vl = ax10.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")
    ax10.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
    ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, :, im])))
    ylim0 = ylim1 - 6.
    ax10.set_ylim(10.**ylim0, 10.**ylim1)
    ax10.set_xlabel("Distance from star [au]")
    ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

    plt11g = ax11.loglog(data.r[it, ...]/c.au,
                         data.SigmaGas[it, ...], label="Gas")
    plt11d = ax11.loglog(data.r[it, ...]/c.au,
                         data.SigmaDusti[it, ...].sum(-1), label="Dust")
    plt11vl = ax11.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")
    ax11.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
    ax11.set_ylim(10.**(sg_max-6), 10.**sg_max)
    ax11.set_xlabel("Distance from star [AU]")
    ax11.set_ylabel("$\Sigma$ [g/cm²]")
    ax11.legend()
    plt11d2g = ax11r.loglog(data.r[it, ...]/c.au,
                            data.eps[it, ...], color="C7", lw=1)
    ax11r.set_ylim(1.e-5, 1.e1)
    ax11r.set_ylabel("Dust-to-gas ratio")

    fig.tight_layout()

    width = ax02.get_position().x1 - ax02.get_position().x0
    fig._widgets = []

    if data.Nt > 2:
        axSliderTime = plt.axes([ax02.get_position().x0 + 0.15 * width,
                                 0.375,
                                 0.75 * width,
                                 0.02], facecolor="lightgoldenrodyellow")
        sliderTime = Slider(axSliderTime, "Time", 0, data.Nt -
                            1, valinit=it, valfmt="%i")
        axSliderTime.set_title("t = {:9.3e} yr".format(data.t[it]/c.year))
        fig._widgets += [sliderTime]

    axSliderMass = plt.axes([ax02.get_position().x0 + 0.15 * width,
                             0.25,
                             0.75 * width,
                             0.02], facecolor="lightgoldenrodyellow")
    sliderMass = Slider(axSliderMass, "Mass", 0,
                        data.Nmi[it]-1, valinit=im, valfmt="%i")
    axSliderMass.set_title("m = {:9.3e} g".format(data.mic[it, im]))
    fig._widgets += [sliderMass]

    axSliderDist = plt.axes([ax02.get_position().x0 + 0.15 * width,
                             0.125,
                             0.75 * width,
                             0.02], facecolor="lightgoldenrodyellow")
    sliderDist = Slider(axSliderDist, "Distance", 0,
                        data.Nr[it]-1, valinit=ir, valfmt="%i")
    axSliderDist.set_title("r = {:9.3e} AU".format(data.r[it, ir]/c.au))
    fig._widgets += [sliderDist]

    def update(val):

        it = 0
        if data.Nt > 2:
            it = int(np.floor(sliderTime.val))
            axSliderTime.set_title("t = {:9.3e} yr".format(data.t[it]/c.year))
        im = int(np.floor(sliderMass.val))
        axSliderMass.set_title("m = {:9.3e} g".format(data.mic[it, im]))
        ir = int(np.floor(sliderDist.val))
        axSliderDist.set_title("r = {:9.3e} AU".format(data.r[it, ir]/c.au))

        for row in plt00Collections:
            ax00.collections.remove(row)
            plt00Collections.remove(row)
        plt00 = ax00.contourf(data.r[it, ...]/c.au,
                              data.mic[it, ...],
                              np.log10(data.sigmaDusti[it, ...].T),
                              levels=np.linspace(sd_max-6, sd_max, 7),
                              cmap="magma",
                              extend="both"
                              )
        for row in plt00.collections:
            plt00Collections.append(row)
        if show_St1:
            for row in plt00StCollections:
                ax00.collections.remove(row)
                plt00StCollections.remove(row)
            plt00St = ax00.contour(data.r[it, ...]/c.au,
                                   data.mic[it, ...],
                                   data.Sti[it, ...].T,
                                   levels=[1.],
                                   colors="white",
                                   linewidths=2
                                   )
            for row in plt00St.collections:
                plt00StCollections.append(row)
        if show_limits:
            for row in plt00DrCollections:
                ax00.collections.remove(row)
                plt00DrCollections.remove(row)
            plt00Dr = ax00.contour(data.r[it, ...]/c.au,
                                   data.mic[it, ...],
                                   (data.Sti - data.StDr[..., None])[it, ...].T,
                                   levels=[0.],
                                   colors="C2",
                                   linewidths=1
                                   )
            for row in plt00Dr.collections:
                plt00DrCollections.append(row)
            for row in plt00FrCollections:
                ax00.collections.remove(row)
                plt00FrCollections.remove(row)
            plt00Fr = ax00.contour(data.r[it, ...]/c.au,
                                   data.mic[it, ...],
                                   (data.Sti - data.StFr[..., None])[it, ...].T,
                                   levels=[0.],
                                   colors="C0",
                                   linewidths=1
                                   )
            for row in plt00Fr.collections:
                plt00FrCollections.append(row)
        plt00vl.set_xdata([data.r[it, ir]/c.au, data.r[it, ir]/c.au])
        plt00hl.set_ydata([data.mic[it, im], data.mic[it, im]])

        plt01[0].set_xdata(data.mic[it, ...])
        plt01[0].set_ydata(data.sigmaDusti[it, ir, :])
        ax01.set_xlim(data.mic[it, 0], data.mic[it, -1])
        ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, ir, :])))
        ylim0 = ylim1 - 6.
        ax01.set_ylim(10.**ylim0, 10.**ylim1)
        plt01vl.set_xdata([data.mic[it, im], data.mic[it, im]])
        plt01vl.set_ydata([0., 1.e100])

        if data.Nt > 2:
            plt02vl.set_xdata([data.t[it]/c.year, data.t[it]/c.year])
        plt10vl.set_xdata([data.r[it, ir]/c.au, data.r[it, ir]/c.au])
        plt11vl.set_xdata([data.r[it, ir]/c.au, data.r[it, ir]/c.au])

        plt10[0].set_xdata(data.r[it, ...]/c.au)
        plt10[0].set_ydata(data.sigmaDusti[it, :, im])
        ax10.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
        ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, :, im])))
        ylim0 = ylim1 - 6.
        ax10.set_ylim(10.**ylim0, 10.**ylim1)
        plt10vl.set_xdata([data.r[it, ir]/c.au, data.r[it, ir]/c.au])
        plt10vl.set_ydata([0., 1.e100])

        plt11g[0].set_xdata(data.r[it, ...]/c.au)
        plt11g[0].set_ydata(data.SigmaGas[it, ...])
        plt11d[0].set_xdata(data.r[it, ...]/c.au)
        plt11d[0].set_ydata(data.SigmaDusti[it, ...].sum(-1))
        plt11vl.set_xdata([data.r[it, ir]/c.au, data.r[it, ir]])
        plt11vl.set_ydata([0., 1.e100])
        plt11d2g[0].set_xdata(data.r[it, ...]/c.au)
        plt11d2g[0].set_ydata(data.eps[it, ...])

    if data.Nt > 2:
        sliderTime.on_changed(update)
    sliderMass.on_changed(update)
    sliderDist.on_changed(update)

    fig.text(0.99, 0.01, "Twopoppy v"+__version__, horizontalalignment="right",
             verticalalignment="bottom")

    plt.show()


def ipanel_new(data, filename="data", extension="hdf5", im=0, ir=0, it=0, show_limits=True, show_St1=True):
    """Simple interactive plotting script for data files or simulation objects.

    Parameters
    ----------
    data : ``twopoppy.Simulation`` or string
        Either instance of ``twopoppy.Simulation`` or path to data directory to be plotted
    filename : string, optional, default : "data"
    extension : string, optional, default : "hdf5"
        Plotting script is looking for files with pattern ``<data>/<filename>*.<extension>``
    im : int, optional, default : 0
        Number of mass bin along which density distribution is plotted
    ir : int, optional, default : 0
        Number of radial grid index along density distribution is plotted
    it : int, optional, default : 0
        Index of snapshot to be plotted
    show_limits : boolean, optional, default : True
        If True growth limits are plotted
    show_St1 : boolean, optional, default : True
        If True St=1 line is plotted"""

    from twopoppy.plot import __version__

    data = _readdata(data, filename=filename, extension=extension)

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, data.Nt-1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data.Nmi[it, ...]-1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...]-1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.sigmaDusti.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max()/c.M_sun)) + 1
    levels_num = 7
    levels = np.linspace(sd_max-(levels_num-1), sd_max, levels_num)

    out1, out2, out3, out4, out5 = widgets.Output(), widgets.Output(), widgets.Output(), \
    widgets.Output(), widgets.Output()

    play = widgets.Play(value=0, min=0, max=int(data.Nt-1), step=1, interval=2000, \
                        description="Press play", disabled=False)
    ui_temp = widgets.IntSlider(description='Temporal index', value=0, min=0, max=int(data.Nt-1));
    widgets.jslink((play, 'value'), (ui_temp, 'value'))
    display(widgets.HBox([play, ui_temp], layout=widgets.Layout(width='100%')))
    ui_rad = widgets.IntSlider(description='Radial index', value=0, min=0, max=int(data.Nr[0, ...]-1));
    ui_mass = widgets.IntSlider(description='Mass index', value=0, min=0, max=int(data.Nmi[0, ...]-2));
    display(widgets.HBox([ui_rad, ui_mass], layout=widgets.Layout(width='100%')))

    with out1:
        def plot1(it, ir, im):

            width = 6.
            height = width/golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax00 = fig.add_subplot(111)

            # Density distribution
            plt00 = ax00.contourf(data.r[it, ...]/c.au,
                                data.mic[it, ...],
                                np.log10(data.sigmaDusti[it, ...].T),
                                levels=levels,
                                cmap="magma",
                                extend="both"
                                )
            plt00Collections = plt00.collections[:]
            if show_St1:
                plt00St = ax00.contour(data.r[it, ...]/c.au,
                                    data.mic[it, ...],
                                    data.Sti[it, ...].T,
                                    levels=[1.],
                                    colors="white",
                                    linewidths=2
                                    )
                plt00StCollections = plt00St.collections[:]
            if show_limits:
                plt00Dr = ax00.contour(data.r[it, ...]/c.au,
                                    data.mic[it, ...],
                                    (data.Sti - data.StDr[..., None])[it, ...].T,
                                    levels=[0.],
                                    colors="C2",
                                    linewidths=1
                                    )
                plt00DrCollections = plt00Dr.collections[:]
                plt00Fr = ax00.contour(data.r[it, ...]/c.au,
                                    data.mic[it, ...],
                                    (data.Sti - data.StFr[..., None])[it, ...].T,
                                    levels=[0.],
                                    colors="C0",
                                    linewidths=1
                                    )
                plt00FrCollections = plt00Fr.collections[:]
            plt00hl = ax00.axhline(data.mic[it, im], color="#AAAAAA", lw=1, ls="--")
            plt00vl = ax00.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")

            cbar00 = plt.colorbar(plt00, ax=ax00)
            cbar00.ax.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
            cbar00ticklabels = []
            for i in levels:
                cbar00ticklabels.append("$10^{{{:d}}}$".format(int(i)))
            cbar00.ax.set_yticklabels(cbar00ticklabels)
            ax00.set_xscale("log")
            ax00.set_yscale("log")
            ax00.set_xlabel("Distance from star [AU]")
            ax00.set_ylabel("Particle mass [g]")

            fig.tight_layout()
            fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
                     verticalalignment="bottom")
            plt.show()

        out = widgets.interactive_output(plot1, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass});
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out2:
        def plot2(it, ir, im):

            width = 6.
            height = width/golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax01 = fig.add_subplot(111)

            passplt01 = ax01.loglog(data.mic[it, ...], data.sigmaDusti[it, ir, :], c="C3")
            plt01vl = ax01.axvline(data.mic[it, im], color="#AAAAAA", lw=1, ls="--")
            ax01.set_xlim(data.mic[it, 0], data.mic[it, -1])
            ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, ir, :])))
            ylim0 = ylim1 - 6.
            ax01.set_ylim(10.**ylim0, 10.**ylim1)
            ax01.set_xlabel("Particle mass [g]")
            ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

            fig.tight_layout()
            fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
                     verticalalignment="bottom")
            plt.show()

        out = widgets.interactive_output(plot2, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass});
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out3:
        def plot3(it, ir, im):

            width = 6.
            height = width/golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax02 = fig.add_subplot(111)

            if data.Nt < 3:
                ax02.set_xticks([0., 1.])
                ax02.set_yticks([0., 1.])
                ax02.text(0.5,
                          0.5,
                          "Not enough data points.",
                          verticalalignment="center",
                          horizontalalignment="center",
                          size="large")
            else:
                ax02.loglog(data.t/c.year, data.Mgas/c.M_sun, c="C0", label="Gas")
                ax02.loglog(data.t/c.year, data.Mdust /
                            c.M_sun, c="C1", label="Dust")
                try:
                    ax02.loglog(data.t/c.year, data.Mplanet/c.M_sun, c="C2", label="Planetesimals")
                except:
                    pass
                plt02vl = ax02.axvline(data.t[it]/c.year, c="#AAAAAA", lw=1, ls="--")
                ax02.set_xlim(data.t[1]/c.year, data.t[-1]/c.year)
                ax02.set_ylim(10.**(Mmax-6.), 10.**Mmax)
                ax02.legend()
            ax02.set_xlabel("Time [yrs]")
            ax02.set_ylabel("Mass [$M_\odot$]")

            fig.tight_layout()
            fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
                     verticalalignment="bottom")
            plt.show()

        out = widgets.interactive_output(plot3, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass});
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out4:
        def plot4(it, im, ir):

            width = 6.
            height = width/golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax10 = fig.add_subplot(111)

            plt10 = ax10.loglog(data.r[it, ...]/c.au,
                                data.sigmaDusti[it, :, im], c="C3")
            plt10vl = ax10.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")
            ax10.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
            ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, :, im])))
            ylim0 = ylim1 - 6.
            ax10.set_ylim(10.**ylim0, 10.**ylim1)
            ax10.set_xlabel("Distance from star [au]")
            ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

            fig.tight_layout()
            fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
                     verticalalignment="bottom")
            plt.show()

        out = widgets.interactive_output(plot4, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass});
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out5:
        def plot5(it, ir, im):

            width = 6.
            height = width/golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax11 = fig.add_subplot(111)
            ax11r = ax11.twinx()

            plt11g = ax11.loglog(data.r[it, ...]/c.au,
                                 data.SigmaGas[it, ...], label="Gas")
            plt11d = ax11.loglog(data.r[it, ...]/c.au,
                                 data.SigmaDusti[it, ...].sum(-1), label="Dust")
            try:
                plt11p = ax11.loglog(data.r[it, ...]/c.au,
                                     data.SigmaPlanet[it, ...], label="Planetesimals")
            except:
                pass
            plt11vl = ax11.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")
            ax11.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
            ax11.set_ylim(10.**(sg_max-6), 10.**sg_max)
            ax11.set_xlabel("Distance from star [AU]")
            ax11.set_ylabel("$\Sigma$ [g/cm²]")
            ax11.legend()
            plt11d2g = ax11r.loglog(data.r[it, ...]/c.au,
                                    data.eps[it, ...], color="C7", lw=1)
            ax11r.set_ylim(1.e-5, 1.e1)
            ax11r.set_ylabel("Dust-to-gas ratio")

            fig.tight_layout()
            fig.text(0.99, 0.01, "TwoPopPy v"+__version__, horizontalalignment="right",
                     verticalalignment="bottom")
            plt.show()

        out = widgets.interactive_output(plot5, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass});
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    tab = widgets.Tab(children = [out1, out2, out4, out3, out5],
              layout=widgets.Layout(width='100%'))
    tab.set_title(0, 'Dust Surface Density')
    tab.set_title(1, 'Sigma Dust [im]')
    tab.set_title(2, 'Sigma Dust [ir]')
    tab.set_title(3, 'Mass Evolution')
    tab.set_title(4, 'Surface Densities')
    display(tab)


def _readdata(data, filename="data", extension="hdf5"):

    ret = {}

    if isinstance(data, Simulation):

        m = data.dust.m[None, ...]
        r = data.grid.r[None, ...]
        ri = data.grid.ri[None, ...]
        Nr = data.grid.Nr[None, ...]
        t = data.t[None, ...]
        Nt = np.array([1])[None, ...]

        SigmaDust = data.dust.Sigma[None, ...]
        SigmaGas = data.gas.Sigma[None, ...]
        eps = data.dust.eps[None, ...]

        cs = data.gas.cs[None, ...]
        delta = data.dust.delta.turb[None, ...]
        OmegaK = data.grid.OmegaK[None, ...]
        St = data.dust.St[None, ...]
        vK = OmegaK[None, ...] * r
        vFrag = data.dust.v.frag[None, ...]
        smin = data.dust.s.min[None, ...]
        smax = data.dust.s.max[None, ...]
        xicalc = data.dust.xi.calc[None, ...]
        rhos = data.dust.rhos[None, ...]
        fill = data.dust.fill[None, ...]
        mfp = data.gas.mfp[None, ...]

        try:
            SigmaPlanet = data.planetesimals.Sigma[None, ...]
        except:
            pass

    elif os.path.isdir(data):

        writer = hdf5writer()

        # Setting up writer
        writer.datadir = data
        writer.extension = extension
        writer.filename = filename

        m = writer.read.sequence("dust.m")
        r = writer.read.sequence("grid.r")
        ri = writer.read.sequence("grid.ri")
        Nr = writer.read.sequence("grid.Nr")
        t = writer.read.sequence("t")
        Nt = np.array([len(t)])

        SigmaDust = writer.read.sequence("dust.Sigma")
        SigmaGas = writer.read.sequence("gas.Sigma")
        eps = writer.read.sequence("dust.eps")

        cs = writer.read.sequence("gas.cs")
        delta = writer.read.sequence("dust.delta.turb")
        OmegaK = writer.read.sequence("grid.OmegaK")
        St = writer.read.sequence("dust.St")
        vK = OmegaK * r
        vFrag = writer.read.sequence("dust.v.frag")
        smin = writer.read.sequence("dust.s.min")
        smax = writer.read.sequence("dust.s.max")
        xicalc = writer.read.sequence("dust.xi.calc")
        rhos = writer.read.sequence("dust.rhos")
        fill = writer.read.sequence("dust.fill")
        mfp = writer.read.sequence("gas.mfp")

        try:
            SigmaPlanet = writer.read.sequence("planetesimals.Sigma")
        except:
            pass

    else:

        raise RuntimeError("Unknown data type.")

    # Masses
    Mgas = (np.pi * (ri[..., 1:]**2 - ri[..., :-1]**2) * SigmaGas[...]).sum(-1)
    Mdust = (np.pi * (ri[..., 1:]**2 - ri[..., :-1]**2)
             * SigmaDust[...].sum(-1)).sum(-1)
    try:
        Mplanet = (np.pi * (ri[..., 1:]**2 - ri[..., :-1]**2) * SigmaPlanet[...]).sum(-1)
    except:
        pass

    # Interpolation of the density distribution over mass grid
    # via distribution exponent
    Nmi = 142
    Nmi = np.array([1])[None, ...] * Nmi
    Nr_len = Nr[0]
    rho = rhos * fill
    # Assumption: Particles of all sizes have same mass density
    rho = np.full((int(Nt), int(Nr_len), int(Nmi - 1)), rho[0, 0, 0])
    mmin = 4./3. * np.pi * rho[:, :, 0] * smin**3
    mmax = 4./3. * np.pi * rho[:, :, 0] * smax**3
    mi = np.full(int(Nmi), np.logspace(np.log10(mmin.min()), np.log10(1.e6 *  mmax.max()), int(Nmi)))
    mi = np.full((int(Nt), int(Nmi)), mi)
    mi0 = mi[..., :-1]
    mi1 = mi[..., 1:]
    mic = 0.5 * (mi0[...] + mi1[...])
    SigmaDustTot = SigmaDust[...].sum(-1)
    SigmaDusti = np.ones((int(Nt), int(Nr_len), int(Nmi-1))) * 1e-100
    for i in range(int(Nt)):
        for j in range(int(Nr_len)):
            for k in range(int(Nmi-1)):
                if mi1[i, k] <= mmax[i, j]:
                    if xicalc[i, j] != -4.:
                        expo = (xicalc[i, j]+4.) / 3.
                        SigmaDusti[i, j, k] = SigmaDustTot[i, j] * \
                        (mi1[i, k]**expo - mi0[i, k]**expo) / \
                        (mmax[i, j]**expo - mmin[i, j]**expo)
                    else:
                        SigmaDusti[i, j, k] = SigmaDustTot[i, j] * \
                        np.log(mi1[i, k] / mi0[i, k]) / \
                        np.log(mmax[i, j] / mmin[i, j])
                else:
                    SigmaDusti[i, j, k] = np.maximum(1.e-100, SigmaDustTot[i, j] - SigmaDusti[i, j].sum())
    if np.abs(SigmaDustTot.sum() - SigmaDusti.sum()) / SigmaDustTot.sum() > 1.e-10:
        print("Error in surface density interpolation!")

    # Transformation of the density distribution
    a = np.array(np.mean(mic[..., 1:] / mic[..., :-1], axis=-1))
    dm = np.array(2. * (a - 1.) / (a + 1.))
    sigmaDusti = SigmaDusti[...] / dm[..., None, None]

    # Calculation of Stokes Number over mass grid
    ai = np.zeros((int(Nt), int(Nr_len), int(Nmi-1)))
    for i in range(int(Nt)):
        ai[i] = np.full((int(Nr_len), int(Nmi-1)), 3 / (4 * np.pi * rho[i, 0, 0]) * mic[i, :]**(1/3))
    Sti = np.zeros((int(Nt), int(Nr_len), int(Nmi-1)))
    for i in range(int(Nt)):
        Sti[i] = dp_dust_f.st_epstein_stokes1(ai[i], mfp[i], rho[i], SigmaGas[i])

    # Fragmentation limit
    b = vFrag**2 / (delta * cs**2)
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings(
            'ignore',
            r'invalid value encountered in sqrt')
        StFr = 1 / (2 * b) * (3 - np.sqrt(9 - 4 * b**2))

    # Drift limit
    p = SigmaGas * OmegaK * cs / np.sqrt(2.*np.pi)
    StDr = np.zeros_like(StFr)
    for i in range(int(Nt)):
        _f = interp1d(np.log10(r[i, ...]), np.log10(
            p[i, ...]), fill_value='extrapolate')
        pi = 10.**_f(np.log10(ri[i, ...]))
        gamma = np.abs(r[i, ...] / p[i, ...] *
                       np.diff(pi) / np.diff(ri[i, ...]))
        StDr[i, ...] = eps[i, ...] / gamma * (vK[i, ...] / cs[i, ...])**2

    ret["mic"] = mic
    ret["Nmi"] = Nmi
    ret["r"] = r
    ret["ri"] = ri
    ret["Nr"] = Nr
    ret["t"] = t
    ret["Nt"] = Nt

    ret["SigmaDusti"] = SigmaDusti
    ret["sigmaDusti"] = sigmaDusti
    ret["SigmaDustTot"] = SigmaDustTot
    ret["SigmaGas"] = SigmaGas
    ret["eps"] = eps

    ret["Mdust"] = Mdust
    ret["Mgas"] = Mgas

    ret["cs"] = cs
    ret["delta"] = delta
    ret["OmegaK"] = OmegaK
    ret["Sti"] = Sti
    ret["StDr"] = StDr
    ret["StFr"] = StFr
    ret["vK"] = vK
    ret["vFrag"] = vFrag
    ret["smin"] = smin
    ret["smax"] = smax
    ret["xicalc"] = xicalc

    try:
        ret["SigmaPlanet"] = SigmaPlanet
        ret["Mplanet"] = Mplanet
    except:
        pass

    return SimpleNamespace(**ret)
