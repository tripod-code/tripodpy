import twopoppy.constants as c
from twopoppy.simulation import Simulation

from types import SimpleNamespace
import numpy as np
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
    im = np.minimum(im, data.Nm[it, ...]-1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...]-1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.SigmaDusti.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max()/c.M_sun)) + 1
    levels = np.linspace(sd_max-(data.Nmi-2), sd_max, data.Nmi-1)
    # default for data.Nmi is 8 in accordance with default sim.ini.grid.Nmbpd in dustpy

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
                          np.log10(data.SigmaDusti[it, ...].T),
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

    ax01.loglog(data.mic[it, ...], data.SigmaDusti[it, ir, :], c="C3")
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

    ax10.loglog(data.r[it, ...]/c.au, data.SigmaDusti[it, :, im], c="C3")
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
    im = np.minimum(im, data.Nm[it, ...]-1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...]-1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.SigmaDusti.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max()/c.M_sun)) + 1
    levels = np.linspace(sd_max-6, sd_max, 7)

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
                          np.log10(data.SigmaDusti[it, ...].T),
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

    plt01 = ax01.loglog(data.mic[it, ...], data.SigmaDusti[it, ir, :], c="C3")
    plt01vl = ax01.axvline(data.mic[it, im], color="#AAAAAA", lw=1, ls="--")
    ax01.set_xlim(data.mic[it, 0], data.mic[it, -1])
    ylim1 = np.ceil(np.log10(np.max(data.SigmaDusti[it, ir, :])))
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
                        data.SigmaDusti[it, :, im], c="C3")
    plt10vl = ax10.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")
    ax10.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
    ylim1 = np.ceil(np.log10(np.max(data.SigmaDusti[it, :, im])))
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
                              np.log10(data.SigmaDusti[it, ...].T),
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
        plt01[0].set_ydata(data.SigmaDusti[it, ir, :])
        ax01.set_xlim(data.mic[it, 0], data.mic[it, -1])
        ylim1 = np.ceil(np.log10(np.max(data.SigmaDusti[it, ir, :])))
        ylim0 = ylim1 - 6.
        ax01.set_ylim(10.**ylim0, 10.**ylim1)
        plt01vl.set_xdata([data.mic[it, im], data.mic[it, im]])
        plt01vl.set_ydata([0., 1.e100])

        if data.Nt > 2:
            plt02vl.set_xdata([data.t[it]/c.year, data.t[it]/c.year])
        plt10vl.set_xdata([data.r[it, ir]/c.au, data.r[it, ir]/c.au])
        plt11vl.set_xdata([data.r[it, ir]/c.au, data.r[it, ir]/c.au])

        plt10[0].set_xdata(data.r[it, ...]/c.au)
        plt10[0].set_ydata(data.SigmaDusti[it, :, im])
        ax10.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
        ylim1 = np.ceil(np.log10(np.max(data.SigmaDusti[it, :, im])))
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


def _readdata(data, filename="data", extension="hdf5"):

    ret = {}

    if isinstance(data, Simulation):

        m = data.dust.m[None, ...]
        Nm = data.grid._Nm_long[None, ...]
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

    elif os.path.isdir(data):

        writer = hdf5writer()

        # Setting up writer
        writer.datadir = data
        writer.extension = extension
        writer.filename = filename

        m = writer.read.sequence("dust.m")
        Nm = writer.read.sequence("grid._Nm_long")
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

    else:

        raise RuntimeError("Unknown data type.")

    # Masses
    Mgas = (np.pi * (ri[..., 1:]**2 - ri[..., :-1]**2) * SigmaGas[...]).sum(-1)
    Mdust = (np.pi * (ri[..., 1:]**2 - ri[..., :-1]**2)
             * SigmaDust[...].sum(-1)).sum(-1)

    # Interpolation of the density distribution over mass grid
    # via distribution exponent
    Nmi = 8 # default for data.Nmi is 8 in accordance with default sim.ini.grid.Nmbpd in dustpy
    mi = np.ones(int(Nt)) * np.logspace(smin.min(), smax.max(), Nmi)
    mi0 = mi[..., :-1]
    mi1 = mi[..., 1:]
    mic = 0.5 * (mi0[...] + mi1[...])
    SigmaDustTot = SigmaDust[...].sum(-1)
    SigmaDustint = np.ones(int(Nt), int(Nr), int(Nmi-1)) * 1e-100
    for i in range(Nt):
        i = int(i)
        for j in range(Nr):
            j = int(j)
            for k in range(Nmi-1):
                k = int(k)
                if mi1[k] <= smax[i, j]:
                    if xicalc[i, j] != -4.:
                        expo = (xicalc[i, j]+4.) / 3.
                        SigmaDustint[i, j, k] = SigmaDustTot[i, j] * \
                        (mi1[k]**expo - mi0[k]**expo) / \
                        (smax[i, j]**expo - smin[i, j]**expo)
                    else:
                        SigmaDustint[i, j, k] = SigmaDustTot[i, j] * \
                        np.log(mi1[k] / mi0[k]) / \
                        np.log(smax[i, j] / smin[i, j])
    Nmi = np.ones(Nt) * Nmi

    # Transformation of the density distribution
    a = np.array(np.mean(mic[..., 1:] / mic[..., :-1], axis=-1))
    dm = np.array(2. * (a - 1.) / (a + 1.))
    SigmaDusti = SigmaDustint[...] / dm[..., None, None]

    # Calculation of Stokes Number over mass grid
    rho = rhos * fill
    ai = 3 / (4 * np.pi * rho) * mic[...]**(1/3)
    Sti = np.zeros(int(Nt))
    for i in range(int(Nt)):
        Sti[i] = dp.dust_f.st_epstein_stokes1(ai, mfp[i], rho, sim.gas.Sigma[i])

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

    ret["m"] = m
    ret["Nm"] = Nm
    ret["r"] = r
    ret["ri"] = ri
    ret["Nr"] = Nr
    ret["t"] = t
    ret["Nt"] = Nt

    ret["SigmaDust"] = SigmaDust
    ret["SigmaGas"] = SigmaGas
    ret["eps"] = eps

    ret["Mdust"] = Mdust
    ret["Mgas"] = Mgas

    ret["cs"] = cs
    ret["delta"] = delta
    ret["OmegaK"] = OmegaK
    ret["St"] = St
    ret["StDr"] = StDr
    ret["StFr"] = StFr
    ret["vK"] = vK
    ret["vFrag"] = vFrag
    ret["smin"] = smin
    ret["smax"] = smax
    ret["xicalc"] = xicalc

    ret["SigmaDustTot"] = SigmaDustTot
    ret["SigmaDustint"] = SigmaDustint
    ret["SigmaDusti"] = SigmaDusti
    ret["mic"] = mic
    ret["Nmi"] = Nmi
    ret["Sti"] = Sti

    return SimpleNamespace(**ret)
