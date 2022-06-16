import twopoppy.constants as c
from twopoppy.plot.read_data import _readdata_tpp as _readdata

import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display
import ipywidgets as widgets
from scipy.constants import golden


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
    it = np.minimum(it, data.Nt - 1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data.Nmi[it, ...] - 1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...] - 1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.sigmaDusti.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max() / c.M_sun)) + 1
    levels_num = 7
    levels = np.linspace(sd_max - (levels_num - 1), sd_max, levels_num)

    out1, out2, out3, out4, out5 = widgets.Output(), widgets.Output(), widgets.Output(), \
                                   widgets.Output(), widgets.Output()

    with out1:
        width = 6.
        height = width / golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax00 = fig.add_subplot(111)

        # Density distribution
        plt00 = ax00.contourf(data.r[it, ...] / c.au,
                              data.mi[it, ir, ...],
                              np.log10(data.sigmaDusti[it, ...].T),
                              levels=levels,
                              cmap="magma",
                              extend="both"
                              )
        if show_St1:
            ax00.contour(data.r[it, ...] / c.au,
                         data.mi[it, ir, ...],
                         data.Sti[it, ...].T,
                         levels=[1.],
                         colors="white",
                         linewidths=2
                         )
        if show_limits:
            ax00.contour(data.r[it, ...] / c.au,
                         data.mi[it, ir, ...],
                         (data.Sti - data.StDr[..., None])[it, ...].T,
                         levels=[0.],
                         colors="C2",
                         linewidths=1
                         )
            ax00.contour(data.r[it, ...] / c.au,
                         data.mi[it, ir, ...],
                         (data.Sti - data.StFr[..., None])[it, ...].T,
                         levels=[0.],
                         colors="C0",
                         linewidths=1
                         )
            ax00.contour(data.r[it, ...] / c.au,
                         data.mi[it, ir, ...],
                         (data.Sti - data.StDf[..., None])[it, ...].T,
                         levels=[0.],
                         colors="gray",
                         linewidths=1
                         )
            ax00.contour(data.r[it, ...] / c.au,
                         data.mi[it, ir, ...],
                         (data.Sti - data.StMax[..., None])[it, ...].T,
                         levels=[0.],
                         colors="yellow",
                         linewidths=1
                         )

        ax00.axhline(data.mi[it, ir, im], color="#AAAAAA", lw=1, ls="--")
        ax00.axvline(data.r[it, ir] / c.au, color="#AAAAAA", lw=1, ls="--")

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
        fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.show()

    with out2:
        width = 6.
        height = width / golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax01 = fig.add_subplot(111)

        ax01.loglog(data.mi[it, ir, ...], data.sigmaDusti[it, ir, :], c="C3")
        ax01.set_xlim(data.mi[it, ir, 0], data.mi[it, ir, -1])
        ax01.set_ylim(10. ** (sd_max - 6.), 10. ** sd_max)
        ax01.set_xlabel("Particle mass [g]")
        ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

        fig.tight_layout()
        fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.show()

    with out3:
        width = 6.
        height = width / golden
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
            ax02.loglog(data.t / c.year, data.Mgas / c.M_sun, c="C0", label="Gas")
            ax02.loglog(data.t / c.year, data.Mdust /
                        c.M_sun, c="C1", label="Dust")
            try:
                ax02.loglog(data.t / c.year, data.Mplanet /
                            c.M_sun, c="C2", label="Planetesimals")
            except:
                pass
            ax02.axvline(data.t[it] / c.year, c="#AAAAAA", lw=1, ls="--")
            ax02.set_xlim(data.t[1] / c.year, data.t[-1] / c.year)
            ax02.set_ylim(10. ** (Mmax - 6.), 10. ** Mmax)
            ax02.legend()
        ax02.set_xlabel("Time [yrs]")
        ax02.set_ylabel("Mass [$M_\odot$]")

        fig.tight_layout()
        fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.show()

    with out4:
        width = 6.
        height = width / golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax10 = fig.add_subplot(111)

        ax10.loglog(data.r[it, ...] / c.au, data.sigmaDusti[it, :, im], c="C3")
        ax10.set_xlim(data.r[it, 0] / c.au, data.r[it, -1] / c.au)
        ax10.set_ylim(10. ** (sd_max - 6.), 10. ** sd_max)
        ax10.set_xlabel("Distance from star [au]")
        ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

        fig.tight_layout()
        fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.show()

    with out5:
        width = 6.
        height = width / golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax11 = fig.add_subplot(111)
        ax11r = ax11.twinx()

        ax11.loglog(data.r[it, ...] / c.au, data.SigmaGas[it, ...], label="Gas")
        ax11.loglog(data.r[it, ...] / c.au,
                    data.SigmaDusti[it, ...].sum(-1), label="Dust")
        try:
            ax11.loglog(data.r[it, ...] / c.au,
                        data.SigmaPlanet[it, ...], label="Planetesimals")
        except:
            pass
        ax11.set_xlim(data.r[it, 0] / c.au, data.r[it, -1] / c.au)
        ax11.set_ylim(10. ** (sg_max - 6), 10. ** sg_max)
        ax11.set_xlabel("Distance from star [AU]")
        ax11.set_ylabel("$\Sigma$ [g/cm²]")
        ax11.legend()
        ax11r.loglog(data.r[it, ...] / c.au, data.eps[it, ...], color="C7", lw=1)
        ax11r.set_ylim(1.e-5, 1.e1)
        ax11r.set_ylabel("Dust-to-gas ratio")

        fig.tight_layout()
        fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.show()

    tab = widgets.Tab(children=[out1, out2, out4, out3, out5],
                      layout=widgets.Layout(width='100%'))
    tab.set_title(0, 'Dust Surface Density')
    tab.set_title(1, 'Sigma Dust [ir]')
    tab.set_title(2, 'Sigma Dust [im]')
    tab.set_title(3, 'Mass Evolution')
    tab.set_title(4, 'Surface Densities')
    display(tab)


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
    it = np.minimum(it, data.Nt - 1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data.Nmi[it, ...] - 1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...] - 1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.sigmaDusti.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max() / c.M_sun)) + 1
    levels_num = 7
    levels = np.linspace(sd_max - (levels_num - 1), sd_max, levels_num)

    out1, out2, out3, out4, out5 = widgets.Output(), widgets.Output(), widgets.Output(), \
                                   widgets.Output(), widgets.Output()

    play = widgets.Play(value=0, min=0, max=int(data.Nt - 1), step=1, interval=2000,
                        description="Press play", disabled=False)
    ui_temp = widgets.IntSlider(description='Temporal index', value=0, min=0, max=int(data.Nt - 1),
                                continuous_update=False)
    widgets.jslink((play, 'value'), (ui_temp, 'value'))
    display(widgets.HBox([play, ui_temp], layout=widgets.Layout(width='100%')))
    ui_rad = widgets.IntSlider(description='Radial index', value=0, min=0, max=int(data.Nr[0, ...] - 1),
                               continuous_update=False)
    ui_mass = widgets.IntSlider(description='Mass index', value=0, min=0, max=int(data.Nmi[0, ...] - 2),
                                continuous_update=False)
    display(widgets.HBox([ui_rad, ui_mass], layout=widgets.Layout(width='100%')))

    with out1:
        def plot1(it, ir, im):

            width = 6.
            height = width / golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax00 = fig.add_subplot(111)

            # Density distribution
            plt00 = ax00.contourf(data.r[it, ...] / c.au,
                                  data.mi[it, ir, ...],
                                  np.log10(data.sigmaDusti[it, ...].T),
                                  levels=levels,
                                  cmap="magma",
                                  extend="both"
                                  )
            if show_St1:
                ax00.contour(data.r[it, ...] / c.au,
                             data.mi[it, ir, ...],
                             data.Sti[it, ...].T,
                             levels=[1.],
                             colors="white",
                             linewidths=2
                             )
            if show_limits:
                ax00.contour(data.r[it, ...] / c.au,
                             data.mi[it, ir, ...],
                             (data.Sti - data.StDr[..., None])[it, ...].T,
                             levels=[0.],
                             colors="C2",
                             linewidths=1
                             )
                ax00.contour(data.r[it, ...] / c.au,
                             data.mi[it, ir, ...],
                             (data.Sti - data.StFr[..., None])[it, ...].T,
                             levels=[0.],
                             colors="C0",
                             linewidths=1
                             )
                ax00.contour(data.r[it, ...] / c.au,
                             data.mi[it, ir, ...],
                             (data.Sti - data.StDf[..., None])[it, ...].T,
                             levels=[0.],
                             colors="gray",
                             linewidths=1
                             )
                ax00.contour(data.r[it, ...] / c.au,
                             data.mi[it, ir, ...],
                             (data.Sti - data.StMax[..., None])[it, ...].T,
                             levels=[0.],
                             colors="yellow",
                             linewidths=1
                             )
            ax00.axhline(data.mi[it, ir, im], color="#AAAAAA", lw=1, ls="--")
            ax00.axvline(data.r[it, ir] / c.au, color="#AAAAAA", lw=1, ls="--")

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
            fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
                     verticalalignment="bottom")
            plt.show()

        out = widgets.interactive_output(plot1, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass})
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out2:
        def plot2(it, ir, im):
            width = 6.
            height = width / golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax01 = fig.add_subplot(111)

            ax01.loglog(data.mi[it, ir, ...], data.sigmaDusti[it, ir, :], c="C3")
            ax01.axvline(data.mi[it, ir, im], color="#AAAAAA", lw=1, ls="--")
            ax01.set_xlim(data.mi[it, ir, 0], data.mi[it, ir, -1])
            ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, ir, :])))
            ylim0 = ylim1 - 6.
            ax01.set_ylim(10. ** ylim0, 10. ** ylim1)
            ax01.set_xlabel("Particle mass [g]")
            ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

            fig.tight_layout()
            fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
                     verticalalignment="bottom")
            plt.show()

        out = widgets.interactive_output(plot2, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass})
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out3:
        def plot3(it, ir, im):

            width = 6.
            height = width / golden
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
                ax02.loglog(data.t / c.year, data.Mgas / c.M_sun, c="C0", label="Gas")
                ax02.loglog(data.t / c.year, data.Mdust /
                            c.M_sun, c="C1", label="Dust")
                try:
                    ax02.loglog(data.t / c.year, data.Mplanet / c.M_sun, c="C2", label="Planetesimals")
                except:
                    pass
                plt02vl = ax02.axvline(data.t[it] / c.year, c="#AAAAAA", lw=1, ls="--")
                ax02.set_xlim(data.t[1] / c.year, data.t[-1] / c.year)
                ax02.set_ylim(10. ** (Mmax - 6.), 10. ** Mmax)
                ax02.legend()
            ax02.set_xlabel("Time [yrs]")
            ax02.set_ylabel("Mass [$M_\odot$]")

            fig.tight_layout()
            fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
                     verticalalignment="bottom")
            plt.show()

        out = widgets.interactive_output(plot3, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass})
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out4:
        def plot4(it, ir, im):
            width = 6.
            height = width / golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax10 = fig.add_subplot(111)

            ax10.loglog(data.r[it, ...] / c.au,
                        data.sigmaDusti[it, :, im], c="C3")
            ax10.axvline(data.r[it, ir] / c.au, color="#AAAAAA", lw=1, ls="--")
            ax10.set_xlim(data.r[it, 0] / c.au, data.r[it, -1] / c.au)
            ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, :, im])))
            ylim0 = ylim1 - 6.
            ax10.set_ylim(10. ** ylim0, 10. ** ylim1)
            ax10.set_xlabel("Distance from star [au]")
            ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

            fig.tight_layout()
            fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
                     verticalalignment="bottom")
            plt.show()

        out = widgets.interactive_output(plot4, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass})
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out5:
        def plot5(it, ir, im):

            width = 6.
            height = width / golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax11 = fig.add_subplot(111)
            ax11r = ax11.twinx()

            ax11.loglog(data.r[it, ...] / c.au,
                        data.SigmaGas[it, ...], label="Gas")
            ax11.loglog(data.r[it, ...] / c.au,
                        data.SigmaDusti[it, ...].sum(-1), label="Dust")
            try:
                ax11.loglog(data.r[it, ...] / c.au,
                            data.SigmaPlanet[it, ...], label="Planetesimals")
            except:
                pass
            ax11.axvline(data.r[it, ir] / c.au, color="#AAAAAA", lw=1, ls="--")
            ax11.set_xlim(data.r[it, 0] / c.au, data.r[it, -1] / c.au)
            ax11.set_ylim(10. ** (sg_max - 6), 10. ** sg_max)
            ax11.set_xlabel("Distance from star [AU]")
            ax11.set_ylabel("$\Sigma$ [g/cm²]")
            ax11.legend()
            ax11r.loglog(data.r[it, ...] / c.au,
                         data.eps[it, ...], color="C7", lw=1)
            ax11r.set_ylim(1.e-5, 1.e1)
            ax11r.set_ylabel("Dust-to-gas ratio")

            fig.tight_layout()
            fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
                     verticalalignment="bottom")
            plt.show()

        out = widgets.interactive_output(plot5, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass})
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    tab = widgets.Tab(children=[out1, out2, out4, out3, out5],
                      layout=widgets.Layout(width='100%'))
    tab.set_title(0, 'Dust Surface Density')
    tab.set_title(1, 'Sigma Dust [ir]')
    tab.set_title(2, 'Sigma Dust [im]')
    tab.set_title(3, 'Mass Evolution')
    tab.set_title(4, 'Surface Densities')
    display(tab)
