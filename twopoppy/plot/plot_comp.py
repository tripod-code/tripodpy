import twopoppy.constants as c
from twopoppy.plot.read_data import _readdata_tpp
from twopoppy.plot.read_data import _readdata_dp

import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display

import ipywidgets as widgets
from scipy.constants import golden


def panel_comp(tpp_data, dp_data, filename="data", extension="hdf5", im=0, ir=0, it=0):
    """Simple plotting script for data files or simulation objects.

    Parameters
    ----------
    tpp_data : ``twopoppy.Simulation`` or string
        Either instance of ``twopoppy.Simulation`` or path to data directory to be plotted
    dp_data : ``dustpy.Simulation`` or string
        Either instance of ``dustpy.Simulation`` or path to data directory to be plotted
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

    data_tpp = _readdata_tpp(tpp_data, filename=filename, extension=extension)
    data_dp = _readdata_dp(dp_data, filename=filename, extension=extension)

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, data_tpp.Nt - 1)
    it = np.minimum(it, data_dp.Nt - 1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data_tpp.Nmi[it, ...] - 1)
    im = np.minimum(im, data_dp.Nm[it, ...] - 1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data_tpp.Nr[it, ...] - 1)
    ir = np.minimum(ir, data_dp.Nr[it, ...] - 1)
    ir = int(ir)

    # Get limits/levels
    sd_max_tpp = np.ceil(np.log10(data_tpp.sigmaDusti.max()))
    sd_max_dp = np.ceil(np.log10(data_dp.sigmaDust.max()))
    sd_max = max(sd_max_tpp, sd_max_dp)
    sg_max_tpp = np.ceil(np.log10(data_tpp.SigmaGas.max()))
    sg_max_dp = np.ceil(np.log10(data_dp.SigmaGas.max()))
    sg_max = max(sg_max_tpp, sg_max_dp)
    Mmax_tpp = np.ceil(np.log10(data_tpp.Mgas.max() / c.M_sun)) + 1
    Mmax_dp = np.ceil(np.log10(data_dp.Mgas.max() / c.M_sun)) + 1
    Mmax = max(Mmax_tpp, Mmax_dp)

    out1, out2, out3, out4 = widgets.Output(), widgets.Output(), widgets.Output(), widgets.Output()

    with out1:
        width = 6.
        height = width / golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax01 = fig.add_subplot(111)

        ax01.loglog(data_tpp.mi[it, ...], data_tpp.sigmaDusti[it, ir, :], c="C3", label="tpp")
        ax01.loglog(data_dp.m[it, ...], data_dp.sigmaDust[it, ir, :], c="C6", label="dp")
        ax01.set_xlim(data_dp.m[it, 0], data_dp.m[it, -1])
        ax01.set_ylim(10. ** (sd_max - 6.), 10. ** sd_max)
        ax01.set_xlabel("Particle mass [g]")
        ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
        ax01.legend()

        fig.tight_layout()
        plt.show()

    with out2:
        width = 6.
        height = width / golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax02 = fig.add_subplot(111)

        if data_dp.Nt < 3:
            ax02.set_xticks([0., 1.])
            ax02.set_yticks([0., 1.])
            ax02.text(0.5,
                      0.5,
                      "Not enough data points.",
                      verticalalignment="center",
                      horizontalalignment="center",
                      size="large")
        else:
            ax02.loglog(data_tpp.t / c.year, data_tpp.Mgas / c.M_sun, c="C0", label="Gas tpp")
            ax02.loglog(data_dp.t / c.year, data_dp.Mgas / c.M_sun, c="C3", label="Gas dp")
            ax02.loglog(data_tpp.t / c.year, data_tpp.Mdust / c.M_sun, c="C1", label="Dust tpp")
            ax02.loglog(data_dp.t / c.year, data_dp.Mdust / c.M_sun, c="C4", label="Dust dp")
            try:
                ax02.loglog(data_tpp.t / c.year, data_tpp.Mplanet / c.M_sun, c="C2", label="Planetesimals tpp")
                ax02.loglog(data_dp.t / c.year, data_dp.Mplanet / c.M_sun, c="C5", label="Planetesimals dp")
            except:
                pass
            ax02.axvline(data_dp.t[it] / c.year, c="#AAAAAA", lw=1, ls="--")
            ax02.set_xlim(data_dp.t[1] / c.year, data_dp.t[-1] / c.year)
            ax02.set_ylim(10. ** (Mmax - 6.), 10. ** Mmax)
            ax02.legend()
        ax02.set_xlabel("Time [yrs]")
        ax02.set_ylabel("Mass [$M_\odot$]")

        fig.tight_layout()
        plt.show()

    with out3:
        width = 6.
        height = width / golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax10 = fig.add_subplot(111)

        ax10.loglog(data_tpp.r[it, ...] / c.au, data_tpp.sigmaDusti[it, :, im], c="C3", label="tpp")
        ax10.loglog(data_dp.r[it, ...] / c.au, data_dp.sigmaDust[it, :, im], c="C6", label="dp")
        ax10.set_xlim(data_dp.r[it, 0] / c.au, data_dp.r[it, -1] / c.au)
        ax10.set_ylim(10. ** (sd_max - 6.), 10. ** sd_max)
        ax10.set_xlabel("Distance from star [au]")
        ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
        ax10.legend()

        fig.tight_layout()
        plt.show()

    with out4:
        width = 6.
        height = width / golden
        fig = plt.figure(dpi=150., figsize=(width, height))
        ax11 = fig.add_subplot(111)
        ax11r = ax11.twinx()

        ax11.loglog(data_tpp.r[it, ...] / c.au, data_tpp.SigmaGas[it, ...], label="Gas tpp")
        ax11.loglog(data_dp.r[it, ...] / c.au, data_dp.SigmaGas[it, ...], label="Gas dp")
        ax11.loglog(data_tpp.r[it, ...] / c.au, data_tpp.SigmaDusti[it, ...].sum(-1), label="Dust tpp")
        ax11.loglog(data_dp.r[it, ...] / c.au, data_dp.SigmaDust[it, ...].sum(-1), label="Dust dp")
        try:
            ax11.loglog(data_tpp.r[it, ...] / c.au, data_tpp.SigmaPlanet[it, ...], label="Planetesimals tpp")
            ax11.loglog(data_dp.r[it, ...] / c.au, data_dp.SigmaPlanet[it, ...], label="Planetesimals dp")
        except:
            pass
        ax11.set_xlim(data_dp.r[it, 0] / c.au, data_dp.r[it, -1] / c.au)
        ax11.set_ylim(10. ** (sg_max - 6), 10. ** sg_max)
        ax11.set_xlabel("Distance from star [AU]")
        ax11.set_ylabel("$\Sigma$ [g/cm²]")
        ax11r.loglog(data_tpp.r[it, ...] / c.au, data_tpp.eps[it, ...], color="C7", lw=1, label="tpp")
        ax11r.loglog(data_dp.r[it, ...] / c.au, data_dp.eps[it, ...], color="C8", lw=1, label="dp")
        ax11r.set_ylim(1.e-5, 1.e1)
        ax11r.set_ylabel("Dust-to-gas ratio")
        ax11.legend()

        fig.tight_layout()
        plt.show()

    tab = widgets.Tab(children=[out4, out2, out1, out3],
                      layout=widgets.Layout(width='100%'))
    tab.set_title(0, 'Surface Densities')
    tab.set_title(1, 'Mass Evolution')
    tab.set_title(2, 'Sigma Dust [ir]')
    tab.set_title(3, 'Sigma Dust [im]')

    display(tab)


def ipanel_comp(tpp_data, dp_data, filename="data", extension="hdf5", im=0, ir=0, it=0):
    """Simple interactive plotting script for data files or simulation objects.

    Parameters
    ----------
    tpp_data : ``twopoppy.Simulation`` or string
        Either instance of ``twopoppy.Simulation`` or path to data directory to be plotted
    dp_data : ``dustpy.Simulation`` or string
        Either instance of ``dustpy.Simulation`` or path to data directory to be plotted
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

    data_tpp = _readdata_tpp(tpp_data, filename=filename, extension=extension)
    data_dp = _readdata_dp(dp_data, filename=filename, extension=extension)

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, data_tpp.Nt - 1)
    it = np.minimum(it, data_dp.Nt - 1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data_tpp.Nmi[it, ...] - 1)
    im = np.minimum(im, data_dp.Nm[it, ...] - 1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data_tpp.Nr[it, ...] - 1)
    ir = np.minimum(ir, data_dp.Nr[it, ...] - 1)
    ir = int(ir)

    # Get limits/levels
    sd_max_tpp = np.ceil(np.log10(data_tpp.sigmaDusti.max()))
    sd_max_dp = np.ceil(np.log10(data_dp.sigmaDust.max()))
    sd_max = max(sd_max_tpp, sd_max_dp)
    sg_max_tpp = np.ceil(np.log10(data_tpp.SigmaGas.max()))
    sg_max_dp = np.ceil(np.log10(data_dp.SigmaGas.max()))
    sg_max = max(sg_max_tpp, sg_max_dp)
    Mmax_tpp = np.ceil(np.log10(data_tpp.Mgas.max() / c.M_sun)) + 1
    Mmax_dp = np.ceil(np.log10(data_dp.Mgas.max() / c.M_sun)) + 1
    Mmax = max(Mmax_tpp, Mmax_dp)

    out1, out2, out3, out4 = widgets.Output(), widgets.Output(), widgets.Output(), widgets.Output()

    play = widgets.Play(value=0, min=0, max=int(data_dp.Nt - 1), step=1, interval=2000, description="Press play",
                        disabled=False)
    ui_temp = widgets.IntSlider(description='Temporal index', value=0, min=0, max=int(data_dp.Nt - 1),
                                continuous_update=False)
    widgets.jslink((play, 'value'), (ui_temp, 'value'))
    display(widgets.HBox([play, ui_temp], layout=widgets.Layout(width='100%')))
    ui_rad = widgets.IntSlider(description='Radial index', value=0, min=0, max=int(data_dp.Nr[0, ...] - 1),
                               continuous_update=False)
    ui_mass = widgets.IntSlider(description='Mass index', value=0, min=0, max=int(data_dp.Nm[0, ...] - 2),
                                continuous_update=False)
    display(widgets.HBox([ui_rad, ui_mass], layout=widgets.Layout(width='100%')))

    with out1:
        def plot1(it, ir, im):
            width = 6.
            height = width / golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax01 = fig.add_subplot(111)

            ax01.loglog(data_tpp.mi[it, ...], data_tpp.sigmaDusti[it, ir, :], c="C3", label="tpp")
            ax01.loglog(data_dp.m[it, ...], data_dp.sigmaDust[it, ir, :], c="C6", label="dp")
            ax01.axvline(data_dp.m[it, im], color="#AAAAAA", lw=1, ls="--")
            ax01.set_xlim(data_dp.m[it, 0], data_dp.m[it, -1])
            ylim1 = np.ceil(np.log10(np.max(data_dp.sigmaDust[it, ir, :])))
            ylim0 = ylim1 - 6.
            ax01.set_ylim(10. ** ylim0, 10. ** ylim1)
            ax01.set_xlabel("Particle mass [g]")
            ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
            ax01.legend()

            fig.tight_layout()
            plt.show()

        out = widgets.interactive_output(plot1, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass})
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out2:
        def plot2(it, ir, im):

            width = 6.
            height = width / golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax02 = fig.add_subplot(111)

            if data_dp.Nt < 3:
                ax02.set_xticks([0., 1.])
                ax02.set_yticks([0., 1.])
                ax02.text(0.5,
                          0.5,
                          "Not enough data points.",
                          verticalalignment="center",
                          horizontalalignment="center",
                          size="large")
            else:
                ax02.loglog(data_tpp.t / c.year, data_tpp.Mgas / c.M_sun, c="C0", label="Gas tpp")
                ax02.loglog(data_dp.t / c.year, data_dp.Mgas / c.M_sun, c="C3", label="Gas dp")
                ax02.loglog(data_tpp.t / c.year, data_tpp.Mdust / c.M_sun, c="C1", label="Dust tpp")
                ax02.loglog(data_dp.t / c.year, data_dp.Mdust / c.M_sun, c="C4", label="Dust dp")
                try:
                    ax02.loglog(data_tpp.t / c.year, data_tpp.Mplanet / c.M_sun, c="C2", label="Planetesimals tpp")
                    ax02.loglog(data_dp.t / c.year, data_dp.Mplanet / c.M_sun, c="C5", label="Planetesimals dp")
                except:
                    pass
                ax02.axvline(data_dp.t[it] / c.year, c="#AAAAAA", lw=1, ls="--")
                ax02.set_xlim(data_dp.t[1] / c.year, data_dp.t[-1] / c.year)
                ax02.set_ylim(10. ** (Mmax - 6.), 10. ** Mmax)
                ax02.legend()
            ax02.set_xlabel("Time [yrs]")
            ax02.set_ylabel("Mass [$M_\odot$]")

            fig.tight_layout()
            plt.show()

        out = widgets.interactive_output(plot2, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass})
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out3:
        def plot3(it, ir, im):
            width = 6.
            height = width / golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax10 = fig.add_subplot(111)

            ax10.loglog(data_tpp.r[it, ...] / c.au, data_tpp.sigmaDusti[it, :, im], c="C3", label="tpp")
            ax10.loglog(data_dp.r[it, ...] / c.au, data_dp.sigmaDust[it, :, im], c="C6", label="dp")
            ax10.axvline(data_dp.r[it, ir] / c.au, color="#AAAAAA", lw=1, ls="--")
            ax10.set_xlim(data_dp.r[it, 0] / c.au, data_dp.r[it, -1] / c.au)
            ylim1 = np.ceil(np.log10(np.max(data_dp.sigmaDust[it, :, im])))
            ylim0 = ylim1 - 6.
            ax10.set_ylim(10. ** ylim0, 10. ** ylim1)
            ax10.set_xlabel("Distance from star [au]")
            ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
            ax10.legend()

            fig.tight_layout()
            plt.show()

        out = widgets.interactive_output(plot3, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass})
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    with out4:
        def plot4(it, ir, im):

            width = 6.
            height = width / golden
            fig = plt.figure(dpi=150., figsize=(width, height))
            ax11 = fig.add_subplot(111)
            ax11r = ax11.twinx()

            ax11.loglog(data_tpp.r[it, ...] / c.au, data_tpp.SigmaGas[it, ...], label="Gas tpp")
            ax11.loglog(data_dp.r[it, ...] / c.au, data_dp.SigmaGas[it, ...], label="Gas dp")
            ax11.loglog(data_tpp.r[it, ...] / c.au, data_tpp.SigmaDusti[it, ...].sum(-1), label="Dust tpp")
            ax11.loglog(data_dp.r[it, ...] / c.au, data_dp.SigmaDust[it, ...].sum(-1), label="Dust dp")
            try:
                ax11.loglog(data_tpp.r[it, ...] / c.au, data_tpp.SigmaPlanet[it, ...], label="Planetesimals tpp")
                ax11.loglog(data_dp.r[it, ...] / c.au, data_dp.SigmaPlanet[it, ...], label="Planetesimals dp")
            except:
                pass
            ax11.axvline(data_dp.r[it, ir] / c.au, color="#AAAAAA", lw=1, ls="--")
            ax11.set_xlim(data_dp.r[it, 0] / c.au, data_dp.r[it, -1] / c.au)
            ax11.set_ylim(10. ** (sg_max - 6), 10. ** sg_max)
            ax11.set_xlabel("Distance from star [AU]")
            ax11.set_ylabel("$\Sigma$ [g/cm²]")
            ax11r.loglog(data_tpp.r[it, ...] / c.au, data_tpp.eps[it, ...], color="C7", lw=1, label="tpp")
            ax11r.loglog(data_dp.r[it, ...] / c.au, data_dp.eps[it, ...], color="C8", lw=1, label="dp")
            ax11r.set_ylim(1.e-5, 1.e1)
            ax11r.set_ylabel("Dust-to-gas ratio")
            ax11.legend()

            fig.tight_layout()
            plt.show()

        out = widgets.interactive_output(plot4, {'it': ui_temp, 'ir': ui_rad, 'im': ui_mass})
        display(widgets.VBox([out], layout=widgets.Layout(width='100%')))

    tab = widgets.Tab(children=[out4, out2, out1, out3],
                      layout=widgets.Layout(width='100%'))
    tab.set_title(0, 'Surface Densities')
    tab.set_title(1, 'Mass Evolution')
    tab.set_title(2, 'Sigma Dust [ir]')
    tab.set_title(3, 'Sigma Dust [im]')

    display(tab)
