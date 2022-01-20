import dustpy as dp


class Simulation(dp.Simulation):

    __name__ = "TwoPopPy"

    def __init__(self, **kwargs):

        super().__init__(self, **kwargs)

        del(self.ini.grid.Nmbpd)
        del(self.ini.grid.mmax)
