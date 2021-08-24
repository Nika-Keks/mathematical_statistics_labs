from math import inf, sin, cos, pi, sqrt, atan
from matplotlib.pyplot import show
import numpy as np
from numpy.core.fromnumeric import mean

class Ellipse:
    """
    """

    def __init__(self, mean: tuple = (0., 0.), disp: tuple = (1., 1.), cov: float = 0., C: float = 1.) -> None:
        self.mean = mean
        self.disp = [d * sqrt(C) for d in disp]
        self.cov = cov


    def __h(self) -> float:
        if self.cov == 0.:
            return float("inf")
        h =  (self.disp[1]**2 - self.disp[0]**2) / (self.cov * self.disp[0] * self.disp[1])
        return h


    def __alfa(self) -> list:
        h = self.__h()
        if h == 0.:
            return pi / 4
        if h == float("inf"):
            return 0.
        return atan(2. / self.__h()) / 2


    def __ab(self) -> tuple:
        alfa = self.__alfa()
        s = sin(alfa)
        c = cos(alfa)
        delx = self.disp[0]
        dely = self.disp[1]
        ro = self.cov

        a = 1 / sqrt(c**2 / delx**2  -  2 * ro * s * c / delx / dely  +  s**2 / dely**2)
        
        b = 1 / sqrt(s**2 / delx**2  +  2 * ro * s * c / delx / dely  + c**2 / dely**2)

        return a, b

    def gragh(self):
        t = np.linspace(0., 2 * pi, 100)
        a, b = self.__ab()

        alfa = self.__alfa()

        coord = np.array([a * np.cos(t), b * np.sin(t)])

        Mrot = np.array([[cos(alfa) , -sin(alfa)],[sin(alfa) , cos(alfa)]])  

        coord = np.dot(Mrot, coord)

        shift = np.array([self.mean for i in range(len(t))]).T

        coord = coord + shift

        return coord