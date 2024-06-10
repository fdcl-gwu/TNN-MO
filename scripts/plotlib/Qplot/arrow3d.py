import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform

class Arrow3Drelative(FancyArrowPatch):

    def __init__(self, p, q, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz1 = (p[0], p[1], p[2]) #starting point
        self._dxdydz = (q[0], q[1], q[2]) #ending point

    def draw(self, renderer):
        x1, y1, z1 = self._xyz1
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz1
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
        
def arrow3Dreletivefunc(ax, p, q, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3Drelative(p, q, *args, **kwargs)
    ax.add_artist(arrow)
        
class Arrow3D(FancyArrowPatch):

    def __init__(self, p, q, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz1 = (p[0], p[1], p[2]) #starting point
        self._dxdydz = (q[0], q[1], q[2]) #ending point

    def draw(self, renderer):
        x1, y1, z1 = self._xyz1
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (dx,  dy,  dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz1
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = ( dx,  dy, dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


        
def arrow3Dfunc(ax, p, q, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(p, q, *args, **kwargs)
    ax.add_artist(arrow)
    
   

