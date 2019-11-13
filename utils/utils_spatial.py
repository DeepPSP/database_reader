"""
utilities for spatial objects and computations, which numpy, scipy, etc. lack
"""

import numpy as np
import math
import numpy.linalg as LA
from numpy import pi as PI
# import cmath
from scipy.spatial import ConvexHull, Delaunay
from numbers import Real
from typing import Union, Optional
import warnings

from .common import ArrayLike, modulo


__all__ = [
    "is_in_hull",
    "convex_hull_inflation",
    "get_rot_mat_2d",
    "ConvexCone2D",
    "Fan2D",
]


EPI = PI/180


def is_in_hull(points:ArrayLike, hull:Union[Delaunay,ConvexHull,ArrayLike]) -> np.ndarray:
    """
    test if points in `points` are in `hull`

    Parameters:
    points: array_like,
        an `NxK` coordinates of `N` points in `K` dimensions
    hull: Delaunay object or ConvexHull object, or array_like,
        the objects which defines the hull, essentially an `MxK` array of the coordinates of `M` points in `K`dimensions
    """
    if isinstance(hull, Delaunay):
        _h = hull
    elif isinstance(hull, ConvexHull):
        _h = Delaunay(hull.points)
    else:
        _h = Delaunay(hull)

    return _h.find_simplex(points) >= 0


def convex_hull_inflation(ch:ConvexHull, inflation_ratio:float=0.2, vertices_only:bool=True) -> ConvexHull:
    """

    Parameters:
    -----------

    Returns:
    --------

    """
    ch_vertices = ch.points[ch.vertices]
    center_of_mass = np.mean(ch.points,axis=0)
    ch_vertices = ch_vertices - center_of_mass  # relative coord.
    ch_vertices = ch_vertices * (1+inflation_ratio)
    ch_vertices = ch_vertices + center_of_mass
    if vertices_only:
        inflated_ch = ConvexHull(ch_vertices)
    else:
        inflated_ch = ConvexHull(ch_vertices,incremental=True)
        inflated_ch.add_points(ch.points[is_in_hull(ch.points, inflated_ch)])
        inflated_ch.close()
    return inflated_ch


class ConvexCone2D(object):
    """
    class of convex cone in 2D space
    """
    def __init__(self, apex:ArrayLike, axis_vec:Optional[ArrayLike]=None, angle:Optional[Real]=None, left_vec:Optional[ArrayLike]=None, right_vec:Optional[ArrayLike]=None, **kwargs):
        """ finished, checked,

        Paramters:
        ----------
        apex: array_like,
            point in 2d Cartesian space, apex of the 2D convex cone
        axis_vec: array_like, optional,
            point in 2d Cartesian space, axis vector of the 2D convex cone
        angle: real number, optional,
            angle of the 2D convex cone, should be within (0,180)
        left_vec: array_like,
            vector of the left (compared to the axis vector) border line
        right_vec:
            vector of the right (compared to the axis vector) border line

        NOTE:
        `axis_vec` and `angle`, or `left_vec` and `right_vec` should be specified at the same time

        TODO: add more functions
        """
        self.apex = np.array(apex).flatten()
        if axis_vec is not None and angle is not None:
            self.axis_vec = np.array(axis_vec).flatten()
            if LA.norm(self.axis_vec) == 0:
                raise ValueError("axis vector should have positive length")
            self.axis_vec = self.axis_vec / LA.norm(self.axis_vec)
            if 0<angle<180:
                self.angle = angle * EPI
            else:
                raise ValueError("angle must be within (0,180)")
        elif left_vec is not None and right_vec is not None:
            if LA.norm(left_vec) == 0 or LA.norm(right_vec) == 0:
                raise ValueError("left and right vectors should have positive length")
            self.axis_vec = left_vec / LA.norm(left_vec) + right_vec / LA.norm(right_vec)
            if LA.norm(self.axis_vec) == 0:
                raise ValueError("left and right vectors should not be oppositive to each other")
            self.axis_vec = self.axis_vec / LA.norm(self.axis_vec)
            self.angle = math.atan2(left_vec[1],left_vec[0]) - math.atan2(right_vec[1],right_vec[0])
            if self.angle <= 0 or self.angle >= PI:
                warnings.warn("left, right are switched to get a convex cone")
                self.angle = math.atan2(right_vec[1],right_vec[0]) - math.atan2(left_vec[1],left_vec[0])
        else:
            raise ValueError("please specify `axis_vec` and `angle` at the same time, or specify `left_vec` and `right_vec` at the same time")
        self.left_vec = np.dot(get_rot_mat_2d(self.angle/2), self.axis_vec)
        self.right_vec = np.dot(get_rot_mat_2d(-self.angle/2), self.axis_vec)
        self.left_vec = self.left_vec / LA.norm(self.left_vec)
        self.right_vec = self.right_vec / LA.norm(self.right_vec)


    def to_relative_polar_coord(self, point:ArrayLike) -> np.ndarray:
        """ finished, checked,
        """
        rel_pos = point - self.apex
        r = LA.norm(rel_pos)
        if r == 0:
            rel_polar_coord = np.array([0,0])
            return rel_polar_coord
        theta = math.atan2(rel_pos[1], rel_pos[0]) - math.atan2(self.axis_vec[1], self.axis_vec[0])
        theta = modulo(theta, dividend=2*PI, val_range_start=-PI)
        rel_polar_coord = np.array([r,theta])
        return rel_polar_coord


    def is_inside(self, point:ArrayLike) -> bool:
        """ finished, checked,
        """
        theta = self.to_relative_polar_coord(point)[1]
        return abs(theta) < self.angle/2


    def plot(self, show:bool=False, **kwargs):
        """ not finished,
        """
        import matplotlib.pyplot as plt
        xlim = kwargs.get("xlim", [-1+self.apex[0],1+self.apex[0]])
        ylim = kwargs.get("ylim", [-1+self.apex[1],1+self.apex[1]])
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xlim(xlim[0],xlim[1])
        ax.set_ylim(ylim[0],ylim[1])
        

        if show:
            plt.show()
        else:
            return fig, ax


class Fan2D(ConvexCone2D):
    """
    """
    def __init__(self, radius:Real, apex:ArrayLike, axis_vec:Optional[ArrayLike]=None, angle:Optional[Real]=None, left_vec:Optional[ArrayLike]=None, right_vec:Optional[ArrayLike]=None, **kwargs):
        """
        """
        super().__init__(apex, axis_vec, angle, left_vec, right_vec, **kwargs)
        self.radius = radius
        self.area = None # to compute

    
    def is_inside(self, point:ArrayLike) -> bool:
        """
        """
        dist_to_apex = LA.norm(np.array(point)-self.apex)
        return dist_to_apex < self.radius and super().is_inside(point)


def get_rot_mat_2d(angle:Real) -> np.ndarray:
    """
    """
    rot_mat = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return rot_mat
