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
from typing import Union, Optional, NoReturn
import warnings

from .commom import ArrayLike
from .commom import modulo


__all__ = [
    "is_in_hull",
    "convex_hull_inflation",
    "get_rot_mat_2d",
    "get_line_2d",
    "is_pass_through_2d",
    "get_perpendicular_line_2d",
    "get_line_intersection_2d",
    "affine_trans_2d",
    "vec2rad",
    "vec2polar",
    "rearrange_vectors_2d",
    "rearrange_convex_contour_points_2d",
    "split_2d_plane_into_convex_cones",
    "ConvexCone2D",
    "Fan2D",
    "Ellipse",
]


def is_in_hull(points:ArrayLike, hull:Union[Delaunay,ConvexHull,ArrayLike]) -> np.ndarray:
    """
    test if points in `points` are in `hull`

    Parameters:
    -----------
    points: array_like,
        an `NxK` coordinates of `N` points in `K` dimensions
    hull: Delaunay object or ConvexHull object, or array_like,
        the objects which defines the hull, essentially an `MxK` array of the coordinates of `M` points in `K`dimensions

    Returns:
    --------
    ndarray of bool
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

    TODO: consider the choice of 'center_of_mass'

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
    """ partly finished, under improving,

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
            angle (in degrees, NOT in radians) of the 2D convex cone,
            should be within (0,180)
        left_vec: array_like,
            vector of the left (compared to the axis vector) border line
        right_vec:
            vector of the right (compared to the axis vector) border line

        NOTE:
        `axis_vec` and `angle`, or `left_vec` and `right_vec` should be specified at the same time

        TODO:
        1. add more functions
        2. when degenerates to a ray, should raise error or just warning?
        """
        self.verbose = kwargs.get("verbose", 0)
        if not self._check_dimensions([apex, axis_vec, left_vec, right_vec]):
            raise ValueError("all points and vectors should be of dimension 2")
        self.apex = np.array(apex).flatten()
        if axis_vec is not None and angle is not None:
            self.axis_vec = np.array(axis_vec).flatten()
            if LA.norm(self.axis_vec) == 0:
                raise ValueError("axis vector should have positive length")
            self.axis_vec = self.axis_vec / LA.norm(self.axis_vec)
            if 0<angle<180:
                self.angle = np.deg2rad(angle)
            else:
                raise ValueError("angle must be within (0,180)")
        elif left_vec is not None and right_vec is not None:
            if LA.norm(left_vec) == 0 or LA.norm(right_vec) == 0:
                raise ValueError("left and right vectors should have positive length")
            self.axis_vec = left_vec / LA.norm(left_vec) + right_vec / LA.norm(right_vec)
            if LA.norm(self.axis_vec) == 0:
                raise ValueError("left and right vectors should not be oppositive to each other")
            self.axis_vec = self.axis_vec / LA.norm(self.axis_vec)
            self.angle = modulo(math.atan2(left_vec[1],left_vec[0]) - math.atan2(right_vec[1],right_vec[0]), dividend=2*PI, val_range_start=0)
            if self.angle >= PI:
                warnings.warn("(names of) left, right vec are switched to get a convex cone")
                self.angle = 2*PI - self.angle
            if self.angle == 0:
                raise ValueError("given `left_vec` and `right_vec`, the convex cone degenerates to a ray")
        else:
            raise ValueError("please specify `axis_vec` and `angle` at the same time, or specify `left_vec` and `right_vec` at the same time")
        self.left_vec = np.dot(get_rot_mat_2d(self.angle/2), self.axis_vec)
        self.right_vec = np.dot(get_rot_mat_2d(-self.angle/2), self.axis_vec)
        self.left_vec = self.left_vec / LA.norm(self.left_vec)
        self.right_vec = self.right_vec / LA.norm(self.right_vec)


    def to_relative_polar_coord(self, point:ArrayLike) -> np.ndarray:
        """ finished, checked,

        Parameters:
        -----------
        point: array_like,
            a point in the 2d cartesian space

        Returns:
        --------
        ndarray,
            polar coordinates of `point`
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


    def is_inside(self, point:ArrayLike, strict:bool=False) -> bool:
        """ finished, checked,
        """
        theta = self.to_relative_polar_coord(point)[1]
        if strict:
            return abs(theta) < self.angle/2
        else:
            return abs(theta) <= self.angle/2


    def plot_cone(self, show:bool=False, kw_ray:Optional[dict]=None, kw_fill:Optional[dict]=None, **kwargs) -> Union[tuple, NoReturn]:
        """ finished, checked, still has some errors

        TODO: 
        1. correct the errors when `xlim` and `ylim` (when given in kwargs) are not so appropriate (sovled!)
        2. (?absurd) support the case where the box formed by `xlim` and `ylim` (when given in kwargs) do not include the `self.apex`

        Parameters:
        -----------
        show: bool, default False,
            to show the plot or return it
        kw_ray: dict, optional,
            settings (params for `ax.plot`) for plotting the two rays of the cone,
        kw_fill: dict, optional,
            settings (params for `ax.fill`) for plotting the area of the cone
        kwargs: dict, optional,
            other key word arguments that can be passed,
            including "fig", "ax", "xlim", "ylim"

        Returns:
        --------
        tuple, or None,
            if `show` is set True, then none will be returned,
            otherwise (fig, ax) is returned
        """
        import matplotlib.pyplot as plt
        ax = kwargs.get("ax", None)
        fig = kwargs.get("fig", None)
        figsize = kwargs.get("figsize", (8,8))
        xlim = sorted(kwargs.get("xlim", [-1+self.apex[0],1+self.apex[0]]))
        ylim = sorted(kwargs.get("ylim", [-1+self.apex[1],1+self.apex[1]]))
        title = kwargs.get("title", "2D Convex Cone")
        fontsize = kwargs.get("fontsize", 20)

        if ax is None or fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(xlim[0],xlim[1])
        ax.set_ylim(ylim[0],ylim[1])
        ax.set_title(title, fontsize=fontsize)

        step_len = min(xlim[1]-xlim[0],ylim[1]-ylim[0])/1000
        x,y = self.apex
        left_ray, right_ray = [[x,y]], [[x,y]]

        step = 1
        while xlim[0]<x<xlim[1] and ylim[0]<y<ylim[1]:
            x,y = self.apex + step*step_len*self.left_vec
            left_ray.append([x,y])
            step += 1

        x,y = self.apex
        step = 1
        while xlim[0]<x<xlim[1] and ylim[0]<y<ylim[1]:
            x,y = self.apex + step*step_len*self.right_vec
            right_ray.append([x,y])
            step += 1
        
        left_ray = np.array(left_ray)
        right_ray = np.array(right_ray)

        kw_ray = kw_ray or {"color": "black"}
        ray_color = kw_ray.get("color", None) or kw_ray.get("c", "black")
        ax.plot(left_ray[:,0], left_ray[:,1], c=ray_color)
        ax.plot(right_ray[:,0], right_ray[:,1], c=ray_color)
        
        left_pt = left_ray[-1]
        right_pt = right_ray[-1]
        corners = [[xlim[0],ylim[0]], [xlim[0],ylim[1]], [xlim[1],ylim[1]], [xlim[1],ylim[0]]]
        border_line = np.array([left_pt] + [p for p in corners if self.is_inside(p)] + [right_pt])

        contour = left_ray.tolist() + border_line.tolist() + right_ray.tolist()
        contour = rearrange_convex_contour_points_2d(contour)

        # fill_x = left_ray[:,0].tolist()+border_line[:,0].tolist()+right_ray[:,0].tolist()
        # fill_y = left_ray[:,1].tolist()+border_line[:,1].tolist()+right_ray[:,1].tolist()
        fill_x = contour[:,0].tolist()
        fill_y = contour[:,1].tolist()
        kw_fill = kw_fill or {"color": "cyan", "alpha": 0.5}
        fill_color = kw_fill.get("color", None) or kw_fill.get("c", "cyan")
        fill_alpha = kw_fill.get("alpha", 0.5)
        
        ax.fill(fill_x, fill_y, c=fill_color, alpha=fill_alpha)

        if show:
            plt.show()
        else:
            return fig, ax

    @classmethod
    def _check_dimensions(cls, points_and_vectors:List[ArrayLike], in_details:bool=False) -> Union[bool, np.ndarray]:
        """
        """
        is_valid = np.array([np.array(item).flatten().shape[0]==2 for item in points_and_vectors if item is not None])
        if in_details:
            return is_valid
        else:
            return is_valid.all()


class Fan2D(ConvexCone2D):
    """ not finished,
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


class Ellipse(object):
    """ not finished,
    seems `sympy` has implemented a class named `Ellipse`
    """
    def __init__(self, center:ArrayLike, axis_vecs:ArrayLike):
        """ not finished,

        """
        pass


def split_2d_plane_into_convex_cones(center:ArrayLike, split_vecs:ArrayLike, **kwargs) -> List[ConvexCone2D]:
    """ finished, checked,

    split the 2d real cartesian space into convex cones

    Parameters:
    -----------
    center: array_like,
        point in 2d space, center of the splitting
    split_vecs: array_like,
        the bound vectors of the convex cones to be constructed
    kwargs: dict,
        other key word arguments, including
        "verbose", and "xlim", "ylim", etc. for plotting the convex cones

    Returns:
    --------
    list of `ConvexCone2D`
    """
    verbose = kwargs.get("verbose", 0)

    if len(split_vecs) <= 2:
        raise ValueError("2d plane cannot be splitted into convex cones by less than 3 vectors")

    svs = rearrange_vectors_2d(split_vecs).tolist()
    svs_radians = [vec2rad(item) for item in svs]
    if (np.diff(svs_radians) >= PI).any() or svs_radians[0]-(svs_radians[-1]-2*PI) >= PI:
        raise ValueError("given the provided `axis_vecs`, concave cone will occur")

    if verbose >= 1:
        print("after rearranging,")
        print("split vectors = {}\ntheir radians = {}".format(svs, svs_radians))

    svs += [svs[0]]

    if verbose >= 1:
        print("after augmentation,")
        print("split vectors = {}".format(svs))

    convex_cones = []
    for idx, s in enumerate(svs[:-1]):
        right_vec = s
        left_vec = svs[idx+1]
        cc = ConvexCone2D(
            apex=center,
            left_vec=left_vec,
            right_vec=right_vec,
            **kwargs
        )
        convex_cones.append(cc)

    if verbose >= 2:
        import matplotlib.pyplot as plt
        figsize = kwargs.get("figsize", (8,8))
        kw = {
            "xlim": kwargs.get("xlim", None),
            "ylim": kwargs.get("ylim", None),
            "title": "2D space\nsplit by {} convex cones".format(len(convex_cones)),
        }
        kw = {k:v for k,v in kw.items() if v is not None}
        fig, ax = plt.subplots(figsize=figsize)
        cm = kwargs.get("cmap", plt.get_cmap("gist_rainbow"))
        for idx, cc in enumerate(convex_cones):
            cc_color = cm(idx/len(convex_cones))
            fig, ax = cc.plot_cone(
                show=False,
                kw_fill={"color":cc_color},
                fig=fig,
                ax=ax,
                **kw
            )
        plt.show()

    return convex_cones


def get_rot_mat_2d(angle:Real) -> np.ndarray:
    """
    angle in radians
    """
    rot_mat = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return rot_mat


def get_line_2d(point1:ArrayLike, point2:ArrayLike) -> np.ndarray:
    """

    get the line passing through `point1` and `point2`

    Returns:
    --------
    line, ndarray,
        the planar function of the line is line[0]*x+line[1]*y+line[2]=0
    """
    p1, p2 = np.array(point1), np.array(point2)
    if (p1==p2).all():
        raise ValueError("`point1` and `point2` should not be identical")
    x1, y1 = p1
    x2, y2 = p2
    line = np.array([y2-y1, x1-x2, y1*x2-x1*y2])
    return line


def is_pass_through_2d(line:np.ndarray, point:ArrayLike) -> np.ndarray:
    """
    """
    return np.dot(line, np.array(point).tolist()+[1]) == 0


def get_perpendicular_line_2d(line:np.ndarray, point:ArrayLike) -> np.ndarray:
    """

    get the line passing through `point` and being perpendicular to `line`

    Parameters:
    -----------
    line: ndarray,
        of the form np.array([a,b,c]), given by ax+by+c=0
    point: array_like,
        a point in the 2d cartesian space
    
    Returns:
    --------
    perp_line: ndarray,
        the line passing through `point` and being perpendicular to `line`
    """
    px,py = np.array(point)
    a,b,_ = line
    return np.array([-b, a, b*px-a*py])
    

def get_line_intersection_2d(line1:np.ndarray, line2:np.ndarray) -> np.ndarray:
    """
    """
    system = np.array([line1, line2])
    A = system[:,:2]
    b = -system[:,2]
    try:
        itc_pt = LA.solve(A, b)
    except Exception:
        raise ValueError("`line1` and `line2` should not be parallel")
    return itc_pt


def affine_trans_2d(points:np.ndarray, shift:np.ndarray, rotation:Real) -> np.ndarray:
    """
    rotation in radians
    """
    rot_mat = get_rot_mat_2d(rotation)
    transformed = np.apply_along_axis(
        func1d=lambda v:np.dot(rot_mat, v),
        axis=-1,
        arr=points
    )
    transformed = transformed + shift
    return transformed


def vec2rad(vec:ArrayLike, val_start_from:Real=0) -> float:
    """
    """
    _v = np.array(vec).flatten()
    if len(_v) != 2:
        raise ValueError("`vec` should be a point in 2d space")
    if LA.norm(_v) == 0:
        raise ValueError("`vec` should have positive length")
    rad = modulo(
        val=math.atan2(_v[1],_v[0]),
        dividend=2*PI,
        val_range_start=val_start_from
    )
    return rad


def vec2polar(vec:ArrayLike) -> np.ndarray:
    """
    """
    _v = np.array(vec).flatten()
    if len(_v) != 2:
        raise ValueError("`vec` should be a point in 2d space")
    r = LA.norm(_v)
    if r == 0:
        raise ValueError("`vec` should have positive length")
    phi = modulo(
        val=math.atan2(_v[1],_v[0]),
        dividend=2*PI
    )
    polar = np.array([r,phi])
    return polar


def rearrange_vectors_2d(vectors:ArrayLike) -> np.ndarray:
    """ rearrange 2d vectors anticlockwise
    """
    vec_radians = [vec2rad(item) for item in vectors]
    return np.array(vectors)[np.argsort(vec_radians)]


def rearrange_convex_contour_points_2d(points:ArrayLike) -> np.ndarray:
    """
    
    rearrange points in a in 2d convex contour anticlockwise

    NOTE: whether or not the points in `points` form a convex contour is NOT checked

    Parameters:
    -----------
    points: array_like,
        array of points in a 2d convex contour
    """
    _p = np.array(points)
    center_of_mass = np.mean(_p, axis=0)
    _p = _p - center_of_mass
    vec_radians = [vec2rad(item) for item in _p]
    return np.array(points)[np.argsort(vec_radians)]
