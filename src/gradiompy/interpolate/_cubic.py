"""Interpolation algorithms using piecewise cubic polynomials."""

import numpy as np
from scipy.interpolate import (Akima1DInterpolator, PchipInterpolator,
                               CubicSpline, PPoly)


__all__ = ["DiscontinuousInterpolator"]


class DiscontinuousInterpolator(PPoly):
    """
    Cubic polynomial interpolator handling discontinuities.
    
    Fit cubic polynomials, given vectors `x` and `y`. The interpolation
    handles discontinuities of either the function or its derivative, at
    known positions `x_knots`.
    
    The result is represented as a `PPoly` instance with breakpoints matching
    the given data.
    
    Parameters
    ----------
    x : array_like
        One-dimensional array containing values of the independent variable.
        Values must be unique, real, finite and in strictly increasing order.
    y : array_like
        One-dimensional array containing values of the dependent variable.
        Values must be finite.
    x_knots : array_like, optional
        One-dimensional array containing the position of the derivative
        discontinuities. Values must lie in the open interval (x[0], x[-1]).
        An empty array leads to the usual cubic interpolator.
    y_knots : list, optional
        Values of the function at `x_knots`. Each element of `y_knots` is
        either a list of two elements or a scalar. In the former case, the
        function is discontinuous at the knot. In the latter case, it is
        continuous but its derivative is discontinuous. Non-finite values of
        `y_knots` are extrapolated from both the left and the right sides of
        the function. Default value is an array of NaNs of the same size as
        `x_knots`, leading to a continuous function with derivative
        discontinuities at the knots.
    kind : str, optional
        Specifies the kind of cubic interpolation. The string has to be one of
        'akima', 'cubic', or 'pchip'. Default is 'cubic'.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first and last
        intervals, or to return NaNs. Default is True.
    assume_clean : bool, optional
        If False, all input arrays are cleaned according to rules above.
		Default is True.
        
    See Also
    --------
    scipy.interpolate.Akima1DInterpolator
    scipy.interpolate.PchipInterpolator
    scipy.interpolate.CubicSpline
    scipy.interpolate.PPoly
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from scipy.interpolate import CubicSpline
    >>> from gradiompy.interpolate import DiscontinuousInterpolator
    
    >>> titles = ['Discontinuous function at x = 1.0',
    >>>           'Discontinuous derivative at x = 1.0']
    >>> for count, title in enumerate(titles):
    >>>     x = np.arange(0, 2, 0.15)
    >>>     y = np.piecewise(x, [x < 1, x >= 1],
    >>>             [lambda x: 1-(x-0.5)**2, lambda x: count-(x-1.5)**2])
    >>>     x_knots = [1]
    >>>     y_knots = [[np.nan, np.nan]] if count == 0 else [np.nan]
    >>>     InterpCubic = CubicSpline(x, y)
    >>>     InterpDiscontinuous = DiscontinuousInterpolator(x, y,
    >>>             x_knots=x_knots, y_knots=y_knots)
    >>>     x_interp = np.linspace(0, 2, 999)
    >>>     if count == 0: plt.clf()
    >>>     plt.subplot(1,2,count+1)
    >>>     plt.plot(x,y,'o')
    >>>     plt.plot(x_interp, InterpCubic(x_interp))
    >>>     plt.plot(x_interp, InterpDiscontinuous(x_interp))
    >>>     plt.legend(['Data', 'CubicSpline', 'DiscontinuousInterpolator'])
    >>>     plt.title(title)
    >>> plt.show()
    
    """
    
    def __init__(self, x, y, x_knots=None, y_knots=None, kind='Cubic',
                 extrapolate=True, assume_clean=True):
        if x_knots is None: x_knots = np.empty(0, float)
        if y_knots is None: y_knots = [np.nan]*np.size(x_knots)
                    
        # Clean input arrays
        x = np.asarray(x)
        y = np.asarray(y)
        x_knots = np.asarray(x_knots)
        if np.isscalar(y_knots): y_knots = [y_knots]
        if not assume_clean:
            x, ind = np.unique(x, return_index=True)
            y = y[ind]
            ind = np.isfinite(x) & np.isfinite(y)
            x = x[ind]
            y = y[ind]
            ind = np.isfinite(x_knots)
            x_knots = x_knots[ind]
            y_knots = [y_knots[i] for i in np.nonzero(ind)[0]]
            if x_knots.size > 0:
                ind = (x_knots > x[0]) & (x_knots < x[-1])
                x_knots = x_knots[ind]
                y_knots = [y_knots[i] for i in np.nonzero(ind)[0]]
                
        # Internal interpolator
        interpolators = {'akima':Akima1DInterpolator, 'cubic':CubicSpline, 
                         'pchip':PchipInterpolator}
        _Interpolator = interpolators[kind.lower()]
        
        # Compute the coefficients and breakpoints
        if x_knots.size == 0:
            _PPoly = _Interpolator(x, y)
            _c = _PPoly.c
            _x = _PPoly.x
        else:
            
            # Characterize all pieces of the data
            piece_x_edges = np.stack((
                np.concatenate(([-np.inf], x_knots)),
                np.concatenate((x_knots, [np.inf]))), axis=1)
            piece_y_edges = np.empty_like(piece_x_edges)
            piece_x = []
            piece_y = []
            for piece_ind, x_edges in enumerate(piece_x_edges):
                ind = (x >= x_edges[0]) & (x <= x_edges[1])
                x_cur = x[ind]
                y_cur = y[ind]
                piece_x.append(x_cur)
                piece_y.append(y_cur)
                if np.sum(ind) > 1:
                    _PPoly = _Interpolator(x_cur, y_cur)
                    piece_y_edges[piece_ind] = _PPoly(x_edges)
                elif np.any(ind):
                    piece_y_edges[piece_ind] = np.full(2, y_cur)
                else:
                    raise ValueError("x should contain at least one point "
                                     "between two consecutive knots.")
                    
            # Value of the function at the knots
            for knot_ind, y0 in enumerate(y_knots):
                yL = piece_y_edges[knot_ind, 1]
                yR = piece_y_edges[knot_ind+1, 0]
                if not np.isscalar(y0):
                    if np.isfinite(y0[0]): yL = y0[0]
                    if np.isfinite(y0[1]): yR = y0[1]
                elif not np.isnan(y0):
                    yL = yR = y0
                else:
                    nL = len(piece_x[knot_ind])
                    nR = len(piece_x[knot_ind+1])
                    x0 = x_knots[knot_ind]
                    dxL = x0 - piece_x[knot_ind][-1]
                    dxR = piece_x[knot_ind+1][0] - x0
                    wL = dxR * np.min([nL-1, 4])
                    wR = dxL * np.min([nR-1, 4])
                    if wL+wR > 0:
                        yL = yR = (wL*yL+wR*yR)/(wL+wR)
                    else:
                        yL = yR = (yL+yR)/2
                piece_y_edges[knot_ind, 1] = yL
                piece_y_edges[knot_ind+1, 0] = yR
                    
            # Merge all piecewise interpolations
            _c = np.empty((4,0))
            _x = [piece_x[0][0]]
            for piece_ind in range(len(x_knots)+1):
                x_cur = piece_x[piece_ind]
                y_cur = piece_y[piece_ind]
                x_edges = piece_x_edges[piece_ind]
                y_edges = piece_y_edges[piece_ind]
                if np.isfinite(x_edges[0]) and (x_edges[0] < x_cur[0]):
                    x_cur = np.concatenate(([x_edges[0]], x_cur))
                    y_cur = np.concatenate(([y_edges[0]], y_cur))
                if np.isfinite(x_edges[1]) and (x_edges[1] > x_cur[-1]):
                    x_cur = np.concatenate((x_cur, [x_edges[1]]))
                    y_cur = np.concatenate((y_cur, [y_edges[1]]))
                _PPoly = _Interpolator(x_cur, y_cur)
                _c = np.concatenate((_c, _PPoly.c), axis=1)
                _x = np.concatenate((_x, _PPoly.x[1:]))
        
        # Set PPoly attributes
        self.c = _c
        self.x = _x
        self.axis = 0
        self.extrapolate = extrapolate