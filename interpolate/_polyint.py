"""Interpolation functions using polynomials."""

import numpy as np


__all__ = ["local_interpolate"]


def local_interpolate(x, y, xi, axis=0, order=3, default_value=np.nan,
                      extrapolate=True, assume_clean=True):
    """
    Local interpolation of a set of points using Lagrange polynomials.
    
    The polynomials are generated from the `order+1` points in `x` that are
    closest to each `xi`. Compared to other interpolation methods, this
    function is particularly well-suited for few query points `xi` and large
    2-D arrays `y`, with `y.shape[1-axis]` larger than ~1000.
    
    Parameters
    ----------
    x : array_like
        One-dimensional array containing values of the independent variable.
        Finite values must be unique.
    y : array_like
        One or two-dimensional array containing values of the dependent
        variable. Values must be finite.
    xi : array_like
        Point or points at which to interpolate the function.
    axis : int, optional
        Axis in the y-array corresponding to the x-coordinate values.
    order : int, optional
        Order of the interpolating polynomials. Default is 3.
    default_value : float, optional
        Default value to be returned if the interpolation fails. Default is
        NaN.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points, or to return the
        default value. Default is True.
    assume_clean : bool, optional
        If False, all input arrays are cleaned according to rules above.
        Default is True.
        
    Returns
    -------
    yi : array_like
        Values of the interpolation at `xi`.
        
    Note
	----
	Due to the Runge phenomenon, the polynomial interpolation is a very
    ill-conditioned process. Therefore, small values of `order` should always
    be passed to this function.
        
    Example
    -------
    >>> from scipy.interpolate import interp1d, barycentric_interpolate
    >>> from gradiompy.interpolate import local_interpolate
    >>> import timeit

    >>> def example_fun(t, x):
    >>>     t = t[:, None]
    >>>     x = x[None, :]
    >>>     return np.cos(5*x-7*t) - np.exp(-50*(x-t)**2)
    >>> t_interp = np.random.random(10)
    >>> t = np.linspace(0, 1, 80)
    >>> x = np.linspace(0, 1, 10000)
    >>> f = example_fun(t, x)

    >>> # Compare 3 interpolation methods
    >>> expressions = ["local_interpolate(t, f, t_interp, order=3)",
    >>>                "barycentric_interpolate(t, f, t_interp)",
    >>>                "interp1d(t, f, axis=0, kind='cubic')(t_interp)"]
    >>> comments = ["", " (Unstable results due to the Runge phenomenon)",""]
    >>> for expression, comment in zip(expressions, comments):
    >>>     f_interp = eval(expression)
    >>>     T = timeit.timeit(expression, number=100, globals=globals())
    >>>     E = np.linalg.norm(f_interp-example_fun(t_interp, x))
    >>>     print('%25s: time = %.3f, error = %.4f%s' %
    >>>           (expression[:str.find(expression,'(')], T, E, comment))
    
    """
    
    # Clean input values
    x = np.asarray(x)
    y = np.asarray(y)
    if axis == 1: y = y.T
    if not assume_clean:
        if y.ndim == 1:
            ind = ~np.isfinite(y)
        else:
            ind = np.any(~np.isfinite(y), axis=1)
        x = x.copy()
        x[ind] = np.inf
        _, ind = np.unique(x, return_index=True)
        ind = np.setdiff1d(np.arange(len(x)), ind, assume_unique=True)
        x[ind] = np.inf
        
    # Interpolation
    if not np.isscalar(xi):
        yi = np.array([local_interpolate(x, y, cur_xi, axis=0, order=order,
            default_value=default_value, extrapolate=extrapolate,
            assume_clean=True) for cur_xi in xi])
    else:
        ind = np.isfinite(x)
        order = np.minimum(order, np.sum(ind)-1)
        if (order == -1) or (not extrapolate and (
                xi < np.min(x[ind]) or xi > np.max(x[ind]))):
            if np.isscalar(default_value):
                yi = np.full_like(y[0], default_value)
            else:
                yi = default_value
        elif order == 0: # Constant
            yi = y[0]
        elif order == 1: # Linear
            i, j = np.argsort(np.abs(x-xi))[:2]
            yi = (y[i] * ((xi-x[j])/(x[i]-x[j])) + 
                  y[j] * ((xi-x[i])/(x[j]-x[i])))
        elif order == 2: # Quadratic
            i, j, k = np.argsort(np.abs(x-xi))[:3]
            yi = (y[i] * ((xi-x[j])*(xi-x[k])/((x[i]-x[j])*(x[i]-x[k]))) +
                  y[j] * ((xi-x[k])*(xi-x[i])/((x[j]-x[k])*(x[j]-x[i]))) +
                  y[k] * ((xi-x[i])*(xi-x[j])/((x[k]-x[i])*(x[k]-x[j]))))
        elif order == 3: # Cubic
            i, j, k, l = np.argsort(np.abs(x-xi))[:4]
            yi = (y[i] * ((xi-x[j])*(xi-x[k])*(xi-x[l])/((x[i]-x[j])*(x[i]-x[k])*(x[i]-x[l]))) +
                  y[j] * ((xi-x[k])*(xi-x[l])*(xi-x[i])/((x[j]-x[k])*(x[j]-x[l])*(x[j]-x[i]))) +
                  y[k] * ((xi-x[l])*(xi-x[i])*(xi-x[j])/((x[k]-x[l])*(x[k]-x[i])*(x[k]-x[j]))) +
                  y[l] * ((xi-x[i])*(xi-x[j])*(xi-x[k])/((x[l]-x[i])*(x[l]-x[j])*(x[l]-x[k]))))
        else:
            ind = np.argsort(np.abs(x-xi))[:order+1]
            yi = 0.
            for count, cur_ind in enumerate(ind):
                cur_ind = ind[count]
                rem_ind = np.delete(ind, count)
                yi += y[cur_ind] * (np.prod(xi-x[rem_ind])/np.prod(x[cur_ind]-x[rem_ind]))
    
    if axis == 1: yi = yi.T
    return yi