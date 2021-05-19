import numpy as np
from scipy.signal import convolve


def composite(y, dx=1, order=3):
    """
    Integrate y(x) for regularly spaced x.
    	
    Parameters
    ----------
    y : array_like
        One-dimensional array to integrate.
    dx : float, optional
        Spacing between elements of `y`.
    order : int, optional
        Order of the interpolating polynomials. Default is 3, and other valid
        values are 1 (trapz), 2 (simps), 4, 5, 6 and 7.
    		
    Returns
    -------
    composite : float
        Definite integral as approximated by the interpolating polynomials.
    		
    See Also
    --------
    scipy.integrate.trapezoid, scipy.integrate.simpson
    gradiompy.integrate.cumulative_composite
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import integrate as sp_integrate
    >>> from gradiompy import integrate as gp_integrate

    >>> x = np.linspace(0,3,num=25)
    >>> dx = x[1]-x[0]
    >>> y = np.sin(3.5*x)
    >>> y_int = -np.cos(3.5*x)/3.5
    >>> y_int = y_int[-1]-y_int[0]
    >>> y_int_approx = sp_integrate.trapezoid(y,dx=dx)
    >>> err = abs(y_int_approx-y_int)
    >>> print('Integration error:\n  trapz   = %.3e' % err)
    >>> for order in range(1,8):
    >>>     y_int_approx = gp_integrate.composite(y,dx,order=order)
    >>>     err = abs(y_int_approx-y_int)
    >>>     print('  order %i = %.3e' % (order,err))
    
    """
    # format inputs
    y = np.asarray(y)
    order = max(1,min(7,order))
    
    # integration coefficients
    coef_dic = {
        1: np.array([-1])/2.,
        2: np.array([-15,4,-1])/24.,
        3: np.array([-16,7,-4,1])/24.,
        4: np.array([-965,462,-336,146,-27])/1440.,
        5: np.array([-981,542,-496,306,-107,16])/1440.,
        6: np.array([-84161,55688,-66109,57024,-31523,9976,-1375])/120960.,
        7: np.array([-85376,64193,-91624,99549,-74048,35491,-9880,1215])/120960.}
    c_boundary = coef_dic[order] * dx
    
    # cumpute the integral
    m = len(c_boundary)
    z = np.sum(y) * dx
    dz_ini = c_boundary.dot(y[:+m])
    dz_end = c_boundary.dot(np.flip(y[-m:]))
    return z + dz_ini + dz_end


def cumulative_composite(y, dx=1, initial=[], order=3):
    """
    Cumulatively integrate y(x) for regularly spaced x.
    
    Parameters
    ----------
    y : array_like
        One-dimensional array to integrate.
    dx : float, optional
        Spacing between elements of `y`.
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        Typically this value should be 0. Default is [], which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration.
    order : int, optional
        Order of the interpolating polynomials. Default is 3, and other valid
        values are 1 (cumtrapz), 2 (cumsimps), 4, 5, 6 and 7. Odd values
        should be preferred over even ones.
        
    Returns
    -------
    cumulative_composite : ndarray
        The result of cumulative integration of `y`. If `initial` is empty or
        zero, the last element of this list is equal to the definite integral
        of y(x).
        
        
    See Also
    --------
    numpy.cumsum, numpy.convolve
    scipy.integrate.cumulative_trapezoid
    gradiompy.integrate.composite
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import integrate as sp_integrate
    >>> from gradiompy import integrate as gp_integrate

    >>> x = np.linspace(0,3,num=25)
    >>> dx = x[1]-x[0]
    >>> y = np.sin(3.5*x)
    >>> y_int = -np.cos(3.5*x)/3.5
    >>> y_int_approx = sp_integrate.cumulative_trapezoid(y,dx=dx,initial=0) + y_int[0]
    >>> err = np.linalg.norm(y_int_approx-y_int)
    >>> print('Norm of residuals:\n  cumtrapz = %.3e' % err)
    >>> for order in range(1,8):
    >>>     y_int_approx = gp_integrate.cumulative_composite(y,dx,initial=y_int[0],order=order)
    >>>     err = np.linalg.norm(y_int_approx-y_int)
    >>>     print('  order %i  = %.3e' % (order,err))
    
    """
    # format inputs
    y = np.asarray(y)
    if np.isscalar(initial): initial = [initial]
    order = max(1,min(7,order))
    
    # integration coefficients
    coef_dic = {
        1: (np.array([[]]),
            np.array([1,1])/2.),  
        2: (np.array([[5,8,-1]])/12.,
            np.array([-1,13,13,-1])/24.),
        3: (np.array([[9,19,-5,1]])/24., 
            np.array([-1,13,13,-1])/24.),
        4: (np.array([[251,646,-264,106,-19],
                      [-19,346,456,-74,11]])/720.,
            np.array([11,-93,802,802,-93,11])/1440.),
        5: (np.array([[475,1427,-798,482,-173,27],
                      [-27,637,1022,-258,77,-11]])/1440.,
            np.array([11,-93,802,802,-93,11])/1440.),
        6: (np.array([[19087,65112,-46461,37504,-20211,6312,-863],
                      [-863,25128,46989,-16256,7299,-2088,271],
                      [271,-2760,30819,37504,-6771,1608,-191]])/60480.,
            np.array([-191,1879,-9531,68323,68323,-9531,1879,-191])/120960.),
        7: (np.array([[36799,139849,-121797,123133,-88547,41499,-11351,1375],
                      [-1375,47799,101349,-44797,26883,-11547,2999,-351],
                      [351,-4183,57627,81693,-20227,7227,-1719,191]])/120960.,
            np.array([-191,1879,-9531,68323,68323,-9531,1879,-191])/120960.)}
    coef = coef_dic[order]
    c_boundary = coef[0] * dx
    c_convolve = coef[1] * dx
    
    # compute the list to be accumulated
    m = c_boundary.shape[1]
    if m > 0:
        dy_ini = c_boundary.dot(y[:m])
        dy_end = np.flip(c_boundary.dot(np.flip(y[-m:])))
    else:
        dy_ini = []
        dy_end = []
    dy_mid = convolve(y,c_convolve,mode='valid',method='direct')
    dy = np.concatenate((initial,dy_ini,dy_mid,dy_end))
    
    # return the cumulative integral
    return np.cumsum(dy)