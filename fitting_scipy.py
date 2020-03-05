# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:21:45 2016

@author: gawe
"""
# ======================================================================== #
# ======================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

import matplotlib.pyplot as _plt
import numpy as _np
from pybaseutils import utils as _ut

from scipy import interpolate as _int
from scipy import ndimage as _ndimage
from scipy.optimize import curve_fit, leastsq


try:
    from. import model_spec as _ms
    from .fitNL import fitNL_base
except:
    from FIT import model_spec as _ms
    from FIT.fitNL import fitNL_base
# end try

# ==== #

# There are annoying differences in the context between scipy version of
# leastsq and curve_fit, and the method least_squares doesn't exist before 0.17
import scipy.version as _scipyversion

# Make a version flag for switching between least squares solvers and contexts
_scipyversion = _scipyversion.version
_scipyversion = _np.float(_scipyversion[0:4])
if _scipyversion >= 0.17:
#    print("Using a new version of scipy")
    from scipy.optimize import least_squares
#else:
#    print("Using an older version of scipy")
# endif

__metaclass__ = type

# ======================================================================== #
# ======================================================================== #


def qparab_fit(XX, *aa, **kwargs):
    return _ms.qparab(XX, *aa, **kwargs)

def dqparabdx(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
    return _ms.deriv_qparab(XX, aa, nohollow)

def dqparabda(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
    return _ms.partial_qparab(XX, aa, nohollow)

# ======================================================================== #


def deriv_bsgaussian(xvar, u, varu, axis=0, nmonti=300, sigma=1, mode='nearest', derivorder=1):
    """
    function [dudx, vardudx] = deriv_bsgaussian(xvar, u, varu, axis=1,
                           nmonti=300, derivorder=1, sigma=1, mode='nearest')

    Calculates the derivative of the input array along the specified axis by
    convolving the input array with the derivative of a Gaussian.

    Inputs:
        xvar - dependent variable
        u    - independent variable
        varu - variance in u
        axis - Axis along which to take the derivative default: axis=1
        nmonti - number of Monte Carlo iterations to use for propagation of
                 uncertainties.  default: nmonti=300
        sigma - Breadth of the Gaussian kernel default: sigma=1
        mode - Extrapolation method at boundary when the Guassian kernel
               passes the convex hull of the data.  default: mode='nearest'
        derivorder - order of the derivative.  default: derivorder=1
            derivorder = 0 - No derivative!  Just a convolution with a gaussian (smoothing)
            derivorder = 1 - First derivative!  Convolution with derivative of a gaussian
            derivorder = 2 or 3 ... second and third derivatives (higher order not supported)
    Outputs
        dudx - derivative of u wrt x of order 'derivorder',
                    derivorder = 2, outputs d2udx2
        vardudx - Estimate of variance in derivative


    The result includes some smoothing, and also has boundary effects where
    the Gaussian touches the boundaries.  The boundary effects are controlled
    by the filter option 'mode'.  Below is a table showing how Gaussian
    filter handles the boundary points.

    From stackoverflow:
    mode       |   Ext   |         Input          |   Ext
    -----------+---------+------------------------+---------
    'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
    'reflect'  | 3  2  1 | 1  2  3  4  5  6  7  8 | 8  7  6
    'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
    'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
    'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3


    derivorder - Order of the derivative to be taken (1st, 2nd, or 3rd)
                 default: derivorder=1 - first derivative = du/dx
                 ...Higher order derivatives are not yet implemented...
    """

    #============================================================== #
    # Input data formatting.  All of this is undone before output
    # xvar, _, _ = _ms._derivative_inputcondition(xvar)

    u, ush, transp = _ms._derivative_inputcondition(u)
    varu, _, _ = _ms._derivative_inputcondition(varu)
    if transp:
        # xvar = xvar.T
        u = u.T
        varu = varu.T
    # endif
    nsh = _np.shape(u)

#    if _np.shape(xvar) != _np.shape(u):
#        xvar = _np.tile(xvar, (nsh[1],))
#    # endif

    if (nsh[0] == 1):
        # xvar = xvar.reshape(nsh[1], nsh[0])
        u = u.reshape(nsh[1], nsh[0])
        varu = varu.reshape(nsh[1], nsh[0])
        nsh = _np.flipud(_np.atleast_1d(nsh))
    # endif
    if (nsh[1] == 1):
        u = u.reshape(nsh[0],)
        varu = varu.reshape(nsh[0],)
        nsh = _np.flipud(_np.atleast_1d(nsh))
    # endif

    # =================================================================== #
    # Estimate the variance by wiggling the input data within it's
    # uncertainties. This is referred to as 'bootstrapping' or 'Monte Carlo'
    # error propagation. It works well for non-linear methods, and
    # computational methods.

    # Pre-allocate
    dudx = _np.zeros((nmonti, nsh[0], nsh[1]), dtype=_np.float64)
    for ii in range(nmonti):

        # Wiggle the input data within its statistical uncertainty
        # utemp = _np.random.normal(0.0, 1.0, _np.size(u, axis=axis))
        # utemp = utemp.reshape(nsh[0], nsh[1])
        utemp = _np.random.normal(0.0, 1.0, _np.shape(u))
        utemp = u + _np.sqrt(varu)*utemp

        # Convolve with the derivative of a Gaussian to get the derivative
        # There is some smoothing in this procedure
        utemp = _ndimage.gaussian_filter1d(utemp, sigma=sigma, order=derivorder,
                                          axis=axis, mode=mode)
        # utemp /= dx  # dx
        dudx[ii, :, :] = utemp.copy()
    # endfor

    # Take mean and variance of the derivative
    vardudx = _np.nanvar(dudx, axis=0)
    dudx = _np.nanmean(dudx, axis=0)

    if _np.size(dudx, axis=0) == 1:
        dudx = dudx.T
        vardudx = vardudx.T
    # endif

    if derivorder > 0:
        # Do the derivative part now
    #    dx = xvar[1]-xvar[0]
        dx = _np.concatenate((_np.diff(xvar[:2], axis=axis),
                              0.5*(_np.diff(xvar[:-1], axis=axis)+_np.diff(xvar[1:], axis=axis)),
                              _np.diff(xvar[-2:], axis=axis)))

        dx = _np.atleast_2d(dx).T
        # if nsh[1]>1:
        if _np.shape(dx) != _np.shape(dudx):
            dx = _np.tile(dx, (1,nsh[1]))
        # endif

        vardudx /= dx**2.0
        dudx /= dx
    # end if

    # Match the input data shape in the output data
    vardudx = vardudx.reshape(ush)
    dudx = dudx.reshape(ush)

    return dudx, vardudx
# end def deriv_bsgaussian

# ======================================================================== #


def fit_leastsq(p0, xdat, ydat, func, **kwargs):
    """
    [pfit, pcov] = fit_leastsq(p0, xdat, ydat, func)

    Least squares fit of input data to an input function using scipy's
    "leastsq" function.

    Inputs:
        p0 - initial guess at fitting parameters
        xdat,ydat - Input data to be fit
        func - Handle to an external fitting function, y = func(p,x)

    Outputs:
        pfit - Least squares solution for fitting paramters
        pcov - Estimate of the covariance in the fitting parameters
                (scaled by residuals)
    """

    def errf(*args):
        p,x,y=(args[:-2],args[-2],args[-1])
        return func(x, _np.asarray(p)) - y
    # end def errf
    # errf = lambda p, x, y: func(p,x) - y

    pfit, pcov, infodict, errmsg, success = \
        leastsq(errf, p0, args=(xdat, ydat), full_output=1,
                epsfcn=0.0001, **kwargs)

    # end if

    if (len(ydat) > len(p0)) and pcov is not None:
        pcov = pcov * ((errf(pfit, xdat, ydat)**2).sum()
                       / (len(ydat)-len(p0)))
    else:
        pcov = _np.inf
    # endif

    return pfit, pcov

    """
    The below uncertainty is not a real uncertainty.  It assumes that there
    is no covariance in the fitting parameters.
    perr = []
    for ii in range(len(pfit)):
        try:
            #This assumes uncorrelated uncertainties (no covariance)
            perr.append(_np.absolute(pcov[ii][ii])**0.5)
        except:
            perr.append(0.00)
        # end try
    # end for
    return pfit, _np.array(perr)

    perr - Estimated uncertainty in fitting parameters
                (scaled by residuals)
    """
# end def fit_leastsq

# ======================================================================== #


def fit_curvefit(p0, xdat, ydat, func, yerr=None, **kwargs):
    """
    [pfit, pcov] = fit_curvefit(p0, xdat, ydat, func, yerr)

    Least squares fit of input data to an input function using scipy's
    "curvefit" method.

    Inputs:
        p0 - initial guess at fitting parameters
        xdat,ydat - Input data to be fit
        func - Handle to an external fitting function, y = func(p,x)
        yerr - uncertainty in ydat (optional input)

    Outputs:
        pfit - Least squares solution for fitting paramters
        pcov - Estimate of the covariance in the fitting parameters
                (scaled by residuals)
    """

    method = kwargs.get('lsqmethod','lm')
    if (_scipyversion >= 0.17) and (yerr is not None):
        pfit, pcov = curve_fit(func, xdat, ydat, p0=p0, sigma=yerr,
                               absolute_sigma = True, method=method)
    else:
        pfit, pcov = curve_fit(func, xdat, ydat, p0=p0, sigma=yerr, **kwargs)

        if (len(ydat) > len(p0)) and (pcov is not None):
            pcov = pcov *(((func(pfit, xdat, ydat)-ydat)**2).sum()
                           / (len(ydat)-len(p0)))
        else:
            pcov = _np.inf
        # endif
    # endif

    return pfit, pcov
    """
    The below uncertainty is not a real uncertainty.  It assumes that there
    is no covariance in the fitting parameters.
    perr = []
    for ii in range(len(pfit)):
        try:
            #This assumes uncorrelated uncertainties (no covariance)
            perr.append(_np.absolute(pcov[ii][ii])**0.5)
        except:
            perr.append(0.00)
        # end try
    # end for
    return pfit, _np.array(perr)

    perr - Estimated uncertainty in fitting parameters
                (scaled by residuals)
    """
# end def fit_curvefit

# ======================================================================== #


def fit_mcleastsq(p0, xdat, ydat, func, yerr_systematic=0.0, nmonti=300):
    """
    function [pfit,perr] = fit_mcleastsq(p0, xdat, ydat, func, yerr_systematic, nmonti)

    This is a Monte Carlo wrapper around scipy's leastsq function that is
    meant to propagate systematic uncertainty from input data into the
    fitting parameters nonlinearly.

    Inputs:
        p0 - initial guess at fitting parameters
        xdat,ydat - Input data to be fit
        func - Handle to an external fitting function, y = func(p,x)
        yerr_systematic - systematic uncertainty in ydat (optional input)

    Outputs:
        pfit - Least squares solution for fitting paramters
        perr - Estimate of the uncertainty in the fitting parameters
                (scaled by residuals)

    """
    def errf(*args):
        p,x,y=(args[:-2],args[-2],args[-1])
        return func(x, _np.asarray(p)) - y
    # end def errf
    # errf = lambda p, x, y: func(x, p) - y

    # Fit first time
    pfit, perr = leastsq(errf, p0, args=(xdat, ydat), full_output=0)

    # Get the stdev of the residuals
    residuals = errf(pfit, xdat, ydat)
    sigma_res = _np.std(residuals)

    # Get an estimate of the uncertainty in the fitting parameters (including
    # systematics)
    sigma_err_total = _np.sqrt(sigma_res**2 + yerr_systematic**2)

    # several hundred random data sets are generated and fitted
    ps = []
    for ii in range(nmonti):
        yy = ydat + _np.random.normal(0., sigma_err_total, len(ydat))

        mcfit, mccov = leastsq(errf, p0, args=(xdat, yy), full_output=0)

        ps.append(mcfit)
    #end for

    # You can choose the confidence interval that you want for your
    # parameter estimates:
    # 1sigma gets approximately the same as methods above
    # 1sigma corresponds to 68.3% confidence interval
    # 2sigma corresponds to 95.44% confidence interval
    ps = _np.array(ps)
    mean_pfit = _np.mean(ps, 0)

    Nsigma = 1.0
    err_pfit = Nsigma * _np.std(ps, 0)

    return mean_pfit, err_pfit
# end fit_mcleastsq

# ======================================================================== #


def spline(xvar, yvar, xf, vary=None, deg=5, bbox=None):
    """
    One-dimensional smoothing spline fit to a given set of data points (scipy).
    Fits a spline y = spl(xf) of degree deg to the provided xvar, yvar data.
    :param xvar: 1-D array of independent input data. Must be increasing.
    :param yvar: 1-D array of dependent input data, of the same length as x.
    :param xf: values at which you request the values of the interpolation
                function
    :param vary: Variance in y, used as weights (1/sqrt(vary)) for spline
                    fitting. Must be positive. If None (default), weights are
                    all equal.
    :param nmonti: Number of Monte Carlo iterations for nonlinear error
                    propagation. Default is 300.
    :param deg: Degree of the smoothing spline. Must be <= 5. Default is k=3,
                a cubic spline.
    :param bbox: 2-sequence specifying the boundary of the approximation
                    interval. If None (default), bbox=[xvar[0], xvar[-1]].

    :type xvar: (N,) array_like
    :type yvar: (N,) array_like
    :type xf: (N,) array_like
    :type vary: (N,) array_like, optional
    :type nmonti: int, optional
    :type deg: int, optional
    :type bbox: (2,) array_like, optional

    :return: the interpolation values at xf and the first derivative at xf
    :rtype: ndarray, ndarray

    .. note::
    The number of data points must be larger than the spline degree deg.
    """

    if bbox is None:
        bbox = [xvar[0], xvar[-1]]
    # end if
    if vary is None:
        vary = _np.ones_like(yvar)
    # end if
    # _ut.interp_irregularities()
    # ============= #

    pobj = _int.UnivariateSpline(xvar, yvar, w=1.0/_np.sqrt(vary), bbox=bbox,
                                 k=deg) #, check_finite=True)
    yf = pobj.__call__(xf)
#
#    pobj = pobj.derivative(n=1)
#    dydx = pobj.__call__(xf)
    dydx = pobj.derivative(n=1).__call__(xf)
    return yf, dydx

# ======================================================================== #


def pchip(xvar, yvar, xf):
    """
    PCHIP 1-d monotonic cubic interpolation (from scipy)
    :param xvar: x values for interpolation
    :param yvar: y values for interpolation
    :param xf: values at which you request the values of the interpolation
                function
    :type xvar: ndarray
    :type yvar: ndarray
    :type xf: ndarray
    :return: the interpolation values at xf and the first derivative at xf
    :rtype: ndarray, ndarray

    .. note::
    The interpolator preserves monotonicity in the interpolation data and does
    not overshoot if the data is not smooth.
    The first derivatives are guaranteed to be continuous, but the second
    derivatives may jump at xf.
    """

    pobj = _int.pchip(xvar, yvar, axis=0)

    yf = pobj.__call__(xf)
#    dydx = pobj.derivative(nu=1).__call__(xf)
    dydx = pobj.__call__(xf, nu=1)
    return yf, dydx


def spline_bs(xvar, yvar, vary, xf=None, func="spline", nmonti=300, deg=3, bbox=None):
    """
    :param xvar: x values for interpolation (2D)
    :param yvar: y values for interpolation (2D)
    :param vary: Variance in y, used as weights (1/sqrt(vary)) for spline
                    fitting. Must be positive. If None (default), weights are
                    all equal and no uncertainties are returned. (2D)
    :param xf: values at which you request the values of the interpolation
                function. Default is None, which just uses the xvar values.
    :param func: Function specification, defaults to spline
    :param nmonti: Number of Monte Carlo iterations for nonlinear error
                    propagation. Default is 300.
    :param deg: Degree of the smoothing spline. Must be <= 5. Default is k=3,
                a cubic spline. Only valid for func="spline".
    :param bbox: 2-sequence specifying the boundary of the approximation
                    interval. If None (default), bbox=[xvar[0], xvar[-1]].
                     Only valid for func="spline".
    :type xvar: (N,) ndarray
    :type yvar: (N,) ndarray
    :type vary: (N,) array_like, optional
    :type xf: (N,) ndarray, optional
    :type func: str, optional
    :type nmonti: int, optional
    :type deg: int, optional
    :type bbox: (2,) array_like, optional
    :return: the interpolation values at xf and the first derivative at xf or,
                if yary is given, the interpolation values at xf + the variance
                and the first derivative at xf + the variance
    :rtype: [ndarray, ndarray] resp. [ndarray, ndarray, ndarray, ndarray]
    """

    if func is None:            func = "spline"     # endif
    if xf is None:              xf = xvar.copy()    # endif
    if deg is None:             deg = 3             # endif
    if nmonti is None:          nmonti = 300        # endif
    nxf = len(xf)

    if xf is None:
        xf = xvar.copy()
    # endif
    nxf = len(xf)

    # ============= #

    yvar = _np.atleast_2d(yvar)
    vary = _np.atleast_2d(vary)
    nsh = _np.shape(yvar)
    nsh = _np.atleast_1d(nsh)
    if nsh[0] == 1:
        yvar = yvar.T
        vary = vary.T
        nsh = _np.flipud(nsh)
    # end if

    # ============= #

    xvar = xvar.reshape((nsh[0],), order='C')

    dydx = _np.zeros( (nmonti, nxf, nsh[1]), dtype=_np.float64)
    yf = _np.zeros( (nmonti, nxf, nsh[1]), dtype=_np.float64)
    for ii in range(nmonti):
        utemp = yvar + _np.sqrt(vary)*_np.random.normal(0.0, 1.0, _np.shape(yvar))
        if func == 'pchip':
            yf[ii, :], dydx[ii, :] = pchip(xvar, utemp, xf)
        else:
            tmp1 = _np.zeros((nxf, nsh[1]), dtype=utemp.dtype)
            tmp2 = _np.zeros_like(tmp1)
            for jj in range(nsh[1]):
#                vtemp = (utemp-yvar)
#                tmp1[:,jj], tmp2[:,jj] = spline(xvar, utemp[:,jj], xf, vary=vary[:,jj], deg=deg, bbox=bbox)
                tmp1[:,jj], tmp2[:,jj] = spline(xvar, utemp[:,jj], xf, vary=None, deg=deg, bbox=bbox)
            # end for
#            print(_np.shape(tmp1))
#            print(nsh)
#            print(_np.shape(yf))
#            print(_np.shape(yvar))
            yf[ii, :] = tmp1.reshape((nxf, nsh[1]), order='C')
            dydx[ii, :] = tmp2.reshape((nxf, nsh[1]), order='C')
        # endif
    # end for

    vardydx = _np.var(dydx, axis=0)
    dydx = _np.mean(dydx, axis=0)
    varf = _np.var( yf, axis=0)
    yf = _np.mean(yf, axis=0)

    return yf, varf, dydx, vardydx

# ======================================================================== #
# ======================================================================== #

def expdecay_fit(tt,sig_in,param):
    #
    # param contains intitial guesses for fitting gaussians,
    # (Amplitude, x value, sigma):
    # param = [[50,40,5],[50,110,5],[100,160,5],[100,220,5],
    #      [50,250,5],[100,260,5],[100,320,5], [100,400,5],
    #      [30,300,150]]  # this last one is our noise estimate

    # Define a function that returns the magnitude of stuff under a gaussian
    # peak (with support for multiple peaks)
    fit = lambda param, xx: _np.sum([_ms.expdecay(xx, param[ii*3], param[ii*3+1],
                                              param[ii*3+2])
                                    for ii in _np.arange(len(param)/3)], axis=0)
    # Define a function that returns the difference between the fitted gaussian
    # and the input signal
    err = lambda param, xx, yy: fit(param, xx)-yy

    # If multiple peaks have been requested do some array manipulation
    param = _np.asarray(param).flatten()
    # end if
    # tt  = xrange(len(sig_in))

    # Least squares fit the exponential peaks: "args" gives the arguments to the
    # err function defined above
    results, value = leastsq(err, param, args=(tt, sig_in))

    for res in results.reshape(-1,3):
        print('Peak detected at: amplitude %f, position %f, sigma %f '%(res[0], res[1], res[2]))
    # end for

    return results.reshape(-1,3)

# ======================================================================== #


def gaussian_peak_width(tt,sig_in,param):
    #
    # param contains intitial guesses for fitting gaussians,
    # (Amplitude, x value, sigma):
    # param = [[50,40,5],[50,110,5],[100,160,5],[100,220,5],
    #      [50,250,5],[100,260,5],[100,320,5], [100,400,5],
    #      [30,300,150]]  # this last one is our noise estimate

    # Define a function that returns the magnitude of stuff under a gaussian
    # peak (with support for multiple peaks)
    fit = lambda param, xx: _np.sum([_ms.gaussian(xx, param[ii*3], param[ii*3+1],
                                              param[ii*3+2])
                                    for ii in _np.arange(len(param)/3)], axis=0)
    # Define a function that returns the difference between the fitted gaussian
    # and the input signal
    err = lambda param, xx, yy: fit(param, xx)-yy

    # If multiple peaks have been requested do some array manipulation
    param = _np.asarray(param).flatten()
    # end if
    # tt  = xrange(len(sig_in))

    # Least squares fit the gaussian peaks: "args" gives the arguments to the
    # err function defined above
    results, value = leastsq(err, param, args=(tt, sig_in))

    for res in results.reshape(-1,3):
        print('Peak detected at: amplitude, position, sigma %f '%(res))
    # end for

    return results.reshape(-1,3)

# ======================================================================== #


def qparab_lsfit(xdata, ydata, vary=None, xx=None,
                 lowbounds=None, upbounds=None, nohollow=False):
    if xx is None:
        xx = _np.linspace(1e-4,1.0,100)
    # endif
    if vary is None:
        weights = None
    else:
        weights = _np.sqrt(vary)
    # endif
    Y0 = _np.max(ydata)  # Guess of the core value
    Y1 = _np.min(ydata)  # Guess of the edge value

    if lowbounds is None:
        lowbounds = [ 0.0, 0.0, -_np.inf, -_np.inf, -_np.inf, -_np.inf]
    # endif
    if upbounds is None:
        upbounds = [ _np.inf, Y0, _np.inf, _np.inf,  _np.inf,  _np.inf]
    # endif

    af = [Y0, Y1/Y0, 1.5, 1.5, 0.0, 0.02]
    if nohollow:
        af.pop(4)
        lowbounds.pop(4)
        upbounds.pop(4)
    # endif

    # Alias to the fitting method that allows passing a static argument to the method.
    FitAlias = lambda *args: qparab_fit(args[0], args[1:], nohollow)

    if _scipyversion < 0.17:
        [af,pcov]=curve_fit( FitAlias, xdata, ydata, p0 = af, sigma = weights)

        af = _np.asarray(af, dtype=_np.float64)
        if (len(ydata) > len(af)) and pcov is not None:
            pcov = pcov * (((FitAlias(xdata,af)-ydata)**2).sum()
                           / (len(ydata)-len(af)))
        else:
            pcov = _np.inf
        # endif
    else:
        [af,pcov]=curve_fit( FitAlias, xdata, ydata,
                                    p0 = af,
                                    sigma = weights,
                                    absolute_sigma = True, \
                                    bounds = (lowbounds,upbounds), \
                                    method = 'trf')
    # endif
    return af, pcov
#    vaf = _np.diag(pcov)
#    return af, vaf
# end def qparab_lsfit


# ======================================================================== #
#
#
#def spline(xvar, yvar, xf, vary=None, nmonti=300, deg=5, bbox=None):
#    """
#    One-dimensional smoothing spline fit to a given set of data points (scipy).
#    Fits a spline y = spl(xf) of degree deg to the provided xvar, yvar data.
#    :param xvar: 1-D array of independent input data. Must be increasing.
#    :param yvar: 1-D array of dependent input data, of the same length as x.
#    :param xf: values at which you request the values of the interpolation
#                function
#    :param vary: Variance in y, used as weights (1/sqrt(vary)) for spline
#                    fitting. Must be positive. If None (default), weights are
#                    all equal and no uncertainties are returned.
#    :param nmonti: Number of Monte Carlo iterations for nonlinear error
#                    propagation. Default is 300.
#    :param deg: Degree of the smoothing spline. Must be <= 5. Default is k=3,
#                a cubic spline.
#    :param bbox: 2-sequence specifying the boundary of the approximation
#                    interval. If None (default), bbox=[xvar[0], xvar[-1]].
#
#    :type xvar: (N,) array_like
#    :type yvar: (N,) array_like
#    :type xf: (N,) array_like
#    :type vary: (N,) array_like, optional
#    :type nmonti: int, optional
#    :type deg: int, optional
#    :type bbox: (2,) array_like, optional
#
#    :return: the interpolation values at xf and the first derivative at xf or,
#                if yary is given, the interpolation values at xf + the variance
#                and the first derivative at xf + the variance
#    :rtype: [ndarray, ndarray] resp. [ndarray, ndarray, ndarray, ndarray]
#
#    .. note::
#    The number of data points must be larger than the spline degree deg.
#    """
#
#    if bbox is None:
#        bbox = [xvar[0], xvar[-1]]
#    # end if
#
#    if vary is None:
#        pobj = _int.UnivariateSpline(xvar, yvar, bbox=bbox, k=deg,
#                                     check_finite=True)
#        yf_val = pobj.__call__(xf)
#        dyf_val = pobj.derivative(n=1).__call__(xf)
#        return yf_val, dyf_val
#    # end if
#
#    # ============= #
#    # Monte Carlo
#    yf = []
#    dyf = []
#    for _ in range(nmonti):
#        yvar_mc = yvar + _np.random.normal(0., _np.sqrt(vary), len(yvar))
#
#        pobj = _int.UnivariateSpline(xvar, yvar_mc, bbox=bbox, k=deg,
#                                     check_finite=True)
#        yf_val = pobj.__call__(xf)
#        dyf_val = pobj.derivative(n=1).__call__(xf)
#
#        yf.append(yf_val)
#        dyf.append(dyf_val)
#    # end for
#
#    return _np.mean(yf, axis=0), _np.var(yf, axis=0), _np.mean(dyf, axis=0), _np.var(dyf, axis=0)
#
#
#def pchip(xvar, yvar, xf, vary=None, nmonti=300):
#    """
#    PCHIP 1-d monotonic cubic interpolation (from scipy)
#    :param xvar: x values for interpolation
#    :param yvar: y values for interpolation
#    :param xf: values at which you request the values of the interpolation
#                function
#    :param vary: Variance in y, used as weights (1/sqrt(vary)) for spline
#                    fitting. Must be positive. If None (default), weights are
#                    all equal and no uncertainties are returned.
#    :param nmonti: Number of Monte Carlo iterations for nonlinear error
#                    propagation. Default is 300.
#    :type xvar: ndarray
#    :type yvar: ndarray
#    :type xf: ndarray
#    :type vary: (N,) array_like, optional
#    :type nmonti: int, optional
#    :return: the interpolation values at xf and the first derivative at xf or,
#                if yary is given, the interpolation values at xf + the variance
#                and the first derivative at xf + the variance
#    :rtype: [ndarray, ndarray] resp. [ndarray, ndarray, ndarray, ndarray]
#
#    .. note::
#    The interpolator preserves monotonicity in the interpolation data and does
#    not overshoot if the data is not smooth.
#    The first derivatives are guaranteed to be continuous, but the second
#    derivatives may jump at xf.
#    """
#
#    if vary is None:
#        pobj = _int.pchip(xvar, yvar, axis=0)
#        yf_val = pobj.__call__(xf)
#        dyf_val = pobj.derivative(n=1).__call__(xf)
#        return yf_val, dyf_val
#    # end if
#
#    # ============= #
#    # Monte Carlo
#    yf = []
#    dyf = []
#    for _ in range(nmonti):
#        yvar_mc = yvar + _np.random.normal(0., _np.sqrt(vary), len(yvar))
#
#        pobj = _int.pchip(xvar, yvar_mc, axis=0)
#        yf_val = pobj.__call__(xf)
#        dyf_val = pobj.derivative(nu=1).__call__(xf)
#
#        yf.append(yf_val)
#        dyf.append(dyf_val)
#    # end for
#
#    return _np.mean(yf, axis=0), _np.var(yf, axis=0), _np.mean(dyf, axis=0), _np.var(dyf, axis=0)
#
#
#def spline_bs(xvar, yvar, vary, xf=None, func="spline", nmonti=300, deg=3,
#              bbox=None):
#    """
#    :param xvar: x values for interpolation (2D)
#    :param yvar: y values for interpolation (2D)
#    :param xf: values at which you request the values of the interpolation
#                function. Default is None, which just uses the xvar values.
#    :param vary: Variance in y, used as weights (1/sqrt(vary)) for spline
#                    fitting. Must be positive. If None (default), weights are
#                    all equal and no uncertainties are returned. (2D)
#    :param nmonti: Number of Monte Carlo iterations for nonlinear error
#                    propagation. Default is 300.
#    :param deg: Degree of the smoothing spline. Must be <= 5. Default is k=3,
#                a cubic spline. Only valid for func="spline".
#    :param bbox: 2-sequence specifying the boundary of the approximation
#                    interval. If None (default), bbox=[xvar[0], xvar[-1]].
#                     Only valid for func="spline".
#    :type xvar: (N,) ndarray
#    :type yvar: (N,) ndarray
#    :type xf: (N,) ndarray, optional
#    :type vary: (N,) array_like, optional
#    :type nmonti: int, optional
#    :type deg: int, optional
#    :type bbox: (2,) array_like, optional
#    :return: the interpolation values at xf and the first derivative at xf or,
#                if yary is given, the interpolation values at xf + the variance
#                and the first derivative at xf + the variance
#    :rtype: [ndarray, ndarray] resp. [ndarray, ndarray, ndarray, ndarray]
#    """
#
#    if xf is None:
#        xf = xvar.copy()
#    # endif
#
#    yvar = _np.atleast_2d(yvar)
#    vary = _np.atleast_2d(vary)
#
#    nsh = _np.shape(yvar)
#    nsh = _np.atleast_1d(nsh)
#    if nsh[0] == 1:
#        yvar = yvar.T
#        vary = vary.T
#        nsh = _np.flipud(nsh)
#
#    # ============= #
#    yf = list()
#    varf = list()
#    dydx = list()
#    vardydx = list()
#
#    if func == 'spline':
#        for ii in range(yvar.shape[1]):
#            tmp1, tmp2, tmp3, tmp4 = spline(xvar, yvar[:, ii], xf, vary=vary[:, ii],
#                         nmonti=nmonti, deg=deg, bbox=bbox)
#            yf.append(tmp1.copy())
#            varf.append(tmp2.copy())
#            dydx.append(tmp3.copy())
#            vardydx.append(tmp4.copy())
#
#    elif func == 'pchip':
#        for ii in range(yvar.shape[1]):
#            tmp1, tmp2, tmp3, tmp4 = pchip(xvar, yvar[:, ii], xf, vary=vary[:, ii], nmonti=nmonti)
#            yf.append(tmp1.copy())
#            varf.append(tmp2.copy())
#            dydx.append(tmp3.copy())
#            vardydx.append(tmp4.copy())
#    else:
#        raise("Unknown func for spline. I know currently only 'spline' and " +
#              "'pchip'")
#
#    yf = _np.asarray(yf)
#    varf = _np.asarray(varf)
#    dydx = _np.asarray(dydx)
#    vardydx = _np.asarray(vardydx)
#    return yf, varf, dydx, vardydx
# ======================================================================== #
# ======================================================================== #

def fit_TSneprofile(QTBdat, rvec, loggradient=True, plotit=False, amin=0.51, returnaf=False):

    roa = QTBdat['roa']
    ne = QTBdat['ne']
    varn =  _np.sqrt(QTBdat['varNL']*QTBdat['varNH'])
    # ne *= 1e-20
    # varn *= (1e-20)**2.0

    def fitqparab(af, XX):
        return qparab_fit(XX, af)

    def fitdqparabdx(af, XX):
        return dqparabdx(XX, af)

    info = _ms.model_qparab(XX=None)
    LB = info.Lbounds
    UB = info.Ubounds

    af0 = _np.asarray([0.30, 0.002, 2.0, 0.7, -0.24, 0.30], dtype=_np.float64)
    NLfit = fitNL(roa, 1e-20*ne, 1e-40*varn, af0, fitqparab, LB=LB, UB=UB)
    NLfit.run()

    nef, gvec, info = _ms.model_qparab(_np.abs(rvec), NLfit.af)
    varnef = NLfit.properror(_np.abs(rvec), gvec)
    varnef = varnef.copy()

    NLfit.func = fitdqparabdx
    dlnnedrho = info.dprofdx / nef
    vardlnnedrho = NLfit.properror(_np.abs(rvec), info.dgdx)
    vardlnnedrho = (dlnnedrho)**2.0 * ( vardlnnedrho/(info.dprofdx**2.0) + varnef/(nef**2.0)  )

    # Convert back to absolute units (particles per m-3 not 1e20 m-3)
    nef = 1e20*nef
    varnef = 1e40*varnef

    varlogne = varnef / nef**2.0
    logne = _np.log(nef)

    # ================== #

    if plotit:
        _plt.figure()
        ax1 = _plt.subplot(2,1,1)
#        ax2 = _plt.subplot(3,1,2, sharex=ax1)
        ax3 = _plt.subplot(2,1,2, sharex=ax1)

        ax1.grid()
#        ax2.grid()
        ax3.grid()

        ax1.set_title(r'Density Profile Info')
        ax1.set_ylabel(r'$n_\mathrm{e}\ \mathrm{in}\ 10^{20}\mathrm{m}^{-3}$')
#        ax2.set_ylabel(r'$\ln(n_\mathrm{e})$')   # ax2.set_ylabel(r'ln(n$_e$[10$^20$m$^-3$])')
        ax3.set_ylabel(r'$a/L_\mathrm{ne}$')
        ax3.set_xlabel(r'$r/a$')

        ax1.errorbar(roa, 1e-20*ne, yerr=1e-20*_np.sqrt(varn), fmt='bo', color='b' )
#        ax2.errorbar(roa, _np.log(ne), yerr=_np.sqrt(varn/ne**2.0), fmt='bo', color='b' )

        ax1.plot(rvec, 1e-20*nef, 'b-', lw=2)
        ax1.plot(rvec, 1e-20*(nef+_np.sqrt(varnef)), 'b--', lw=1)
        ax1.plot(rvec, 1e-20*(nef-_np.sqrt(varnef)), 'b--', lw=1)
#        ax2.plot(rvec, logne, 'b-', lw=2)
#        ax2.plot(rvec, logne+_np.sqrt(varlogne), 'b--', lw=1)
#        ax2.plot(rvec, logne-_np.sqrt(varlogne), 'b--', lw=1)

        # _, _, nint, nvar = _ut.trapz_var(rvec, dlnnedrho, vary=vardlnnedrho)

        if loggradient:
            idx = _np.where(_np.abs(rvec) < 0.05)
            plotdlnnedrho = dlnnedrho.copy()
            plotvardlnnedrho = vardlnnedrho.copy()
            plotdlnnedrho[idx] = _np.nan
            plotvardlnnedrho[idx] = _np.nan
            plotdlnnedrho *= -1*amin
            plotvardlnnedrho *= amin**2.0
            ax3.plot(rvec, plotdlnnedrho, 'b-',
                     rvec, plotdlnnedrho+_np.sqrt(plotvardlnnedrho), 'b--',
                     rvec, plotdlnnedrho-_np.sqrt(plotvardlnnedrho), 'b--')

            # nint += (_np.log(ne[0]) - _ut.interp(rvec, nint, xo=roa[0]))
            # nint = _np.exp(nint)
        else:
            vardlnnedrho = (dlnnedrho/logne)**2.0 * (vardlnnedrho/dlnnedrho**2.0+varlogne/logne**2.0)
            dlnnedrho = dlnnedrho/logne
            plotdlnnedrho = -1*amin * dlnnedrho.copy()
            plotvardlnnedrho = (amin**2.0) * vardlnnedrho.copy()

            ax3.plot(rvec, plotdlnnedrho, 'b-',
                     rvec, plotdlnnedrho+_np.sqrt(plotvardlnnedrho), 'b--',
                     rvec, plotdlnnedrho-_np.sqrt(plotvardlnnedrho), 'b--')

            # nint += (ne[0] - _ut.interp(rvec, nint, xo=roa[0]))
        # end if
        # ax1.plot(rvec, 1e-20*nint, 'b--', lw=1)
        # ax2.plot(rvec, _np.log(nint), 'b--', lw=1)
        _plt.tight_layout()

    # end if plotit

    # ==================== #
    if returnaf:
        return logne, varlogne, dlnnedrho, vardlnnedrho, NLfit.af
    return logne, varlogne, dlnnedrho, vardlnnedrho

# ======================================================================== #

def fit_TSteprofile(QTBdat, rvec, loggradient=True, plotit=False, amin=0.51, returnaf=False):

    roa = QTBdat['roa']
    Te = QTBdat['Te']
    varT =  _np.sqrt(QTBdat['varTL']*QTBdat['varTH'])
    # ne *= 1e-20
    # varn *= (1e-20)**2.0

    def fitqparab(af, XX):
        return qparab_fit(XX, af)

    def fitdqparabdx(af, XX):
        return dqparabdx(XX, af)

    info = _ms.model_qparab(XX=None)
    LB = info.Lbounds
    UB = info.Ubounds
    af0 = _np.asarray([4.00, 0.07, 5.0, 2.0, 0.04, 0.50], dtype=_np.float64)
    NLfit = fitNL(roa, Te, varT, af0, fitqparab, LB=LB, UB=UB)
    NLfit.run()

    Tef, gvec, info = _ms.model_qparab(_np.abs(rvec), NLfit.af)
    varTef = NLfit.properror(_np.abs(rvec), gvec)
    varTef = varTef.copy()

    NLfit.func = fitdqparabdx
    dlnTedrho = info.dprofdx / Tef
    vardlnTedrho = NLfit.properror(_np.abs(rvec), info.dgdx)
    vardlnTedrho = (dlnTedrho)**2.0 * ( vardlnTedrho/(info.dprofdx**2.0) + varTef/(Tef**2.0)  )

    varlogTe = varTef / Tef**2.0
    logTe = _np.log(Tef)

    # ================== #

    if plotit:
        _plt.figure()
        ax1 = _plt.subplot(2,1,1)
#        ax2 = _plt.subplot(3,1,2, sharex=ax1)
        ax3 = _plt.subplot(2,1,2, sharex=ax1)

        ax1.grid()
#        ax2.grid()
        ax3.grid()

        ax1.set_title(r'Temperature Profile Info')
        ax1.set_ylabel(r'T$_e$ in KeV')
#        ax2.set_ylabel(r'$\ln(T_\mathrm{e})$')   # ax2.set_ylabel(r'ln(T$_e$)')
        ax3.set_ylabel(r'$a/L_\mathrm{Te}$')
        ax3.set_xlabel(r'$r/a$')

        ax1.errorbar(roa, Te, yerr=_np.sqrt(varT), fmt='ro', color='r' )
#        ax2.errorbar(roa, _np.log(Te), yerr=_np.sqrt(varT/Te**2.0), fmt='bo', color='b' )

        ax1.plot(rvec, Tef, 'r-', lw=2)
        ax1.plot(rvec, (Tef+_np.sqrt(varTef)), 'r--', lw=1)
        ax1.plot(rvec, (Tef-_np.sqrt(varTef)), 'r--', lw=1)
#        ax2.plot(rvec, logTe, 'b-', lw=2)
#        ax2.plot(rvec, logTe+_np.sqrt(varlogTe), 'r--', lw=1)
#        ax2.plot(rvec, logTe-_np.sqrt(varlogTe), 'r--', lw=1)

        # _, _, Tint, Tvar = _ut.trapz_var(rvec, dlnTedrho, vary=vardlnTedrho)

        if loggradient:
            idx = _np.where(_np.abs(rvec) < 0.05)
            plotdlnTedrho = dlnTedrho.copy()
            plotvardlnTedrho = vardlnTedrho.copy()
            plotdlnTedrho[idx] = _np.nan
            plotvardlnTedrho[idx] = _np.nan
            plotdlnTedrho *= -1*amin
            plotvardlnTedrho *= amin**2.0
            ax3.plot(rvec, plotdlnTedrho, 'r-',
                     rvec, plotdlnTedrho+_np.sqrt(plotvardlnTedrho), 'r--',
                     rvec, plotdlnTedrho-_np.sqrt(plotvardlnTedrho), 'r--')

            # Tint += (_np.log(Te[0]) - _ut.interp(rvec, Tint, xo=roa[0]))
            # Tint = _np.exp(Tint)
        else:
            vardlnTedrho = (dlnTedrho/logTe)**2.0 * (vardlnTedrho/dlnTedrho**2.0+varlogTe/logTe**2.0)
            dlnTedrho = dlnTedrho/logTe
            plotdlnTedrho = -1*amin * dlnTedrho.copy()
            plotvardlnTedrho = (amin**2.0) * vardlnTedrho.copy()

            ax3.plot(rvec, plotdlnTedrho, 'r-',
                     rvec, plotdlnTedrho+_np.sqrt(plotvardlnTedrho), 'r--',
                     rvec, plotdlnTedrho-_np.sqrt(plotvardlnTedrho), 'r--')

            # Tint += (Te[0] - _ut.interp(rvec, Tint, xo=roa[0]))
        # end if
        # ax1.plot(rvec, Tint, 'b--', lw=1)
        # ax2.plot(rvec, _np.log(Tint), 'b--', lw=1)
        _plt.tight_layout()

    # end if plotit

    # ==================== #
    # ==================== #
    if returnaf:
        return logTe, varlogTe, dlnTedrho, vardlnTedrho, NLfit.af
    return logTe, varlogTe, dlnTedrho, vardlnTedrho


# ======================================================================== #

class fitNL(fitNL_base):
    """
    To use this first generate a class that is a child of this one
    class Prob(fitNL)

        def __init__(self, xdata, ydata, yvar, options, **kwargs)
            # call super init
            super(fitNL, self).__init__(xdat, ydat, vary, af0, self.func, options, kwargs)
        # end def

        def func(self, af):
            return y-2*af[0]+af[1]

    """
    def __init__(self, xdat, ydat, vary, af0, func, fjac=None, **kwargs):

        # call super init
        super(fitNL, self).__init__(xdat, ydat, vary, af0, self.func, **kwargs)

        # 1) Least-squares, 2) leastsq, 3) Curve_fit
        self.lsqfitmethod = kwargs.get("lsqfitmethod", 'lm')
        if _scipyversion >= 0.17:
            self.lsqmethod = kwargs.get("lsqmethod", int(1))
        else:
            self.lsqmethod = kwargs.get("lsqmethod", int(2))
        # end if

        # ========================== #

        if self.lsqmethod == 1:
            self.lmfit = self.__use_least_squares
        elif self.lsqmethod == 2:
            self.lmfit = self.__use_leastsq
        elif self.lsqmethod == 3:
            self.lmfit = self.__use_curvefit
        # end if

    # end def

    # ========================== #

    def __use_least_squares(self, **options):
        """
        Wrapper around the scipy least_squares function
        """
        lsqfitmethod = options.get("lsqfitmethod", 'lm')

        if _np.isscalar(self.af0):
            self.af0 = [self.af0]
        self.numfit = len(self.af0)

        res = least_squares(self.calc_chi2, self.af0, bounds=(self.LB, self.UB),
                          method=lsqfitmethod, **options)
                          # args=(self.xdat,self.ydat,self.vary), kwargs)
        self.af = res.x
        # chi2
        #resid = res.fun
        jac = res.jac

        # Make a final call to the fitting function to update object values
        self.calc_chi2(self.af)

        # Estimate of covariance in solution
        # jac = _np.full(jac) #Sparse matrix form
        # resid*
        self.covmat = (_np.eye(self.numfit)) / self.numfit / \
            _np.dot(jac[:, 0:self.numfit].T, jac[:, 0:self.numfit])

        return self.af, self.covmat
    # end def __use_least_squares

    # ========================== #

    def __use_curvefit(self, **kwargs):
        """
        Wrapper around scipy's curve_fit function
        """
        lsqfitmethod = kwargs.get("lsqfitmethod", 'lm')
        def calcchi2(xdat, *af):
            af = _np.asarray(af)
            return self.calc_chi2(af)
        # end def calcchi2

        if _scipyversion >= 0.17:
            pfit, pcov = \
                curve_fit(calcchi2, self.xdat, self.ydat, p0=self.af0,
                          sigma=_np.sqrt(self.vary), epsfcn=0.0001,
                          absolute_sigma=True, bounds=(self.LB, self.UB),
                          method=lsqfitmethod, **kwargs)
        else:
            pfit, pcov = \
                curve_fit(calcchi2, self.xdat, self.ydat, p0=self.af0,
                          sigma=_np.sqrt(self.vary), **kwargs)
        # end if

        self.af = _np.asarray(pfit)
        if _np.isfinite(pcov) == 0:
            print('FAILED in curvefitting!')
        # end if
        self.covmat = _np.asarray(pcov)
        return self.af, self.covmat
    # end def __use_curvefit

    # ========================== #

    def __use_leastsq(self, **kwargs):
        """
        Wrapper for the leastsq function from scipy
        """
        lsqfitmethod = kwargs.get("lsqfitmethod", 'lm')
        if _scipyversion >= 0.17:
            pfit, pcov, infodict, errmsg, success = \
                leastsq(self.calc_chi2, self.af0, full_output=1, ftol=1e-8,
                        xtol=1e-8, maxfev=1e3, epsfcn=0.0001,
                        method=lsqfitmethod)

            # self.covmat = (resid*_np.eye[self.numfit]) / self.numfit \
            #     / _np.dot(jac[:, 0:self.numfit].T, jac[:, 0:self.numfit])
        else:
            pfit, pcov, infodict, errmsg, success = \
                leastsq(self.calc_chi2, x0=self.af0, full_output=1, **kwargs)
        # end if

        self.af = _np.asarray(pfit, dtype=_np.float64)
        if (len(self.ydat) > len(self.af)) and pcov is not None:
            pcov = pcov * ((self.calc_chi2(self.af)**2).sum()
                           / (len(self.ydat)-len(self.af)))
        else:
            pcov = _np.inf
        # endif

        self.covmat = _np.asarray(pcov, dtype=_np.float64)
        return self.af, self.covmat
    # end def __use_leastsq

    # ========================== #
    # ========================== #
# end class fitNL

# ======================================================================== #
# ======================================================================== #


def test_dat(multichannel=True):
#    x = _np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#                  dtype=_np.float64)
#    y = _np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47,
#                   98.36, 112.25, 126.14, 140.03], dtype=_np.float64)


    x = _np.linspace(-1.0, 1.0, 32)
    x += _np.random.normal(0.0, 0.03, _np.shape(x))

    x = _np.hstack((x, _np.linspace(-0.3, -0.5, 16)))

    y = 3.0*_np.ones_like(x)
    y -= 10.0*x**2.0
    y += 4.0*x**3.0
    y += 5*x**8
    y += 1.3*x

    vary = (0.3*_np.ones_like(y))**2.0
    vary *= _np.max(y)**2.0

    isort = _np.argsort(x)
    x = x[isort]
    y = y[isort]
    vary = vary[isort]

    y = _np.transpose( _np.vstack((y, _np.sin(x))) )
    vary = _np.transpose(_np.vstack((vary, (1e-5+0.1*_np.sin(x))**2.0)))

#    y = _np.hstack((y, _np.sin(x)))
#    vary = _np.hstack((vary, 0.1*_np.sin(x)))

    # ======================= #

    if multichannel:
        y = _np.transpose( _np.vstack((y, _np.sin(x))) )
        vary = _np.transpose(_np.vstack((vary, (1e-5+0.1*_np.sin(x))**2.0)))

#        y = _np.hstack((y, _np.sin(x)))
#        vary = _np.hstack((vary, 0.1*_np.sin(x)))
    # endif

    return x, y, vary


def test_derivatives():

    x, y, vary = test_dat(multichannel=True)

    # ======================= #

    # xd = _np.linspace(0, 15, 100)
    # _plt.plot(x, y, "o")

#    p , e = curve_fit(_ms.piecewise_2line, x, y)
#    _plt.plot(xd, _ms.piecewise_2line(xd, *p))

#    from model_spec import model_chieff as fmodel
#
#    mn = 1 # model_number
#    np = 4 # npoly
#    yd, gd, info = fmodel(af=None, XX=xd, model_number=mn, npoly=np)
#    p0 = info.af
#
#    p, e = fit_leastsq(p0, x, y, fmodel, (mn, np, 1) )
#    _plt.plot(xd, fmodel(*p, XX=xd, model_number=mn, npoly=np, nargout=1) )

#    yxp, yxvp, dydxp, vardydxp = spline_bs(x, y, vary, x, nmonti=30000, func="pchip")
#    yxp, yxvp, dydxp, vardydxp = spline_bs(x, y, vary, x, nmonti=30000, func="pchip")
#    yxp, yxvp, dydxp, vardydxp = spline_bs(x, y, vary, x, nmonti=300, func="spline")

    dydx, vardydx = deriv_bsgaussian(x, y, vary, axis=0, nmonti=300,
                                     sigma=0.8, mode='nearest')

    yxp, yxvp = deriv_bsgaussian(x, y, vary, axis=0, nmonti=300,
                                 sigma=0.8, mode='nearest', derivorder=0)


    _, _, yx, yxv = _ut.trapz_var(x, dydx, None, vardydx)
    yx += (y[0] - _ut.interp(x, yx, ei=None, xo=x[0]))

    # ==== #

    _plt.figure()
    ax1 = _plt.subplot(2,1,1)
    ax1.plot(x, y, "ko")
    # Integrals
    ax1.plot(x, yx, 'k-',
             x, yx+_np.sqrt(yxv), 'k--',
             x, yx-_np.sqrt(yxv), 'k--')
    ax1.plot(x, yxp, 'g-',
             x, yxp+_np.sqrt(yxvp), 'g--',
             x, yxp-_np.sqrt(yxvp), 'g--')
    # Derivatives
    ax2 = _plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(x, dydx, 'k-',
             x, dydx+_np.sqrt(vardydx), 'k--',
             x, dydx-_np.sqrt(vardydx), 'k--')
# end main()


if __name__=="__main__":
    test_derivatives()

#    x = _np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15], dtype=float)
#    y = _np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])
#
#    p , e = curve_fit(_ms.piecewise_2line, x, y)
#
#    xd = _np.linspace(0, 15, 100)
#    _plt.plot(x, y, "o")
#    _plt.plot(xd, _ms.piecewise_2line(xd, *p))
#endif

# ======================================================================== #
# ======================================================================== #













