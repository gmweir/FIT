# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:21:45 2016

@author: gawe
"""
# ======================================================================== #
# ======================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

from scipy import interpolate as _int
from scipy.optimize import curve_fit, leastsq
import matplotlib.pyplot as _plt
import numpy as _np

#from pybaseutils.Struct import Struct
from pybaseutils import utils as _ut

try:
    from FIT import model_spec as _ms
    from FIT import derivatives as _dd
    from FIT.fitNL import fitNL, modelfit
except:
    from . import model_spec as _ms
    from . import derivatives as _dd
    from .fitNL import fitNL, modelfit
# end try


# ==== #

# There are annoying differences in the context between scipy version of
# leastsq and curve_fit, and the method least_squares doesn't exist before 0.17
import scipy.version as _scipyversion

# Make a version flag for switching between least squares solvers and contexts
_scipyversion = _scipyversion.version
try:
    _scipyversion = _np.float(_scipyversion[0:4])
except:
    _scipyversion = _np.float(_scipyversion[0:3])
# end try

__metaclass__ = type

# ======================================================================== #
# ======================================================================== #

# =============================== #
# ---- no scipy dependence ------ #
# =============================== #

def linreg(X, Y, verbose=True, varY=None, varX=None, cov=False, plotit=False):
    """
    Returns coefficients to the regression line "y=ax+b" from x[] and
    y[].  Basically, solves
        Sxx a + Sx b = Sxy
         Sx a +  N b = Sy
    where Sxy = \sum_i x_i y_i, Sx = \sum_i x_i, and Sy = \sum_i y_i.  The
    solution is
        a = (Sxy N - Sy Sx)/det
        b = (Sxx Sy - Sx Sxy)/det
    where det = Sxx N - Sx^2.  In addition,
        Var|a| = s^2 |Sxx Sx|^-1 = s^2 | N  -Sx| / det
           |b|       |Sx  N |          |-Sx Sxx|
        s^2 = {\sum_i (y_i - \hat{y_i})^2 \over N-2}
            = {\sum_i (y_i - ax_i - b)^2 \over N-2}
            = residual / (N-2)
        R^2 = 1 - {\sum_i (y_i - \hat{y_i})^2 \over \sum_i (y_i - \mean{y})^2}
            = 1 - residual/meanerror

    It also prints to <stdout> few other data,
        N, a, b, R^2, s^2,
    which are useful in assessing the confidence of estimation.
    """
    N = len(X)
    if len(X) != len(Y):  raise ValueError('unequal length')

    if varX is not None:
        a, b, _, _ = linreg(X, Y, varY=varY)
        varY += a*varX
    # endif

    if varY is None:
        weights = _np.ones_like(Y)
    else:
        weights = 1.0/varY
#        weights /= _np.sum(1.0/varY)   # normalized
    # endif

    Sw = Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y, w in zip(X, Y, weights):
        Sw += w
        Sx += w*x
        Sy += w*y
        Sxx += w*x*x
        Syy += w*y*y
        Sxy += w*x*y
    det = Sxx * Sw - Sx * Sx         # delta
    a, b = (Sxy * Sw - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det

    meanerror = residual = 0.0
    for x, y, w in zip(X, Y, weights):
#        meanerror += (w*y - Sy/Sw)**2.0
#        residual += (w/Sw)*(y - a * x - b)**2.0
        meanerror += (y - _np.mean(Y))**2.0
        residual += (y - a * x - b)**2.0
    RR = 1.0 - residual/meanerror
    ss = residual / (N-2.0)
    Var_a, Var_b = ss * Sw / det, ss * Sxx / det

    Cov_ab = _np.sqrt(Var_a)*_np.sqrt(Var_b)
    Cov_ab *= RR

#    tst = _np.cov( _np.vstack((X, Y)) )
#    corrcoef = tst[0,1]/_np.sqrt(tst[0,0]*tst[1,1])
#    Cov_ab *= corrcoef

    if verbose:
        print("y=ax+b")
        print("N= %d" % N )
        print("a= %g \\pm t_{%d;\\alpha/2} %g" % (a, N-2, _np.sqrt(Var_a)) )
        print("b= %g \\pm t_{%d;\\alpha/2} %g" % (b, N-2, _np.sqrt(Var_b)) )
        print("R^2= %g" % RR)
        print("s^2= %g" % ss)
        print(r"$\chi^2_\nu$ = %g, $\nu$ = %g" % (ss*1.0 / (_np.sum(weights)/N), N-2))
    # end if

    if plotit:
        xo = _np.linspace(_np.min(X)-_np.abs(0.05*_np.min(X)), _np.max(X)+_np.abs(0.05*_np.min(X)))

        _plt.figure()
        if varX is not None:
            _plt.errorbar(X.flatten(), Y.flatten(),
                          xerr=_np.sqrt(varX.flatten()),
                          yerr=_np.sqrt(varY.flatten()), fmt='bo' )
        elif varY is not None:
            _plt.errorbar(X.flatten(), Y.flatten(),
                          yerr=_np.sqrt(varY.flatten()), fmt='bo' )
        else:
            _plt.plot(X.flatten(), Y.flatten(), 'bo')
        # endif
        yo = a*xo+b
        vyo = Var_a*(xo)**2.0 + Var_b*(1.0)**2.0
#        vyo += 2.0*Cov_ab
        _plt.plot(xo, yo, 'r-')
        _plt.plot(xo, yo+_np.sqrt(vyo), 'r--')
        _plt.plot(xo, yo-_np.sqrt(vyo), 'r--')
        _plt.title('Linear regression')
    # endif

    if cov:
        return _np.asarray([a, b]), _np.asarray([[Var_a, Cov_ab],[Cov_ab, Var_b]])
    else:
        return a, b, Var_a, Var_b
    # end if
# end def linreg

# ======================================================================== #


def weightedPolyfit(xvar, yvar, xo, vary=None, deg=1, nargout=2):
    if vary is None:
        weights = _np.ones_like(yvar)
        vary = _np.abs(0.025*_np.nanmean(yvar[~_np.isinf(yvar)])*_np.ones_like(yvar))
    else:
        if (vary==0).any():
            vary[vary==0] = _np.finfo(float).eps
        # endif
        weights = 1.0/vary
    # end if

    # intel compiler error with infinite weight (zero variance ... fix above)
    if _np.isinf(weights).any():
#        xvar = xvar[~_np.isinf(weights)]
#        yvar = yvar[~_np.isinf(weights)]
        weights = weights[~_np.isinf(weights)]
#        weights[_np.isinf(weights)] = 1.0/_np.finfo(float).eps

    if len(xvar) == 2 and deg == 1:
        af = _np.polyfit(xvar, yvar, deg=deg, full=False, w=weights, cov=False)
        Vcov = _np.full((deg+1, deg+1), _np.nan)
        yf = af[0] * xo + af[1]
        varyf = _np.full_like(yf, _np.nan)
        dydf = af[0] * xo
        vardydf = _np.full_like(dydf, _np.nan)
        if nargout == 0:
            return af, Vcov
        elif nargout == 2:
            return yf, varyf
        elif nargout == 4:
            return yf, varyf, dydf, vardydf

#    af, Vcov = _np.polyfit(xvar, yvar, deg=deg, full=False, w=weights, cov=True)
#
#    # end try
##    if (len(xvar) - deg - 2.0) == 0.0:
#    if _np.isinf(Vcov).all():
##        print('insufficient data points (d.o.f.) for true covariance calculation in fitting routine')
#        # this concatenation effectively reduces the number of degrees of freedom ... it's a kluge
#        af, Vcov = _np.polyfit(_np.hstack((xvar, xvar[-1:])),
#                               _np.hstack((yvar, yvar[-1:])),
#                               deg=deg, full=False,
#                               w=_np.hstack( (weights, _np.finfo(float).eps) ),
#                               cov=True)
#    # endif
#
#    if nargout == 0:
#        return af, Vcov
#    # endif

    def _func(xvec, af, **fkwargs):
        return _ms.model_poly(xvec, af, npoly=deg)

    fitter = modelfit(xvar, yvar, ey=_np.sqrt(vary), XX=xo, func=_func)

    if nargout == 0:
        return fitter.params, fitter.covmat # fitter.perror
    elif nargout == 1:
        return fitter.prof
    elif nargout == 2:
        return fitter.prof, fitter.varprof
    elif nargout == 4:
        return fitter.prof, fitter.varprof, fitter.dprofdx, fitter.vardprofdx
# end def weightedPolyfit


# ======================================================================== #


def savitzky_golay(y, window_size, order, deriv=0):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.

    This code has been taken from http://www.scipy.org/Cookbook/SavitzkyGolay
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over an odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.savefig('images/golay.png')
    #plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = _np.abs(_np.int(window_size))
        order = _np.abs(_np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = _np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = _np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - _np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + _np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = _np.concatenate((firstvals, y, lastvals))
    return _np.convolve( m, y, mode='valid')

# ========================================================================== #

# =============================== #
# ---- leastsq dependent--------- #
# =============================== #


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


def gaussian_peak_width(tt,sig_in,param):
    """
     param contains intitial guesses for fitting gaussians,
     (Amplitude, x value, sigma):
     param = [[50,40,5],[50,110,5],[100,160,5],[100,220,5],
          [50,250,5],[100,260,5],[100,320,5], [100,400,5],
          [30,300,150]]  # this last one is our noise estimate

     Define a function that returns the magnitude of stuff under a gaussian
     peak (with support for multiple peaks)
    """
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

# =============================== #
# ---- curvefit dependent-------- #
# =============================== #

# ========================= #

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
    FitAlias = lambda *args: _ms.qparab_fit(args[0], args[1:], nohollow)

#    bounds = (lowbounds,upbounds)
#    method = 'trf'
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

# =================================== #
# --- scipy interpolate dependent --- #
# =================================== #


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

    pobj = _int.UnivariateSpline(xvar, yvar, w=1.0/_np.sqrt(vary), bbox=bbox, k=deg) #, check_finite=True)
    yf = pobj.__call__(xf)
    dydx = pobj.derivative(n=1).__call__(xf)
    return yf, dydx

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

    niterate = len(xvar)
    niterate *= nmonti

    dydx = _np.zeros( (niterate, nxf, nsh[1]), dtype=_np.float64)
    yf = _np.zeros( (niterate, nxf, nsh[1]), dtype=_np.float64)
    cc = -1
    for ii in range(niterate):
        utemp = yvar.copy()
        vtemp = vary.copy()
        cc += 1
        if cc >= nsh[0]:
            cc = 0
        # end if
        utemp[cc,:] += _np.sqrt(vary[cc,:])*_np.random.normal(0.0, 1.0, _np.shape(vary[cc,:]))
        vtemp[cc,:] = (utemp[cc,:]-yvar[cc,:])**2.0

        if func == 'pchip':
            yf[ii, :], dydx[ii, :] = pchip(xvar, utemp, xf)
        else:
            tmp1 = _np.zeros((nxf, nsh[1]), dtype=utemp.dtype)
            tmp2 = _np.zeros_like(tmp1)
            for jj in range(nsh[1]):
                tmp1[:,jj], tmp2[:,jj] = spline(xvar, utemp[:,jj], xf, vary=vtemp[:,jj], deg=deg, bbox=bbox)
            # end for
            yf[ii, :] = tmp1.reshape((nxf, nsh[1]), order='C')
            dydx[ii, :] = tmp2.reshape((nxf, nsh[1]), order='C')
        # endif
    # end for

    vardydx = _np.var(dydx, axis=0)
    dydx = _np.mean(dydx, axis=0)
    varf = _np.var( yf, axis=0)
    yf = _np.mean(yf, axis=0)

    return yf, varf, dydx, vardydx

# ======================================================================= #

# =================================== #
# ---------- fitNL dependent -------- #
# =================================== #


def fit_profile(rdat, pdat, vdat, rvec, **kwargs):
    scale_by_data = kwargs.get('scale_problem',True)
    arescale = kwargs.get('arescale',1.0)
    agradrho = kwargs.get('agradrho', 1.0)
    bootstrappit = kwargs.get('bootstrappit',False)
    af0 = kwargs.get('af0', None)
    LB = kwargs.get('LB', None)
    UB = kwargs.get('UB', None)

    if 1:
        modelfunc = kwargs.get('modelfunc', _ms.model_qparab);
    else:
        modelfunc = kwargs.get('modelfunc', _ms.model_2power);
    # end if

    # ==== #

    def func(af, XX):
        prof, _, _ = modelfunc(_np.abs(XX), af)
        return prof
#        return fitfunc(XX, af)

    def fgvec(af, XX):
        _, gvec, info = modelfunc(_np.abs(XX), af)
        # _, gvec, info = returngvec(_np.abs(XX), af)
        return gvec, info.dprofdx, info.dgdx

#    def derivfunc(af, XX):
#        _, _, info = modelfunc(_np.abs(XX), af)
#        # _, gvec, info = returngvec(_np.abs(XX), af)
#        return info.dprofdx
##        return fitderivfunc(XX, af)
    # ==== #

    info = modelfunc(XX=None)
    if af0 is None:
        af0 = info.af
    if LB is None:
        LB = info.Lbounds
    if UB is None:
        UB = info.Ubounds
    # end if

    # constrain things to be within r/a of 1.0 here, then undo it later
    rdat /= arescale

    if scale_by_data:
        pdat, vdat, slope, offset = _ms.rescale_problem(pdat, vdat)
#        af0[0] = 1.0
#        af0[1] = 0.0
    # end if

    isort = _np.argsort(_np.abs(rdat))
    rdat = _ut.cylsym_odd(rdat[isort].copy())
    pdat = _ut.cylsym_even(pdat[isort].copy())
    vdat = _ut.cylsym_even(vdat[isort].copy())

    options = dict()
#    options.setdefault('xtol', 1e-16) #
#    options.setdefault('ftol', 1e-16) #
#    options.setdefault('gtol', 1e-16) #
#    options.setdefault('nprint', 10) #
#    # Consider making epsfcn just 0.5 * nanmean(dx)... the max or min here doesn't make sense in most cases
##    options.setdefault('epsfcn', None) # 5e-4
#    options.setdefault('epsfcn', max((_np.nanmean(_np.diff(rdat.copy())),1e-3))) # 5e-4
#    options.pop('epsfcn') # 5e-4
#    options.setdefault('factor',100) # 100
#    options.setdefault('maxiter',1200)
    NLfit = fitNL(rdat, pdat, vdat, af0, func, LB=LB, UB=UB, **options)
    NLfit.run()

    if bootstrappit:
        NLfit.gvecfunc = fgvec
        NLfit.bootstrapper(xvec=_np.abs(rvec), weightit=False)

#        prof = NLfit.mfit
#        varp = NLfit.vfit
#        varp = varp.copy()
#
#        dprofdx = NLfit.dprofdx.copy()
#        vardprofdx = NLfit.vdprofdx.copy()
#    else:
#        prof, gvec, info = modelfunc(_np.abs(rvec), NLfit.af)
#        varp = NLfit.properror(_np.abs(rvec), gvec)
#        varp = varp.copy()
#
#        dprofdx = info.dprofdx.copy()
#        vardprofdx = NLfit.properror(_np.abs(rvec), info.dgdx)
    # end if
    prof, gvec, info = modelfunc(_np.abs(rvec), NLfit.af)
    varp = NLfit.properror(_np.abs(rvec), gvec)
#    varp = _np.abs(varp)
    varp = varp.copy()

    dprofdx = info.dprofdx.copy()
    vardprofdx = NLfit.properror(_np.abs(rvec), info.dgdx)
#    vardprofdx = _np.abs(vardprofdx)
    af = NLfit.af.copy()

    if scale_by_data:
        # slope = _np.nanmax(pdat)-_np.nanmin(pdat)
        # offset = _np.nanmin(pdat)

        info.prof = _np.copy(prof)
        info.varp = _np.copy(varp)
        info.dprofdx = _np.copy(dprofdx)
        info.vardprofdx = _np.copy(vardprofdx)
        info.af = _np.copy(af)
        info.slope = slope
        info.offset = offset
        prof, varp, dprofdx, vardprofdx, af = _ms.rescale_problem(info=info, nargout=5)
    # end if
    dlnpdrho = dprofdx / prof
    vardlnpdrho = (dlnpdrho)**2.0 * ( vardprofdx/(dprofdx**2.0) + varp/(prof**2.0)  )

    # ================== #
    # back to the grid where r/a can be greater than 1
    rdat *= arescale  # this is done locally here. only plot limits changed in fit_TS*
    rvec *= arescale

    dlnpdrho /= arescale       # This is a suspicious step
    vardlnpdrho /= arescale**2.0

#    if _np.isnan(prof[0]):
#        prof[0] = af[0].copy()
#        varp[0] = varp[1].copy()
#        dlnpdrho[0] = 0.0
#        vardlnpdrho[0] = vardlnpdrho[1].copy()
#    # end if

    if 0:
        _plt.figure()
        ax1 = _plt.subplot(2,1,1)
        ax3 = _plt.subplot(2,1,2, sharex=ax1)

        ax1.grid()
        ax3.grid()

        ax1.set_ylabel(r'Fit')
        ax3.set_ylabel(r'$a/L_\mathrm{f}$')
        ax3.set_xlabel(r'$r/a$')

        ax1.errorbar(rdat[rdat>0], pdat[rdat>0], yerr=_np.sqrt(vdat[rdat>0]), fmt='ko', color='k' )

        ax1.plot(rvec, prof, 'k-', lw=2)
#        ax1.plot(rvec, prof+_np.sqrt(varp), 'k--', lw=1)
#        ax1.plot(rvec, prof-_np.sqrt(varp), 'k--', lw=1)
        ax1.fill_between(rvec, prof-_np.sqrt(varp),
                               prof+_np.sqrt(varp),
                               interpolate=True, color='k', alpha=0.3)

        ax3.plot(rvec, agradrho*dlnpdrho, 'k-'),
#                 rvec, agradrho*(dlnpdrho+_np.sqrt(vardlnpdrho)), 'k--',
#                 rvec, agradrho*(dlnpdrho-_np.sqrt(vardlnpdrho)), 'k--')
        ax1.fill_between(rvec, agradrho*(dlnpdrho-_np.sqrt(vardlnpdrho)),
                               agradrho*(dlnpdrho+_np.sqrt(vardlnpdrho)),
                               interpolate=True, color='k', alpha=0.3)

    # end if
    return prof, varp, dlnpdrho, vardlnpdrho, af


def fit_TSneprofile(QTBdat, rvec, **kwargs):
    edge = kwargs.get('set_edge',None)
    iuse_ts = kwargs.get('iuse_ts', _np.ones( QTBdat["roa"].shape, dtype=bool))
    loggradient = kwargs.get('loggradient', True)
    plot_fd = kwargs.get('plot_fd', False)
    plotit = kwargs.get('plotit', False)
    agradrho = kwargs.get('agradrho',1.0)
    returnaf = kwargs.get('returnaf',False)
    arescale = kwargs.get('arescale', 1.0)
    bootstrappit = kwargs.get('bootstrappit',False)
    plotlims = kwargs.get('plotlims', None)
    fitin = kwargs.get('fitin', None)
    af0 = kwargs.get('af0', None)
    rescale_by_linavg = kwargs.get('rescale_by_linavg',False)

    QTBdat = QTBdat.copy()
    nkey = 'roa' if 'roan' not in QTBdat else 'roan'
    rvec = _np.copy(rvec)
    roa = _np.copy(QTBdat[nkey])
    QTBdat["roa"] = roa.copy()

    # set the edge density if necessary, then switch back to the fat grid
    # where r/a<1.0 (mostly) ... also discard channels based on settings in
    # clean_fitdict - used several places
    QTBdat = build_TSfitdict(QTBin=QTBdat, set_edge=edge,
                iuse_ts=iuse_ts*(~_np.isinf(roa))*(~_np.isnan(roa))*(QTBdat["ne"]>1e10))
    roa = _np.copy(QTBdat["roa"])  # fat grid!
    ne = _np.copy(QTBdat['ne'])    # [part/m3]
    varNL = _np.copy(QTBdat['varNL'])
    varNH = _np.copy(QTBdat['varNH'])
    if 'varRL' in QTBdat:
        varRL = _np.copy(QTBdat['varRL'])
        varRH = _np.copy(QTBdat['varRH'])
    # end if
    varn =  _np.copy(_np.sqrt(varNL*varNH))

    if fitin is None:
        nef, varnef, dlnnedrho, vardlnnedrho, af = fit_profile(
            roa.copy(), 1e-20*ne.copy(), 1e-40*varn.copy(), rvec,
            arescale=arescale, bootstrappit=bootstrappit, af0=af0)

        if rescale_by_linavg:
            rescale_by_linavg *= 1e-20
            iuse = ~(_np.isinf(nef) + _np.isnan(nef))
            nfdl, vnfdl, _, _ = _ut.trapz_var(rvec[iuse], nef[iuse], vary=varnef[iuse])
            nfdl /= _np.abs(_np.max(rvec)-_np.min(rvec))
            vnfdl /= _np.abs(_np.max(rvec)-_np.min(rvec))**2.0

            nef *= rescale_by_linavg/nfdl  # [1e20*part/m3]
            varnef *= (rescale_by_linavg/nfdl)**2.0

            af[0] *= rescale_by_linavg/af[0]   # [1e20*part/m3]
        # end if
    else:
        nef = 1e-20*fitin['prof']  # [1e20*part/m3]
        varnef = 1e-40*fitin['varprof']
        dlnnedrho = fitin['dlnpdrho']
        vardlnnedrho = fitin['vardlnpdrho']
        af = fitin['af']  # [1e20*part/m3]
    # end if

    # TODO!: watch out for nans here where r/a>=1
    # Convert back to absolute units (particles per m-3 not 1e20 m-3)
    nef *= 1e20
    varnef *= 1e40

    varlogne = varnef / nef**2.0
    logne = _np.log(nef)

    # ================== #

    if plotlims is None:
        plotlims = [-0.05, 1.05]
    # end if
    if arescale:
        plotlims[0] *= arescale  # fat grid
        plotlims[1] *= arescale
    # end if

    # ================== #

    if plotit:
        if plot_fd:
            if len(_np.atleast_1d(agradrho)) ==1:
                agr = agradrho*_np.ones_like(roa)
            else:
                agr = _ut.interp(rvec,agradrho,None,roa)
            dndx, vdndx = _dd.findiff1d(roa.copy(), ne.copy(), varn.copy())
            vdndx = agr**2.0 * (dndx / ne)**2.0 * ( vdndx/(dndx**2.0) + varn/(ne**2.0)  )
            dndx = -1.0*agr*dndx / ne
        # end if

        _plt.figure()
        ax1 = _plt.subplot(2,1,1)
        ax3 = _plt.subplot(2,1,2, sharex=ax1)

        ax1.grid()
        ax3.grid()

        ax1.set_title(r'Density Profile Info')
        ax1.set_ylabel(r'$n_\mathrm{e}\ \mathrm{in}\ 10^{20}\mathrm{m}^{-3}$')
        ax3.set_ylabel(r'$a/L_\mathrm{ne}$')
        ax3.set_xlabel(r'$r/a$')

        if 'varRL' in QTBdat:
            ax1.errorbar(roa, 1e-20*ne, xerr=[_np.sqrt(varRL), _np.sqrt(varRH)],
                     yerr=[1e-20*_np.sqrt(varNL), 1e-20*_np.sqrt(varNH)], fmt='bo') #, ecolor='r', elinewidth=2)
        else:
            ax1.errorbar(roa, 1e-20*ne, yerr=[1e-20*_np.sqrt(varNL), 1e-20*_np.sqrt(varNH)], fmt='bo') #, ecolor='r', elinewidth=2)
        # end if

        ax1.plot(rvec, 1e-20*nef, 'b-', lw=2)
        ylims = ax1.get_ylim()
        ax1.fill_between(rvec, 1e-20*(nef-_np.sqrt(varnef)), 1e-20*(nef+_np.sqrt(varnef)),
                                interpolate=True, color='b', alpha=0.3)
        ax1.set_xlim((plotlims[0],plotlims[1]))
#        ax1.set_ylim((0,1.1*ylims[1]))
        ax1.set_ylim((0.0, 6.0))
        if loggradient:
            idx = _np.where(_np.abs(rvec) < 0.05)
            plotdlnnedrho = dlnnedrho.copy()
            plotvardlnnedrho = vardlnnedrho.copy()
            plotdlnnedrho[idx] = _np.nan
            plotvardlnnedrho[idx] = _np.nan
            plotdlnnedrho *= -1*agradrho
            plotvardlnnedrho *= (agradrho)**2.0
        else:
            vardlnnedrho = (dlnnedrho/logne)**2.0 * (vardlnnedrho/dlnnedrho**2.0+varlogne/logne**2.0)
            dlnnedrho = dlnnedrho/logne
            plotdlnnedrho = -1*(agradrho) * dlnnedrho.copy()
            plotvardlnnedrho = ((agradrho)**2.0) * vardlnnedrho.copy()
         # end if
        if plot_fd:
            ax3.errorbar(roa, dndx, yerr=_np.sqrt(vdndx), fmt='bo')
        # end if
        ax3.plot(rvec, plotdlnnedrho, 'b-', lw=2)
        ax3.fill_between(rvec, plotdlnnedrho-_np.sqrt(plotvardlnnedrho),
                               plotdlnnedrho+_np.sqrt(plotvardlnnedrho),
                               interpolate=True, color='b', alpha=0.3)
        ax3.set_xlim((plotlims[0],plotlims[1]))
#        maxyy = min((_np.nanmax(1.05*(plotdlnnedrho+_np.sqrt(plotvardlnnedrho))),15))
        maxyy = 10.0
        ax3.set_ylim((0,maxyy))
        _plt.tight_layout()
    # end if plotit

    # ==================== #
    if returnaf:
        return logne, varlogne, dlnnedrho, vardlnnedrho, af
    return logne, varlogne, dlnnedrho, vardlnnedrho

# ======================================================================= #


def fit_TSteprofile(QTBdat, rvec, **kwargs):
    edge = kwargs.get('set_edge',None)
    iuse_ts = kwargs.get('iuse_ts', _np.ones( QTBdat["roa"].shape, dtype=bool))
    loggradient = kwargs.get('loggradient', True)
    plot_fd = kwargs.get('plot_fd', False)
    plotit = kwargs.get('plotit', False)
    agradrho = kwargs.get('agradrho', 1.00)
    returnaf = kwargs.get('returnaf',False)
    arescale = kwargs.get('arescale', 1.0)
    bootstrappit = kwargs.get('bootstrappit',False)
    plotlims = kwargs.get('plotlims', None)
    fitin = kwargs.get('fitin', None)
    af0 = kwargs.get('af0', None)

    QTBdat = QTBdat.copy()
    rvec = _np.copy(rvec)

    # set the edge Temperature if necessary, then switch back to the fat grid
    # where r/a<1.0 (mostly) ... also discard channels based on settings in
    # clean_fitdict - used several places
    QTBdat = build_TSfitdict(QTBin=QTBdat, set_edge=edge, iuse_ts=iuse_ts)
    roa = _np.copy(QTBdat['roa'])
    Te = _np.copy(QTBdat['Te'])
    varTL = _np.copy(QTBdat['varTL'])
    varTH = _np.copy(QTBdat['varTH'])
    if 'varRL' in QTBdat:
        varRL = _np.copy(QTBdat['varRL'])
        varRH = _np.copy(QTBdat['varRH'])
    # end if
    varT =  _np.copy(_np.sqrt(varTL*varTH))

    if fitin is None:
        Tef, varTef, dlnTedrho, vardlnTedrho, af = fit_profile(
            roa.copy(), Te.copy(), varT.copy(), rvec,
            arescale=arescale, bootstrappit=bootstrappit, af0=af0)
    else:
        Tef = fitin['prof']
        varTef = fitin['varprof']
        dlnTedrho = fitin['dlnpdrho']
        vardlnTedrho = fitin['vardlnpdrho']
        af = fitin['af']
    # end if
    varlogTe = _np.abs(varTef) / Tef**2.0
    logTe = _np.log(Tef)

    # ================== #

    if plotlims is None:
        plotlims = [-0.05, 1.05]
    # end if
    if arescale:
        plotlims[0] *= arescale
        plotlims[1] *= arescale
    # end if

    # ================== #

    if plotit:
        if plot_fd:
            if len(_np.atleast_1d(agradrho)) ==1:
                agr = agradrho*_np.ones_like(roa)
            else:
                agr = _ut.interp(rvec,agradrho,None,roa)
            dTdx, vdTdx = _dd.findiff1d(roa.copy(), Te.copy(), varT.copy())
            vdTdx = agr**2.0 * (dTdx / Te)**2.0 * ( vdTdx/(dTdx**2.0) + varT/(Te**2.0)  )
            dTdx = -1.0*agr*dTdx / Te
        # end if

        _plt.figure()
        ax1 = _plt.subplot(2,1,1)
        ax3 = _plt.subplot(2,1,2, sharex=ax1)

        ax1.grid()
        ax3.grid()

        ax1.set_title(r'Temperature Profile Info')
        ax1.set_ylabel(r'T$_e$ in KeV')
        ax3.set_ylabel(r'$a/L_\mathrm{Te}$')
        ax3.set_xlabel(r'$r/a$')

        if 'varRL' in QTBdat:
            ax1.errorbar(roa, Te, xerr=[_np.sqrt(varRL), _np.sqrt(varRH)],
                     yerr=[_np.sqrt(varTL), _np.sqrt(varTH)], fmt='ro') #, ecolor='r', elinewidth=2)
        else:
            ax1.errorbar(roa, Te, yerr=[_np.sqrt(varTL), _np.sqrt(varTH)], fmt='ro') #, ecolor='r', elinewidth=2)
        # end if

        ax1.plot(rvec, Tef, 'r-', lw=2)
        ylims = ax1.get_ylim();
        maxyy = min((ylims[1],12)) #min((_np.max(1.05*(Te+_np.sqrt(varTH))),12))
        ax1.fill_between(rvec, Tef-_np.sqrt(varTef), Tef+_np.sqrt(varTef),
                                    interpolate=True, color='r', alpha=0.3) # TODO!: watch for inf/nan's
        ax1.set_xlim((plotlims[0],plotlims[1]))
#        ax1.set_ylim((0,maxyy))

        if loggradient:
            idx = _np.where(_np.abs(rvec) < 0.05)
            plotdlnTedrho = dlnTedrho.copy()
            plotvardlnTedrho = vardlnTedrho.copy()
            plotdlnTedrho[idx] = _np.nan
            plotvardlnTedrho[idx] = _np.nan
            plotdlnTedrho *= -1*agradrho
            plotvardlnTedrho *= agradrho**2.0
        else:
            vardlnTedrho = (dlnTedrho/logTe)**2.0 * (vardlnTedrho/dlnTedrho**2.0+varlogTe/logTe**2.0)
            dlnTedrho = dlnTedrho/logTe
            plotdlnTedrho = -1*(agradrho) * dlnTedrho.copy()
            plotvardlnTedrho = (agradrho**2.0) * vardlnTedrho.copy()
        # end if
        if plot_fd:
            ax3.errorbar(roa, dTdx, yerr=_np.sqrt(vdTdx), fmt='ro')
        # end if

        ax3.plot(rvec, plotdlnTedrho, 'r-', lw=2)
        ax3.fill_between(rvec, plotdlnTedrho-_np.sqrt(plotvardlnTedrho),
                               plotdlnTedrho+_np.sqrt(plotvardlnTedrho),
                               interpolate=True, color='r', alpha=0.3)
        ax3.set_xlim((plotlims[0],plotlims[1]))
        maxyy = min((_np.nanmax(1.05*(plotdlnTedrho+_np.sqrt(plotvardlnTedrho))),15))
#        ax3.set_ylim((0,maxyy))
        _plt.tight_layout()
    # end if plotit

    # ==================== #
    # ==================== #
    if returnaf:
        return logTe, varlogTe, dlnTedrho, vardlnTedrho, af
    return logTe, varlogTe, dlnTedrho, vardlnTedrho

# ======================================================================= #
# ======================================================================= #


def extrap_TS(QTBdat, plotit=False):
    # Extrapolate along the line of sight of the Thomson to get edge channels
    if (QTBdat["roa"]>1.0).any() or _np.isnan(QTBdat["roa"]).any() or _np.isinf(QTBdat["roa"]).any():

        # Major radial coordinate of each Thomson channel
        rr = _np.sqrt( QTBdat['xyz'][:,0]**2.0 + QTBdat['xyz'][:,1]**2.0 )
        QTBdat['R'] = rr.copy()

        # get line-of-sight (by minimum and maximum channel, not by view coord)
        # cartesian coordinate / "length" along line-of-sight
        [rr, ll] = _ut.endpnts2vectors( QTBdat['xyz'][_np.argmax(rr),:],
                                        QTBdat['xyz'][_np.argmin(rr),:], nn = 200)
        rr = _np.sqrt(rr[:,0]**2.0 + rr[:,1]**2.0) # major radial coordinate

        # extrapolate r/a along line-of-sight to get r/a > 1.0 at R-coordinates outside LCFS
        isort, iunsort = _ut.argsort(QTBdat['R'].squeeze(), iunsort=True)

        iuse = ~((QTBdat["roa"][isort]>1.0) + _np.isnan(QTBdat["roa"][isort]) + _np.isinf(QTBdat["roa"][isort]))
        iuse = iuse.flatten()
        QTBdat["roa"] = _ut.interp((QTBdat['R'][isort])[iuse], (QTBdat["roa"][isort])[iuse], ei=None, xo=QTBdat['R'][isort])[iunsort]

        if plotit:
            _plt.figure()
            _plt.plot((QTBdat['R'][isort])[iuse], (QTBdat["roa"][isort])[iuse], 'x')
            _plt.plot(QTBdat['R'][isort], QTBdat["roa"][isort], 'o')
            _plt.xlabel('R [m]')
            _plt.xlabel('r/a')
            _plt.title('Thomson channel extrapolation')
        # end if
    # end if
    return QTBdat
# end def

def extrap_ECE(QMEdat, cls, rescale_amin=1.0, plotit=False):
    # Is this right?
#    qme_xyz_origin = _np.asarray([6.4632, 0.6907, 0.35],dtype=float) # [6.5, 6.1, 0.35], in cyl .... nearly 0 deg toroidally: AEE41 is ~-45 deg
#    qme_xyz_target = _np.asarray([5.52478, 0.506607, 0.0576283],dtype=float) # [-17,-5] in aim ang.
#
    qme_xyz_origin = _np.asarray([-4.731, -4.572, 0.272],dtype=float)
    qme_xyz_target = _np.asarray([-4.092, -3.704, 0.150],dtype=float)

    #Connect the origin and target by 'nn' points ... returning the
    # cartesian position and length along the central ray of the ECE diagnostic
    cls.targ2vec(qme_xyz_origin, qme_xyz_target)
    cls.setCoords(cls.rr)
    cls.roa *= rescale_amin   # r/a is now greater than 1.0 if starting on a weird fat grid
    cls.roa[_np.argmin(cls.roa):] *= -1.0   # HFS is negative by convention
    cls.Omegace = 1.6e-19*cls.modB/(9.11e-31)    # [rad/s], electron cyclotron frequency

    # Extrapolate along the line of sight of the ECE to get edge channels
    rr = _np.copy(cls.rr)
    roa = _np.copy(cls.roa).squeeze()

    fece = 1e-9*(2.0*cls.Omegace/(2*_np.pi))   # GHz
    fece = fece.squeeze()

    RR = _np.sqrt(cls.rr[:,0]**2.0+cls.rr[:,1]**2.0) # major radius
    Rmax = 6.15 # [m]
    iuse = (RR<Rmax).squeeze()
    if plotit:
        _plt.figure()
        _plt.subplot(2,1,1)
        _plt.plot(RR, cls.modB, '-')
        _plt.plot(RR[iuse], cls.modB[iuse], 'r-')
        _plt.ylabel('|B| [T]')
        _plt.subplot(2,1,2)
        _plt.plot(RR, fece, '-')
        _plt.plot(RR[iuse], fece[iuse], 'r-')
        _plt.xlabel('R [m]')
        _plt.ylabel(r'2f$_{ece}$ [GHz]')
    # end if

    if 1:
        # extrapolate r/a along line-of-sight to get r/a > 1.0 at R-coordinates outside LCFS
        # Do this by fitting the magnetic field along the view and interpolating B vs r/a
        isort, iunsort = _ut.argsort(QMEdat['ece_freq'].squeeze(), iunsort=True)
        iuse = ~((_np.abs(roa)>1.0) + _np.isnan(roa) + _np.isinf(roa))  # use w/in LCFS for fitting
        iuse = iuse*(RR<Rmax)
        iuse = iuse.squeeze()

        # interpolate to upper bandwidth bound (lower r/a for r/a increasing to LFS)
        ece_lr = weightedPolyfit(xvar=fece[iuse].squeeze(), yvar=roa[iuse],
                xo=(QMEdat["ece_freq"]+0.5*QMEdat["ece_bw"]).squeeze()[isort], vary=None, deg=3, nargout=1).squeeze()[iunsort]
        # interpolate to lower bandwidth bound
        ece_ur = weightedPolyfit(xvar=fece[iuse].squeeze(), yvar=roa[iuse],
                xo=(QMEdat["ece_freq"]-0.5*QMEdat["ece_bw"]).squeeze()[isort], vary=None, deg=3, nargout=1).squeeze()[iunsort]
        # interpolate to center frequency
        ece_roa = weightedPolyfit(xvar=fece[iuse].squeeze(), yvar=roa[iuse],
                xo=(QMEdat["ece_freq"]).squeeze()[isort], vary=None, deg=3, nargout=1).squeeze()[iunsort]

        # interpolate to real space coordinates along view for cold plasma resonances
        QMEdat["xyz"] = _np.array( (
            weightedPolyfit(xvar=fece[iuse].squeeze(), yvar=(rr[:,0].squeeze())[iuse],
              xo=(QMEdat["ece_freq"]).squeeze()[isort], vary=None, deg=3, nargout=1).squeeze()[iunsort],
            weightedPolyfit(xvar=fece[iuse].squeeze(), yvar=(rr[:,1].squeeze())[iuse],
              xo=(QMEdat["ece_freq"]).squeeze()[isort], vary=None, deg=3, nargout=1).squeeze()[iunsort],
            weightedPolyfit(xvar=fece[iuse].squeeze(), yvar=(rr[:,2].squeeze())[iuse],
              xo=(QMEdat["ece_freq"]).squeeze()[isort], vary=None, deg=3, nargout=1).squeeze()[iunsort]
                                   ), dtype=_np.float64).squeeze().T[iunsort]
    # end if
    # for plotting purposes make these something like error bars
    ece_lr = _np.abs(ece_roa - ece_lr)
    ece_ur = _np.abs(ece_roa - ece_ur)

    if plotit:
        _plt.figure()
        _plt.plot(fece[iuse], roa[iuse], 'x')
#        _plt.plot((QMEdat["ece_freq"]).squeeze(), ece_roa, 'o')
        _plt.errorbar((QMEdat["ece_freq"]).squeeze(), ece_roa,
                      xerr=[0.5*QMEdat["ece_bw"].squeeze(), 0.5*QMEdat["ece_bw"].squeeze()],
                      yerr=[ece_lr, ece_ur], fmt='o')
        _plt.title("Cold plasma ECE interpolation")
        _plt.ylabel('r/a')
        _plt.xlabel('f [GHz]')

        _plt.figure()
        _plt.subplot(3,1,1)
        _plt.plot(fece[iuse], (cls.rr[:,0])[iuse], 'x')
        _plt.plot((QMEdat["ece_freq"]).squeeze(), QMEdat["xyz"][:,0], 'o')
        _plt.ylabel('x [m]')
        _plt.title("Cold plasma ECE interpolation")
        _plt.subplot(3,1,2)
        _plt.plot(fece[iuse], (cls.rr[:,1])[iuse], 'x')
        _plt.plot((QMEdat["ece_freq"]).squeeze(), QMEdat["xyz"][:,1], 'o')
        _plt.ylabel('y [m]')
        _plt.subplot(3,1,3)
        _plt.plot(fece[iuse], (cls.rr[:,2])[iuse], 'x')
        _plt.plot((QMEdat["ece_freq"]).squeeze(), QMEdat["xyz"][:,2], 'o')
        _plt.ylabel('z [m]')
        _plt.xlabel('f [GHz]')
    # end if
    # by convention things outside of LCFS get r/a=10 in TRAVIS, follow that here
    ece_lr[_np.isnan(ece_lr)] = 10.0
    ece_ur[_np.isnan(ece_ur)] = 10.0
    ece_roa[_np.isnan(ece_roa)] = 10.0

    QMEdat['ece_roa'] = ece_roa
    QMEdat['ece_lr'] = ece_lr
    QMEdat['ece_ur'] = ece_ur
    return QMEdat

def add_sysTSerr(QTBin, sysTSerr=0.20, nargout=6):
#def add_sysTSerr(Te, TeL, TeH, ne, neL, neH, sysTSerr=0.20, nargout=6):
    Te = _np.copy(QTBin["Te"])
    TeL = _np.sqrt(QTBin['varTL'].copy())
    TeH = _np.sqrt(QTBin['varTH'].copy())
    ne = _np.copy(QTBin["ne"])
    neL = _np.sqrt(QTBin['varNL'].copy())
    neH = _np.sqrt(QTBin['varNH'].copy())

    Te = Te.flatten()
    TeL = TeL.flatten()
    TeH = TeH.flatten()
    ne = ne.flatten()
    neL = neL.flatten()
    neH = neH.flatten()

   # Add in an estimate of the systematic error
    izero = (Te==0)+(TeL==0)+(TeH==0)
    Te[Te==0], TeL[izero], TeH[izero] = 1e-3, min((_np.nanmean(TeL[~izero]),_np.max(TeL))), min((_np.nanmean(TeH[~izero]),_np.max(TeH)))
    TeL = _np.sqrt(Te**2.0*((TeL/Te)**2.0 + sysTSerr**2.0))# + 0.5**2.0)
    TeH = _np.sqrt(Te**2.0*((TeH/Te)**2.0 + sysTSerr**2.0))# + 0.5**2.0)

    izero = (ne==0)+(neL==0)+(neH==0)
    ne[ne==0], neL[izero], neH[izero] = 1e16, min((_np.nanmean(neL[~izero]),_np.max(neL))), min((_np.nanmean(neH[~izero]),_np.max(neH)))
    neL = _np.sqrt(ne**2.0*((neL/ne)**2.0 + sysTSerr**2.0))
    neH = _np.sqrt(ne**2.0*((neH/ne)**2.0 + sysTSerr**2.0))

    TeL[_np.isnan(TeL)] = _np.nanmean(TeL[~_np.isnan(TeL)])
    TeH[_np.isnan(TeL)] = _np.nanmean(TeH[~_np.isnan(TeH)])
    neL[_np.isnan(TeL)] = _np.nanmean(neL[~_np.isnan(neL)])
    neH[_np.isnan(TeL)] = _np.nanmean(neH[~_np.isnan(neH)])

    if nargout==1:
        QTBout = QTBin.copy()
        QTBout["Te"] = Te
        QTBout["TeL"] = TeL
        QTBout["TeH"] = TeH
        QTBout["varTL"] = TeL**2.0
        QTBout["varTH"] = TeH**2.0
        QTBout["ne"] = ne
        QTBout["neL"] = neL
        QTBout["neH"] = neH
        QTBout["varNL"] = neL**2.0
        QTBout["varNH"] = neH**2.0
        return QTBout
    return Te, TeL, TeH, ne, neL, neH

def build_TSfitdict(QTBin, set_edge=None, iuse_ts=None, rescale_amin=1.0):
    if iuse_ts is not None:
        iuse_ts = _np.ones(_np.shape(QTBin['roa']), dtype=bool)
    # end if

    dictdat = dict()
    if 'varRL' in QTBin:
        dictdat['varRL'] =  _np.hstack((QTBin['varRL'].copy(), _np.nanmean(QTBin['varRL'].copy())))
        dictdat['varRH'] =  _np.hstack((QTBin['varRH'].copy(), _np.nanmean(QTBin['varRH'].copy())))
    # end if
    dictdat['roa'] =  QTBin['roa'].copy()  # on the r/a > 1.0 grid
    dictdat['Te'] =   QTBin["Te"].copy()   # []
    dictdat['ne'] =  QTBin["ne"].copy()    # [part/m3]

    if "TeL" in dictdat:
        dictdat['varTL'] = QTBin["TeL"].copy()**2.0
        dictdat['varTH'] = QTBin["TeH"].copy()**2.0
        dictdat['varNL'] = QTBin["neL"].copy()**2.0
        dictdat['varNH'] = QTBin["neH"].copy()**2.0
    else:
        dictdat['varTL'] = QTBin["varTL"].copy()
        dictdat['varTH'] = QTBin["varTH"].copy()
        dictdat['varNL'] = QTBin["varNL"].copy()
        dictdat['varNH'] = QTBin["varNH"].copy()
    # end if
    if set_edge is not None:
        redge = set_edge[0]
        nedge = set_edge[1]
        Tedge = set_edge[2]

        dictdat['roa'] =  _np.hstack((dictdat['roa'], max(redge,0.1+_np.max(dictdat['roa']))))
        dictdat['Te'] =  _np.hstack((dictdat['Te'], Tedge)) # Te
        dictdat['ne'] =  _np.hstack((dictdat['ne'], nedge)) # ne

        dictdat['varTL'] = _np.hstack((dictdat['varTL'], _np.nanmean(QTBin["TeL"].copy())**2.0))
        dictdat['varTH'] = _np.hstack((dictdat['varTH'], _np.nanmean(QTBin["TeH"].copy())**2.0))
        dictdat['varNL'] = _np.hstack((dictdat['varNL'], _np.nanmean(QTBin["neL"].copy())**2.0))
        dictdat['varNH'] = _np.hstack((dictdat['varNH'], _np.nanmean(QTBin["neH"].copy())**2.0))

#        dictdat['TeL'] = _np.sqrt(dictdat["varTL"])
#        dictdat['TeH'] = _np.sqrt(dictdat["varTH"])
#        dictdat['neL'] = _np.sqrt(dictdat["varNL"])
#        dictdat['neH'] = _np.sqrt(dictdat["varNH"])

        iuse_ts = _np.hstack((iuse_ts,True))
    else:
        Tedge = 0.0
    # end if
    dictdat['roa'] /= rescale_amin # back to the fat grid (most points r/a<1.0)
    dictdat['roan'] = dictdat['roa'].copy()

    dictdat, iuse = clean_fitdict(dictdat, iuse=iuse_ts, rmax=9.0, Tmin=0.5*Tedge, Tmax=9.0)
    return dictdat

def concat_Tdat(dictdat, newdat=None):
    addDat = False
    if newdat is not None:
        addDat=True
        if "chmsk" in newdat:
            chmsk = newdat["chmsk"]
        else:
            chmsk = _np.ones( newdat["roa"].shape, dtype=bool)
        # end if
        rho_use = _np.abs(newdat["roa"].copy())
        rho_min = newdat["roaL"].copy()
        rho_max = newdat["roaH"].copy()
        Tdat = newdat["Te"]
        if "varTL" in newdat:
            Tvar = _np.sqrt(newdat["varTL"]*newdat["varTH"])
        else:
            Tvar = _np.abs(newdat["TeL"]*newdat["TeH"])
    # end if

    if addDat:
        dictdat['roa'] = _np.hstack((rho_use[chmsk].copy(), dictdat['roa'].copy()))
        dictdat['Te'] = _np.hstack((Tdat[chmsk].copy(), dictdat['Te'].copy())) # Te
        if "varTL" in dictdat:
            dictdat['varTL'] = _np.hstack((Tvar[chmsk].copy(), dictdat['varTL'])) #TeL
            dictdat['varTH'] = _np.hstack((Tvar[chmsk].copy(), dictdat['varTH'])) #TeH
        else:
            dictdat['TeL'] = _np.hstack((_np.sqrt(Tvar[chmsk].copy()), dictdat['TeL'])) #TeL
            dictdat['TeH'] = _np.hstack((_np.sqrt(Tvar[chmsk].copy()), dictdat['TeH'])) #TeH
        # end if
        if 'varRL' in dictdat:
            dictdat['varRL'] = _np.hstack((rho_min[chmsk].copy()**2.0, dictdat['varRL']))
            dictdat['varRH'] = _np.hstack((rho_max[chmsk].copy()**2.0, dictdat['varRH']))
        # end if
    # end if
    return dictdat

def sort_fitdict(dictdat):
    isort = _np.argsort(_np.abs(dictdat['roa']).squeeze())
    dictdat['roa'] = _np.abs(dictdat['roa'][isort])
    dictdat['Te'] = dictdat['Te'][isort]
    if 'varTL' in dictdat:
        dictdat['varTL'] = dictdat['varTL'][isort]
        dictdat['varTH'] = dictdat['varTH'][isort]
    else:
        dictdat['TeL'] = dictdat['TeL'][isort]
        dictdat['TeH'] = dictdat['TeH'][isort]
    # end if
    if 'varRL' in dictdat:
        dictdat['varRL'] = dictdat['varRL'][isort]
        dictdat['varRH'] = dictdat['varRH'][isort]
    # end if
    return dictdat

def clean_fitdict(dictdat, iuse=None, rmax=9.0, Tmin=0.0, Tmax=9.0):
    if iuse is None:
        iuse = _np.ones(_np.shape(dictdat['roa']), dtype=bool)
    # end if
    iuse = iuse*(~_np.isinf(dictdat['roa']))*(~_np.isnan(dictdat['roa']))*(dictdat['Te']>Tmin)*(_np.abs(dictdat['roa'])<rmax)*(_np.abs(dictdat['Te'])<Tmax)*(dictdat['roa']!=0.0)
    dictdat['roa'] = dictdat['roa'][iuse]
    dictdat['Te'] = dictdat['Te'][iuse]
    if "TeL" in dictdat:
        dictdat['TeL'] = dictdat['TeL'][iuse]
        dictdat['TeH'] = dictdat['TeH'][iuse]

#        dictdat['varTL'] = dictdat['TeL']**2.0
#        dictdat['varTH'] = dictdat['TeH']**2.0
    else:
        dictdat['varTL'] = dictdat['varTL'][iuse]
        dictdat['varTH'] = dictdat['varTH'][iuse]

#        dictdat['TeL'] = _np.sqrt(dictdat['varTL'])
#        dictdat['TeH'] = _np.sqrt(dictdat['varTH'])
    # end if

    if 'varRL' in dictdat:
        dictdat['varRL'] = dictdat['varRL'][iuse]
        dictdat['varRH'] = dictdat['varRH'][iuse]
    # end if
    if "ne" in dictdat and dictdat['ne'].shape==iuse.shape:
        dictdat['ne'] = dictdat['ne'][iuse]
        if "neL" in dictdat:
            dictdat['neL'] = dictdat['neL'][iuse]
            dictdat['neH'] = dictdat['neH'][iuse]

#            dictdat['varNL'] = dictdat['neL']**2.0
#            dictdat['varNH'] = dictdat['neH']**2.0
        else:
            dictdat['varNL'] = dictdat['varNL'][iuse]
            dictdat['varNH'] = dictdat['varNH'][iuse]

#            dictdat['neL'] = _np.sqrt(dictdat['varNL'])
#            dictdat['neH'] = _np.sqrt(dictdat['varNH'])
        # end if
    # end if
    return dictdat, iuse

def prep_Tdat(rdat, Tdat, Tvar, rlow=None, rhigh=None,
              edge=None, rescale_amin=1.0):

    rdat = _np.abs(rdat.copy())/rescale_amin
    isort = _np.argsort(rdat.squeeze())
    rdat = rdat[isort]
    Tdat = Tdat.copy()[isort]
    Tvar = Tvar.copy()[isort]

    if edge is not None:
        redge = edge[0]/rescale_amin
        Tedge = edge[1]

        rdat = _np.hstack((rdat, max(redge,redge*max(rdat))))
        Tdat = _np.hstack((Tdat, Tedge))
        Tvar = _np.hstack((Tvar, _np.nanmean(Tvar)))
    else:
        Tedge = 0.0
    # end if

    rdat = _ut.cylsym_odd(rdat)
    Tdat = _ut.cylsym_even(Tdat)
    Tvar = _np.sqrt(_ut.cylsym_even(Tvar))

    dictout, iuse = clean_fitdict({"roa":rdat, "Te":Tdat, "varTL":Tvar, "varTH":Tvar}, Tmin=0.5*Tedge)
    rdat = dictout["roa"]
    Tdat = dictout["Te"]
    Tvar = dictout["varTL"]

    if rlow is not None:
        rlow = rlow.copy()/rescale_amin
        rlow = rlow[isort]
        rlow = _np.hstack((rlow, _np.nanmean(rlow)))
        rlow = _ut.cylsym_even(rlow)
        rlow = rlow[iuse]
    if rhigh is not None:
        rhigh = rhigh.copy()/rescale_amin
        rhigh = rhigh[isort]
        rhigh = _np.hstack((rhigh, _np.nanmean(rlow)))
        rhigh = _ut.cylsym_even(rhigh)
        rhigh = rhigh[iuse]
    if rlow is not None and rhigh is not None:
        return rdat, Tdat, Tvar, rlow, rhigh
    elif rlow is not None:
        return rdat, Tdat, Tvar, rlow
    else:
        return rdat, Tdat, Tvar
    # end if
# end def

# ======================================================================= #
# ======================================================================= #


# =========================== #
# ---------- testing -------- #
# =========================== #

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


def test_linreg():
    x = _np.linspace(-2, 15, 100)
    af = [0.2353335600009, 3.1234563234]
    y =  af[0]* x + af[1]
    y += 0.1*_np.sin(0.53 * 2*_np.pi*x/(_np.max(x)-_np.min(x)))
#    vary = None
    vary = ( 0.3*_np.mean(y)*_np.random.normal(0.0, 1.0, len(y)) )**2.0

    a, b, Var_a, Var_b = linreg(x, y, verbose=True, varY=vary, varX=None, cov=False, plotit=True)
    print(a-af[0], b-af[1] )

def test_derivatives():

    x, y, vary = test_dat(multichannel=True)

    # ======================= #

    # xd = _np.linspace(0, 15, 100)
    # _plt.plot(x, y, "o")

#    p , e = curve_fit(piecewise_2line, x, y)
#    _plt.plot(xd, piecewise_2line(xd, *p))

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

    dydx, vardydx = _dd.deriv_bsgaussian(x, y, vary, axis=0, nmonti=300,
                                     sigma=0.8, mode='nearest')

    yxp, yxvp = _dd.deriv_bsgaussian(x, y, vary, axis=0, nmonti=300,
                                 sigma=0.8, mode='nearest', derivorder=0)

#
#    dydx0, vardydx0 = _dd.findiff1d(x, y, vary, order=1)
##    dydx2,vardydx2 = _dd.findiff1d(x, y, vary, order=2)
##    dydx4,vardydx4 = _dd.findiff1d(x, y, vary, order=4)
#
#    ndydx0, nvardydx0 = _dd.findiffnp( x, y, vary, order=1 )
#    ndydx2, nvardydx2 = _dd.findiffnp( x, y, vary, order=2 )
##    dydx4,vardydx4 = _dd.findiff1dr(x, y, vary)
#
#    # integrate derivative and compare to source
#    _, _, yx, yxv = _ut.trapz_var(x, dydx, None, vardydx)
#    yx += (y[0] - _ut.interp(x, yx, ei=None, xo=x[0]))
#
#    _, _, yx0, yxv0 = _ut.trapz_var(x, dydx0, None, vardydx0)
#    yx0 += (y[0] - _ut.interp(x, yx0, ei=None, xo=x[0]))
#
#    _, _, nyx0, nyxv0 = _ut.trapz_var(x, ndydx0, None, nvardydx0)
#    nyx0 += (y[0] - _ut.interp(x, nyx0, ei=None, xo=x[0]))
#
##    _, _, yx2, yxv2 = _ut.trapz_var(x, dydx2, None, vardydx2)
##    yx2 += (y[0] - _ut.interp(x, yx2, ei=None, xo=x[0]))
#
#    _, _, nyx2, nyxv2 = _ut.trapz_var(x, ndydx2, None, nvardydx2)
#    nyx2 += (y[0] - _ut.interp(x, nyx2, ei=None, xo=x[0]))
#
##    _, _, yx4, yxv4 = _ut.trapz_var(x, dydx4, None, vardydx4)

    _, _, yx, yxv = _ut.trapz_var(x, dydx, None, vardydx)
    yx += (y[0] - _ut.interp(x, yx, ei=None, xo=x[0]))
#
#    _, _, yx0, yxv0 = _ut.trapz_var(x, dydx0, None, vardydx0)
#    yx0 += (y[0] - _ut.interp(x, yx0, ei=None, xo=x[0]))
#
#    _, _, nyx0, nyxv0 = _ut.trapz_var(x, ndydx0, None, nvardydx0)
#    nyx0 += (y[0] - _ut.interp(x, nyx0, ei=None, xo=x[0]))
#
##    _, _, yx2, yxv2 = _ut.trapz_var(x, dydx2, None, vardydx2)
##    yx2 += (y[0] - _ut.interp(x, yx2, ei=None, xo=x[0]))
#
#    _, _, nyx2, nyxv2 = _ut.trapz_var(x, ndydx2, None, nvardydx2)
#    nyx2 += (y[0] - _ut.interp(x, nyx2, ei=None, xo=x[0]))
#
##    _, _, yx4, yxv4 = _ut.trapz_var(x, dydx4, None, vardydx4)
##    yx4 += (y[0] - _ut.interp(x, yx4, ei=None, xo=x[0]))


    # ==== #

    _plt.figure()

    ax1 = _plt.subplot(2,1,1)
    ax1.plot(x, y, "ko")

    # Integrals
    ax1.plot(x, yx, 'k-',
             x, yx+_np.sqrt(yxv), 'k--',
             x, yx-_np.sqrt(yxv), 'k--')
#
    ax1.plot(x, yxp, 'g-',
             x, yxp+_np.sqrt(yxvp), 'g--',
             x, yxp-_np.sqrt(yxvp), 'g--')


#    ax1.plot(x, yx0, 'r-',
#             x, yx0+_np.sqrt(yxv0), 'r--',
#             x, yx0-_np.sqrt(yxv0), 'r--')

#    ax1.plot(x, nyx0, 'b-',
#             x, nyx0+_np.sqrt(nyxv0), 'b--',
#             x, nyx0-_np.sqrt(nyxv0), 'b--')

#    ax1.plot(x, yxpc, 'm-',
#             x, yxpc+_np.sqrt(yxvpc), 'm--',
#             x, yxpc-_np.sqrt(yxvpc), 'm--')

#    ax1.plot(x, yx2, 'g-',
#             x, yx2+_np.sqrt(yxv2), 'g--',
#             x, yx2-_np.sqrt(yxv2), 'g--')

#    ax1.plot(x, nyx2, 'm-',
#             x, nyx2+_np.sqrt(nyxv2), 'm--',
#             x, nyx2-_np.sqrt(nyxv2), 'm--')

#    ax1.plot(x, yx4, 'y-',
#             x, yx4+_np.sqrt(yxv4), 'y--',
#             x, yx4-_np.sqrt(yxv4), 'y--')

    # Derivatives
    ax2 = _plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(x, dydx, 'k-',
             x, dydx+_np.sqrt(vardydx), 'k--',
             x, dydx-_np.sqrt(vardydx), 'k--')

#    ax2.plot(x, dydxp, 'g-',
#             x, dydxp+_np.sqrt(vardydxp), 'g--',
#             x, dydxp-_np.sqrt(vardydxp), 'g--')

#    ax2.plot(x, dydxpc, 'm-',
#             x, dydxpc+_np.sqrt(vardydxpc), 'm--',
#             x, dydxpc-_np.sqrt(vardydxpc), 'm--')

#    ax2.plot(x, dydx0, 'r-',
#             x, dydx0+_np.sqrt(vardydx0), 'r--',
#             x, dydx0-_np.sqrt(vardydx0), 'r--')

#    ax2.plot(x, ndydx0, 'b-',
#             x, ndydx0+_np.sqrt(nvardydx0), 'b--',
#             x, ndydx0-_np.sqrt(nvardydx0), 'b--')

#    ax2.plot(x, dydx2, 'g-',
#             x, dydx2+_np.sqrt(vardydx2), 'g--',
#             x, dydx2-_np.sqrt(vardydx2), 'g--')

#    ax2.plot(x, ndydx2, 'm-',
#             x, ndydx2+_np.sqrt(nvardydx2), 'm--',
#             x, ndydx2-_np.sqrt(nvardydx2), 'm--')

##    ax2.plot(x, dydx4, 'y-',
##             x, dydx4+_np.sqrt(vardydx4), 'y--',
##             x, dydx4-_np.sqrt(vardydx4), 'y--')
# end main()


if __name__=="__main__":
    try:
        from FIT.model_spec import model_qparab
    except:
        from .model_spec import model_qparab
    # end try
    info = model_qparab(XX=None)
    af = info.af
    af = _np.asarray([af[ii]+0.05*af[ii]*_np.random.normal(0.0, 1.0, 1) for ii in range(len(af))],
                     dtype=_np.float64)

    QTBdat = {}
    QTBdat['roa'] = _np.linspace(0.05, 1.05, num=10)
    QTBdat['ne'], _, _ = model_qparab(QTBdat['roa'], af)
    QTBdat['ne'] *= 1e20
    QTBdat['varNL'] = (0.1*QTBdat['ne'])**2.0
    QTBdat['varNH'] = (0.1*QTBdat['ne'])**2.0

    aT = _np.asarray([4.00, 0.07, 5.0, 2.0, 0.04, 0.50], dtype=_np.float64)
    aT = _np.asarray([aT[ii]+0.05*aT[ii]*_np.random.normal(0.0, 1.0, 1) for ii in range(len(aT))],
                     dtype=_np.float64)
    QTBdat['Te'], _, _ = model_qparab(QTBdat['roa'], aT)
    QTBdat['varTL'] = (0.1*QTBdat['Te'])**2.0
    QTBdat['varTH'] = (0.1*QTBdat['Te'])**2.0

    nout = fit_TSneprofile(QTBdat, _np.linspace(0, 1.05, num=51), plotit=True,
                           agradrho=1.20, returnaf=False, bootstrappit=False)

    Tout = fit_TSteprofile(QTBdat, _np.linspace(0, 1.05, num=51), plotit=True,
                           agradrho=1.20, returnaf=False, bootstrappit=False)
#    test_linreg()
#    test_derivatives()

#    x = _np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15], dtype=float)
#    y = _np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])
#
#    p , e = curve_fit(piecewise_2line, x, y)
#
#    xd = _np.linspace(0, 15, 100)
#    _plt.plot(x, y, "o")
#    _plt.plot(xd, piecewise_2line(xd, *p))
#endif

# ======================================================================== #
# ======================================================================== #














