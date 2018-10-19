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
from scipy import ndimage as _ndimage
from scipy.optimize import curve_fit, leastsq
import matplotlib.pyplot as _plt
import numpy as _np

#from pybaseutils.Struct import Struct
from pybaseutils import utils as _ut

try:
    from FIT import model_spec as _ms
    from FIT.fitNL import fitNL, modelfit
except:
    from . import model_spec as _ms
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
#if _scipyversion >= 0.17:
##    print("Using a new version of scipy")
#    from scipy.optimize import least_squares
##else:
##    print("Using an older version of scipy")
## endif

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

    af, Vcov = _np.polyfit(xvar, yvar, deg=deg, full=False, w=weights, cov=True)

    # end try
#    if (len(xvar) - deg - 2.0) == 0.0:
    if _np.isinf(Vcov).all():
#        print('insufficient data points (d.o.f.) for true covariance calculation in fitting routine')
        # this concatenation effectively reduces the number of degrees of freedom ... it's a kluge
        af, Vcov = _np.polyfit(_np.hstack((xvar, xvar[-1:])),
                               _np.hstack((yvar, yvar[-1:])),
                               deg=deg, full=False,
                               w=_np.hstack( (weights, _np.finfo(float).eps) ),
                               cov=True)
    # endif

    if nargout == 0:
        return af, Vcov
    # endif

    def _func(xvec, af, **fkwargs):
        return _ms.model_poly(xvec, af, npoly=deg)

    fitter = modelfit(xvar, yvar, ey=_np.sqrt(vary), XX=xo, func=_func)

    if nargout == 1:
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


# ======================================================================== #


def _derivative_inputcondition(xvar):
    """
    function [xvar, xsh, transp] = _derivative_inputcondition(xvar)

    Required for the derivative functions below.

    Inputs:
        xvar - input array of shape less than 3D

    Outputs:
        xvar - the new array of size (#channels,len(data))
            if shape(xvar) == (len(data),), xvar has shape (1,len(data,1))
        xsh  - original shape of the input array (len(data),)
        transp - Boolean - was the data transposed?

    Converts the input to a minimum 2D numpy array (for multi-channel
    compatability), then checks whether the data is stored column-wise or
    row-wise.  Column-wise is faster, but the current incarnation of the
    finite difference and derivative codes are written so that data is stored
    row-wise.  This should be changed in the future.

    The final two arguments are used for matching the output data format to
    the input data format.
        ex from findiff1d //

        def findiff1d(xvar, yvar, vary, ...)
            xvar, _, _ = _derivative_inputcondition(xvar)
            yvar, ysh, ytransp = _derivative_inputcondition(yvar)
            ...
            [code body]
            ...
            vardydx = vardydx.reshape(ysh)
            dydx = dydx.reshape(ysh)

            return dydx, vardydx
    """
    xsh = _np.shape(xvar)
    xvar = _np.atleast_2d(xvar)
    transp = False
    if _np.size(xvar, axis=0) > _np.size(xvar, axis=1):
        transp = True
        xvar = xvar.T
    # endif
    return xvar, xsh, transp
# end def _derivative_inputcondition


# ======================================================================== #


def findiffnp(xvar, u, varu=None, order=1):

    # Input data formatting.  All of this is undone before output
    # xvar, _, _ = _derivative_inputcondition(xvar)
    if varu is None: varu = _np.zeros_like(u)   # endif
    u, ush, transp = _derivative_inputcondition(u)
    varu, _, _ = _derivative_inputcondition(varu)
    if transp:
        # xvar = xvar.T
        u = u.T
        varu = varu.T
    # endif
    nsh = _np.shape(u)

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

    # Works with Matrices:
    dx = _np.gradient(xvar)

    # Pre-allocate
    nmonti = 300
    dudx = _np.zeros((nmonti, nsh[0], nsh[1]), dtype=_np.float64)
    for ii in range(nmonti):

        # Wiggle the input data within its statistical uncertainty
        # utemp = _np.random.normal(0.0, 1.0, _np.size(u, axis=axis))
        # utemp = utemp.reshape(nsh[0], nsh[1])
        utemp = _np.random.normal(0.0, 1.0, _np.shape(u))
        utemp = u + _np.sqrt(varu)*utemp

        utemp = _np.gradient(utemp, dx)

        dudx[ii, :, :] = (utemp.copy()).reshape(nsh[0], nsh[1])
    # endfor

    # Take mean and variance of the derivative
    vardudx = _np.nanvar(dudx, axis=0)
    dudx = _np.nanmean(dudx, axis=0)

    if _np.size(dudx, axis=0) == 1:
        dudx = dudx.T
        vardudx = vardudx.T
    # endif

    return dudx.reshape(ush), vardudx.reshape(ush)

# ===================================================================== #


def findiff1d(xvar, u, varu=None, order=1):
    """
    function [dudx,ngradu,u,xvar]=findiff1d(xvar, u, varu, order)

    Calculates the 1D derivative using hybrid finited differences

    Inputs:
       xvar - The independent variable
       u    - The dependent variable
       varu - variance in dependent variable
       order - Accuracy of the finite difference scheme (2nd, 4th, or 6th)

    Outputs:
       dudx   - Approximation to the derivative of u with respect to xvar
       vardudx - Variance in derivative (error propagation from varu)

    A 1D finite-differencing scheme that keeps the most points.
    Centered finite difference on all interior points, and forward
    difference on the left-boundary, backward difference on the right
    boundary.
    """

    # ================================================== #

    xvar, _, _ = _derivative_inputcondition(xvar)
    u, ush, utransp = _derivative_inputcondition(u)

    # ================================================== #

    nr, nc = _np.shape(u)
    dudx = _np.zeros((nr, nc), dtype=_np.float64)
    if nr > 1:
        nr = nr-1  # Python compatability
    # end if

    # Works with Matrices:
    ii = 0
    while _np.isinf(xvar[:, ii+1]-xvar[:, ii]).any():  # avoid ln(0) = Inf
        ii = ii + 1
        if ii > 5:
            return
        # end if
    # end while
    dx = xvar[0, ii+1]-xvar[0, ii]

    dudx[0:nr, 0] = (u[0:nr, 1] - u[0:nr, 0])/dx
    dudx[0:nr, -1] = (u[0:nr, -1] - u[0:nr, -2])/dx
    dudx[0:nr, 1:-1] = _np.diff(u[0:nr, :-1], n=1, axis=1)/dx
    dudx[0:nr, 1:-1] += _np.diff(u[0:nr, 1:], n=1, axis=1)/dx
    dudx[0:nr, 1:-1] = 0.5*dudx[0:nr, 1:-1]
    # 0.5*(_np.diff(u[0:nr, :-1], n=1, axis=1)/dx
    #      + _np.diff(u[0:nr,1:],n=1,axis=1)/dx )

    if nc > 5 and order > 2:
        # 4th order accurate centered finite differencing across middle
        dudx[0:nr, 2:-3] = (8*(u[0:nr, 3:-2] - u[0:nr, 1:-4])
                            + (u[0:nr, 4:-1] - u[0:nr, 0:-5]))/(12*dx)
    # endif

    if nc > 7 and order > 4:
        # 6th order accurate across middle:
        dudx[0:nr, 3:-4] = (45*(u[0:nr, 4:-3] - u[0:nr, 2:-5])
                            - 9*(u[0:nr, 5:-2] - u[0:nr, 1:-6])
                            + (u[0:nr, 6:-1] - u[0:nr, 0:-7]))/(60*dx)
    # endif

    # =============================================================== #

    if utransp:
        dudx = dudx.T
    # endif
    dudx = dudx.reshape(ush)

    if varu is not None:
        varu, _, _ = _derivative_inputcondition(varu)

        vardudx = _np.zeros_like(u)
        vardudx[0:nr, 0] = (varu[0:nr, 1] + varu[0:nr, 0])/dx**2
        vardudx[0:nr, -1] = (varu[0:nr, -1] + varu[0:nr, -2])/dx**2
        vardudx[0:nr, 1:-1] = \
            (0.5/dx)**2*(varu[0:nr, 0:-2]+2*varu[0:nr, 1:-1]+varu[0:nr, 2:])

        if nc > 5 and order > 2:
            # 4th order accurate centered finite differencing across middle
            vardudx[0:nr, 2:-3] = \
                ((8**2)*(varu[0:nr, 3:-2] + varu[0:nr, 1:-4])
                 + (1**2)*(varu[0:nr, 4:-1] + varu[0:nr, 0:-5]))/(12*dx)**2
        # endif

        if nc > 7 and order > 4:
            vardudx[0:nr, 3:-4] = \
                ((45**2)*(varu[0:nr, 4:-3] + varu[0:nr, 2:-5])
                 + (9**2)*(varu[0:nr, 5:-2]+varu[0:nr, 1:-6])
                 + (1**2)*(varu[0:nr, 6:-1]+varu[0:nr, 0:-7]))/(60*dx)**2
        # endif

        # ================================ #

        if utransp:
            vardudx = vardudx.T
        # endif
        vardudx = vardudx.reshape(ush)

        return dudx, vardudx
    else:
        return dudx
    # endif
# end def findiff1d

# ======================================================================== #


def findiff1dr(rvar, u, varu=None):
    """
    [dudr, vardudr] = findiff1dr(rvar, u, varu)

    Calculates the radial component of the divergence 1/r d/dr (ru)
    in cylindrical coordinates

    Inputs:
       rvar - The independent variable
       u    - The dependent variable
       varu - variance in dependent variable
    Outputs:
       dudr   - Approximation to the radial divergence of u wrt rvar
       vardudr - Variance in derivative (error propagation from varu)

    A 1D finite-differencing scheme that keeps the most points.
    Centered finite difference on all interior points, and forward
    difference on the left-boundary, backward difference on the right
    boundary.
    """

    # ================================== #

    rvar, _, _ = _derivative_inputcondition(rvar)
    u, ush, utransp = _derivative_inputcondition(u)

    # ================================== #

    dudr = _np.zeros(_np.shape(u), dtype=_np.float64)

    r1 = 0.5*(rvar[0, 1] + rvar[0, 0])
    re = 0.5*(rvar[0, -1] + rvar[0, -2])
    # Works with Matrices:
    dudr[:, 0] = ((rvar[:, 1]*u[:, 1] - rvar[:, 0]*u[:, 0])
                  / (rvar[:, 1] - rvar[:, 0]) / r1)
    dudr[:, -1] = ((rvar[:, -1]*u[:, -1] - rvar[:, -2]*u[:, -2])
                   / (rvar[:, -1] - rvar[:, -2]) / re)
    dudr[:, 1:-1] = (0.5 * (_np.diff(rvar[:, :-1]*u[:, :-1], n=1, axis=1)
                     / _np.diff(rvar[:, :-1], n=1, axis=1)
                     + _np.diff(rvar[:, 1:]*u[:, 1:], n=1, axis=1)
                     / _np.diff(rvar[:, 1:], n=1, axis=1)) / rvar[:, 1:-1])

    if utransp:
        dudr = dudr.T
    # endif
    dudr = dudr.reshape(ush)

    if varu is not None:
        varu, _, _ = _derivative_inputcondition(varu)

        vardudr = _np.zeros_like(u)
        vardudr[:, 0] = ((varu[:, 1]*rvar[:, 1]**2
                         + varu[:, 0]*rvar[:, 0]**2)
                         / ((rvar[:, 1] - rvar[:, 0]) / r1)**2)
        vardudr[:, -1] = ((varu[:, -1]*rvar[:, -1]**2
                           + varu[:, -2]*rvar[:, -2]**2)
                          / ((rvar[:, -1] - rvar[:, -2]) / re)**2)
        vardudr[:, 1:-1] = \
            ((0.5/rvar[:, 1:-1])**2 * (
             (varu[:, 0:-2]*rvar[:, 0:-2]**2 + varu[:, 1:-1]*rvar[:, 1:-1]**2)
             / _np.diff(rvar[:, :-1], n=1, axis=1)**2
             + (varu[:, 2:]*rvar[:, 2:]**2 + varu[:, 1:-1]*rvar[:, 1:-1]**2)
             / _np.diff(rvar[:, 1:], n=1, axis=1)**2))

        if utransp:
            vardudr = vardudr.T
        # endif
        vardudr = vardudr.reshape(ush)
        return dudr, vardudr
    else:
        return dudr
    # endif
# end def findiff1dr

# ======================================================================== #


def findiff2d(x, y, u):
    """
    [dudx, dudy] = findiff2d(x, y, u)

    Calculates the partial deriavtives of a 2D function on a uniform grid
    using hybrid finite differencing.

    Inputs:
       x,y    - The independent variables
       u      - The dependent variable  size(u) ~ length(x),length(y)

    Outputs:
       dudx   - Approx. to the partial derivative of u with respect to x
       dudy   - Approx. to partial derivative of u wrt y
       ngradux - Normalized gradient of u wrt x, ngradux = - (1/u)dudx ~ 1/Lx
       ngraduy - Normalized gradient of u wrt y, ngraduy = - (1/u)dudy ~ 1/Ly

    A 2D finite-differencing scheme that keeps the most points.
    Centered finite difference on all interior points, and forward
    difference on the left-boundary, backward difference on the right
    boundary.
    """

    nx = len(x)
    ny = len(y)

    gradux = _np.zeros(_np.shape(u), dtype=_np.float64)
    graduy = gradux.copy()

    for jj in range(ny):  # jj=1:ny
        for ii in range(nx):  # ii = 1:nx
            if jj == 0:  # Lower Boundary
                dy = y[jj+1]-y[jj]
                graduy[ii, jj] = (u[ii, jj+1] - u[ii, jj])/dy

                if ii == 0:  # Left Boundary
                    dx = x[ii+1]-x[ii]
                    gradux[ii, jj] = (u[ii+1, jj] - u[ii, jj])/dx
                elif ii == nx-1:  # Right Boundary
                    dx = x[ii]-x[ii-1]
                    gradux[ii, jj] = (u[ii, jj] - u[ii-1, jj])/dx
                else:  # Interior
                    dx = x[ii+1]-x[ii-1]
                    gradux[ii, jj] = (u[ii+1, jj]-u[ii-1, jj])/dx
                # endif
            elif jj == ny-1:  # Upper boundary
                dy = y[jj]-y[jj-1]
                graduy[ii, jj] = (u[ii, jj] - u[ii, jj-1])/dy

                if ii == 0:  # Left Boundary
                    dx = x[ii+1]-x[ii]
                    gradux[ii, jj] = (u[ii+1, jj] - u[ii, jj])/dx
                elif ii == nx-1:  # Right Boundary
                    dx = x[ii]-x[ii-1]
                    gradux[ii, jj] = (u[ii, jj] - u[ii-1, jj])/dx
                else:  # Interior
                    dx = x[ii+1]-x[ii-1]
                    gradux[ii, jj] = (u[ii+1, jj]-u[ii-1, jj])/dx
                # endif
            else:  # Interior
                dy = y[jj+1]-y[jj-1]
                graduy[ii, jj] = (u[ii, jj+1] - u[ii, jj-1])/dy

                if ii == 0:  # Left Boundary
                    dx = x[ii+1]-x[ii]
                    gradux[ii, jj] = (u[ii+1, jj] - u[ii, jj])/dx
                elif ii == nx-1:  # Right Boundary
                    dx = x[ii]-x[ii-1]
                    gradux[ii, jj] = (u[ii, jj] - u[ii-1, jj])/dx
                else:  # Interior
                    dx = x[ii+1]-x[ii-1]
                    gradux[ii, jj] = (u[ii+1, jj]-u[ii-1, jj])/dx
                # endif
            # endif
        # endfor
    # endfor

    dudx = gradux
    dudy = graduy

    # #Now calculate normalized gradient.
    # ngradux    =  -gradux /  u
    # ngraduy    =  -graduy /  u

    return dudx, dudy
#  end def findiff2d

# ======================================================================== #
# ======================================================================== #

# =============================== #
# ---- ndimage dependent--------- #
# =============================== #


def interp_profile(roa, ne, varne, rvec, loggradient=True):

    # ================= #
    # Preconditioning   #
    # ================= #

    xfit = _ut.cylsym_odd(roa)
    if loggradient:
        yfit = _ut.cylsym_even(_np.log(ne))
        vfit = _ut.cylsym_even(varne/(ne**2.0))
    else:
        yfit = _ut.cylsym_even(ne)
        vfit = _ut.cylsym_even(varne)
    # endif
    ne = _np.atleast_2d(ne)
    roa = _np.atleast_2d(roa)
    rvec = _np.atleast_2d(rvec)
    xfit = _np.atleast_2d(xfit)
    yfit = _np.atleast_2d(yfit)
    vfit = _np.atleast_2d(vfit)
    if _np.size(ne, axis=0) == 1: ne = ne.T # endif
    if _np.size(roa, axis=0) == 1: roa = roa.T # endif
    if _np.size(rvec, axis=0) == 1: rvec = rvec.T # endif
    if _np.size(xfit, axis=0) == 1: xfit = xfit.T # endif
    if _np.size(yfit, axis=0) == 1: yfit = yfit.T # endif
    if _np.size(vfit, axis=0) == 1: vfit = vfit.T # endif

    ysh = _np.shape(yfit)
    if _np.shape(yfit) != _np.shape(xfit):
        xfit = _np.tile(xfit, ysh[1])
    if _np.size(rvec, axis=1) != _np.size(yfit, axis=1):
        rvec = _np.tile(rvec, ysh[1])
    if _np.size(roa, axis=1) != _np.size(ne, axis=1):
        roa = _np.tile(roa, ysh[1])
    # endif

    # =============== #
    dyfdr = _np.zeros( (len(rvec),ysh[1]),dtype=_np.float64)
    vardyfdr = _np.zeros_like(dyfdr)
    yf = _np.zeros_like(dyfdr)
    varyf = _np.zeros_like(vardyfdr)
    for ii in range(_np.size(dyfdr, axis=1)):
#        if ii == 1:
#            print('what')
#        # endif

        dydr, vardydr = deriv_bsgaussian(xfit[:,ii], yfit[:,ii], vfit[:,ii])

        dyfdr[:,ii], vardyfdr[:,ii] = _ut.interp(xfit[:,ii], dydr, _np.sqrt(vardydr), rvec[:,ii])

        _, _, yf[:,ii], varyf[:,ii] = _ut.trapz_var(rvec[:,ii], dyfdr[:,ii], None, vardyfdr[:,ii], dim=0)

        yf[:,ii] += (ne[0,ii]-_ut.interp(rvec[:,ii], yf[:,ii], None, roa[0,ii]))
    # endfor

    if _np.size(yf, axis=1) == 1:
        yf = yf.reshape(len(rvec),)
        dyfdr = dyfdr.reshape(len(rvec),)
        varyf = varyf.reshape(len(rvec),)
        vardyfdr = vardyfdr.reshape(len(rvec),)
    # end if

    return yf, dyfdr, varyf, vardyfdr

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
    # xvar, _, _ = _derivative_inputcondition(xvar)

    u, ush, transp = _derivative_inputcondition(u)
    varu, _, _ = _derivative_inputcondition(varu)
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
# ======================================================================== #

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

# =============================== #
# ---- curvefit dependent-------- #
# =============================== #


#def twopower(XX, aa):
#    return _ms.twopower(XX, aa)
#
#def expedge(XX, aa):
#    return _ms.expedge(XX, aa)
#
#def qparab_fit(XX, *aa, **kwargs):
#    return _ms.qparab(XX, *aa, **kwargs)
#
#def dqparabdx(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
#    return _ms.deriv_qparab(XX, aa, nohollow)
#
#def dqparabda(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
#    return _ms.partial_qparab(XX, aa, nohollow)

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

#        af = _np.delete(af,4)
#        lowbounds = _np.delete(lowbounds,4)
#        upbounds = _np.delete(upbounds,4)
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

    pobj = _int.UnivariateSpline(xvar, yvar, w=1.0/_np.sqrt(vary), bbox=bbox,
                                 k=deg) #, check_finite=True)
    yf = pobj.__call__(xf)
#
#    pobj = pobj.derivative(n=1)
#    dydx = pobj.__call__(xf)
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
        vtemp = (utemp-yvar)**2.0
        if func == 'pchip':
            yf[ii, :], dydx[ii, :] = pchip(xvar, utemp, xf)
        else:
            tmp1 = _np.zeros((nxf, nsh[1]), dtype=utemp.dtype)
            tmp2 = _np.zeros_like(tmp1)
            for jj in range(nsh[1]):
#                vtemp = (utemp-yvar)
                tmp1[:,jj], tmp2[:,jj] = spline(xvar, utemp[:,jj], xf, vary=vtemp[:,jj], deg=deg, bbox=bbox)
#                tmp1[:,jj], tmp2[:,jj] = spline(xvar, utemp[:,jj], xf, vary=None, deg=deg, bbox=bbox)
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

# ======================================================================= #

# =================================== #
# ---------- fitNL dependent -------- #
# =================================== #


def fit_profile(rdat, pdat, vdat, rvec, **kwargs):
    arescale = kwargs.get('arescale',1.0)
    bootstrappit = kwargs.get('bootstrappit',True)
    af0 = kwargs.get('af0', None)
    LB = kwargs.get('LB', None)
    UB = kwargs.get('UB', None)

    # ==== #

    def fitqparab(af, XX):
        return _ms.qparab(XX, af)

    def returngvec(af, XX):
        _, gvec, info = _ms.model_qparab(_np.abs(XX), af)
        return gvec, info.dprofdx, info.dgdx

    def fitdqparabdx(af, XX):
        return _ms.deriv_qparab(XX, af)

    # ==== #

    info = _ms.model_qparab(XX=None)
    if af0 is None:
        af0 = info.af
    if LB is None:
        LB = info.Lbounds
    if UB is None:
        UB = info.Ubounds
    # end if

#    af0 = _np.asarray([0.30, 0.002, 2.0, 0.7, -0.24, 0.30], dtype=_np.float64)
#    LB = _np.array([  0.0, 0.0,-100,-100,-1,-1], dtype=_np.float64)
#    UB = _np.array([ 20.0, 1.0, 100, 100, 1, 1], dtype=_np.float64)

    options = dict()
    options.setdefault('epsfcn', 1e-3) # 5e-4
    options.setdefault('factor',100)
    options.setdefault('maxiter',200)
    NLfit = fitNL(rdat, pdat, vdat, af0, fitqparab, LB=LB, UB=UB, **options)
    NLfit.run()

    if bootstrappit:
        NLfit.gvecfunc = returngvec
    #    NLfit.bootstrapper(xvec=_np.abs(rvec), weightit=True)
        NLfit.bootstrapper(xvec=_np.abs(rvec), weightit=False)
        prof = NLfit.mfit
        varp = NLfit.vfit
        varp = varp.copy()

        dprofdx = NLfit.dprofdx.copy()
        vardlnpdrho = NLfit.vdprofdx.copy()
    else:
        prof, gvec, info = _ms.model_qparab(_np.abs(rvec), NLfit.af)
        varp = NLfit.properror(_np.abs(rvec), gvec)
        varp = varp.copy()

        dprofdx = info.dprofdx.copy()
        vardlnpdrho = NLfit.properror(_np.abs(rvec), info.dgdx)
    # end if
    af = NLfit.af.copy()
    dlnpdrho = dprofdx / prof
    vardlnpdrho = (dlnpdrho)**2.0 * ( vardlnpdrho/(dprofdx**2.0) + varp/(prof**2.0)  )

    # ================== #
    rdat *= arescale
    rvec *= arescale

    dlnpdrho *= arescale
    vardlnpdrho *= arescale**2.0

    if _np.isnan(prof[0]):
        prof[0] = af[0].copy()
        varp[0] = varp[1].copy()
        dlnpdrho[0] = 0.0
        vardlnpdrho[0] = vardlnpdrho[1].copy()
    # end if

    if 0:
        _plt.figure()
        ax1 = _plt.subplot(2,1,1)
        ax3 = _plt.subplot(2,1,2, sharex=ax1)

        ax1.grid()
        ax3.grid()

        ax1.set_ylabel(r'Fit')
        ax3.set_ylabel(r'$a/L_\mathrm{f}$')
        ax3.set_xlabel(r'$r/a$')

        ax1.errorbar(rdat, pdat, yerr=_np.sqrt(vdat), fmt='ko', color='k' )

        ax1.plot(rvec, prof, 'k-', lw=2)
        ax1.plot(rvec, prof+_np.sqrt(varp), 'k--', lw=1)
        ax1.plot(rvec, prof-_np.sqrt(varp), 'k--', lw=1)

        ax3.plot(rvec, dlnpdrho, 'k-',
                 rvec, dlnpdrho+_np.sqrt(vardlnpdrho), 'k--',
                 rvec, dlnpdrho-_np.sqrt(vardlnpdrho), 'k--')
    # end if

    return prof, varp, dlnpdrho, vardlnpdrho, af


def fit_TSneprofile(QTBdat, rvec, **kwargs):
    loggradient = kwargs.get('loggradient', True)
    plotit = kwargs.get('plotit', False)
    gradrho = kwargs.get('gradrho',1.0)
    amin = kwargs.get('amin', 0.51)
    returnaf = kwargs.get('returnaf',False)
    arescale = kwargs.get('arescale', 1.0)
    bootstrappit = kwargs.get('bootstrappit',False)
    plotlims = kwargs.get('plotlims', None)
    fitin = kwargs.get('fitin', None)
    af0 = kwargs.get('af0', None)
    rescale_by_linavg = kwargs.get('rescale_by_linavg',False)

    nkey = 'roa' if 'roan' not in QTBdat else 'roan'
    rvec = _np.copy(rvec)
    roa = _np.copy(QTBdat[nkey])
    ne = _np.copy(QTBdat['ne'])
    varNL = _np.copy(QTBdat['varNL'])
    varNH = _np.copy(QTBdat['varNH'])
    if 'varRL' in QTBdat:
        varRL = _np.copy(QTBdat['varRL'])
        varRH = _np.copy(QTBdat['varRH'])
    # end if
    varn =  _np.copy(_np.sqrt(varNL*varNH))

    iuse = (~_np.isinf(roa))*(~_np.isnan(roa))*(ne>1e10)
    ne = ne[iuse]
    roa = roa[iuse]
    varNL = varNL[iuse]
    varNH = varNH[iuse]
    if 'varRL' in QTBdat:
        varRL = varRL[iuse]
        varRH = varRH[iuse]
    # end if
    varn = varn[iuse]

    if fitin is None:
        isort = _np.argsort(roa)

        nef, varnef, dlnnedrho, vardlnnedrho, af = fit_profile(
            roa[isort], 1e-20*ne[isort], 1e-40*varn[isort], rvec, arescale=arescale,
            bootstrappit=bootstrappit, af0=af0)

        if rescale_by_linavg:
            rescale_by_linavg *= 1e-20
            iuse = ~(_np.isinf(nef) + _np.isnan(nef))
            nfdl, vnfdl, _, _ = _ut.trapz_var(rvec[iuse], nef[iuse], vary=varnef[iuse])
            nfdl /= _np.abs(_np.max(rvec)-_np.min(rvec))
            vnfdl /= _np.abs(_np.max(rvec)-_np.min(rvec))**2.0

            nef *= rescale_by_linavg/nfdl
            varnef *= (rescale_by_linavg/nfdl)**2.0

            af[0] *= rescale_by_linavg/af[0]
        # end if

    else:
        nef = 1e-20*fitin['prof']
        varnef = 1e-40*fitin['varprof']
        dlnnedrho = fitin['dlnpdrho']
        vardlnnedrho = fitin['vardlnpdrho']
        af = fitin['af']
    # end if

    # Convert back to absolute units (particles per m-3 not 1e20 m-3)
    nef = 1e20*nef
    varnef = 1e40*varnef  # TODO!: watch out for nans here

    varlogne = varnef / nef**2.0
    logne = _np.log(nef)

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

        if 'varRL' in QTBdat:
            ax1.errorbar(roa, 1e-20*ne, xerr=[_np.sqrt(varRL), _np.sqrt(varRH)],
                     yerr=[1e-20*_np.sqrt(varNL), 1e-20*_np.sqrt(varNH)], fmt='bo') #, ecolor='r', elinewidth=2)
        else:
            ax1.errorbar(roa, 1e-20*ne, yerr=[1e-20*_np.sqrt(varNL), 1e-20*_np.sqrt(varNH)], fmt='bo') #, ecolor='r', elinewidth=2)
#        ax1.errorbar(roa, 1e-20*ne, yerr=1e-20*_np.sqrt(varn), fmt='bo', color='b' )
##        ax2.errorbar(roa, _np.log(ne), yerr=_np.sqrt(varn/ne**2.0), fmt='bo', color='b' )

        ax1.plot(rvec, 1e-20*nef, 'b-', lw=2)
        ylims = ax1.get_ylim()
#        ax1.plot(rvec, 1e-20*(nef+_np.sqrt(varnef)), 'b--', lw=1)
#        ax1.plot(rvec, 1e-20*(nef-_np.sqrt(varnef)), 'b--', lw=1)
#        ax2.plot(rvec, logne, 'b-', lw=2)
#        ax2.plot(rvec, logne+_np.sqrt(varlogne), 'b--', lw=1)
#        ax2.plot(rvec, logne-_np.sqrt(varlogne), 'b--', lw=1)
        ax1.fill_between(rvec, 1e-20*(nef-_np.sqrt(varnef)), 1e-20*(nef+_np.sqrt(varnef)),
                                interpolate=True, color='b', alpha=0.3)
        # _, _, nint, nvar = _ut.trapz_var(rvec, dlnnedrho, vary=vardlnnedrho)
        ax1.set_xlim((plotlims[0],plotlims[1]))
        ax1.set_ylim((0,1.1*ylims[1]))

        if loggradient:
            idx = _np.where(_np.abs(rvec) < 0.05)
            plotdlnnedrho = dlnnedrho.copy()
            plotvardlnnedrho = vardlnnedrho.copy()
            plotdlnnedrho[idx] = _np.nan
            plotvardlnnedrho[idx] = _np.nan
            plotdlnnedrho *= -1*amin*gradrho
            plotvardlnnedrho *= (amin*gradrho)**2.0
            ax3.plot(rvec, plotdlnnedrho, 'b-', lw=2)
#            ax3.plot(rvec, plotdlnnedrho+_np.sqrt(plotvardlnnedrho), 'b--')
#            ax3.plot(rvec, plotdlnnedrho-_np.sqrt(plotvardlnnedrho), 'b--')
            ax3.fill_between(rvec, plotdlnnedrho-_np.sqrt(plotvardlnnedrho),
                                   plotdlnnedrho+_np.sqrt(plotvardlnnedrho),
                                   interpolate=True, color='b', alpha=0.3)
            # nint += (_np.log(ne[0]) - _ut.interp(rvec, nint, xo=roa[0]))
            # nint = _np.exp(nint)
            ax3.set_xlim((plotlims[0],plotlims[1]))
            ax3.set_ylim((0,15))
        else:
            vardlnnedrho = (dlnnedrho/logne)**2.0 * (vardlnnedrho/dlnnedrho**2.0+varlogne/logne**2.0)
            dlnnedrho = dlnnedrho/logne
            plotdlnnedrho = -1*(amin*gradrho) * dlnnedrho.copy()
            plotvardlnnedrho = ((amin*gradrho)**2.0) * vardlnnedrho.copy()

            ax3.plot(rvec, plotdlnnedrho, 'b-', lw=2)
#            ax3.plot(rvec, plotdlnnedrho+_np.sqrt(plotvardlnnedrho), 'b--')
#            ax3.plot(rvec, plotdlnnedrho-_np.sqrt(plotvardlnnedrho), 'b--')
            ax3.fill_between(rvec, plotdlnnedrho-_np.sqrt(plotvardlnnedrho),
                                   plotdlnnedrho+_np.sqrt(plotvardlnnedrho),
                                   interpolate=True, color='b', alpha=0.3)
            # nint += (ne[0] - _ut.interp(rvec, nint, xo=roa[0]))
            ax3.set_xlim((plotlims[0],plotlims[1]))
            ax3.set_ylim((0,30))
        # end if
        # ax1.plot(rvec, 1e-20*nint, 'b--', lw=1)
        # ax2.plot(rvec, _np.log(nint), 'b--', lw=1)
        _plt.tight_layout()

    # end if plotit

    # ==================== #
    if returnaf:
        return logne, varlogne, dlnnedrho, vardlnnedrho, af
    return logne, varlogne, dlnnedrho, vardlnnedrho

# ======================================================================= #


def fit_TSteprofile(QTBdat, rvec, **kwargs):
    loggradient = kwargs.get('loggradient', True)
    plotit = kwargs.get('plotit', False)
    amin = kwargs.get('amin', 0.51)
    returnaf = kwargs.get('returnaf',False)
    arescale = kwargs.get('arescale', 1.0)
    bootstrappit = kwargs.get('bootstrappit',False)
    plotlims = kwargs.get('plotlims', None)
    fitin = kwargs.get('fitin', None)
    af0 = kwargs.get('af0', None)

    rvec = _np.copy(rvec)
    roa = _np.copy(QTBdat['roa'])
    Te = _np.copy(QTBdat['Te'])
    varTL = _np.copy(QTBdat['varTL'])
    varTH = _np.copy(QTBdat['varTH'])
    if 'varRL' in QTBdat:
        varRL = _np.copy(QTBdat['varRL'])
        varRH = _np.copy(QTBdat['varRH'])
    # end if
    varT =  _np.copy(_np.sqrt(varTL*varTH))

    iuse = (~_np.isinf(roa))*(~_np.isnan(roa))*(Te>1e-3) #*(Te<9.0)
    Te = Te[iuse]
    roa = roa[iuse]
    varTL = varTL[iuse]
    varTH = varTH[iuse]
    if 'varRL' in QTBdat:
        varRL = varRL[iuse]
        varRH = varRH[iuse]
    # end if
    varT = varT[iuse]

    if fitin is None:
        isort = _np.argsort(roa)
        Tef, varTef, dlnTedrho, vardlnTedrho, af = fit_profile(
            roa[isort], Te[isort], varT[isort], rvec,
            arescale=arescale, bootstrappit=bootstrappit, af0=af0)
    else:
        Tef = fitin['prof']
        varTef = fitin['varprof']
        dlnTedrho = fitin['dlnpdrho']
        vardlnTedrho = fitin['vardlnpdrho']
        af = fitin['af']
    # end if
    varlogTe = varTef / Tef**2.0
    logTe = _np.log(Tef)

    if plotlims is None:
        plotlims = [-0.05, 1.05]
    # end if
    if arescale:
        plotlims[0] *= arescale
        plotlims[1] *= arescale
    # end if

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

        if 'varRL' in QTBdat:
            ax1.errorbar(roa, Te, xerr=[_np.sqrt(varRL), _np.sqrt(varRH)],
                     yerr=[_np.sqrt(varTL), _np.sqrt(varTH)], fmt='ro') #, ecolor='r', elinewidth=2)
        else:
            ax1.errorbar(roa, Te, yerr=[_np.sqrt(varTL), _np.sqrt(varTH)], fmt='ro') #, ecolor='r', elinewidth=2)

#        ax1.errorbar(roa, Te, yerr=_np.sqrt(varT), fmt='ro', color='r' )
##        ax2.errorbar(roa, _np.log(Te), yerr=_np.sqrt(varT/Te**2.0), fmt='bo', color='b' )

        ax1.plot(rvec, Tef, 'r-', lw=2)
#        ylims = ax1.get_ylim()
#        ax1.plot(rvec, (Tef+_np.sqrt(varTef)), 'r--', lw=1)
#        ax1.plot(rvec, (Tef-_np.sqrt(varTef)), 'r--', lw=1)
#        ax2.plot(rvec, logTe, 'b-', lw=2)
#        ax2.plot(rvec, logTe+_np.sqrt(varlogTe), 'r--', lw=1)
#        ax2.plot(rvec, logTe-_np.sqrt(varlogTe), 'r--', lw=1)
        ax1.fill_between(rvec, Tef-_np.sqrt(varTef), Tef+_np.sqrt(varTef),
                                    interpolate=True, color='r', alpha=0.3) # TODO!: watch for inf/nan's

        # _, _, Tint, Tvar = _ut.trapz_var(rvec, dlnTedrho, vary=vardlnTedrho)
        ax1.set_xlim((plotlims[0],plotlims[1]))
        maxyy = min((_np.max(1.05*(Te+_np.sqrt(varTH))),12))
        ax1.set_ylim((0,maxyy))

        if loggradient:
            idx = _np.where(_np.abs(rvec) < 0.05)
            plotdlnTedrho = dlnTedrho.copy()
            plotvardlnTedrho = vardlnTedrho.copy()
            plotdlnTedrho[idx] = _np.nan
            plotvardlnTedrho[idx] = _np.nan
            plotdlnTedrho *= -1*amin
            plotvardlnTedrho *= amin**2.0
            ax3.plot(rvec, plotdlnTedrho, 'r-', lw=2)
#            ax3.plot(rvec, plotdlnTedrho+_np.sqrt(plotvardlnTedrho), 'r--',
#            ax3.plot(rvec, plotdlnTedrho-_np.sqrt(plotvardlnTedrho), 'r--')
            ax3.fill_between(rvec, plotdlnTedrho-_np.sqrt(plotvardlnTedrho),
                                   plotdlnTedrho+_np.sqrt(plotvardlnTedrho),
                                   interpolate=True, color='r', alpha=0.3)

            # Tint += (_np.log(Te[0]) - _ut.interp(rvec, Tint, xo=roa[0]))
            # Tint = _np.exp(Tint)
            ax3.set_xlim((plotlims[0],plotlims[1]))
#            ax3.set_ylim((0,15))
#            maxyy = min((_np.max(1.05*(plotdlnTedrho+_np.sqrt(plotvardlnTedrho))),15))
            maxyy = _np.max(1.05*(plotdlnTedrho+_np.sqrt(plotvardlnTedrho)))
            ax3.set_ylim((0,maxyy))
        else:
            vardlnTedrho = (dlnTedrho/logTe)**2.0 * (vardlnTedrho/dlnTedrho**2.0+varlogTe/logTe**2.0)
            dlnTedrho = dlnTedrho/logTe
            plotdlnTedrho = -1*amin * dlnTedrho.copy()
            plotvardlnTedrho = (amin**2.0) * vardlnTedrho.copy()

            ax3.plot(rvec, plotdlnTedrho, 'r-', lw=2)
#            ax3.plot(rvec, plotdlnTedrho+_np.sqrt(plotvardlnTedrho), 'r--',
#            ax3.plot(rvec, plotdlnTedrho-_np.sqrt(plotvardlnTedrho), 'r--')
            ax3.fill_between(rvec, plotdlnTedrho-_np.sqrt(plotvardlnTedrho),
                                   plotdlnTedrho+_np.sqrt(plotvardlnTedrho),
                                   interpolate=True, color='r', alpha=0.3)
            # Tint += (Te[0] - _ut.interp(rvec, Tint, xo=roa[0]))
            ax3.set_xlim((plotlims[0],plotlims[1]))
#            ax3.set_ylim((0,30))
#            maxyy = min((_np.max(1.05*(plotdlnTedrho+_np.sqrt(plotvardlnTedrho))),30))
            maxyy = _np.max(1.05*(plotdlnTedrho+_np.sqrt(plotvardlnTedrho)))
            ax3.set_ylim((0,maxyy))
        # end if
        # ax1.plot(rvec, Tint, 'b--', lw=1)
        # ax2.plot(rvec, _np.log(Tint), 'b--', lw=1)
        _plt.tight_layout()

    # end if plotit

    # ==================== #
    # ==================== #
    if returnaf:
        return logTe, varlogTe, dlnTedrho, vardlnTedrho, af
    return logTe, varlogTe, dlnTedrho, vardlnTedrho


# ======================================================================= #
# ======================================================================= #


# function [pFit,vF,info] = weightedpoly(xvar,yvar,polyorder,xfit,vary,plotit);
#
#inputs
#   xvar - independent variables for fitting, predictors
#   yvar - dependent variable for fitting
#   polyorder - polynomial order for fitting,   def: linear, polyorder = 1
#   xfit - predictors to evaluate the fits at,  def: xfit = 0:0.01:1
#   vary - variance in each dependent variable, def: vary = 0;
#   plotit - boolean for plotting,              def: plotit = 'False'
#
#outputs
#   pFit - polynomial coefficients, same format as p = polyfit(xvar,yvar)
#   vF   - Estimate of the variance in each coefficient from fit
#   info is a structure containing the following:
#     yf    - fit evaluated at xfit
#     varyf - estimate of variance in yf at xfit
#     dyfit - derivative of yf at xfit
#     vardy - estimate of variance in dyfit at xfit
#
#def weightedpoly(xvar,yvar, polyorder = 1, xfit = 0.0, vary = 0.0, plotit = 'False' ):
#
#    if xfit == 0.0
#        xfit = range(np.min(xvar),np.max(xvar),(np.max(xvar)-np.min(xvar))/99)
#    #endif
#
#    if np.length( vary ) == 1:
#        vary = vary*np.ones_like( yvar )
#    #endif
#    vary( vary == 0   )    = np.nan
#    vary( np.isinf(yvar) ) = np.nan
#    yvar( np.isinf(yvar) ) = np.nan
#
#    #Remove data that can't be fit:
#    del vary( np.isnan(yvar) )
#    del xvar( np.isnan(yvar) )
#    del yvar( np.isnan(yvar) )
#
#    if np.length(xvar)>(polyorder):
#            vary( np.isnan(vary) ) = 1
#            AA(:,polyorder+1) = ones(length(xvar),1)
#            for jj = polyorder:-1:1
#                AA(:,jj) = (xvar.T)*AA(:,jj+1)
#            #endfor
#            # [pFit,sF,mse,covS] = lscov(AA,yvar.T)
#
#            weights = 1/vary.T
#            # weights = (1./vary.T)/np.nanmean(1/vary.T) #sum to 1
#            [pFit,sF,mse,covS] = lscov(AA,yvar.T,weights)
#            vF = sF**2 #= diag(covS) )
#    #end
#
#
#    #I'm manually fitting the data so I can simultaneously propagate the error
#    #
#    #Fit the line and the derivative
#    info.yf    = zeros(1,length(xfit)) #info.varyf = info.yf
#    info.dyfit = zeros(1,length(xfit)) #info.vardy = info.dyfit
#    ii = 0
#    for kk = polyorder:-1:0
#        #the line itself
#        ii = ii+1
#        info.yf = info.yf    + pFit(ii)*(xfit**kk)
#
#        ###
#
#        if kk>0:
#            #The derivative
#            info.dyfit = info.dyfit + (kk)*pFit(ii)*(xfit**(kk-1) )
#        #endif
#    #endfor
#
#    # The coefficients of the polynomial fit depend on each other:
#    # y = c+bx
#    # y = (yavg-bxavg)+bx
#    # y = yavg + b(x-xavg)
#    #
#    # Intuitive but wrong: info.varyf = info.varyf + vF(  ii)*(xfit**(2*kk))
#    # info.varyf = info.varyf + vF(  ii)*(xfit.^(2*kk));
#    #
#    # Matrix style error propagation from
#    # http://www.stanford.edu/~kimth/www-mit/8.13/error_propagation.pdf
#    dcovS = covS(1:(end-1),1:(end-1))
#
#    #Pre-allocate
#    info.varyf = zeros(size(info.yf))
#    info.vardy = zeros(size(info.dyfit))
#
#    for jj = 1:length(xfit)
#        gvec  = ones(polyorder+1,1)
#        dgvec = ones(polyorder  ,1)
#
#        for kk = polyorder:-1:1
#            gvec(kk) = xfit(jj)*gvec(kk+1)
#            if kk<polyorder:
#                dgvec(kk) = xfit(jj)*dgvec(kk+1)
#            #endif
#        #endfor
#        info.varyf(jj) = (gvec.T )*covS *gvec
#        info.vardy(jj) = (dgvec.T)*dcovS*dgvec
#    #endfor
#    ###
#    else:
#        print('not enough data for fit in weightedpoly function')
#        pFit = np.nan(1,polyorder)
#        vF   = pFit
#
#        info.yf    = np.nan(1,length(xfit))
#        info.varyf = info.yf
#        info.dyfit = np.nan(1,length(xfit))
#        info.vardy = info.yf
#    #end
#    ##########################################
#    if plotit:
#        figure, hold all,
#        set(gca,'FontSize',14,'FontName','Arial')
#        errorbar(xfit,info.yf,np.sqrt( info.varyf ),'k')
#        errorbar(xvar,yvar,np.sqrt( vary ),'ko','MarkerSize',6,'LineWidth',2)
#    #end plotit
#
#    return pFit, vF, info

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

    dydx, vardydx = deriv_bsgaussian(x, y, vary, axis=0, nmonti=300,
                                     sigma=0.8, mode='nearest')

    yxp, yxvp = deriv_bsgaussian(x, y, vary, axis=0, nmonti=300,
                                 sigma=0.8, mode='nearest', derivorder=0)

#
#    dydx0, vardydx0 = findiff1d(x, y, vary, order=1)
##    dydx2,vardydx2 = findiff1d(x, y, vary, order=2)
##    dydx4,vardydx4 = findiff1d(x, y, vary, order=4)
#
#    ndydx0, nvardydx0 = findiffnp( x, y, vary, order=1 )
#    ndydx2, nvardydx2 = findiffnp( x, y, vary, order=2 )
##    dydx4,vardydx4 = findiff1dr(x, y, vary)
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

    nout = fit_TSneprofile(QTBdat, _np.linspace(0, 1.05, num=51), plotit=True, amin=0.51, returnaf=False)

    Tout = fit_TSteprofile(QTBdat, _np.linspace(0, 1.05, num=51), plotit=True, amin=0.51, returnaf=False)
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














