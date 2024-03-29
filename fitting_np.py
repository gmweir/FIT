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

try:
    from . import model_spec as _ms
    from .fitNL import fitNL_base
except:
    from FIT import model_spec as _ms
    from FIT.fitNL import fitNL_base
# end try

__metaclass__ = type

# ======================================================================== #
# ======================================================================== #


def findiffnp(xvar, u, varu=None, order=1):

    # Input data formatting.  All of this is undone before output
    # xvar, _, _ = _ms._derivative_inputcondition(xvar)
    if varu is None: varu = _np.zeros_like(u)   # endif
    u, ush, transp = _ms._derivative_inputcondition(u)
    varu, _, _ = _ms._derivative_inputcondition(varu)
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

    xvar, _, _ = _ms._derivative_inputcondition(xvar)
    u, ush, utransp = _ms._derivative_inputcondition(u)

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
        varu, _, _ = _ms._derivative_inputcondition(varu)

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

    rvar, _, _ = _ms._derivative_inputcondition(rvar)
    u, ush, utransp = _ms._derivative_inputcondition(u)

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
        varu, _, _ = _ms._derivative_inputcondition(varu)

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
# ------------------------------------------------------------------------ #
# ======================================================================== #


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


def weightedPolyfit(xvar, yvar, xo, vary=None, deg=1, nargout=2, xbnds=None, sortit=True):

    if xbnds is not None:
        iuse = (xvar>xbnds[0])*(xvar<xbnds[1])

        xvar = xvar[iuse]
        yvar = yvar[iuse, ...]
        vary = vary[iuse, ...]
    if sortit:

        isort = _np.argsort(xvar)
        xvar = xvar[isort]
        yvar = yvar[isort, ...]
        vary = vary[isort, ...]
    # end if

    if vary is None:
        weights = _np.ones_like(yvar)
    # end if

    if (vary==0).any():
        vary[vary==0] = _np.finfo(float).eps
    # endif

    weights = 1.0/vary

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

    def _func(af, xvec):
        yf, _, _ = _ms.model_poly(xvec, af, npoly=deg)
        return yf

    _, gvec, info = _ms.model_poly(xo, af, npoly=deg)
    # The g-vector contains the partial derivatives used for error propagation
    # f = a1*x^2+a2*x+a3
    # dfda1 = x^2;
    # dfda2 = x;
    # dfda3 = 1;
    # gvec(1,1:nx) = XX**2;
    # gvec(2,1:nx) = XX   ;
    # gvec(3,1:nx) = 1;

    fitter = fitNL(xvar, yvar, vary, af0=af, func=_func, fjac=None)
    fitter.af = af
    fitter.covmat = Vcov
    varyf = fitter.properror(xo, gvec)
    yf = fitter.mfit

    if nargout == 2:
        return yf, varyf

    elif nargout == 4:

        def _derivfunc(af, xvec):
            if deg-1 == 0:
                dydx = af[0]*_np.ones_like(xvec)
            else:
                dydx, _, _ = _ms.model_poly(xvec, af, npoly=deg-1)
            return dydx

        # The g-vector for the derivative, ex for quadratic:
        # dfdx = 2*a1*x+a2
        # dfda1 = 2*x;
        # dfda2 = 1;
        # gvec(1,1:nx) = 2*XX
        # gvec(2,1:nx) = 1.0

        gvec = _np.zeros((deg, len(xo)), dtype=_np.float64)
        for ii in range(deg):  # ii=1:num_fit
            # kk = num_fit - (ii + 1)
            kk = deg - (ii + 1)
            gvec[ii, :] = (kk+1)*xo**kk
        # endfor

        af = _np.polyder(af, m=1)
        fitter = fitNL(xvar, yvar, vary, af0=af, func=_derivfunc, fjac=None)
        fitter.af = af  # first derivative

        Vcov = _np.atleast_3d(Vcov)
        fitter.covmat = _np.squeeze(Vcov[:-1, :-1, :])

        vardydf = fitter.properror(xo, gvec)
        dydf = fitter.mfit
        # dydf = info.dprofdx

        return yf, varyf, dydf, vardydf
# end def weightedPolyfit

# ======================================================================== #
# ------------------------------------------------------------------------ #
# ======================================================================== #


def fit_profile(roa, ne, varne, rvec, loggradient=True, derivfunc=None):
    if derivfunc is None:  derivfunc = findiffnp # endif

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

        dydr, vardydr = derivfunc(xfit[:,ii], yfit[:,ii], vfit[:,ii])
#        dydr, vardydr = deriv_bsgaussian(xfit[:,ii], yfit[:,ii], vfit[:,ii])

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
# ======================================================================== #


class fitNL(fitNL_base):
    """
    To use this first generate a class that is a chil of this one
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


    # end def __init__

    # ========================== #


# end class fitNL

# ======================================================================= #


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


# ======================================================================= #
# ======================================================================= #


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


    dydx0, vardydx0 = findiff1d(x, y, vary, order=1)
    dydx2,vardydx2 = findiff1d(x, y, vary, order=2)
    dydx4,vardydx4 = findiff1d(x, y, vary, order=4)
#
    ndydx0, nvardydx0 = findiffnp( x, y, vary, order=1 )
    ndydx2, nvardydx2 = findiffnp( x, y, vary, order=2 )
    dydx5,vardydx5 = findiff1dr(x, y, vary)

    # integrate derivative and compare to source
    _, _, yx0, yxv0 = _ut.trapz_var(x, dydx0, None, vardydx0)
    yx0 += (y[0] - _ut.interp(x, yx0, ei=None, xo=x[0]))

    _, _, nyx0, nyxv0 = _ut.trapz_var(x, ndydx0, None, nvardydx0)
    nyx0 += (y[0] - _ut.interp(x, nyx0, ei=None, xo=x[0]))

    # 2nd order
    _, _, yx2, yxv2 = _ut.trapz_var(x, dydx2, None, vardydx2)
#    _, _, yx2, yxv2 = _ut.trapz_var(x, yx2, None, yxv2)
    yx2 += (y[0] - _ut.interp(x, yx2, ei=None, xo=x[0]))

    _, _, nyx2, nyxv2 = _ut.trapz_var(x, ndydx2, None, nvardydx2)
#    _, _, nyx2, nyxv2 = _ut.trapz_var(x, nyx2, None, nyxv2)
    nyx2 += (y[0] - _ut.interp(x, nyx2, ei=None, xo=x[0]))

    # 4th order
    _, _, yx4, yxv4 = _ut.trapz_var(x, dydx4, None, vardydx4)
    yx4 += (y[0] - _ut.interp(x, yx4, ei=None, xo=x[0]))

    # Cylinrdical
    _, _, yx5, yxv5 = _ut.trapz_var(x, dydx5, None, vardydx5)
    yx5 += (y[0] - _ut.interp(x, yx5, ei=None, xo=x[0]))

    # ==== #

    _plt.figure()

    ax1 = _plt.subplot(2,1,1)
    ax1.plot(x, y, "ko")

    # Integrals
    ax1.plot(x, yx0, 'r-',
             x, yx0+_np.sqrt(yxv0), 'r--',
             x, yx0-_np.sqrt(yxv0), 'r--')

    ax1.plot(x, nyx0, 'b-',
             x, nyx0+_np.sqrt(nyxv0), 'b--',
             x, nyx0-_np.sqrt(nyxv0), 'b--')

    ax1.plot(x, yx2, 'g-',
             x, yx2+_np.sqrt(yxv2), 'g--',
             x, yx2-_np.sqrt(yxv2), 'g--')

    ax1.plot(x, nyx2, 'm-',
             x, nyx2+_np.sqrt(nyxv2), 'm--',
             x, nyx2-_np.sqrt(nyxv2), 'm--')

    ax1.plot(x, yx4, 'y-',
             x, yx4+_np.sqrt(yxv4), 'y--',
             x, yx4-_np.sqrt(yxv4), 'y--')

    ax1.plot(x, yx5, 'g.-',
             x, yx5+_np.sqrt(yxv5), 'g.-',
             x, yx5-_np.sqrt(yxv5), 'g.-')


    # Derivatives
    ax2 = _plt.subplot(2,1,2)
    ax2.plot(x, dydx0, 'r-',
             x, dydx0+_np.sqrt(vardydx0), 'r--',
             x, dydx0-_np.sqrt(vardydx0), 'r--')

    ax2.plot(x, ndydx0, 'b-',
             x, ndydx0+_np.sqrt(nvardydx0), 'b--',
             x, ndydx0-_np.sqrt(nvardydx0), 'b--')

    ax2.plot(x, dydx2, 'g-',
             x, dydx2+_np.sqrt(vardydx2), 'g--',
             x, dydx2-_np.sqrt(vardydx2), 'g--')

    ax2.plot(x, ndydx2, 'm-',
             x, ndydx2+_np.sqrt(nvardydx2), 'm--',
             x, ndydx2-_np.sqrt(nvardydx2), 'm--')

    ax2.plot(x, dydx4, 'y-',
             x, dydx4+_np.sqrt(vardydx4), 'y--',
             x, dydx4-_np.sqrt(vardydx4), 'y--')

    ax2.plot(x, dydx5, 'g.-',
             x, dydx5+_np.sqrt(vardydx5), 'g.-',
             x, dydx5-_np.sqrt(vardydx5), 'g.-')

# end main()


if __name__=="__main__":
    test_linreg()
    test_derivatives()
#endif

# ======================================================================== #
# ======================================================================== #














