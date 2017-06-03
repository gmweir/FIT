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

from ..Struct import Struct
from .. import utils as _ut   # for normal use

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


def polyeval(a, x):
    """
    p(x) = polyeval(a, x)
         = a[0] + a[1]x + a[2]x^2 +...+ a[n-1]x^{n-1} + a[n]x^n
         = a[0] + x(a[1] + x(a[2] +...+ x(a[n-1] + a[n]x)...)
    """
    p = 0
    for coef in a[::-1]:
        p = p * x + coef
    return p
# end def polyeval


def polyderiv(a):
    """
    p'(x) = polyderiv(a)
          = b[0] + b[1]x + b[2]x^2 +...+ b[n-2]x^{n-2} + b[n-1]x^{n-1}
    where
        b[i] = (i+1)a[i+1]
    """
    b = [i * x for i,x in enumerate(a)][1:]
    return b
# end def polyderiv

def polyreduce(a, root):
    """
    Given x = r is a root of n'th degree polynomial p(x) = (x-r)q(x),
    divide p(x) by linear factor (x-r) using the same algorithm as
    polynomial evaluation.  Then, return the (n-1)'th degree quotient
    q(x) = polyreduce(a, r)
         = c[0] + c[1]x + c[2]x^2 +...+ c[n-2]x^{n-2} + c[n-1]x^{n-1}
    """
    c, p = [], 0
    a.reverse()
    for coef in a:
        p = p * root + coef
        c.append(p)
    a.reverse()
    c.reverse()
    return c[1:]
# end def polyreduce


# ======================================================================== #


def piecewise_2line(x, x0, y0, k1, k2):
    """
    function y = piecewise_2line(x, x0, y0, k1, k2)

    Model for a piecewise linear function with one break
    
    Inputs:
        x - model independent variable
        x0 - position of the break in slope 
        y0 - offset at the break in slope 
        k1 - slope of first line (dydx[x<x0])
        k2 - slope of second line (dydx[x>x0])

    Outputs:
        y(x) - model at the input positions specified by x
    """
    return _np.piecewise(x, [x < x0],
                         [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
# end def piecewise_2line

# def piecewise_linear(x, x0, y0, k1, k2):
#
#   yy = _np.piecewise(x, [x < x0],
#                       [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
#   return yy
# # end def piecewise_linear   
    
# ========================================================================== #



def linreg(X, Y, verbose=True):
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
    if len(X) != len(Y):  raise ValueError('unequal length')

    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx += x
        Sy += y
        Sxx += x*x
        Syy += y*y
        Sxy += x*y
    det = Sxx * N - Sx * Sx
    a, b = (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det

    meanerror = residual = 0.0
    for x, y in zip(X, Y):
        meanerror += (y - Sy/N)**2
        residual += (y - a * x - b)**2
    RR = 1 - residual/meanerror
    ss = residual / (N-2)
    Var_a, Var_b = ss * N / det, ss * Sxx / det

    if verbose:
        print("y=ax+b")
        print("N= %d" % N )
        print("a= %g \\pm t_{%d;\\alpha/2} %g" % (a, N-2, _np.sqrt(Var_a)) )
        print("b= %g \\pm t_{%d;\\alpha/2} %g" % (b, N-2, _np.sqrt(Var_b)) )
        print("R^2= %g" % RR)
        print("s^2= %g" % ss)
    # end if
    
    return a, b, Var_a, Var_b
# end def linreg

    
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

def fit_profile(roa, ne, varne, rvec, loggradient=True):

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

    
def deriv_bsgaussian(xvar, u, varu, axis=0, nmonti=300, sigma=1, mode='nearest'):
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
    derivorder=1
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

    # Match the input data shape in the output data
    vardudx = vardudx.reshape(ush)
    dudx = dudx.reshape(ush)

    return dudx, vardudx
# end def deriv_bsgaussian

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


def gaussian(xx, AA, x0, ss):
    return AA*_np.exp(-(xx-x0)**2/(2.0*ss**2))

def gaussian_peak_width(tt,sig_in,param):
    #
    # param contains intitial guesses for fitting gaussians, 
    # (Amplitude, x value, sigma):
    # param = [[50,40,5],[50,110,5],[100,160,5],[100,220,5],
    #      [50,250,5],[100,260,5],[100,320,5], [100,400,5],   
    #      [30,300,150]]  # this last one is our noise estimate

    # Define a function that returns the magnitude of stuff under a gaussian 
    # peak (with support for multiple peaks)
    fit = lambda param, xx: _np.sum([gaussian(xx, param[ii*3], param[ii*3+1], 
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

# Set the plasma density, temperature, and Zeff profiles (TRAVIS INPUTS)      
def qparab_fit(XX, *aa, nohollow=False):
    """
    ex// ne_parms = [0.30, 0.002, 2.0, 0.7, -0.24, 0.30]
    This subfunction calculates the quasi-parabolic fit
    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-xx^aa[2])^aa[3]+aa[4]*(1-exp(-xx^2/aa[5]^2)) 
        xx - r/a
    aa[0] - Y0 - function value on-axis
    aa[1] - gg - Y1/Y0 - function value at edge over core
    aa[2],aa[3]-  pp, qq - power scaling parameters
    aa[4],aa[5]-  hh, ww - hole depth and width
    """
    XX = _np.abs(XX)
    if (type(aa) is tuple) and (len(aa) == 1):
        aa = aa[0]
    # endif
    aa = _np.asarray(aa, dtype=_np.float64)
    if nohollow and (_np.size(aa)==4):
        aa = _np.vstack((aa,_np.atleast_1d(0.0)))
        aa = _np.vstack((aa,_np.atleast_1d(1.0)))
    elif nohollow:
        aa[4] = 0.0
        aa[5] = 1.0
    # endif
    prof = aa[0]*( aa[1]-aa[4]
                   + (1.0-aa[1]+aa[4])*_np.abs(1.0-XX**aa[2])**aa[3]
                   + aa[4]*(1.0-_np.exp(-XX**2.0/aa[5]**2.0)) )
    return prof
# end def qparab_fit

def dqparabdx(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
    """    
    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
    This subfunction calculates the quasi-parabolic fit
    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-xx^aa[2])^aa[3]+aa[4]*(1-exp(-xx^2/aa[5]^2)) 
        xx - r/a
    aa[0] - Y0 - function value on-axis
    aa[1] - gg - Y1/Y0 - function value at edge over core
    aa[2],aa[3]-  pp, qq - power scaling parameters
    aa[4],aa[5]-  hh, ww - hole depth and width
    """    
    XX = _np.abs(XX)
    aa = _np.asarray(aa,dtype=_np.float64)
    if nohollow and (_np.size(aa)==4):
        aa = _np.vstack((aa,_np.atleast_1d(0.0)))
        aa = _np.vstack((aa,_np.atleast_1d(1.0)))
    elif nohollow:
        aa[4] = 0.0
        aa[5] = 1.0
    # endif  
    dpdx = aa[0]*( (1.0-aa[1]+aa[4])*(-1.0*aa[2]*XX**(aa[2]-1.0))*aa[3]*(1.0-XX**aa[2])**(aa[3]-1.0)
                   - aa[4]*(-2.0*XX/aa[5]**2.0)*_np.exp(-XX**2.0/aa[5]**2.0) )
    return dpdx
# end def dqparabdx
    
def dqparabda(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
    """    
    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
    This subfunction calculates the quasi-parabolic fit
    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-xx^aa[2])^aa[3]+aa[4]*(1-exp(-xx^2/aa[5]^2)) 
        xx - r/a
    aa[0] - Y0 - function value on-axis
    aa[1] - gg - Y1/Y0 - function value at edge over core
    aa[2],aa[3]-  pp, qq - power scaling parameters
    aa[4],aa[5]-  hh, ww - hole depth and width
    """    
    XX = _np.abs(XX)
    aa = _np.asarray(aa,dtype=_np.float64)
    if nohollow and (_np.size(aa)==4):
        aa = _np.vstack((aa,_np.atleast_1d(0.0)))
        aa = _np.vstack((aa,_np.atleast_1d(1.0)))
    elif nohollow:
        aa[4] = 0.0
        aa[5] = 1.0
    # endif    
    gvec = _np.zeros( (6,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = ( aa[1]-aa[4]
                   + (1.0-aa[1]+aa[4])*_np.abs(1.0-XX**aa[2])**aa[3]
                   + aa[4]*(1.0-_np.exp(-XX**2.0/aa[5]**2.0)) )    
    gvec[1,:] = aa[0]
    gvec[2,:] = aa[0]*(1.0-aa[1]+aa[4])*(-1.0*_np.log(XX)*XX**aa[2])*aa[2]*_np.abs(1.0-XX**aa[2])**(aa[3]-1.0)
    gvec[3,:] = aa[0]*(1.0-aa[1]+aa[4])*_np.log(1.0-XX**aa[2])*_np.abs(1.0-XX**aa[2])**aa[3]
    gvec[4,:] = aa[0]*( -1.0
                   + _np.abs(1.0-XX**aa[2])**aa[3]
                   + (1.0-_np.exp(-XX**2.0/aa[5]**2.0)) )
    gvec[5,:] = aa[0]*aa[4]*( -1.0*_np.exp(-XX**2.0/aa[5]**2.0) )*( 2.0*XX**2.0/aa[5]**3 )                        

    return gvec
# end def dqparabda

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
    FitAlias = lambda *args: qparab_fit(args[0], args[1:], nohollow)
        
    if _scipyversion < 0.17:        
        [af,pcov]=curve_fit( FitAlias, xdata, ydata, p0 = af, sigma = weights)

        af = _np.asarray(af, dtype=_np.float64)
        if (len(ydata) > len(af)) and pcov is not None:
            pcov = pcov * (((qparab_fit(xdata,af)-ydata)**2).sum()
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
    function [pfit,perr] = fit_mcleastsq(p0, xdat, ydat, func, 
                                         yerr_systematic, nmonti)

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


def spline(xvar, yvar, xf, vary=None, nmonti=300, deg=5, bbox=None):
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

    # ============= #

    pobj = _int.UnivariateSpline(xvar, yvar, w=1.0/_np.sqrt(vary), bbox=bbox,
                                 k=deg, check_finite=True)
    yf = pobj.__call__(xf)

    pobj = pobj.derivative(n=1)
    dydx = pobj.__call__(xf)

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
    dydx = pobj.derivative(xf, der=1)            
    return yf, dydx


def spline_bs(xvar, yvar, vary, xf=None, func="spline", nmonti=300, deg=3,
              bbox=None):  
    """
    
    """  

    if xf is None:
        xf = xvar.copy()
    # endif        
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
    
    dydx = _np.zeros( (nmonti, nsh[0], nsh[1]), dtype=_np.float64)    
    yf = _np.zeros( (nmonti, nsh[0], nsh[1]), dtype=_np.float64)
    for ii in range(nmonti):        
        utemp = yvar + _np.sqrt(vary)*_np.random.normal(0.0, 1.0, _np.shape(yvar))
        if func == 'pchip':
            yf[ii, :], dydx[ii, :] = pchip(xvar, utemp, xf)        
        else:
            tmp1 = _np.zeros_like(utemp)
            tmp2 = _np.zeros_like(utemp)
            for jj in range(nsh[1]):
                tmp1[:,jj], tmp2[:,jj] = spline(xvar, utemp[:,jj], xf, vary[:,jj], nmonti, deg, bbox)
            # end for
            yf[ii, :] = tmp1.reshape((nsh[0], nsh[1]), order='C')
            dydx[ii, :] = tmp2.reshape((nsh[0], nsh[1]), order='C')

        # endif
    # end for
        
    vardydx = _np.var(dydx, axis=0)
    dydx = _np.mean(dydx, axis=0)
    varf = _np.var( yf, axis=0)
    yf = _np.mean(yf, axis=0)
    
    return yf, varf, dydx, vardydx


# ======================================================================== #


class fitNL(Struct):
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

    def __init__(self, xdat, ydat, vary, af0, func, fjac=None,
                 options={}):

        self.xdat = xdat
        self.ydat = ydat
        self.vary = vary
        self.af0 = af0
        self.func = func
        self.fjac = fjac

        # ========================== #

        # options.update(kwargs)

        options["nmonti"] = options.get("nmonti", 300)
        options["af0"] = options.get("af0", self.af0)
        options["LB"] = options.get("LB", -_np.Inf*_np.ones_like(self.af0))
        options["UB"] = options.get("UB",  _np.Inf*_np.ones_like(self.af0))

        # 1) Least-squares, 2) leastsq, 3) Curve_fit
        options["lsqfitmethod"] = options.get("lsqfitmethod", 'lm')
        if _scipyversion >= 0.17:
            options["lsqmethod"] = options.get("lsqmethod", int(1))
        else:
            options["lsqmethod"] = options.get("lsqmethod", int(2))
        # end if

        # ========================== #

        # Pull out the run data from the options dictionary
        #  possibilities include
        #   - lsqfitmetod - from leastsquares - 'lm' (levenberg-marquardt,etc.) 
        #   - LB, UB - Lower and upper bounds on fitting parameters (af)
        self.__dict__.update(options)

    # end def __init__
        
    # ========================== #

    def run(self, **kwargs):
        self.__dict__.update(kwargs)
        
        if self.lsqmethod == 1:
            self.__use_least_squares(kwargs)
        elif self.lsqmethod == 2:
            self.__use_leastsq(kwargs)
        elif self.lsqmethod == 3:
            self.__use_curvefit(kwargs)
        return self.af, self.covmat
    # end def run

    # ========================== #

    def calc_chi2(self, af):
        self.chi2 = (self.func(af, self.xdat) - self.ydat)
        self.chi2 = self.chi2 / _np.sqrt(self.vary)
        return self.chi2

    # ========================== #
        
    def __use_least_squares(self, options):
        """
        Wrapper around the scipy least_squares function
        """
        lsqfitmethod = options.get("lsqfitmethod", 'lm')

        self.numfit = len(self.af0)        

        res = least_squares(self.calc_chi2, self.af0, bounds=(self.LB, self.UB),
                          method=lsqfitmethod, **options)
                          # args=(self.xdat,self.ydat,self.vary), kwargs)
        self.af = res.x
        # chi2 
        resid = res.fun
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

    def bootstrapper(self, xvec, **kwargs):
        self.__dict__.update(kwargs)
        
        niterate = 1
        if self.nmonti > 1:
            niterate = self.nmonti
            # niterate *= len(self.xdat)
        # endif

        nch = len(self.xdat)
        numfit = len(self.af0)
        xsav = self.xdat.copy()
        ysav = self.ydat.copy()
        vsav = self.vary.copy()
        af = _np.zeros((niterate, numfit), dtype=_np.float64)
        chi2 = _np.zeros((niterate,), dtype=_np.float64)

        nx = len(xvec)
        self.mfit = self.func(self.af, xvec)       
        mfit = _np.zeros((niterate, nx), dtype=_np.float64)
        for mm in range(niterate):

            self.ydat = ysav.copy()
            self.vary = vsav.copy()

            self.ydat += _np.sqrt(self.vary)*_np.random.normal(0.0,1.0,_np.shape(self.ydat))
            self.vary = (self.ydat-ysav)**2
            
#            cc = 1+_np.floor((mm-1)/self.nmonti)
#            if self.nmonti > 1:
#                self.ydat[cc] = ysav[cc].copy() 
#                self.ydat[cc] += _np.sqrt(vsav[cc]) * _np.random.normal(0.0,1.0,_np.shape(vsav[cc]))
#                self.vary[cc] = (self.ydat[cc]-ysav[cc])**2
#                    # _np.ones((1,nch), dtype=_np.float64)*
#            # endif

            af[mm, :], _ = self.run()    
            chi2[mm] = _np.sum(self.chi2)/(numfit-nch-1)
            
            mfit[mm, :] = self.func(af[mm,:], xvec)        
        # endfor
        self.xdat = xsav
        self.ydat = ysav
        self.vary = vsav
        
        self.vfit = _np.var(mfit, axis=0)        
        self.mfit = _np.mean(mfit, axis=0)
        
        # straight mean and covariance
        self.covmat = _np.cov(af, rowvar=False)
        self.af = _np.mean(af, axis=0)
        
#        # weighted mean and covariance
#        aw = 1.0/(1.0-chi2) # chi2 close to 1 is good, high numbers good in aweights
#        covmat = _np.cov( af, rowvar=False, aweights=aw)
        
        # Weighting by chi2
        # chi2 = _np.sqrt(chi2)
        # af = _np.sum( af/(chi2*_np.ones((1,numfit),dtype=_np.float64)), axis=0)
        # af = af/_np.sum(1/chi2, axis=0)

        # self.covmat = covmat
        # self.af = af                
        return self.af, self.covmat

    # ========================== #

    def properror(self, xvec, gvec):  # (x-positions, gvec = dqparabda)
        if gvec is None: gvec = self.fjac # endif
        sh = _np.shape(xvec)

        nx = len(xvec)
        self.mfit = self.func(self.af, xvec)
        
        self.vfit = _np.zeros(_np.shape(xvec), dtype=_np.float64)       
        for ii in range(nx):
            # Required to propagate error from model            
            self.vfit[ii] = _np.dot(_np.atleast_2d(gvec[:,ii]), _np.dot(self.covmat, gvec[:,ii]))
        # endfor
        self.vfit = _np.reshape(self.vfit, sh)            
        return self.vfit

    # ========================== #
        
# end class fitNL        
        
# ------------------------------------------------------------------------- #
        

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
    polynomial of high order over a odd-sized window centered at
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


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #


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

# ------------------------------------------------------------------------ #
    
def test_derivatives(): 

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

    yxp, yxvp, dydxp, vardydxp = spline_bs(x, y, vary, x, nmonti=300, func="spline")

    dydx, vardydx = deriv_bsgaussian(x, y, vary, axis=0, nmonti=300,
                                     sigma=1, mode='nearest')

    dydx0, vardydx0 = findiff1d(x, y, vary, order=1)
#    dydx2,vardydx2 = findiff1d(x, y, vary, order=2)
#    dydx4,vardydx4 = findiff1d(x, y, vary, order=4)

    ndydx0, nvardydx0 = findiffnp( x, y, vary, order=1 )
    ndydx2, nvardydx2 = findiffnp( x, y, vary, order=2 )
#    dydx4,vardydx4 = findiff1dr(x, y, vary)

    # integrate derivative and compare to source
    _, _, yx, yxv = _ut.trapz_var(x, dydx, None, vardydx) 
    yx += (y[0] - _ut.interp(x, yx, ei=None, xo=x[0]))

    _, _, yx0, yxv0 = _ut.trapz_var(x, dydx0, None, vardydx0) 
    yx0 += (y[0] - _ut.interp(x, yx0, ei=None, xo=x[0]))

    _, _, nyx0, nyxv0 = _ut.trapz_var(x, ndydx0, None, nvardydx0) 
    nyx0 += (y[0] - _ut.interp(x, nyx0, ei=None, xo=x[0]))
        
#    _, _, yx2, yxv2 = _ut.trapz_var(x, dydx2, None, vardydx2) 
#    yx2 += (y[0] - _ut.interp(x, yx2, ei=None, xo=x[0]))

    _, _, nyx2, nyxv2 = _ut.trapz_var(x, ndydx2, None, nvardydx2) 
    nyx2 += (y[0] - _ut.interp(x, nyx2, ei=None, xo=x[0]))
    
#    _, _, yx4, yxv4 = _ut.trapz_var(x, dydx4, None, vardydx4) 
#    yx4 += (y[0] - _ut.interp(x, yx4, ei=None, xo=x[0]))


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
             
    ax1.plot(x, yx0, 'r-',
             x, yx0+_np.sqrt(yxv0), 'r--',    
             x, yx0-_np.sqrt(yxv0), 'r--')     

    ax1.plot(x, nyx0, 'b-',
             x, nyx0+_np.sqrt(nyxv0), 'b--',    
             x, nyx0-_np.sqrt(nyxv0), 'b--')     


#    ax1.plot(x, yx2, 'g-',
#             x, yx2+_np.sqrt(yxv2), 'g--',    
#             x, yx2-_np.sqrt(yxv2), 'g--')     

    ax1.plot(x, nyx2, 'm-',
             x, nyx2+_np.sqrt(nyxv2), 'm--',    
             x, nyx2-_np.sqrt(nyxv2), 'm--')     

#    ax1.plot(x, yx4, 'y-',
#             x, yx4+_np.sqrt(yxv4), 'y--',    
#             x, yx4-_np.sqrt(yxv4), 'y--')     
             
    # Derivatives
    ax2 = _plt.subplot(2,1,2, sharex=ax1)

    ax2.plot(x, dydx, 'k-',
             x, dydx+_np.sqrt(vardydx), 'k--',
             x, dydx-_np.sqrt(vardydx), 'k--')

    ax2.plot(x, dydxp, 'g-',
             x, dydxp+_np.sqrt(vardydxp), 'g--',
             x, dydxp-_np.sqrt(vardydxp), 'g--')
             
    ax2.plot(x, dydx0, 'r-',
             x, dydx0+_np.sqrt(vardydx0), 'r--',
             x, dydx0-_np.sqrt(vardydx0), 'r--')

    ax2.plot(x, ndydx0, 'b-',
             x, ndydx0+_np.sqrt(nvardydx0), 'b--',
             x, ndydx0-_np.sqrt(nvardydx0), 'b--')
    
#    ax2.plot(x, dydx2, 'g-',
#             x, dydx2+_np.sqrt(vardydx2), 'g--',
#             x, dydx2-_np.sqrt(vardydx2), 'g--')

    ax2.plot(x, ndydx2, 'm-',
             x, ndydx2+_np.sqrt(nvardydx2), 'm--',
             x, ndydx2-_np.sqrt(nvardydx2), 'm--')
             
#    ax2.plot(x, dydx4, 'y-',
#             x, dydx4+_np.sqrt(vardydx4), 'y--',
#             x, dydx4-_np.sqrt(vardydx4), 'y--')
# end main()
    
    
if __name__=="__main__":
    test_derivatives()

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














