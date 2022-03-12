# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:43:52 2018

@author: gawe
"""


import numpy as _np
from scipy import ndimage as _ndimage
from pybaseutils import utils as _ut

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


# def differentiateCSM(f, dx):
#     """
#     Calculate the derivative of a field using the complex step method
#     f        is a matrix of the field
#     dx       is the cell size
#     returns the (matrix) derivative of f in the x-direction

#     This  function uses a 2nd order accurate complex step formula for a
#     real-valued function
#         f(x) = f(xo) + dfdx
#     """
#     # directions for np.roll()
#     R = -1   # right
#     L = 1    # left
#     return ( _np.roll(f,R,axis=0) - _np.roll(f,L,axis=0) ) / (2*dx)

def getGradient(f, dx):
    """
    Calculate the gradients of a field
    f        is a matrix of the field
    dx       is the cell size
    returns the (matrix) derivative of f in the x-direction

    This  function uses a 2nd order centered finite difference formula
    """
    # directions for np.roll()
    R = -1   # right
    L = 1    # left
    return ( _np.roll(f,R,axis=0) - _np.roll(f,L,axis=0) ) / (2*dx)

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
    nmonti = 1000   # TODO!: convert this into iterating over shape ... wiggle one parameter at a time: resampling with replacement!
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
        dudx[0:nr, 2:-3] = (8.0*(u[0:nr, 3:-2] - u[0:nr, 1:-4])
                            + (u[0:nr, 4:-1] - u[0:nr, 0:-5]))/(12.0*dx)
    # endif

    if nc > 7 and order > 4:
        # 6th order accurate across middle:
        dudx[0:nr, 3:-4] = (45.0*(u[0:nr, 4:-3] - u[0:nr, 2:-5])
                            - 9.0*(u[0:nr, 5:-2] - u[0:nr, 1:-6])
                            + (u[0:nr, 6:-1] - u[0:nr, 0:-7]))/(60.0*dx)
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


def interp_profile(roa, ne, varne, rvec, loggradient=True, **kwargs):

    # ================= #
    # Preconditioning   #
    # ================= #

    roa = roa.copy()
    ne = ne.copy()
    varne = varne.copy()
    rvec = rvec.copy()
    if roa[0] == 0.0:
        xfit = _np.concatenate((-_np.flipud(roa),roa[1:]))
        yfit = _np.concatenate(( _np.flipud(ne),ne[1:]))
        vfit = _np.concatenate(( _np.flipud(varne),varne[1:]))
    else:
        xfit = _ut.cylsym_odd(roa)
        yfit = _ut.cylsym_even(ne)
        vfit = _ut.cylsym_even(varne)
    # end if

    if loggradient:
        yfit = _np.log(ne)
        vfit = varne/(ne**2.0)
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
        y, vy = deriv_bsgaussian(xfit[:,ii], yfit[:,ii], vfit[:,ii], derivorder=0, **kwargs)
        dydr, vardydr = deriv_bsgaussian(xfit[:,ii], yfit[:,ii], vfit[:,ii], derivorder=1, **kwargs)

        yf[:,ii], varyf[:,ii] = _ut.interp(xfit[:,ii], y, _np.sqrt(vy), rvec[:,ii])
        dyfdr[:,ii], vardyfdr[:,ii] = _ut.interp(xfit[:,ii], dydr, _np.sqrt(vardydr), rvec[:,ii])

#        _, _, yf[:,ii], varyf[:,ii] = _ut.trapz_var1d(rvec[:,ii], dyfdr[:,ii], None, vardyfdr[:,ii])
##        _, _, yf[:,ii], varyf[:,ii] = _ut.trapz_var(rvec[:,ii], dyfdr[:,ii], None, vardyfdr[:,ii], dim=0)
#
#        yf[:,ii] += (ne[0,ii]-_ut.interp(rvec[:,ii], yf[:,ii], None, roa[0,ii]))
    # endfor

    if _np.size(yf, axis=1) == 1:
        yf = yf.reshape(len(rvec),)
        dyfdr = dyfdr.reshape(len(rvec),)
        varyf = varyf.reshape(len(rvec),)
        vardyfdr = vardyfdr.reshape(len(rvec),)
    # end if

    return yf, dyfdr, varyf, vardyfdr

# ======================================================================== #


def deriv_bsgaussian(xvar, u, varu, axis=0, nmonti=1000, sigma=1, mode='nearest', derivorder=1):
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
#    if (nsh[1] == 1):
#        u = u.reshape(nsh[0],)
#        varu = varu.reshape(nsh[0],)
#        nsh = (nsh[0],1)
#    # endif

    # =================================================================== #
    # Estimate the variance by wiggling the input data within it's
    # uncertainties. This is referred to as 'bootstrapping' or 'Monte Carlo'
    # error propagation. It works well for non-linear methods, and
    # computational methods.

    # Pre-allocate
#    if nmonti>1:

    niterate = nsh[axis]*nmonti
    _np.random.seed(1)
    if 1:
        cc = -1
        dudx = _np.zeros((niterate, nsh[0], nsh[1]), dtype=_np.float64)
        for ii in range(niterate):
            cc += 1
            if cc>=nsh[axis]:
                cc = 0
            # end if
            # Wiggle the input data within its statistical uncertainty
            # utemp = _np.random.normal(0.0, 1.0, _np.size(u, axis=axis))
            # utemp = utemp.reshape(nsh[0], nsh[1])
#            utemp = _np.random.normal(0.0, 1.0, _np.shape(u))
#            utemp = _np.random.randn(_np.shape(u))
#            utemp = u + _np.sqrt(varu)*utemp
            vtemp = varu.copy()
            utemp = u.copy()
            utemp[cc] += _np.sqrt(vtemp[cc])*_np.random.randn(1)
            vtemp[cc] = (utemp[cc]-u[cc])**2.0
#            if axis == 0:
#                utemp[cc,:] += _np.sqrt(vtemp[cc,:])*_np.random.randn(1)
#                vtemp[cc,:] = (utemp[cc,:]-u[cc,:])**2.0
#            elif axis == 1:

            # end if

            # Convolve with the derivative of a Gaussian to get the derivative
            # There is some smoothing in this procedure
            utemp = _ndimage.gaussian_filter1d(utemp.copy(), sigma=sigma, order=derivorder,
                                              axis=axis, mode=mode)
            # utemp /= dx  # dx
            dudx[ii, :, :] = utemp.copy()
        # endfor

        # Take mean and variance of the derivative
        vardudx = _np.nanvar(dudx, axis=0)
        dudx = _np.nanmean(dudx, axis=0)
#    else:
#        dudx = _ndimage.gaussian_filter1d(utemp, sigma=sigma, order=derivorder,
#                                           axis=axis, mode=mode)
    # end if

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