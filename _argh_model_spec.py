# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:59:28 2016

@author: gawe
"""
# ========================================================================== #
# ========================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

# ========================================================================== #
# ========================================================================== #

import numpy as _np
import matplotlib.pyplot as _plt
#import pybaseutils as _pyut
from pybaseutils.Struct import Struct
from pybaseutils.utils import sech, interp_irregularities
from pybaseutils.utils import log as utlog

# ========================================================================== #
# ========================================================================== #

def randomize_initial_conditions(LB, UB):
    LB, UB = LB.copy(), UB.copy()
    if _np.isinf(LB).any():
        LB[_np.isinf(LB)] = -1e3
    if _np.isinf(UB).any():
        UB[_np.isinf(UB)] = 1e3

    af = _np.zeros_like(LB)
    for ii in range(len(LB)):
        af[ii] = _np.random.uniform(low=LB[ii], high=UB[ii], size=None)
    # end for
    return af

def rescale_xlims(XX, forward=True, ascl=None):
    if ascl is None:
        ascl = max(_np.max(XX), 1.0)
    # end if
    if forward:
        return XX/ascl
    else:
        return XX*ascl

def rescale_problem(pdat=0, vdat=0, info=None, nargout=1):
    """
    When fitting this problem it is convenient to scale it to order 1:
        slope = _np.nanmax(pdat)-_np.nanmin(pdat)
        offset = _np.nanmin(pdat)

        pdat = (pdat-offset)/slope
        vdat = vdat/slope**2.0

    After fitting, then the algebra necessary to unscale the problem to original
    units is:
        prof = slope*prof+offset
        varp = varp*slope**2.0

        dprofdx = (slope*dprofdx)
        vardprofdx = slope**2.0 * vardprofdx

    Note that when scaling the problem, it is best to propagate errors from
    covariance / gvec / dgdx before rescaling
    """
    if info is None:
        slope = _np.nanmax(pdat)-_np.nanmin(pdat)
        offset = _np.nanmin(pdat)

        pdat = (pdat.copy()-offset)/slope
        vdat = vdat.copy()/_np.abs(slope)**2.0

        return pdat, vdat, slope, offset
    elif info is not None:
        slope = info.slope
        offset = info.offset

        if hasattr(info,'pdat'):
            info.pdat = info.pdat*slope+offset
            info.vdat = info.vdat * (slope**2.0)
        # end if
        if hasattr(info,'dprofdx'):
            info.dprofdx = slope*info.dprofdx
            info.vardprofdx = (slope**2.0)*info.vardprofdx
        # end if

        if hasattr(info,'prof'):
            info.prof = slope*info.prof+offset
            info.varp = (slope**2.0)*info.varp
        # end if

        info.af = info.unscaleaf(info.af, info.slope, info.offset)
        if nargout == 1:
            return info
        else:
            return info.prof, info.varp, info.dprofdx, info.vardprofdx, info.af
    else:
#        print('for a backward unscaling: you must provide the info Struct from the fit!')
        raise ValueError('for a backward unscaling: you must provide the info Struct from the fit!')
    # end if
# end def rescale problem

# ========================================================================== #
# ========================================================================== #


def line(XX, a):
    y = a[0]*XX+a[1]
    return y

def line_gvec(XX, a):
    gvec = _np.zeros( (2,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = XX.copy() # aa[0]
    gvec[1,:] = 1.0       # aa[1]
    return gvec

def deriv_line(XX, a):
    dydx = a[0]*_np.ones_like(XX)
    return dydx

def partial_deriv_line(XX, a):
    dgdx = _np.zeros( (2,_np.size(XX)), dtype=_np.float64)
    dgdx[0,:] = _np.ones_like(XX)  # aa[0]
    dgdx[1,:] = _np.zeros_like(XX) # aa[1]
    return dgdx

def model_line(XX, af=None):
    """
     y = a * x + b

     if the data is scaled, then unscaling it goes like this:
         (y-miny)/(maxy-miny) = (y-offset)/slope
         (y-offset)/slope = a' x + b'

         y-offset = slope* a' * x + slope * b'
         y = slope* a' * x + slope * b' + offset
         a = slope*a'
         b = slope*b' + offset
    """
    if af is None:
        af = _np.asarray([2.0,1.0], dtype=_np.float64)
    # endif

    info = Struct()
    info.Lbounds = _np.array([-_np.inf, -_np.inf], dtype=_np.float64)
    info.Ubounds = _np.array([_np.inf, _np.inf], dtype=_np.float64)
    info.af = af

    def unscaleaf(ain, slope, offset=0.0):
        aout = _np.copy(ain)
        aout[0] = slope*aout[0]
        aout[1] = slope*aout[1] + offset
        return aout
    info.unscaleaf = unscaleaf
    if XX is None:
        return info

    prof = line(XX, af)
    gvec = line_gvec(XX, af)

    info.prof = prof
    info.gvec = gvec
    info.dprofdx = deriv_line(XX, af)
    info.dgdx = partial_deriv_line(XX, af)
    return prof, gvec, info

# ========================================================================== #

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


def expdecay(tt, Y0, t0, tau):
    return Y0*_np.exp(-(tt-t0)/tau)

# ========================================================================== #

def gaussian(xx, AA, x0, ss):
    return AA*_np.exp(-(xx-x0)**2/(2.0*ss**2))

def partial_gaussian(xx, AA, x0, ss):
    gvec = _np.zeros( (3,len(xx)), dtype=_np.float64)
    gvec[0,:] = _np.exp(-(xx-x0)**2/(2.0*ss**2))
    gvec[1,:] = AA*(xx-x0)*_np.exp((-0.5*(xx-x0)**2.0)/(ss**2.0))/(ss**2.0)
    gvec[2,:] = AA*((xx-x0)**2.0) *_np.exp((-0.5*(xx-x0)**2.0)/(ss**2.0))/(ss**3.0)
    return gvec

def deriv_gaussian(xx, AA, x0, ss):
    return -1.0*AA*(xx-x0)*_np.exp((-0.5*(xx-x0)**2.0)/(ss**2.0))/(ss**2.0)

def partial_deriv_gaussian(xx, AA, x0, ss):
    gvec = _np.zeros( (3,len(xx)), dtype=_np.float64)

    gvec[0,:] = -1.0*(xx-x0)*_np.exp(-0.5*(xx-x0)**2/(ss**2))/(ss**2.0)
    gvec[1,:] = AA*_np.exp(-0.5*(xx-x0)**2/(ss**2))*(ss*2.0-xx*2.0+2*xx*x0-x0**2.0)/(ss**4.0)
    gvec[2,:] = AA*_np.exp(-0.5*(xx-x0)**2/(ss**2))*(-1.0*xx*3.0+3.0*(xx**2.0)*x0-3.0*xx*(x0**2.0) + 2.0*xx*(ss**2.0) + x0**3.0 - 2.0*x0*(ss**2.0))/(ss**5.0)
    return gvec

def model_gaussian(XX, af=None):
    """
    A gaussian with three free parameters:
        xx - x - independent variable
        af - magnitude, shift, width

        af[0]*_np.exp(-(xx-af[1])**2/(2.0*af[2]**2))

    If fitting with scaling, then the algebra necessary to unscale the problem
    to original units is:
        af[0] = slope*af[0]
        offset = 0.0  (in practice, this is necessary for this fit)

    found in the info Structure
    """

    if af is None:
        af = 0.7*_np.ones((3,), dtype=_np.float64)
    # endif

    info = Struct()
    info.Lbounds = _np.array([-_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    info.Ubounds = _np.array([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    info.af = af

    def unscaleaf(ain, slope, offset=0.0):
        aout = _np.copy(ain)
        aout[0] = slope*aout[0]
        return aout
    info.unscaleaf = unscaleaf
    if XX is None:
        return info

    prof = gaussian(XX, af[0], af[1], af[2])
    gvec = partial_gaussian(XX, af[0], af[1], af[2])

    info.prof = prof
    info.gvec = gvec
    info.dprofdx = deriv_gaussian(XX, af[0], af[1], af[2])
    info.dgdx = partial_deriv_gaussian(XX, af[0], af[1], af[2])
    return prof, gvec, info

# ========================================================================== #

def edgepower(XX, af):
    """
    model a two-power fit
        b+(1-b)*(1-XX^c)^d
        first-half of a quasi-parabolic (hole depth, no width or decaying edge)

        y = edge/core + (1-edge/core)
        a = amplitude of core
        b = ( edge/core - hole depth)
        c = power scaling factor 1
        d = power scaling factor 2
    """
    b = af[0]
    c = af[1]
    d = af[2]
    return b+(1-b)*_np.power((1-_np.power(XX,c)), d)

def partial_edgepower(XX, af):
    """
    This subfunction calculates the jacobian of a two-power edge fit
    (partial derivatives of the fit)
    """
    b = af[0]
    c = af[1]
    d = af[2]

    gvec = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = 1.0 - _np.power(1-_np.power(XX,c),d)
    gvec[1,:] = -1.0 * (1.0-b)*d*_np.power(XX,c)*utlog(XX)*_np.power(1.0-_np.power(XX,c),d-1.0)
#    gvec[2,:] = -1.0*(b-1.0)*_np.power((1-_np.power(XX,c)), d)*_np.log(1.0-_np.power(XX,c))
    gvec[2,:] = -1.0*(b-1.0)*_np.power((1-_np.power(XX,c)), d)*utlog(-1.0*_np.power(XX,c), a1p=True)
    return gvec

def deriv_edgepower(XX, af):
    """"
    This subfunction calculates the first derivative of a two-power edge fit
    """
    b = af[0]
    c = af[1]
    d = af[2]
    return -1.0*(1-b)*c*d*_np.power(XX,c-1)*_np.power((1-_np.power(XX,c)), d-1)

def partial_deriv_edgepower(XX, af):
    """"
    This subfunction calculates the jacobian of the second derivative of a
    two-power edge fit (partial derivatives of the second derivative of a fit)
    """
    b = af[0]
    c = af[1]
    d = af[2]

    gvec = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = c*d*_np.power(XX,c-1.0)*_np.power(1.0-_np.power(XX,c),d-1.0)
    gvec[1,:] = (
        -1.0*(1-b)*d*_np.power(XX,c-1.0)*_np.power(1.0-_np.power(XX,c), d-1.0)
        - (1.0-b)*d*c*_np.power(XX,c-1.0)*utlog(XX)*_np.power(1.0-_np.power(XX,c),d-1.0)
        + (1.0-b)*(d-1.0)*d*c*_np.power(XX,2*c-1.0)*utlog(XX)*_np.power(1.0-_np.power(XX,c),d-2.0)
        )
    gvec[2,:] = (
        -1.0*(1.0-b)*c*_np.power(XX,c-1.0)*_np.power(1.0-_np.power(XX,c),d-1.0)
        - (1.0-b)*c*d*_np.power(XX,c-1.0)*_np.power(1.0-_np.power(XX,c),d-1.0)*utlog(-_np.power(XX,c), a1p=True)
#        - (1.0-b)*c*d*_np.power(XX,c-1.0)*_np.power(1.0-_np.power(XX,c),d-1.0)*_np.log(1.0-_np.power(XX,c))
        )
    return gvec

def deriv2_edgepower(XX, af):
    """"
    This subfunction calculates the second derivative of a two-power edge fit
    """
    b = af[0]
    c = af[1]
    d = af[2]
    return ((b-1)*(c-1)*c*d*_np.power(XX,c-2)*_np.power(1.0-_np.power(XX,c),d-1.0)
     - (b-1.0)*_np.power(c,2.0)*(d-1.0)*d*_np.power(XX,2*c-2.0)*_np.power(1.0-_np.power(XX,c),d-2.0))

def partial_deriv2_edgepower(XX, af):
    """"
    This subfunction calculates the jacobian of the second derivative of a
    two-power edge fit (partial derivatives of the second derivative of a fit)
    """
    b = af[0]
    c = af[1]
    d = af[2]

    gvec = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = (
        (c-1.0)*c*d*_np.power(XX,c-2.0)*_np.power(1.0-_np.power(XX,c),d-1.0)
      - _np.power(c, 2.0)*(d-1.0)*d*_np.power(XX, 2*c-2.0)*_np.power(1.0-_np.power(XX,c), d-2.0)
      )
    gvec[1,:] = (
          (b-1.0)*d*(c-1.0)*_np.power(XX,c-2.0)*_np.power(1.0-_np.power(XX,c),d-1.0)
        + (b-1.0)*d*c*_np.power(XX,c-2)*_np.power(1.0-_np.power(XX,c),d-1.0)
        + (b-1.0)*d*(c-1.0)*c*_np.power(XX,c-2.0)*utlog(XX)*_np.power(1.0-_np.power(XX,c),d-1.0)
        + (b-1.0)*(d-2.0)*(d-1.0)*d*_np.power(c,2.0)*_np.power(XX,3*c-2.0)*utlog(XX)*_np.power(1.0-_np.power(XX,c),d-3.0)
        - 2.0*(b-1.0)*(d-1.0)*d*_np.power(c,2.0)*_np.power(XX,2*c-2.0)*utlog(XX)*_np.power(1.0-_np.power(XX,c),d-2.0)
        - 2.0*(b-1.0)*(d-1.0)*d*c*_np.power(XX,2*c-2.0)*_np.power(1-_np.power(XX,c),d-2.0)
        - (b-1.0)*(d-1.0)*d*(c-1.0)*c*_np.power(XX,2*c-2.0)*utlog(XX)*_np.power(1 - _np.power(XX,c),d-2.0)
        )
    gvec[2,:] = (
        -1.0*(b-1.0)*_np.power(c,2.0)*(d-1.0)*_np.power(XX,2*c-2.0)*_np.power(1.0-_np.power(XX,c),d-2.0)
        - (b-1.0)*_np.power(c,2.0)*d*_np.power(XX,2*c-2.0)*_np.power(1.0-_np.power(XX,2),d-2.0)
        - (b-1.0)*_np.power(c,2.0)*(d-1.0)*d*_np.power(XX,2*c-2.0)*_np.power(1.0-_np.power(XX,c),d-2.0)*utlog(-_np.power(XX,c), a1p=True)
#        - (b-1.0)*_np.power(c,2.0)*(d-1.0)*d*_np.power(XX,2*c-2.0)*_np.power(1.0-_np.power(XX,c),d-2.0)*_np.log(1.0-_np.power(XX,c))
        + (b-1.0)*(c-1.0)*c*_np.power(XX,c-2.0)*_np.power(1.0-_np.power(XX,c),d-1.0)
        + (b-1.0)*(c-1.0)*c*d*_np.power(XX,c-2.0)*_np.power(1.0-_np.power(XX,c),d-1.0)*utlog(-_np.power(XX,c), a1p=True)
#        + (b-1.0)*(c-1.0)*c*d*_np.power(XX,c-2.0)*_np.power(1.0-_np.power(XX,c),d-1.0)*_np.log(1-_np.power(XX,c))
        )
    return gvec

def model_edgepower(XX, af=None):
    """
... this is identical to model_2power and model_twopower, just different formulations

    model a two-power fit
        a*(b+(1-b)*(1-XX^c)^d)
        first-half of a quasi-parabolic (hole depth, no width or decaying edge)

        y/a = edge/core + (1-edge/core)
        af[0] = a = amplitude of core
        af[0] = b = ( edge/core - hole depth)
        af[1] = c = power scaling factor 1
        af[2] = d = power scaling factor 2

        xx - x - independent variable

    It is kind of dumb to try and scale / unscale this fit.  It is already scaled mostly.

    If fitting with scaling, then the algebra necessary to unscale the problem
    to original units is:
        a = core
        b = edge/core;   b = offset/(slope+offset)

        y/a = edge/core + (1-edge/core)*(1-x^c)^d
        y = edge + (core-edge)*(1-x^c)^d
        prof = offset + slope*(1-x^c)^d

        # Edge = offset
        # Core - Edge = slope;  Core = slope+offset;
        # b = edge/core = offset/(slope+offset)

        af[0] = edge/core = offset/(slope+offset)

        (y/a -b)/(1-b) = (1-x^c)^d

    found in the info Structure
    """

    if af is None:
        af = 0.1*_np.ones((3,), dtype=_np.float64)
    # endif

    info = Struct()
    info.Lbounds = _np.array([0.0, -20.0, -20.0], dtype=_np.float64)
    info.Ubounds = _np.array([_np.inf, 20.0, 20.0], dtype=_np.float64)
    info.af = af

    def unscaleaf(ain, slope, offset=0.0):
        aout = _np.copy(ain)
        aout[0] = slope*aout[0]
        return aout
    info.unscaleaf = unscaleaf
    if XX is None:
        return info

    prof = edgepower(XX, af)
    gvec = partial_edgepower(XX, af)

    info.prof = prof
    info.gvec = gvec
    info.dprofdx = deriv_edgepower(XX, af)
    info.dgdx = partial_deriv_edgepower(XX, af)
    info.d2profdx2 = deriv2_edgepower(XX,af)
    info.d2gdx2 = partial_deriv2_edgepower(XX,af)
    return prof, gvec, info

# ========================================================================== #

def twopower(XX, af):
    """
    model a two-power fit
        first-half of a quasi-parabolic (no hole depth width or decaying edge)
                .... dumb: reproduced everywhere: y-offset = slope*(1-x^b)^c
        y = a*(1.0 - x**b)**c

        y = edge/core + (1-edge/core)
        a = amplitude of core
        b = power scaling factor 1
        c = power scaling factor 2
    """
    a = af[0]
    b = af[1]
    c = af[2]
    return a*_np.power((1.0-_np.power(XX,b)), c)

def partial_twopower(XX, af):
    a = af[0]
    b = af[1]
    c = af[2]

    gvec = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = _np.power(1.0-_np.power(XX,b),c)
    gvec[1,:] = (
        -1.0*a*c*_np.power(XX,b)*utlog(XX)*_np.power(1.0-_np.power(XX,b),c-1.0)
        )
    gvec[2,:] = (
        a*_np.power(1.0-_np.power(XX,b),c)*utlog(-_np.power(XX,b), a1p=True)
#        a*_np.power(1.0-_np.power(XX,b),c)*_np.log(1.0-_np.power(XX,b))
        )
    return gvec

def deriv_twopower(XX, af):
    a = af[0]
    b = af[1]
    c = af[2]
    return -1.0*a*b*c*_np.power(XX,b-1)*_np.power((1-_np.power(XX,b)), c-1)

def partial_deriv_twopower(XX, af):
    a = af[0]
    b = af[1]
    c = af[2]

    gvec = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = -1.0*b*c*_np.power(XX,b-1.0)*_np.power(1.0-_np.power(XX,b), c-1.0)
    gvec[1,:] = (
        -1.0*a*c*_np.power(XX,b-1.0)*_np.power(1.0-_np.power(XX,b),c-1.0)
        - a*c*b*_np.power(XX,b-1.0)*utlog(XX)*_np.power(1.0-_np.power(XX,b),c-1.0)
        + a*(c-1.0)*c*b*_np.power(XX,2*b-1.0)*utlog(XX)*_np.power(1.0-_np.power(XX,b),c-2.0)
        )
    gvec[2,:] = (
        -1.0*a*b*_np.power(XX,b-1.0)*_np.power(1.0-_np.power(XX,b),c-1.0)*(c*utlog(-_np.power(XX,b), a1p=True)+1.0)
#        -1.0*a*b*_np.power(XX,b-1.0)*_np.power(1.0-_np.power(XX,b),c-1.0)*(c*_np.log(1.0-_np.power(XX,b))+1.0)
        )
    return gvec

def model_twopower(XX, af=None):
    """
    model a two-power fit
        first-half of a quasi-parabolic (no hole depth width or decaying edge)
                .... dumb: reproduced everywhere: y-offset = slope*(1-x^b)^c
        y = a*(1.0 - x**b)**c

        y = edge/core + (1-edge/core)
        a = amplitude of core
        b = power scaling factor 1
        c = power scaling factor 2

        To unscale the problem
            prof = slope*(1-x^b)^c+offset
            slope = (a-b) ~ core-edge
            offset = b = edge
    """

    info = Struct()
    info.Lbounds = _np.array([0.0, -20.0, -20.0], dtype=_np.float64)
    info.Ubounds = _np.array([20, 20.0, 20.0], dtype=_np.float64)
    if af is None:
        af = _np.array([1.0, 12.0, 3.0], dtype=_np.float64)
#        af *= 0.1*_np.random.normal(0.0, 1.0, 1)
    # endif
    info.af = af

    def unscaleaf(ain, slope, offset=0.0):
        aout = _np.copy(ain)
        aout[0] = slope
        return aout
    info.unscaleaf = unscaleaf
    if XX is None:
        return info

    prof = twopower(XX, af)          # a2+(a1-a2)*(1.0 - XX**a3)**a4
    gvec = partial_twopower(XX, af)  # _np.atleast_2d(prof / af)

    info.prof = prof
    info.gvec = gvec
    info.dprofdx = deriv_twopower(XX, af)
    info.dgdx = partial_deriv_twopower(XX, af)
    return prof, gvec, info

# ========================================================================== #

def expedge(XX, af):
    """
    model an exponential edge
        second-half of a quasi-parabolic (no edge, or power factors)
        a = amplitude of core
        e = hole width
        h = hole depth
    """
    e = af[0]
    h = af[1]
    return e*(1-_np.exp(-_np.square(XX)/h))

# ========= Quasi-parabolic model ========== #

def model_qparab(XX, af=None, nohollow=False, prune=False, rescale=False, info=None):
    """
    ex// ne_parms = [0.30, 0.002, 2.0, 0.7, -0.24, 0.30]
    This function calculates the quasi-parabolic fit
    Y/Y0 = af[1]-af[4]+(1-af[1]+af[4])*(1-xx^af[2])^af[3]
                + af[4]*(1-exp(-xx^2/af[5]^2))
    xx - r/a
    af[0] - Y0 - function value on-axis
    af[1] - gg - Y1/Y0 - function value at edge over core
    af[2],af[3]-  pp, qq - power scaling parameters
    af[4],af[5]-  hh, ww - hole depth and width

    If fitting with scaling, then the algebra necessary to unscale the problem
    to original units is:
            af[0] is Y0, af[1] if Y1/Y0; Y1 = af[1]*af[0]

        af[1] = (slope*af[1]*af[0]+offset)/(slope*af[0]+offset)
        af[0] = slope*af[0]+offset
    found in the info Structure

    """
    if info is None:
        info = Struct()  # Custom class that makes working with dictionaries easier
#    info.Lbounds = _np.array([    0.0, 0.0,-_np.inf,-_np.inf,-_np.inf,-_np.inf], dtype=_np.float64)
#    info.Ubounds = _np.array([_np.inf, _np.inf, _np.inf, _np.inf, _np.inf, _np.inf], dtype=_np.float64)
#    info.Lbounds = _np.array([  0.0, 0.0,-20,-20,-1,-1], dtype=_np.float64)
#    info.Ubounds = _np.array([ 20.0, 1.0, 20, 20, 1, 1], dtype=_np.float64)
    info.Lbounds = _np.array([  0.0, 0.0,-10,-10,-1,-1], dtype=_np.float64)
    info.Ubounds = _np.array([ 20.0, 1.0, 10, 10, 1, 1], dtype=_np.float64)

    if af is None:
        af = _np.array([1.0, 0.002, 2.0, 0.7, -0.24, 0.30], dtype=_np.float64)
#        af = randomize_initial_conditions(info.Lbounds, info.Ubounds)
        if nohollow:
            af[4] = 0.0
            af[5] = 1.0
        # endif
    if len(af) == 4:
        nohollow = True
        af = _np.hstack((af,0.0))
        af = _np.hstack((af,1.0))
    # endif
    info.af = _np.copy(af)

    def unscaleaf(ain, slope, offset):
        aout = _np.copy(ain)
        aout[1] = (slope*ain[1]*ain[0]+offset)/(slope*ain[0]+offset)
        aout[0] = slope*ain[0]+offset
        return aout
    info.unscaleaf = unscaleaf
    if XX is None:
        if prune:
            info.af = info.af[:4]
            info.Lbounds = info.Lbounds[:4]
            info.Ubounds = info.Ubounds[:4]
        # end if
        return info
    # endif

    # ========= #
    XX = _np.abs(XX)
    if rescale:
        XX = rescale_xlims(XX, forward=True, ascl=rescale)
    else:
        rescale = 1.0
    # end if

    # ========= #

    af = af.reshape((len(af),))
    if _np.isfinite(af).any() == 0:
        print("checkit! No finite values in fit coefficients! (from model_spec: model_qparab)")
#    print(_np.shape(af))

    try:
        prof = qparab(XX, af, nohollow)
#        prof = interp_irregularities(prof, corezero=False)
        info.prof = prof

        gvec = partial_qparab(XX*rescale, af, nohollow)
#        gvec = interp_irregularities(gvec, corezero=False)  # invalid slice
        info.gvec = gvec

        info.dprofdx = deriv_qparab(XX, af, nohollow)
#        info.dprofdx = interp_irregularities(info.dprofdx, corezero=True)

        info.dgdx = partial_deriv_qparab(XX*rescale, af, nohollow)
#        info.dgdx = interp_irregularities(info.dgdx, corezero=False)
    except:
        pass
        raise

    if prune:
        af = af[:4]
        info.Lbounds = info.Lbounds[:4]
        info.Ubounds = info.Ubounds[:4]

        gvec = gvec[:4, :]
        info.dgdx = info.dgdx[:4, :]
    # endif
    info.af = af

    return prof, gvec, info
# end def model_qparab

# ========= Subfunctions of the quasi-parabolic model ========== #


# Set the plasma density, temperature, and Zeff profiles (TRAVIS INPUTS)
def qparab(XX, *aa, **kwargs):
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
    options = {}
    options.update(kwargs)
    nohollow = options.get('nohollow', False)
    aedge = options.get('aedge', 1.0)
    if len(aa)>6:
        nohollow = aa.pop(6)
    XX = _np.abs(XX)/aedge
    if (type(aa) is tuple) and (len(aa) == 2):
        nohollow = aa[1]
        aa = aa[0]
    elif (type(aa) is tuple) and (len(aa) == 1):
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

    if aa[2] == -1.0:
        aa[2] += 1e-100
    if aa[3] == 1.0 or aa[3] == 2.0:
        aa[3] += 1e-100
    if aa[5] == 0.0:
        aa[5] += 1e-100
    # end if

    prof = aa[0]*( aa[1]-aa[4]
                   + (1.0-aa[1]+aa[4])*_np.abs(1.0-XX**aa[2])**aa[3]
                   + aa[4]*-1.0*_np.expm1(-XX**2.0/aa[5]**2.0) )
#                   + aa[4]*(1.0-_np.exp(-XX**2.0/aa[5]**2.0)) )
    return prof
# end def qparab

def deriv_qparab(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
    """
    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
    This subfunction calculates the derivative of a quasi-parabolic fit
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

    if aa[2] == -1.0:
        aa[2] += 1e-100
    if aa[3] == 1.0 or aa[3] == 2.0:
        aa[3] += 1e-100
    if aa[5] == 0.0:
        aa[5] += 1e-100
    # end if

    dpdx = aa[0]*( (1.0-aa[1]+aa[4])*(-1.0*aa[2]*XX**(aa[2]-1.0))*aa[3]*(1.0-XX**aa[2])**(aa[3]-1.0)
                   - aa[4]*(-2.0*XX/aa[5]**2.0)*_np.exp(-XX**2.0/aa[5]**2.0) )

    return dpdx
# end def derive_qparab

def deriv2_qparab(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
    """
    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
    This subfunction calculates the second derivative of a quasi-parabolic fit
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

    if aa[2] == -1.0:
        aa[2] += 1e-100
    if aa[3] == 1.0 or aa[3] == 2.0:
        aa[3] += 1e-100
    if aa[5] == 0.0:
        aa[5] += 1e-100
    # end if

    d2pdx2 = aa[3]*(aa[2]**2.0)*(aa[3]-1.0)*(1.0+aa[4]-aa[1])*(XX**(2.*aa[2]-2.0))*(1-XX**aa[2])**(aa[3]-2.0)
    d2pdx2 -= (aa[2]-1.0)*aa[2]*aa[3]*(1.0+aa[4]-aa[1])*(XX**(aa[2]-2.0))*(1-XX**aa[2])**(aa[3]-1.0)
    d2pdx2 += (2.0*aa[4]*_np.exp(-XX**2.0/(aa[5]**2.0)))/(aa[5]**2.0)
    d2pdx2 -= (4*aa[4]*(XX**2.0)*_np.exp(-XX**2.0/(aa[5]**2.0)))/(aa[5]**4.0)
    d2pdx2 *= aa[0]
    return d2pdx2
# end def derive_qparab

def partial_qparab(XX,aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
    """
    ex// ne_parms = [0.30, 0.002 2.0, 0.7 -0.24 0.30]
    This subfunction calculates the jacobian of a quasi-parabolic fit

    quasi-parabolic fit:
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
    Y0 = aa[0]
    g = aa[1]
    p = aa[2]
    q = aa[3]
    h = aa[4]
    w = aa[5]

    if p == -1.0:
        p += 1e-100
    if q == 1.0 or q == 2.0:
        q += 1e-100
    if w == 0:
        w += 1e-100
    # end if

    gvec = _np.zeros( (6,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = g-h+(1.0-g+h)*_np.abs(1.0-XX**p)**q + h*(1.0-_np.exp(-XX**2.0/w**2.0))
    gvec[1,:] = Y0*( 1.0-_np.abs(1.0-XX**p)**q )    # aa[0]
    gvec[2,:] = -1.0*Y0*q*(g-h-1.0)*(XX**p)*utlog(XX)*(_np.abs(1-XX**p)**q)/(XX**p-1.0)
    gvec[3,:] = Y0*(-g+h+1.0)*(_np.abs(1-XX**p)**q)*utlog(-XX**p, a1p=True)
#    gvec[3,:] = Y0*(-g+h+1.0)*(_np.abs(1-XX**p)**q)*_np.log(1.0-XX**p)
    gvec[4,:] = Y0*(_np.abs(1.0-XX**p)**q) - Y0*_np.exp(-(XX/w)**2.0)
    gvec[5,:] = -2.0*h*(XX**2.0)*Y0*_np.exp(-(XX/w)**2.0) / w**3.0

    return gvec
# end def partial_qparab


def partial_deriv_qparab(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
    """
    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
    This subfunction calculates the jacobian of the derivative of a
    quasi-parabolic fit (partial derivatives of the derivative of a quasi-parabolic fit)

    quasi-parabolic fit:
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
    Y0 = aa[0]
    g = aa[1]
    p = aa[2]
    q = aa[3]
    h = aa[4]
    w = aa[5]

    if p == -1.0:
        p += 1e-100
    if q == 1.0 or q == 2.0:
        q += 1e-100
    if w == 0:
        w += 1e-100
    # end if

    gvec = _np.zeros( (6,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = 2.0*h*XX*_np.exp(-(XX/w)**2.0)/(w**2.0) - p*q*(-g+h+1.0)*(XX**(p-1.0))*((1.0-XX**p)**(q-1.0))
    gvec[1,:] = p*q*Y0*(XX**(p-1.0))*((1-XX**p)**(q-1.0))

    gvec[2,:] = q*Y0*(-1.0*(g-h-1.0))*((1.0-XX**p)**(q-2.0))
    gvec[2,:] *= p*utlog(XX)*((q-1.0)*(XX**(2.0*p-1.0))
                    - (XX**(p-1.0))*(1.0-XX**p))+(XX**p-1.0)*XX**(p-1.0)

    gvec[3,:] = p*Y0*(g-h-1.0)*(XX**(p-1.0))*((1.0-XX**p)**(q-1.0))*(q*utlog(-XX**p, a1p=True)+1.0)
#    gvec[3,:] = p*Y0*(g-h-1.0)*(XX**(p-1.0))*((1.0-XX**p)**(q-1.0))*(q*_np.log(1.0-XX**p)+1.0)
    gvec[4,:] = (2.0*XX*Y0*_np.exp(-1.0*(XX/w)**2.0))/(w**2.0) - p*q*Y0*(XX**(p-1.0))*((1.0-XX**p)**(q-1.0))

    gvec[5,:] = h*Y0*_np.exp(-1.0*(XX/w)**2.0)*((4.0*(XX**3.0))/(w**5.0)-(4.0*XX)/(w**3.0))

    return gvec
# end def partial_deriv_qparab

def partial_deriv2_qparab(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
    """
    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
    This subfunction calculates the jacobian of the second derivative of a
    quasi-parabolic fit (partial derivatives of the second derivative of a quasi-parabolic fit)

    quasi-parabolic fit:
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
    Y0 = aa[0]
    g = aa[1]
    p = aa[2]
    q = aa[3]
    h = aa[4]
    w = aa[5]

    if p == -1.0:
        p += 1e-100
    if q == 1.0 or q == 2.0:
        q += 1e-100
    if w == 0:
        w += 1e-100
    # end if

    gvec = _np.zeros( (6,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = deriv2_qparab(XX, aa, nohollow) / Y0
    gvec[1,:] = -p*q*Y0*(XX**(p-2.0))*(1.0-XX**p)**(q-2.0)*(p*(q*(XX**p)-1.0)-XX**p+1.0)
    gvec[2,:] = p*utlog(XX)*(p*((q**2.0)*(XX**(2.0*p))-3.0*q*(XX**p)+XX**p+1.0)-(XX**p-1.0)*(q*XX**p-1.0))
    gvec[2,:] += (XX**p-1.0)*(2.0*p*(q*(XX**p)-1.0)-XX**p+1.0)
    gvec[2,:] *= q*Y0*(g-h-1.0)*(XX**(p-2.0))*((1.0-XX**p)**(q-3.0))
    gvec[3,:] = p*Y0*(-(g-h-1.0))*(XX**(p-2.0))*((1.0-XX**p)**(q-2.0))*(p*(2.0*q*XX**p-1.0)+q*(p*(q*XX**p-1.0)-XX**p+1.0)*utlog(-XX**p, a1p=True)-XX**p+1.0)
#    gvec[3,:] = p*Y0*(-(g-h-1.0))*(XX**(p-2.0))*((1.0-XX**p)**(q-2.0))*(p*(2.0*q*XX**p-1.0)+q*(p*(q*XX**p-1.0)-XX**p+1.0)*_np.log(1.0-XX**p)-XX**p+1.0)
    gvec[4,:] = Y0*(p*q*(XX**(p-2.0))*((1.0-XX**p)**(q-2.0))*(p*(q*XX**p-1.0)-XX**p+1.0)+(2.0*_np.exp(-XX**2.0/w**2.0)*(w**2.0-2.0*XX**2.0))/w**4.0)
    gvec[5,:] = -(4.0*h*Y0*_np.exp(-XX**2.0/w**2.0)*(w**4.0-5*w**2.0*XX**2.0+2.0*XX**4.0))/w**7.0

    return gvec
# end def partial_deriv2_qparab

# ========================================================================== #
# ========================================================================== #


def model_ProdExp(XX, af=None, npoly=4):
    """
    --- Product of Exponentials ---
    Model - chi ~ prod(af(ii)*XX^(polyorder-ii))
    af    - estimate of fitting parameters
    XX    - independent variable
        npoly is overruled by the shape of af.  It is only used if af is None
    """
    info = Struct()
    info.Lbounds = -_np.inf*_np.ones((npoly+1,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((npoly+1,), dtype=_np.float64)

    if af is None:
        af = 0.9*_np.ones((npoly+1,), dtype=_np.float64)
#        af *= _np.random.normal(0.0, 1.0, npoly+1.0)
#        af = randomize_initial_conditions(info.Lbounds, info.Ubounds)
    # endif
    npoly = _np.size(af)-1
    info.Lbounds = -_np.inf*_np.ones((npoly+1,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((npoly+1,), dtype=_np.float64)
    info.af = af

    nx = _np.size(XX)
    num_fit = _np.size(af)  # Number of fitting parameters

    # Polynomial of order num_fit
    pp = _np.poly1d(af)
    prof = pp(XX)

    # Could be just an exponential fit
    # prof=af[-1]*_np.exp(prof)
    prof = _np.exp(prof)

    #The derivative of chi with respect to rho is analytic as well:
    # f = exp(a1x^n+a2x^(n-1)+...a(n+1))
    # f = exp(a1x^n)exp(a2x^(n-1))...exp(a(n+1)));
    # dfdx = (n*a1*x^(n-1)+(n-1)*a2*x^(n-2)+...a(n))*f
    ad = pp.deriv()
    info.dprofdx = ad(XX)
    info.dprofdx = prof*info.dprofdx

    # The g-vector contains the partial derivatives used for error propagation
    # f = exp(a1*x^2+a2*x+a3)
    # dfda1 = x^2*f;
    # dfda2 = x  *f;
    # dfda3 = f;
    # gvec(0,1:nx) = XX**2.*prof;
    # gvec(1,1:nx) = XX   .*prof;
    # gvec(2,1:nx) =        prof;
    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(num_fit):  # 1:num_fit
        # Formulated this way, there is an analytic jacobian:
        kk = num_fit - (ii + 1)
        gvec[ii, :] = (XX**kk)*prof
    # endif
    info.gvec = gvec

    # The g-vector (jacobian) for the derivative
    # dfdx = (...+2*a1*x + a2)*exp(...+a1*x^2+a2*x+a3)
    #
    # Product rule:  partial derivatives of the exponential term times the leading derivative polynomial
    dgdx = gvec.copy() * (_np.ones((num_fit,1), dtype=float) * _np.atleast_2d(ad(XX)))

    # Product rule:  exponential term times the partial derivatives of the derivative polynomial
    for ii in range(num_fit-1):  # 1:num_fit
        # Formulated this way, there is an analytic jacobian:
        kk = num_fit-1 - (ii + 1)
        dgdx[ii, :] += (kk+1)*(XX**kk)*prof
    # endif
    info.dgdx = dgdx

    return prof, gvec, info
# end def model_ProdExp()
# ========================================================================== #
# ========================================================================== #


def model_poly(XX, af=None, npoly=4):
    """
    --- Straight Polynomial ---
    Model - chi ~ sum( af(ii)*XX^(polyorder-ii))
    af    - estimate of fitting parameters
    XX    - independent variable

     if the data is scaled, then unscaling it goes like this:
         (y-miny)/(maxy-miny) = (y-offset)/slope
         (y-offset)/slope = sum_i(a_i'*x^i)

         y = slope* sum_i(a_i'*x^i) + offset
         a_i = slope*a'
    """
    info = Struct()
    info.Lbounds = -_np.inf*_np.ones((npoly+1,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((npoly+1,), dtype=_np.float64)

    if af is None:
#        af = randomize_initial_conditions(info.Lbounds, info.Ubounds)
        af = 0.1*_np.ones((npoly+1,), dtype=_np.float64)
#        af *= _np.random.normal(0.0, 1.0, npoly+1.0)
    # endif
    npoly = _np.size(af)-1
    info.Lbounds = -_np.inf*_np.ones((npoly+1,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((npoly+1,), dtype=_np.float64)
    info.af = af

    def unscaleaf(ain, slope, offset=0.0):
        aout = _np.copy(ain)
        aout = ain*slope
        return aout
    info.unscaleaf = unscaleaf

    num_fit = _np.size(af)  # Number of fitting parameters
    nx = _np.size(XX)

    # Polynomial of order num_fit
    pp = _np.poly1d(af)
    prof = pp(XX)
    info.prof = prof

    ad = pp.deriv()
    info.dprofdx = ad(XX)

    # The g-vector contains the partial derivatives used for error propagation
    # f = a1*x^2+a2*x+a3
    # dfda1 = x^2;
    # dfda2 = x;
    # dfda3 = 1;
    # gvec(0,1:nx) = XX**2;
    # gvec(1,1:nx) = XX   ;
    # gvec(2,1:nx) = 1;

    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(num_fit):  # ii=1:num_fit
        kk = num_fit - (ii + 1)
        gvec[ii, :] = XX**kk
    # endfor
    info.gvec = gvec

    # The jacobian for the derivative
    # f = a1*x^2+a2*x+a3
    # dfdx = 2*a1*x+a2
    # dfda1 = 2*x;
    # dfda2 = 1;
    # dfda3 = 0;
    # dgdx(1,1:nx) = 2*XX;
    # dgdx(2,1:nx) = 1.0;
    # dgdx(3,1:nx) = 0.0;
    dgdx = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(num_fit-1):
        kk = num_fit-1 - (ii + 1)
        dgdx[ii,:] = (kk+1)*(XX**kk)
    # end for
    info.dgdx = dgdx

    return prof, gvec, info
# end def model_poly()

# ========================================================================== #
# ========================================================================== #


def model_evenpoly(XX, af=None, npoly=4):
    """
    --- Polynomial with only even powers ---
    Model - chi ~ sum( af(ii)*XX^2*(numfit-ii))
    af    - estimate of fitting parameters (npoly=4, numfit=3, poly= a0*x^4+a1*x^2+a3)
    XX    - independent variable
    """
    nx = _np.size(XX)

    info = Struct()
    info.Lbounds = -_np.inf*_np.ones((npoly//2+1,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((npoly//2+1,), dtype=_np.float64)

    if af is None:
        af = 0.1*_np.ones((npoly//2+1,), dtype=_np.float64)
#        af *= _np.random.normal(0.0, 1.0, npoly//2+1.0)
#        af = randomize_initial_conditions(info.Lbounds, info.Ubounds)
    # endif
    num_fit = _np.size(af)  # Number of fitting parameters
    npoly = _np.int(2*(num_fit-1))  # Polynomial order from input af
    info.Lbounds = -_np.inf*_np.ones((npoly//2+1,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((npoly//2+1,), dtype=_np.float64)

    info.af = af

    # Even Polynomial of order num_fit, Insert zeros for the odd powers
    a0 = _np.insert(af, _np.linspace(1, num_fit-1, 2), 0.0)
    pp = _np.poly1d(a0)
    prof = pp(XX)
    info.prof = prof

    ad = pp.deriv()
    info.dprofdx = ad(XX)

    # The g-vector contains the partial derivatives used for error propagation
    # f = a1*x^4+a2*x^2+a3
    # dfda1 = x^4;
    # dfda2 = x^2;
    # dfda3 = 1;
    # gvec(1,1:nx) = XX**4;
    # gvec(2,1:nx) = XX**2;
    # gvec(3,1:nx) = 1;

    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(num_fit):  # ii=1:num_fit
        #2*(num_fit-1)
        kk = num_fit - (ii + 1)
        kk *= 2
        gvec[ii, :] = XX**kk
    # endfor
    info.gvec = gvec

    # The jacobian for the derivative
    # f = a1*x^4+a2*x^2+a3
    # dfdx = 4*a1*x^3 + 2*a2*x + 0
    # dfdxda1 = 4*x^3;
    # dfda2 = 2*x^1;
    # dfda3 = 0;
    # dgdx(1,1:nx) = 4*XX**3;
    # dgdx(2,1:nx) = 2*XX;
    # dgdx(3,1:nx) = 0.0;
    dgdx = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(num_fit-1):
        kk = num_fit - (ii + 1)    # ii=0, kk = num_fit-1;   ii=num_fit-2, kk=+1
        kk *= 2                    #       kk = 2*num_fit-2;               kk=+2
        dgdx[ii,:] = kk * XX**(kk-1)
    # end for
    info.dgdx = dgdx

    return prof, gvec, info
# end def model_evenpoly()

# ========================================================================== #
# ========================================================================== #


def model_PowerLaw(XX, af=None, npoly=4):
    """
    --- Power Law w/exponential cut-off ---
    Model - fc = x^(a1*x^(n)+a2*x^(n-1)+...a(n))
            chi ~ a(n+2)*fc*exp(a(n+1)*x)
            # chi ~ a(n+1)*x^(a1*x^n+a2*x^(n-1)+...an)
    af    - estimate of fitting parameters
    XX    - independent variable
    """
    info = Struct()
    info.Lbounds = _np.hstack((-_np.inf * _np.ones((npoly,), dtype=_np.float64), -_np.inf,       0))
    info.Ubounds = _np.hstack(( _np.inf * _np.ones((npoly,), dtype=_np.float64),  _np.inf, _np.inf))

    if af is None:
        af = 0.1*_np.ones((npoly+2,), dtype=_np.float64)
#        af *= _np.random.normal(0.0, 1.0, npoly+2.0)    # endif
#        af = randomize_initial_conditions(info.Lbounds, info.Ubounds)
    # end if
    num_fit = _np.size(af)  # Number of fitting parameters
    npoly = num_fit-3
    info.Lbounds = _np.hstack((-_np.inf * _np.ones((npoly,), dtype=_np.float64), -_np.inf,       0))
    info.Ubounds = _np.hstack(( _np.inf * _np.ones((npoly,), dtype=_np.float64),  _np.inf, _np.inf))

    nx = _np.size(XX)


    info.af = af

    if len(af) != info.Lbounds.shape[0]:
        print('pause')
    # end if
    # Curved power-law:
    # fc = x^(a1*x^(n+1)+a2*x^n+...a(n+1))
    # With exponential cut-off:
    # f  = a(n+2)*fc(x)*exp(a(n+1)*XX);
    pp = _np.poly1d(af[:npoly+1])
    polys = pp(XX)
    exp_factor = _np.exp(af[num_fit-2]*XX)
    prof = af[num_fit-1]*(XX**polys)*exp_factor
    info.prof = prof

    # dfdx = dfcdx*(an*e^an1x) +an1*f(x);
    # dfcdx = XX^(-1)*prof*(polys+ddx(poly)*XX*log(XX),
    # log is natural logarithm
    dcoeffs = _np.poly1d(af[:npoly+1])
    dcoeffs = dcoeffs.deriv()
    dpolys = dcoeffs(XX)
    dcoeffs = dcoeffs.coeffs

    info.dprofdx = polys/XX + utlog(XX)*dpolys
    info.dprofdx *= prof

    # The g-vector contains the partial derivatives used for error propagation
    # fc = x^( a1*x^(n+1)+a2*x^n+...a(n+1) )
    # f  = a(n+2)*fc(x)*exp(a(n+1)*XX)
    # dfda_n = dfc/da_n * (f/fc)
    # dfda_n+1 = XX*f
    # dfda_n+2 = f/a_n+2
    # gvec(0,1:nx) = XX**(n+1)*utlog(XX);
    # gvec(1,1:nx) = XX   ;
    # gvec(2,1:nx) = 1;
    # ...
    # gvec(num_fit-1, 1:nx) = 1;
    # gvec(num_fit  , 1:nx) = d1;

    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(npoly+1):  # ii=1:num_fit
        kk = npoly+1 - (ii + 1)
        gvec[ii, :] = prof*utlog(XX)*XX**kk
    # endfor
    gvec[num_fit-1,:] = prof/af[num_fit-1]
    gvec[num_fit-1, :] = prof*XX     # TODO: CHECK THIS
#    gvec[num_fit, :] = prof*XX     # TODO: CHECK THIS
    info.gvec = gvec

    # The jacobian of the derivative
    dgdx = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(npoly+1):  # ii=1:(npoly-1)
        kk = npoly+1 - (ii + 1)
        dgdx[ii, :] = info.dprofdx*utlog(XX)*(XX**kk)
#        dgdx[ii, :] += prof*af[num_fit]*utlog(XX)*(XX**kk)
        dgdx[ii, :] += prof*af[num_fit-1]*utlog(XX)*(XX**kk)  # TODO: check this

        if ii<npoly:
            dgdx[ii, :] += prof*(XX**(kk-1))*(1.0 + kk*utlog(XX))     # 3 = dcoeffs / af[:npoly+1]
        else:
            dgdx[ii, :] += prof*(XX**(kk-1))
        # endif
    # endfor
    dgdx[num_fit-2, :] = (info.dprofdx/(af[num_fit-2]) + af[num_fit-1])*prof/af[num_fit-1]
    dgdx[num_fit-1, :] = prof*( af[num_fit-1]*XX + 1.0 + XX*info.dprofdx )
#    dgdx[num_fit-1, :] = (info.dprofdx/(af[num_fit-1]) + af[num_fit])*prof/af[num_fit-1]
#    dgdx[num_fit  , :] = prof*( af[num_fit]*XX + 1.0 + XX*info.dprofdx )
    info.dgdx = dgdx

    return prof, gvec, info
# end def model_PowerLaw()

# ========================================================================== #
# ========================================================================== #


def model_Exponential(XX, af=None, npoly=None):
    """
    --- Exponential on Background ---
    Model - chi ~ a1*(exp(a2*xx^a3) + XX^a4)
    af    - estimate of fitting parameters
    XX    - independent variables
    """
#    num_fit = npoly+3
    num_fit = 4

    info = Struct()
    info.Lbounds = -_np.inf*_np.ones((num_fit,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((num_fit,), dtype=_np.float64)
    info.Lbounds[0] = 0

    if af is None:
        af = 0.1*_np.ones((num_fit,), dtype=_np.float64)
#        af *= _np.random.normal(0.0, 1.0, num_fit)
#        af = randomize_initial_conditions(info.Lbounds, info.Ubounds)
    # endif
    num_fit = _np.size(af)  # Number of fitting parameters
    info.Lbounds = -_np.inf*_np.ones((num_fit,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((num_fit,), dtype=_np.float64)
    info.Lbounds[0] = 0

    nx = _np.size(XX)

    info.af = af

    # f     = a1*(exp(a2*xx^a3) + XX^a4) = f1+f2;
    # dfdx  = a1*(a2*a3*xx^(a3-1)*exp(a2*xx^a3) + a4*xx^(a4-1));
    # dfda1 = f/a1;
    # dfda2 = xx^a3*f1;
    # dfda3 = f1*xx^a3*log10(xx)
    # dfda4 = a1*xx^a4*log10(xx) = log10(xx)*f2;
    prof1 = af[0]*_np.exp(af[1]*XX**af[2])
    dprof1dx = af[1]*af[2]*XX**(af[2]-1.0)*prof1

    prof2 = af[0]*XX**af[3]
    dprof2dx = af[0]*af[3]*(XX**(af[3]-1))

    prof = prof1 + prof2
    info.prof = prof
    info.dprofdx = dprof1dx + dprof2dx

    gvec = _np.zeros( (num_fit,nx), dtype=float)
    gvec[0, :] = prof/af[0]
    gvec[1, :] = prof1*(XX**af[2])
    gvec[2, :] = prof1*af[1]*utlog(XX)*(XX**af[2])
    gvec[3, :] = af[0]*utlog(XX)*(XX**af[3])

    dgdx = _np.zeros( (num_fit,nx), dtype=float)
    dgdx[0, :] = info.dprofdx / af[0]
    dgdx[1, :] = dprof1dx*(XX**af[2]) + dprof1dx/af[1]
    dgdx[2, :] = dprof1dx*(af[1]*utlog(XX)*(XX**af[2]) + utlog(XX) + 1.0/af[2] )
    dgdx[3, :] = dprof2dx*utlog(XX) + af[0]*XX**(af[3]-1.0)

    # gvec = _np.zeros( (num_fit,nx), dtype=float)
    # dgdx = _np.zeros( (num_fit,nx), dtype=float)
    # for ii in range(3, num_fit)  # ii = 1:(num_fit-3)
    #     kk = npoly+1 - (ii-2)
    #     prof2 += af[ii]*XX**kk
    #     gvec[ii,:] = af[0]*XX**kk
    #
    #     if ii < num_fit:
    #       dprof2dx += kk*af[ii]*XX**(kk-1)
    #       dgdx[ii,:] = kk*af[0]*XX**(kk-1)
    # # end


    return prof, gvec, info
# end def model_Exponential()

# ========================================================================== #
# ========================================================================== #


def model_parabolic(XX, af):
    """
    A parabolic profile with one free parameters:
        f(x) ~ a*(1.0-x^2)
        xx - x - independent variable
        af - a - central value of the plasma parameter
    """

    if af is None:
        af = _np.array([1.0], dtype=_np.float64)
#        af *= 0.1*_np.random.normal(0.0, 1.0, 1)
    # endif

    info = Struct()
    info.Lbounds = _np.array([0.0], dtype=_np.float64)
    info.Ubounds = _np.array([_np.inf], dtype=_np.float64)
    info.af = af

    prof = af*(1.0 - XX**2.0)
    gvec = _np.atleast_2d(prof / af)

    info.prof = prof
    info.gvec = gvec
    info.dprofdx = -2.0*af*XX
    info.dgdx = -2.0*XX
    return prof, gvec, info

# ========================================================================== #
# ========================================================================== #


def model_flattop(XX, af):
    """
    A flat-top plasma parameter profile with three free parameters:
        a, b, c
    prof ~ f(x) = a / (1 + (x/b)^c)
        af[0] - a - central value of the plasma parameter
        af[1] - b - determines the gradient location
        af[2] - c - the gradient steepness
    The profile is constant near the plasma center, smoothly descends into a
    gradient near (x/b)=1 and tends to zero for (x/b)>>1
    """
    if af is None:
        af = _np.array([1.0, 0.4, 5.0], dtype=_np.float64)
#        af *= 0.1*_np.random.normal(0.0, 1.0, len(af))
    # endif

    nx = len(XX)
    XX = _np.abs(XX)

    info = Struct()
    info.Lbounds = _np.array([0.0, 0.0, 1.0], dtype=_np.float64)
    info.Ubounds = _np.array([_np.inf, 1.0, _np.inf], dtype=_np.float64)
    info.af = af

    temp = (XX/af[1])**af[2]
    prof = af[0] / (1.0 + temp)
    info.prof = prof

    gvec = _np.zeros((3, nx), dtype=_np.float64)
    gvec[0, :] = prof / af[0]
    gvec[1, :] = af[0]*af[2]*temp / (af[1]*(1.0+temp)**2.0)
    gvec[2, :] = af[0]*temp*utlog(XX/af[1]) / (1.0+temp)**2.0
    info.gvec = gvec

    info.dprofdx = -1.0*af[0]*af[2]*temp/(XX*(1.0+temp)**2.0)

    dgdx = _np.zeros((3, nx), dtype=_np.float64)
    dgdx[0, :] = info.dprofddx / af[0]
    dgdx[1, :] = prof * info.dprofdx * (XX/temp) * (af[2]/af[0]) * (temp-1.0) / (af[1]*af[1])
    dgdx[2, :] = info.dprofdx/af[2]
    dgdx[2, :] += info.dprofdx*utlog(XX/af[1])
    dgdx[2, :] -= 2.0*(info.dprofdx**2.0)*(utlog(XX/af[1])/prof)
    info.dgdx = dgdx
    return prof, gvec, info

# ========================================================================== #
# ========================================================================== #


def model_massberg(XX, af):
    """
    Commonly referred to the Massberg profile and used in a lot of W7-AS
    analyses.
        Four free parameters a, b, c and h,
        af[0] - a - central value of the plasma parameter
        af[1] - b - determines the gradient location
        af[2] - c - the gradient steepness
        af[3] - h - profile peaking / hollowness (core linear slope)
            prof = a * (1-h*(x/b)) / (1+(x/b)^c)
                 = flattop*(1-h*(x/b))
    Similar to the FlatTopProfile, but allows for finite slope near core.
    The slope can be positive (hollow profile, h < 0) or
                     negative (peaked profile, h > 0).
    """
    if af is None:
        af = _np.array([1.0, 0.4, 5.0, 2.0], dtype=_np.float64)
#        af *= 0.1*_np.random.normal(0.0, 1.0, len(af))
    # endif

    nx = len(XX)
    XX = _np.abs(XX)

    info = Struct()
    info.Lbounds = _np.array([0.0, 0.0, 1.0, -_np.inf], dtype=_np.float64)
    info.Ubounds = _np.array(
        [_np.inf, 1.0, _np.inf, _np.inf], dtype=_np.float64)
    info.af = af

    prft, gft, inft = model_flattop(XX, af)

    temp = XX/af[1]
    prof = prft * (1-af[3]*temp)
    info.prof = prof
    info.dprofdx = inft.dprofdx*(1-af[3]*temp) - inft.prof*af[3]/af[1]


    gvec = _np.zeros((4, nx), dtype=_np.float64)
    gvec[0, :] = prof / af[0]
    gvec[1, :] = gft[1,:]*(1.0-af[3]*temp) + inft.prof*af[3]*XX/(af[1]**2.0)
    gvec[2, :] = gft[2,:]*(1.0-af[3]*temp)
    gvec[3, :] = (-1.0*XX / af[1])*inft.prof
    info.gvec = gvec

    dgdx = _np.zeros((4, nx), dtype= float)
    dgdx[0,:] = info.dprofdx / af[0]
    dgdx[1,:] = inft.dgdx[1,:]*(1.0-af[3]*temp) + inft.dprofdx * af[3]*XX/(af[1]**2.0)
    dgdx[1,:] += inft.prof*af[3]/(af[1]**2.0) - (af[3]/af[1])*gft[1,:]
    dgdx[2,:] = inft.dgdx[2,:]*(1.0-af[3]*temp) - gft[2, :]*af[3]/af[1]
    dgdx[3,:] = -1.0*(XX/af[1])*inft.prof
    info.dgdx = dgdx

    return prof, gvec, info

# ========================================================================== #


def model_2power(XX, af=None):
    """
    A two power profile fit with four free parameters:
    prof ~ f(x) = (Core-Edge)*(1-x^pow1)^pow2 + Edge
        af[0] - Core - central value of the plasma parameter
        af[1] - Edge - edge value of the plasma parameter
        af[2] - pow1 - first power
        af[3] - pow2 - second power


    If fitting with scaling, then the algebra necessary to unscale the problem
    to original units is:
        y = (a-b)*(1- x^c)^d + b
             = a*(1- x^c)^d + b*(1 - (1- x^c)^d)
         (y-miny)/(maxy-miny) = (y-offset)/slope
         (y-offset)/slope = (a'-b')*(1- x^c')^d' + b'

         y = (slope*a'-slope*b')*(1- x^c')^d' + slope*b'+offset
           = slope*a'*(1- x^c')^d'+ slope*b'*(1-(1- x^c')^d')+offset
         y-offset = slope*a'*(1- x^c')^d'+ slope*b'*(1-(1- x^c')^d')
         a = slope*a'
         b = slope*b' + offset  ... offset pushes into prof, messes up error prop. a little bit
         c = c'
         d = d'

        ... to  make this actually work you need:
           f(x) = (Core-Edge1)*(1-x^pow1)^pow2 + Edge0
           where
            af[4] - Edge0 - offset value subtracted from fit... should be fixed to zero, but necessary for rescaling
    found in the info Structure
    """

    info = Struct()
    info.Lbounds = _np.array([0.0, 0.0, -_np.inf, -_np.inf], dtype=_np.float64)
    info.Ubounds = _np.array([_np.inf, _np.inf, _np.inf, _np.inf], dtype=_np.float64)
#    info.Lbounds = _np.array([0.0, 0.0, -20.0, -20.0], dtype=_np.float64)
#    info.Ubounds = _np.array([15.0, 1.0, 20.0, 20.0], dtype=_np.float64)

    if af is None:
        af = _np.array([1.0, 0.0, 2.0, 1.0], dtype=_np.float64)
#        af *= 0.1*_np.random.normal(0.0, 1.0, len(af))
    # endif

    info.af = _np.copy(af)

    def unscaleaf(ain, slope, offset):
        """ this cannot reproduce the original fit do to the offset"""
        aout = _np.copy(ain)
        aout[0] = slope*ain[0]
        aout[1] = slope*ain[1]
        return aout
    info.unscaleaf = unscaleaf
    if XX is None:
        return info
    # endif

    nx = len(XX)
    XX = _np.abs(XX)

    # f(x) = (Core-Edge)*(1-x^pow1)^pow2 + Edge
    prof = (af[0]-af[1])*(1.0-XX**af[2])**af[3] + af[1]
    info.prof = prof
    #
    #               d(b^x) = b^x *ln(b)
    # dfdx = -(a0-a1)*a2*a3*x^(a2-1)*(1-x^a2)^(a3-1)
    # dfda0 = (1-x^a2)^a3
    # dfda1 = 1-(1-x^a2)^a3 = 1-dfda0
    # dfda2 = -(a0-a1)*(ln(x)*x^a2)*a3*(1-x^a2)^(a3-1)
    # dfda3 = (a0-a1)*(1-x^a2)^a3*ln(1-x^a2)

    info.dprofdx = -1.0*(af[0]-af[1])*af[2]*af[3]*(XX)**(af[2]-1.0)
    info.dprofdx *= (1.0-XX**af[2])**(af[3]-1.0)

    gvec = _np.zeros((4, nx), dtype=_np.float64)
    gvec[0, :] = (prof-af[1])/(af[0]-af[1])
    gvec[1, :] = 1.0-gvec[0, :].copy()
    gvec[2, :] = -1.0*af[3]*(af[0]-af[1])*(1.0-XX**af[2])**(af[3]-1.0)
    gvec[2, :] *= XX**(af[2])*utlog(XX)
    gvec[3, :] = (af[0]-af[1])*utlog(-XX**af[2], a1p=True)
#    gvec[3, :] = (af[0]-af[1])*_np.log(1.0-XX**af[2])
    gvec[3, :] *= (1.0-XX**af[2])**af[3]
    info.gvec = gvec

    dgdx = _np.zeros((4, nx), dtype=_np.float64)
    dgdx[0, :] = -af[2]*af[3]*(XX**(af[2]-1.0))*(1.0-XX**af[2])**(af[3]-1.0)
    dgdx[1, :] = -1.0*dgdx[0, :].copy()
    dgdx[2, :] = af[3]*(af[0]-af[1])*XX**(af[2]-1.0)*(1.0-XX**af[2])**af[3]
    dgdx[2, :] *= af[2]*af[3]*XX**af[2]+utlog(XX)+XX**af[2]-af[2]*utlog(XX)-1.0
    dgdx[2, :] /= (XX**af[2] - 1.0)**2.0
    dgdx[3, :] = info.dprofdx / af[3]
    dgdx[3, :] *= 1.0 + af[3]*utlog(-XX**af[2], a1p=True)
#    dgdx[3, :] *= 1.0 + af[3]*_np.log(1.0-XX**af[2])
    info.dgdx = dgdx

    return prof, gvec, info

# ========================================================================== #
# ========================================================================== #
# These two haven't been checked yet!!! also need to add analytic jacobian
# for the derivatives

def model_Heaviside(XX, af=None, npoly=4, rinits=[0.30, 0.35]):
    """
    --- Polynomial with a Heaviside function (Logistics function) ---

    A smooth approximation to the heaviside function is the logistics
    function H(x) = 1/2 + 1/2*tanh(kx),
    k controls sharpness of transition (k large -> very sharp)

    H(x1)-H(x2) ~ 1/2*(tanh(ka5) - tanh(ka6))

    Model - chi ~ a1x^2+a2x+a3+a4*(XX>a5)*(XX<a6)
    af    - estimate of fitting parameters
    XX    - independent variable
    """
#    rinits = [0.00,0.35]
#    rinits = [0.30,0.35] #Miniature-Island, hah!
#    rinits = [0.30,0.50] #Discontinuity
#    rinits = [0.12,0.20] #ITB?, hah!
#    rinits = [0.10,0.35] #ITB?, hah!
#    rinits = [0.16,0.27] #ITB?, hah!
    rinits = _np.array(rinits, dtype=_np.float64)

    if af is None:
        af = _np.hstack(
            (1.0*_np.ones((npoly,), dtype=_np.float64), 2.0, 0.3, 0.4))
#        af *= 0.1*_np.random.normal(0.0, 1.0, len(af))
    # endif
    npoly = _np.size(af)-4
    num_fit = _np.size(af)  # Number of fitting parameters
    nx = _np.size(XX)

    info = Struct()
    info.Lbounds = _np.hstack(
        (-_np.inf*_np.ones((num_fit-3,), dtype=_np.float64), 0, 0, -_np.inf))
    info.Ubounds = _np.hstack(
        (_np.inf*_np.ones((num_fit-3,), dtype=_np.float64), 1, 1, _np.inf))
    info.Lbounds[0] = 0
    info.af = af

    zz = 1e3
    # offset = 0.1;

    #    H(x1)-H(x2) ~ 1/2*(tanh(ka5) - tanh(ka6))
    # f = a1x^2+a2x+a3+a4*(XX>a5)*(XX<a6)
    prof = _np.zeros((nx,), dtype=_np.float64)
    for ii in range(npoly+1):  # ii=1:(num_fit-3)
        kk = npoly + 1 - (ii + 1)
        prof = prof+af[ii]*(XX**kk)
    # endfor
    # prof = prof + af(num_fit)*(XX>af(num_fit-1))*(XX<af(num_fit-2));
    prof = prof + 0.5*af[num_fit-3]*(
                        _np.tanh(zz*(XX-af[num_fit-2]))
                        - _np.tanh(zz*(XX-af[num_fit-1])))

    # d(tanh(x))/dx = 1-tanh(x)^2 = sech(x)^2
    # dfdx  = (a1*2*x^1+a2+0) + 0.5*k*a4*(sech(k*(x-a5))^2 - sech(k*(x-a6))^2)
    info.dprofdx = _np.zeros((nx,), dtype=_np.float64)
    for ii in range(npoly):  # ii = 1:(num_fit-4)
        kk = npoly - (ii+1)
        info.dprofdx = info.dprofdx+af[ii]*kk*(XX**(kk-1))
    # endfor
    info.dprofdx = info.dprofdx + 0.5*af[num_fit-3]*zz*(
                      (sech(zz*(XX-af[num_fit-2]))**2)
                      - (sech(zz*(XX-af[num_fit-1]))**2))

    # dfda1 = x^2
    # dfda2 = x
    # dfda3 = 1
    # dfda4 = a4<XX<a5 = 0.5*(tanh(kk*(x-a5)) - tanh(kk*(x-a5)))
    # dfda5 = -0.5*a4*kk*sech(kk*(x-a5))^2
    # dfda6 = -0.5*a4*kk*sech(kk*(x-a6))^2
    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(npoly+1):  # ii=1:(num_fit-3)
        kk = npoly+1 - (ii+1)
        gvec[ii, :] = (XX**kk)
    # endfor

    gvec[num_fit-3, :] = (0.5*(_np.tanh(zz*(XX-af[num_fit-2]))
                          - _np.tanh(zz*(XX-af[num_fit-1]))))
    gvec[num_fit-2, :] = (-1.0*0.5*af[num_fit-3]*zz
                          * sech(zz*(XX-af[num_fit-2]))**2)
    gvec[num_fit-1, :] = (-1.0*0.5*af[num_fit-3]*zz*(-1
                          * sech(zz*(XX-af[num_fit-1]))**2))

    return prof, gvec, info
# end def model_Heaviside()

# ========================================================================== #
# ========================================================================== #


def model_StepSeries(XX, af=None, npoly=4):
    """
    --- Series of step functions ---

     A smooth approximation to the step function is the hyperbolic
     tangent function S(x) = 1/2 + 1/2*tanh(zx),
     z controls sharpness of transition (z large -> very sharp)

     H(x1)-H(x2) ~ 1/2*(tanh(z*(x-x1)) - tanh(z*(x-x2)))

    Model - chi ~ sum_k(1/2 + 1/2*tanh(z*(x-ak)))
    af    - estimate of fitting parameters
    XX    - independent variable
    """

    if af is None:
        af = _np.hstack((5.0, 1.0*_np.random.randn(npoly,)))
#        af *= 0.1*_np.random.normal(0.0, 1.0, len(af))
    # endif
    npoly = _np.size(af)-1
    num_fit = _np.size(af)  # Number of fitting parameters
    nx = _np.size(XX)
    zz = 1e3 # 50

    info = Struct()
    info.Lbounds = _np.hstack(
        (0, -_np.inf*_np.ones((num_fit-1,), dtype=_np.float64)))
    info.Ubounds = _np.hstack(
        (10, _np.inf*_np.ones((num_fit-1,), dtype=_np.float64)))
    info.af = af

    # The central step and the derivative of the transition
    prof = _np.ones((nx,), dtype=_np.float64)
    info.dprofdx = _np.zeros((nx,), dtype=_np.float64)
    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)

    gvec[0, :] = prof.copy()
    prof = af[0]*prof

    # ba = 1
    for ii in range(1, num_fit):  # ii=1:(num_fit-1)
        bb = ii/(num_fit-1)

        # Analytic jacobian
        # f = a1*(0.5 + 0.5*tanh(zz*(x-b^a1)))
        #             + a2*(0.5 + 0.5*tanh(zz*(x-b^a2))) + ...
        #             + an*(0.5 + 0.5*tanh(zz*(x-b^an)))
        #   = 0.5*sum_k(ak) + 0.5*sum_k(ak*tanh(zz*(x-b^ak))
        # dfdx = 0.5*zz*sum_k(ak*sech(zz*(x-b^ak))^2)
        # dfdak = 0.5*(1 + tanh(zz*(x-b^ak))
        #                - 0.5*zz*ln(b)*b^ak*sech(zz*(x-b^ak))^2)

        # f    = a1*tanh(zz(x-x1))+a2*tanh(zz(x-x2))+...an*tanh(zz(x-xn))
        temp = _np.tanh(zz*(XX-bb**af[ii]))
        prof = prof + 0.5*af[ii]*(1 + temp)

        info.dprofdx = info.dprofdx+0.5*af[ii]*zz*(1 - temp**2)
        # info.dprofdx = info.dprofdx+0.5*af[ii]*zz*sech(zz*(XX-bb**af[ii]))**2

        gvec[ii, :] = (0.5*(1 + temp)
                       - 0.5*zz*utlog(bb)*(bb**af[ii])*(1 - temp**2))

#        #indice of transitions
#        bx = _np.floor(1+bb/(XX[2]-XX[1]))
#        gvec[num_fit-1,ba-1:bx-1] = (zz*utlog(bb)*(-bb)**af[num_fit-1]
#                            * sech(zz*(XX[ba-1:bx-1]-bb**af[num_fit-1]))**2
#        ba = _np.floor(1+bb/(XX(2)-XX(1)))
    # endfor

    return prof, gvec, info
# end def model_StepSeries()


# ========================================================================== #
# ========================================================================== #


def model_profile(af=None, XX=None, model_number=7, npoly=4, nargout=1, verbose=False):
    """
    function [prof, gvec, info] = model_chieff(af,XX ,model_number,npoly)

     af - estimate of fitting parameters
     XX - independent variable
     model_number:

       1 - Product of exponentials   - f(x) ~ prod(af(ii)*XX^(polyorder-ii))
       2 - Straight Polynomial       - f(x) ~ sum( af(ii)*XX^(polyorder-ii))
       3 - Power law fit             - f(x) ~ a(n+1)*x^(a1*x^n+a2*x^(n-1)+...an)
       4 - Exponential on Background - f(x) ~ a1*(exp(a2*xx^a3) + XX^a4)
       5 - Polynomial with a Heaviside (Logistics)
                                        - f(x) ~ a1x^2+a2x+a3+a4*(XX>a5)*(XX<a6)
       6 - Series of Step functions  - H(x1)-H(x2) ~ 1/2*(tanh(ka5)-tanh(ka6))
       7 - Quasi-parabolic fit       - f(x) / af[0] ~ af[1]-af[4]+(1-af[1]+af[4])*(1-xx^af[2])^af[3]
                                                + af[4]*(1-exp(-xx^2/af[5]^2))
       8 - Even order polynomial     - f(x) ~ sum( af(ii)*XX^2*(polyorder-ii))
       9 - 2-power profile           - f(x) ~ (Core-Edge)*(1-x^pow1)^pow2 + Edge
       10 - Parabolic fit            - f(x) ~ a*(1.0-x^2)
       11 - Flat top profile         - f(x) ~ a / (1 + (x/b)^c)
       12 - Massberg profile         - f(x) ~ a * (1-h*(x/b)) / (1+(x/b)^c) = flattop*(1-h*(x/b))
    """
    if af is not None:
        if (af==0).any():
            af[_np.where(af==0)[0]] = 1e-14
        # end if
    # end if
    if XX is None:
        XX = _np.linspace(1e-4, 1, 200)
    # endif
    if len([XX])>1 and (XX==0).any():
        XX[_np.where(XX==0)[0]] = 1e-14
    # end if

    # ====================================================================== #

    if model_number == 1:
        if verbose: print('Modeling with an order %i product of Exponentials'%(npoly,))  # endif
        [prof, gvec, info] = model_ProdExp(XX, af, npoly)
        info.func = model_ProdExp

    elif model_number == 2:
        if verbose: print('Modeling with an order %i polynomial'%(npoly,))  # endif
        [prof, gvec, info] = model_poly(XX, af, npoly)
        info.func = model_poly

    elif model_number == 3:
        if verbose: print('Modeling with an order %i power law'%(npoly,))  # endif
        [prof, gvec, info] = model_PowerLaw(XX, af, npoly)
        info.func = model_PowerLaw

    elif model_number == 4:
        if verbose: print('Modeling with an exponential on order %i polynomial background'%(npoly,))  # endif
        [prof, gvec, info] = model_Exponential(XX, af, npoly)
        info.func = model_Exponential

    elif model_number == 5:
        if verbose: print('Modeling with an order %i polynomial+Heaviside fn'%(npoly,))  # endif
        [prof, gvec, info] = model_Heaviside(XX, af, npoly)
        info.func = model_Heaviside

    elif model_number == 6:
        if verbose: print('Modeling with a %i step profile'%(npoly,))  # endif
        [prof, gvec, info] = model_StepSeries(XX, af, npoly)
        info.func = model_StepSeries

    elif model_number == 7:
        if verbose: print('Modeling with a quasiparabolic profile')  # endif
        [prof, gvec, info] = model_qparab(XX, af)
        info.func = model_qparab

    elif model_number == 8:
        if verbose: print('Modeling with an order %i even polynomial'%(npoly,))  # endif
        [prof, gvec, info] = model_evenpoly(XX, af, npoly)
        info.func = model_evenpoly

    elif model_number == 9:  # Two power fit
        if verbose: print('Modeling with a 2-power profile')  # endif
        [prof, gvec, info] = model_2power(XX, af)
        info.func = model_2power

    elif model_number == 10:
        if verbose: print('Modeling with a parabolic profile')  # endif
        [prof, gvec, info] = model_parabolic(XX, af)
        info.func = model_parabolic

    elif model_number == 11:
        if verbose: print('Modeling with a flat-top profile')  # endif
        [prof, gvec, info] = model_flattop(XX, af)
        info.func = model_flattop

    elif model_number == 12:
        if verbose: print('Modeling with a Massberg-style profile')  # endif
        [prof, gvec, info] = model_massberg(XX, af)
        info.func = model_massberg

    # end switch-case

    if nargout == 3:
        return prof, gvec, info
    elif nargout == 2:
        return prof, gvec
    elif nargout == 1:
        return prof
# end def model_profile()

# ========================================================================== #


def model_chieff(af=None, XX=None, model_number=1, npoly=4, nargout=1, verbose=False):
    """
    function [chi_eff, gvec, info] = model_chieff(af,XX ,model_number,npoly)

     af - estimate of fitting parameters
     XX - independent variable
     model_number:
       1 - Product of exponentials   - chi ~ prod(af(ii)*XX^(polyorder-ii))
       2 - Straight Polynomial       - chi ~ sum( af(ii)*XX^(polyorder-ii))
       3 - Power law fit             - chi ~ a(n+1)*x^(a1*x^n+a2*x^(n-1)+...an)
       4 - Exponential on Background - chi ~ a1*(exp(a2*xx^a3) + XX^a4)
       5 - Polynomial with a Heaviside (Logistics)
                                        - chi ~ a1x^2+a2x+a3+a4*(XX>a5)*(XX<a6)
       6 - Series of Step functions  - H(x1)-H(x2) ~ 1/2*(tanh(ka5)-tanh(ka6))
       7 - Quasi-parabolic fit       - 10.0 - qparab
       8 - Even order polynomial     - chi ~ sum( af(ii)*XX^2*(polyorder-ii))
       9 - 2-power profile           - chi ~ (Core-Edge)*(1-x^pow1)^pow2 + Edge
    """
    if af is not None:
        if (af==0).any():
            af[_np.where(af==0)[0]] = 1e-14
        # end if
    # end if

    if XX is None:
        XX = _np.linspace(1e-4, 1, 200)
    # endif
    if len([XX])>1 and (XX==0).any():
        XX[_np.where(XX==0)[0]] = 1e-14
    # end if

    # ====================================================================== #

    if model_number == 1:
        if verbose: print('Modeling with an order %i product of Exponentials'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_ProdExp(XX, af, npoly)
        info.func = model_ProdExp

    elif model_number == 2:
        if verbose: print('Modeling with an order %i polynomial'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_poly(XX, af, npoly)
        info.func = model_poly

    elif model_number == 3:
        if verbose: print('Modeling with an order %i power law'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_PowerLaw(XX, af, npoly)
        info.func = model_PowerLaw

    elif model_number == 4:
        if verbose: print('Modeling with an exponential on order %i polynomial background'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_Exponential(XX, af, npoly)
        info.func = model_Exponential

    elif model_number == 5:
        if verbose: print('Modeling with an order %i polynomial+Heaviside fn'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_Heaviside(XX, af, npoly)
        info.func = model_Heaviside

    elif model_number == 6:
        if verbose: print('Modeling with a %i step profile'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_StepSeries(XX, af, npoly)
        info.func = model_StepSeries

    elif model_number == 7:
        if verbose: print('Modeling with the derivative of a quasiparabolic profile')  # endif
        def tfunc(XX, af, npoly=None):
            _, _, info = model_qparab(XX, af)

            info.prof = deriv_qparab(XX, info.af)
            info.gvec = partial_deriv_qparab(XX, info.af)
            info.dprofdx = deriv2_qparab(XX, info.af)
            info.dgdx = partial_deriv2_qparab(XX, info.af)

            info.prof = -1.0*info.prof
            info.dprofdx = -1.0*info.dprofdx
            info.gvec = -1.0*info.gvec
            info.dgdx = -1.0*info.dgdx
            return info.prof, info.gvec, info

        def tfunc_plus(XX, af, npoly=None):
            if af is None:
                _, _, info = tfunc(XX, af, npoly)
                af = _np.asarray(info.af.tolist()+[3.0], dtype=_np.float64)
            else:
                _, _, info = tfunc(XX, af[:-1], npoly)
            # end if
            info.af = _np.copy(af)
            info.prof += af[-1].copy()
            info.gvec = _np.insert(info.gvec, [-1], _np.ones(_np.shape(info.gvec[0,:]), dtype=_np.float64), axis=0)
            info.dgdx = _np.insert(info.dgdx, [-1], _np.zeros(_np.shape(info.dgdx[0,:]), dtype=_np.float64), axis=0)
            info.Lbounds = _np.asarray(info.Lbounds.tolist()+[-20.0], dtype=_np.float64)
            info.Ubounds = _np.asarray(info.Ubounds.tolist()+[ 20.0], dtype=_np.float64)
            return info.prof, info.gvec, info
        [chi_eff, gvec, info] = tfunc(XX, af)
        info.func = tfunc

    elif model_number == 8:
        if verbose: print('Modeling with an order %i even polynomial'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_evenpoly(XX, af, npoly)
        info.func = model_evenpoly

    elif model_number == 9:
        if verbose: print('Modeling with a 2-power profile')  # endif
        def tfunc(XX, af, npoly=None):
            return model_2power(XX, af)
        [chi_eff, gvec, info] = tfunc(XX, af)
        info.func = tfunc

    # end switch-case

    if nargout == 3:
        info.dchidx = info.dprofdx
        del info.dprofdx
        return chi_eff, gvec, info
    elif nargout == 2:
        return chi_eff, gvec
    elif nargout == 1:
        return chi_eff
# end def model_chieff()

# ========================================================================== #
# ========================================================================== #


def normalize_test_prof(xvar, dPdx, dVdrho):
    dPdx = dPdx/_np.trapz(dPdx, x=xvar)     # Test profile shape : dPdroa

    dPdx = _np.atleast_2d(dPdx).T
    dVdrho = _np.atleast_2d(dVdrho).T

    # Test power density profile : dPdVol
    dPdV = dPdx/dVdrho
    dPdV[_np.where(_np.isnan(dPdV))] = 0.0
    return dPdV
# end def normalize_test_prof

def get_test_Prad(xvar, exppow=14.0, dVdrho=None):
    if dVdrho is None:        dVdrho = _np.ones_like(xvar)    # endif
    dPdx = _np.exp(xvar**exppow)-1.0       # Radiation profile shape [MW/rho]

    # Radiation emissivity profile in MW/m3
    dPdV = normalize_test_prof(xvar, dPdx, dVdrho)
    return dPdV

def get_test_Pdep(xvar, rloc=0.1, rhalfwidth=0.05, dVdrho=None, A0=None):
    sh = _np.shape(xvar)
    if dVdrho is None:        dVdrho = _np.ones_like(xvar)    # endif
    # Normalized gaussian for the test power deposition shape : dPdroa
    Amp = 2*_np.pi *rhalfwidth * rhalfwidth
    dPdx = gaussian(xvar, 1.0/Amp, rloc, rhalfwidth/_np.sqrt(2))
    dPdx = _np.sqrt(dPdx)

#    dPdx = (_np.exp(-0.5*((xvar-rloc)/rhalfwidth)**2)
#            / (rhalfwidth*_np.sqrt(2*_np.pi)))
    # Test ECRH power density in MW/m3
    dPdV = normalize_test_prof(xvar, dPdx, dVdrho)
    dPdV = dPdV.reshape(sh)
    if A0 is not None:
        dPdV *= A0
    # end if
    return dPdV
# end def

# ========================================================================== #
# ========================================================================== #

#def sech(x):
#    """
#    sech(x)
#    Uses numpy's cosh(x).
#    """
#    return 1.0/_np.cosh(x)

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

# ========================================================================== #
# ========================================================================== #


if __name__ == '__main__':

    XX = _np.linspace(0, 1, 200)
    npoly = 5
    model_number = 6
#    af = [0.1,0.1,0.1]
    af = None
#    af = [2.0,0.1,0.1,0.5]

    [chi_eff, gvec, info] = \
        model_chieff(af=af, XX=XX, model_number=model_number, npoly=npoly, nargout=3, verbose=True)

#    [chi_eff, gvec, info] = \
#        model_profile(af=af, XX=XX, model_number=model_number, npoly=npoly, nargout=3, verbose=True)
#    info.dchidx = info.dprofdx

    varaf = (0.1*info.af)**2
    varchi = _np.zeros_like(chi_eff)
    for ii in range(_np.size(gvec, axis=0)):
        varchi = varchi + varaf[ii]*(gvec[ii, :]**2)
    # endfor

    _plt.figure()
    _plt.plot(XX, chi_eff, 'k-')
    _plt.plot(XX, chi_eff+_np.sqrt(varchi), 'k--')
    _plt.plot(XX, chi_eff-_np.sqrt(varchi), 'k--')

    _plt.figure()
    _plt.plot(XX, info.dchidx, '-')

    _plt.figure()
    for ii in range(_np.size(gvec, axis=0)):
        _plt.plot(XX, gvec[ii, :], '-')
    # endfor
#    _plt.plot(XX, gvec[1, :], '-')
#    _plt.plot(XX, gvec[2, :], '-')

# endif

# ========================================================================== #
# ========================================================================== #






