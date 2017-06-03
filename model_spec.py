# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:59:28 2016

@author: gawe
"""
# ========================================================================== #    

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

# ========================================================================== #    
# ========================================================================== #    

import numpy as _np
import matplotlib.pyplot as _plt
#import pybaseutils as _pyut
from ..Struct import Struct

# =========================================================================== #
# =========================================================================== #


def model_qparab(XX, af=None):
    """
    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
    This function calculates the quasi-parabolic fit
    Y/Y0 = af[1]-af[4]+(1-af[1]+af[4])*(1-xx^af[2])^af[3]
                + af[4]*(1-exp(-xx^2/af[5]^2))
    xx - r/a
    af[0] - Y0 - function value on-axis
    af[1] - gg - Y1/Y0 - function value at edge over core
    af[2],af[3]-  pp, qq - power scaling parameters
    af[4],af[5]-  hh, ww - hole depth and width

    """

    if af is None:
        af = _np.array([5.0, 0.002, 2.0, 0.7, -0.24, 0.30], dtype=_np.float64)
    # endif

    info = Struct()
    info.Lbounds = _np.array([0.0, 0.0, -_np.inf, -_np.inf,
                              -_np.inf, -_np.inf], dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones_like(af)
    info.af = af

    XX = _np.abs(XX)
    af = af.reshape((6,))
    if _np.isfinite(af).any() == 0:
        print("checkit!")
#    print(_np.shape(af))
    prof = af[0]*(af[1]-af[4]
                  + (1.0-af[1]+af[4])*_np.abs(1.0-XX**af[2])**af[3]
                  + af[4]*(1.0-_np.exp(-XX**2.0/af[5]**2.0)))

    gvec = _np.zeros((6, _np.size(XX)), dtype=_np.float64)
    gvec[0, :] = (af[1]-af[4]
                  + (1.0-af[1]+af[4])*_np.abs(1.0-XX**af[2])**af[3]
                  + af[4]*(1.0-_np.exp(-XX**2.0/af[5]**2.0)))
    gvec[1, :] = af[0]
    gvec[2, :] = (af[0]*(1.0-af[1]+af[4])*(-1.0*_np.log(XX)*XX**af[2])
                  * af[2]*_np.abs(1.0-XX**af[2])**(af[3]-1.0))
    gvec[3, :] = (af[0]*(1.0-af[1]+af[4])*_np.log(1.0-XX**af[2])
                  * _np.abs(1.0-XX**af[2])**af[3])
    gvec[4, :] = (af[0]*(-1.0 + _np.abs(1.0-XX**af[2])**af[3]
                  + (1.0-_np.exp(-XX**2.0/af[5]**2.0))))
    gvec[5, :] = (af[0]*af[4]*(-1.0*_np.exp(-XX**2.0/af[5]**2.0))
                  * (2.0*XX**2.0/af[5]**3))

    info.dprofdx = (af[0]*((1.0-af[1]+af[4])*(-1.0*af[2]*XX**(af[2]-1.0))
                    * af[3]*(1.0-XX**af[2])**(af[3]-1.0)
                    - af[4]*(-2.0*XX/af[5]**2.0)*_np.exp(-XX**2.0/af[5]**2.0)))

#    
#    info.gdx = _np.zeros_like(info.dprofdx)
#    info.gdx[0, :] = info.dprofdx / af[0]
#    info.gdx[1, :] = (af[0]*((1.0-af[1]+af[4])*(-1.0*af[2]*XX**(af[2]-1.0))
#                    * af[3]*(1.0-XX**af[2])**(af[3]-1.0)
#                    - af[4]*(-2.0*XX/af[5]**2.0)*_np.exp(-XX**2.0/af[5]**2.0)))
#                    

    return prof, gvec, info
# end def model_qparab

# =========================================================================== #
# =========================================================================== #


def model_ProdExp(XX, af=None, npoly=4):
    """
    --- Product of Exponentials ---
    Model - chi ~ prod(af(ii)*XX^(polyorder-ii))
    af    - estimate of fitting parameters
    XX    - independent variable
    """
    if af is None:
        af = 0.1*_np.ones((npoly+1,), dtype=_np.float64)
    # endif
    npoly = _np.size(af)-1

    info = Struct()
    info.Lbounds = _np.array([], dtype=_np.float64)
    info.Ubounds = _np.array([], dtype=_np.float64)
    info.af = af

    nx = _np.size(XX)
    num_fit = _np.size(af)  # Number of fitting parameters

    # Polynomial of order num_fit
    pp = _np.poly1d(af)
    chi_eff = pp(XX)

    # Could be just an exponential fit
    # chi_eff=af[-1]*_np.exp(chi_eff)
    chi_eff = _np.exp(chi_eff)

    ad = pp.deriv()
    info.dchidx = ad(XX)

#     info.dchidx  = zeros(1, nx);
#     for ii=1:num_fit-1
#         #The derivative of chi with respect to rho is analytic as well:
#         # f = exp(a1x^n+a2x^(n-1)+...a(n+1))
#         # f = exp(a1x^n)exp(a2x^(n-1))...exp(a(n+1)));
#         # dfdx = (n*a1*x^(n-1)+(n-1)*a2*x^(n-2)+...a(n))*f
#         kk = num_fit - ii;
#         info.dchidx = info.dchidx+kk*af(ii)*XX**(kk-1);
#     end
    info.dchidx = chi_eff*info.dchidx

    #
    # The g-vector contains the partial derivatives used for error propagation
    # f = exp(a1*x^2+a2*x+a3)
    # dfda1 = x^2*f;
    # dfda2 = x  *f;
    # dfda3 = f;
    # gvec(1,1:nx) = XX**2.*chi_eff;
    # gvec(2,1:nx) = XX   .*chi_eff;
    # gvec(3,1:nx) =        chi_eff;
    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(num_fit):  # 1:num_fit
        # Formulated this way, there is an analytic jacobian:
        kk = num_fit - (ii + 1)
        gvec[ii, :] = (XX**kk)*chi_eff
    # endif

    return chi_eff, gvec, info
# end def model_ProdExp()

# =========================================================================== #
# =========================================================================== #


def model_poly(XX, af=None, npoly=4):
    """
    --- Straight Polynomial ---
    Model - chi ~ sum( af(ii)*XX^(polyorder-ii))
    af    - estimate of fitting parameters
    XX    - independent variable
    """

    if af is None:
        af = 1*_np.ones((npoly+1,), dtype=_np.float64)
    # endif
    npoly = _np.size(af)-1

    info = Struct()
    info.Lbounds = -_np.inf*_np.ones((npoly+1,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((npoly+1,), dtype=_np.float64)
    info.af = af

    num_fit = _np.size(af)  # Number of fitting parameters
    nx = _np.size(XX)

    # Polynomial of order num_fit
    pp = _np.poly1d(af)
    chi_eff = pp(XX)

    ad = pp.deriv()
    info.dchidx = ad(XX)

    # The g-vector contains the partial derivatives used for error propagation
    # f = a1*x^2+a2*x+a3
    # dfda1 = x^2;
    # dfda2 = x;
    # dfda3 = 1;
    # gvec(1,1:nx) = XX**2;
    # gvec(2,1:nx) = XX   ;
    # gvec(3,1:nx) = 1;

    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(num_fit):  # ii=1:num_fit
        kk = num_fit - (ii + 1)
        gvec[ii, :] = XX**kk
    # endfor

    return chi_eff, gvec, info
# end def model_poly()

# =========================================================================== #
# =========================================================================== #


def model_evenpoly(XX, af=None, npoly=4):
    """
    --- Polynomial with only even powers ---
    Model - chi ~ sum( af(ii)*XX^2*(polyorder-ii))
    af    - estimate of fitting parameters
    XX    - independent variable
    """
    if af is None:
        af = 1*_np.ones((npoly/2+1,), dtype=_np.float64)
    # endif
    npoly = _np.int(2*(_np.size(af)-1))  # Polynomial order from input af

    info = Struct()
    info.Lbounds = -_np.inf*_np.ones((npoly/2+1,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((npoly/2+1,), dtype=_np.float64)
    info.af = af

    num_fit = _np.size(af)  # Number of fitting parameters
    nx = _np.size(XX)

    # Even Polynomial of order num_fit, Insert zeros for the odd powers
    a0 = _np.insert(af, _np.linspace(1, num_fit-1, 2), 0.0)
    pp = _np.poly1d(a0)
    chi_eff = pp(XX)

    ad = pp.deriv()
    info.dchidx = ad(XX)

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
        kk = num_fit - (ii + 1)
        gvec[ii, :] = XX**(2*kk)
    # endfor

    return chi_eff, gvec, info
# end def model_evenpoly()

# =========================================================================== #
# =========================================================================== #


def model_PowerLaw(XX, af=None, npoly=4):
    """
    --- Power Law w/exponential cut-off ---
    Model - fc = x^(a1*x^(n+1)+a2*x^n+...a(n+1))
            chi ~ a(n+2)*fc*exp(a(n+1)*x)
            # chi ~ a(n+1)*x^(a1*x^n+a2*x^(n-1)+...an)
    af    - estimate of fitting parameters
    XX    - independent variable
    """
    if af is None:
        af = 1*_np.ones((npoly+2,), dtype=_np.float64)
    # endif
    num_fit = _np.size(af)  # Number of fitting parameters
    npoly = num_fit-2
    nx = _np.size(XX)

    info = Struct()
    info.Lbounds = _np.hstack(
        (-_np.inf * _np.ones((npoly,), dtype=_np.float64), -_np.inf, 0))
    info.Ubounds = _np.hstack(
        (_np.inf * _np.ones((npoly,), dtype=_np.float64), _np.inf, _np.inf))
    info.af = af

    # Curved power-law:
    # fc = x^(a1*x^(n+1)+a2*x^n+...a(n+1))
    # With exponential cut-off:
    # f  = a(n+2)*fc(x)*exp(a(n+1)*XX);
    pp = _np.poly1d(af[0:npoly-1])
    polys = pp(XX)
    exp_factor = _np.exp(af[num_fit-2]*XX)
    chi_eff = af[num_fit-1]*(XX**polys)*exp_factor

    # dfdx = dfcdx*(an*e^an1x) +an1*f(x);
    # dfcdx = XX^(-1)*chi_eff*(polys+ddx(poly)*XX*log(XX),
    # log is natural logarithm
    dpolys = _np.poly1d(af[0:npoly-1])
    dpolys = dpolys.deriv()
    dpolys = dpolys(XX)
    #  dpolys = zeros(1, nx);
    #  for ii=1:npoly-2
    #      #The derivative of chi with respect to rho is analytic as well:
    #      kk = npoly - 2 - ii;
    #      dpolys = dpolys+kk*af(ii)*XX**kk;
    #  # end for
    info.dchidx = ((chi_eff/XX)*(polys + dpolys*XX*_np.log(XX))
                   + chi_eff*af[num_fit-2])

    # The g-vector contains the partial derivatives used for error propagation
    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(npoly-1):  # ii=1:(npoly-1)
        kk = npoly - 1 - (ii + 1)
        gvec[ii, :] = chi_eff*_np.log(XX)*XX**kk
    # endfor
    gvec[num_fit-2, :] = XX*chi_eff
    gvec[num_fit-1, :] = chi_eff/af[num_fit-1]

    return chi_eff, gvec, info
# end def model_PowerLaw()

# =========================================================================== #
# =========================================================================== #


def model_Exponential(XX, af=None, npoly=4):
    """
    --- Exponential on Background ---
    Model - chi ~ a1*(exp(a2*xx^a3) + XX^a4)
    af    - estimate of fitting parameters
    XX    - independent variable
    """

#    num_fit = npoly+3
    num_fit = 4
    if af is None:
        af = 1*_np.ones((num_fit,), dtype=_np.float64)
    # endif
    num_fit = _np.size(af)  # Number of fitting parameters
    nx = _np.size(XX)

    info = Struct()
    info.Lbounds = -_np.inf*_np.ones((num_fit,), dtype=_np.float64)
    info.Ubounds = _np.inf*_np.ones((num_fit,), dtype=_np.float64)
    info.Lbounds[0] = 0
    info.af = af

    # f     = a1*(exp(a2*xx^a3) + XX^a4) = f1+f2;
    # dfdx  = a1*(a2*a3*xx^(a3-1)*exp(a2*xx^a3) + a4*xx^(a4-1));
    # dfda1 = f/a1;
    # dfda2 = xx^a3*f1;
    # dfda3 = f1*xx^a3*log10(xx)
    # dfda4 = a1*xx^a4*log10(xx) = log10(xx)*f2;
    chi_eff1 = _np.exp(af[1]*XX**af[2])

    #
    # chi_eff2 = zeros(1, nx);
    # for ii = 1:(num_fit-3)
    #     chi_eff2 = chi_eff2+XX**ii;
    # end
    chi_eff = af[0]*(chi_eff1+XX**af[3])
    info.dchidx = (af[0]*(af[1]*af[2]*XX**(af[2]-1)*chi_eff1
                   + af[3]*XX**(af[3]-1)))

    gvec = _np.zeros((4, nx), dtype=_np.float64)
    gvec[0, :] = chi_eff/af[0]
    gvec[1, :] = af[0]*chi_eff1*XX**af[2]
    gvec[2, :] = af[0]*chi_eff1*_np.log10(XX)*XX**af[2]
    gvec[3, :] = af[0]*_np.log10(XX)*XX**af[3]

    return chi_eff, gvec, info
# end def model_Exponential()

# =========================================================================== #
# =========================================================================== #


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
    chi_eff = _np.zeros((nx,), dtype=_np.float64)
    for ii in range(npoly+1):  # ii=1:(num_fit-3)
        kk = npoly + 1 - (ii + 1)
        chi_eff = chi_eff+af[ii]*(XX**kk)
    # endfor
    # chi_eff = chi_eff + af(num_fit)*(XX>af(num_fit-1))*(XX<af(num_fit-2));
    chi_eff = chi_eff + 0.5*af[num_fit-3]*(
                        _np.tanh(zz*(XX-af[num_fit-2]))
                        - _np.tanh(zz*(XX-af[num_fit-1])))

    # d(tanh(x))/dx = 1-tanh(x)^2 = sech(x)^2
    # dfdx  = (a1*2*x^1+a2+0) + 0.5*k*a4*(sech(k*(x-a5))^2 - sech(k*(x-a6))^2)
    info.dchidx = _np.zeros((nx,), dtype=_np.float64)
    for ii in range(npoly):  # ii = 1:(num_fit-4)
        kk = npoly - (ii+1)
        info.dchidx = info.dchidx+af[ii]*kk*(XX**(kk-1))
    # endfor
    info.dchidx = info.dchidx + 0.5*af[num_fit-3]*zz*(
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

    return chi_eff, gvec, info
# end def model_Heaviside()

# =========================================================================== #
# =========================================================================== #


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
#        af = 5.0*_np.ones((npoly+1,), dtype=_np.float64)
    # endif
    npoly = _np.size(af)-1
    num_fit = _np.size(af)  # Number of fitting parameters
    nx = _np.size(XX)
    zz = 50

    info = Struct()
    info.Lbounds = _np.hstack(
        (0, -_np.inf*_np.ones((num_fit-1,), dtype=_np.float64)))
    info.Ubounds = _np.hstack(
        (10, _np.inf*_np.ones((num_fit-1,), dtype=_np.float64)))
    info.af = af

    # The central step and the derivative of the transition
    chi_eff = _np.ones((nx,), dtype=_np.float64)
    info.dchidx = _np.zeros((nx,), dtype=_np.float64)
    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)

    gvec[0, :] = chi_eff.copy()
    chi_eff = af[0]*chi_eff

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
        chi_eff = chi_eff + 0.5*af[ii]*(1 + temp)

        info.dchidx = info.dchidx+0.5*af[ii]*zz*(1 - temp**2)
        # info.dchidx = info.dchidx+0.5*af[ii]*zz*sech(zz*(XX-bb**af[ii]))**2

        gvec[ii, :] = (0.5*(1 + temp)
                       - 0.5*zz*_np.log(bb)*(bb**af[ii])*(1 - temp**2))

#        #indice of transitions
#        bx = _np.floor(1+bb/(XX[2]-XX[1]))
#        gvec[num_fit-1,ba-1:bx-1] = (zz*_np.log(bb)*(-bb)**af[num_fit-1]
#                            * sech(zz*(XX[ba-1:bx-1]-bb**af[num_fit-1]))**2
#        ba = _np.floor(1+bb/(XX(2)-XX(1)))
    # endfor

    return chi_eff, gvec, info
# end def model_StepSeries()

# =========================================================================== #
# =========================================================================== #


def model_parabolic(XX, af):
    """
    A parabolic profile with one free parameters:
        f(x) ~ a*(1.0-x^2)
        xx - x - independent variable
        af - a - central value of the plasma parameter         
    """

    if af is None:
        af = _np.array([1.0], dtype=_np.float64)
    # endif

    info = Struct()
    info.Lbounds = _np.array([0.0], dtype=_np.float64)
    info.Ubounds = _np.array([_np.inf], dtype=_np.float64)
    info.af = af

    prof = af*(1.0 - XX**2.0)
    gvec = _np.atleast_2d(prof / af)
    info.dprofdx = -2.0*af*XX

    return prof, gvec, info

# =========================================================================== #
# =========================================================================== #


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
    # endif

    nx = len(XX)
    XX = _np.abs(XX)

    info = Struct()
    info.Lbounds = _np.array([0.0, 0.0, 1.0], dtype=_np.float64)
    info.Ubounds = _np.array([_np.inf, 1.0, _np.inf], dtype=_np.float64)
    info.af = af

    temp = (XX/af[1])**af[2]
    prof = af[0] / (1.0 + temp)

    gvec = _np.zeros((3, nx), dtype=_np.float64)
    gvec[0, :] = prof / af[0]
    gvec[1, :] = af[0]*af[2]*temp / (af[1]*(1.0+temp)**2.0)
    gvec[2, :] = af[0]*temp*_np.log(XX/af[1]) / (1.0+temp)**2.0

    info.dprofdx = -1.0*af[0]*af[2]*temp/(XX*(1.0+temp)**2.0)

    return prof, gvec, info

# =========================================================================== #
# =========================================================================== #


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
    # endif

    nx = len(XX)
    XX = _np.abs(XX)

    info = Struct()
    info.Lbounds = _np.array([0.0, 0.0, 1.0, -_np.inf], dtype=_np.float64)
    info.Ubounds = _np.array(
        [_np.inf, 1.0, _np.inf, _np.inf], dtype=_np.float64)
    info.af = af

    temp = XX/af[1]
    temp1 = temp**af[2]
    tempd = af[1]*(1.0 + temp1)
#    temp2
    prof = af[0] * (1 - af[3]*(XX/af[2])) / (1.0 + temp1)

    gvec = _np.zeros((4, nx), dtype=_np.float64)
    gvec[0, :] = prof / af[0]
    gvec[1, :] = af[0]*XX*(
                        af[2]*temp**(af[2]-1.0)*(1-af[3]*temp)/tempd
                        + af[3]/af[1]) / tempd
    gvec[2, :] = (-1.0*af[0]*temp1*(af[1] - af[3]*XX)
                  * _np.log(temp) / (tempd*(1.0 + temp1)))
    gvec[3, :] = -1.0*af[0]*XX / tempd

    info.dprofdx = (-1.0*af[0]/tempd)*(
                    af[2]*temp**(af[2] - 1.0)*(1.0 - af[3]*temp)/(1.0 + temp1)
                    + af[3])

    return prof, gvec, info

# =========================================================================== #


def model_2power(XX, af):
    """
    A two power profile fit with four free parameters:
    prof ~ f(x) = (Core-Edge)*(1-x^pow1)^pow2 + Edge
        af[0] - Core - central value of the plasma parameter
        af[1] - Edge - edge value of the plasma parameter
        af[2] - pow1 - first power
        af[3] - pow2 - second power
    """
    if af is None:
        af = _np.array([1.0, 0.0, 2.0, 1.0], dtype=_np.float64)
    # endif

    nx = len(XX)
    XX = _np.abs(XX)

    info = Struct()
    info.Lbounds = _np.array([0.0, 0.0, -_np.inf, -_np.inf], dtype=_np.float64)
    info.Ubounds = _np.array(
                    [_np.inf, _np.inf, _np.inf, _np.inf], dtype=_np.float64)
    info.af = af

    # f(x) = (Core-Edge)*(1-x^pow1)^pow2 + Edge
    prof = (af[0]-af[1])*(1.0-XX**af[2])**af[3] + af[1]
    #
    #               d(b^x) = b^x *ln(b)
    # dfdx = -(a0-a1)*a2*a3*x^(a2-1)*(1-x^a2)^(a3-1)
    # dfda0 = (1-x^a2)^a3
    # dfda1 = 1-(1-x^a2)^a3 = 1-dfda0
    # dfda2 = -(a0-a1)*(ln(x)*x^a2)*a3*(1-x^a2)^(a3-1)
    # dfda3 = (a0-a1)*(1-x^a2)^a3*ln(1-x^a2)
    #
    gvec = _np.zeros((4, nx), dtype=_np.float64)
    gvec[0, :] = (prof-af[1])/(af[0]-af[1])
    gvec[1, :] = 1.0-gvec[0, :].copy()
    gvec[2, :] = (-(af[0]-af[1])*(_np.log(XX)*XX**af[2])
                  * af[3]*(1.0-XX**af[2])**(af[3]-1.0))
    gvec[3, :] = ((af[0]-af[1])*_np.log(1.0-XX**af[2])
                  * (1.0-XX**af[2])**af[3])

    info.dprofdx = ((af[0]-af[1])
                    * -1.0*af[2]*XX**(af[2]-1.0)
                    * af[3]*(1.0-XX**af[2])**(af[3]-1.0))

    return prof, gvec, info

# =========================================================================== #
# =========================================================================== #


def model_profile(af=None, XX=None, model_number=2, npoly=4, nargout=1, verbose=True):
    """
    function [chi_eff, gvec, info] = model_chieff(af,XX ,model_number,npoly)

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

    if XX is None:
        XX = _np.linspace(0, 1, 200)
    # endif

    # ====================================================================== #
    # ====================================================================== #

    if model_number == 1:
        if verbose: print('Modeling with an order %i product of Exponentials'%(npoly,))  # endif
        [prof, gvec, info] = model_ProdExp(XX, af, npoly)
        info.dprofdx = info.dchidx        

    elif model_number == 2:
        if verbose: print('Modeling with an order %i polynomial'%(npoly,))  # endif        
        [prof, gvec, info] = model_poly(XX, af, npoly)
        info.dprofdx = info.dchidx
        
    elif model_number == 3:
        if verbose: print('Modeling with an order %i power law'%(npoly,))  # endif                
        [prof, gvec, info] = model_PowerLaw(XX, af, npoly)
        info.dprofdx = info.dchidx
        
    elif model_number == 4:
        if verbose: print('Modeling with an exponential on order %i polynomial background'%(npoly,))  # endif                
        [prof, gvec, info] = model_Exponential(XX, af, npoly)        
        info.dprofdx = info.dchidx
        
    elif model_number == 5:
        if verbose: print('Modeling with an order %i polynomial+Heaviside fn'%(npoly,))  # endif                
        [prof, gvec, info] = model_Heaviside(XX, af, npoly)        
        info.dprofdx = info.dchidx
        
    elif model_number == 6:
        if verbose: print('Modeling with a %i step profile'%(npoly,))  # endif        
        [prof, gvec, info] = model_StepSeries(XX, af, npoly)
        info.dprofdx = info.dchidx
        
    elif model_number == 7:
        if verbose: print('Modeling with a quasiparabolic profile')  # endif                        
        [prof, gvec, info] = model_qparab(XX, af)
        
    elif model_number == 8:
        if verbose: print('Modeling with an order %i even polynomial'%(npoly,))  # endif                        
        [prof, gvec, info] = model_evenpoly(XX, af, npoly)
        info.dprofdx = info.dchidx
        
    elif model_number == 9:  # Two power fit
        if verbose: print('Modeling with a 2-power profile')  # endif                    
        [prof, gvec, info] = model_2power(XX, af)
        
    elif model_number == 10:
        if verbose: print('Modeling with a parabolic profile')  # endif                                
        [prof, gvec, info] = model_parabolic(XX, af)
                
    elif model_number == 11:
        if verbose: print('Modeling with a flat-top profile')  # endif                        
        [prof, gvec, info] = model_flattop(XX, af)
        
    elif model_number == 12:
        if verbose: print('Modeling with a Massberg-style profile')  # endif                                
        [prof, gvec, info] = model_massberg(XX, af)       
        
    # end switch-case

    if nargout == 3:
        return prof, gvec, info
    elif nargout == 2:
        return prof, gvec
    elif nargout == 1:
        return prof
# end def model_profile()

# =========================================================================== #


def model_chieff(af=None, XX=None, model_number=1, npoly=4, nargout=1, verbose=True):
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

    if XX is None:
        XX = _np.linspace(0, 1, 200)
    # endif

    # ====================================================================== #
    # ====================================================================== #

    if model_number == 1:
        if verbose: print('Modeling with an order %i product of Exponentials'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_ProdExp(XX, af, npoly)

    elif model_number == 2:
        if verbose: print('Modeling with an order %i polynomial'%(npoly,))  # endif        
        [chi_eff, gvec, info] = model_poly(XX, af, npoly)

    elif model_number == 3:
        if verbose: print('Modeling with an order %i power law'%(npoly,))  # endif                
        [chi_eff, gvec, info] = model_PowerLaw(XX, af, npoly)

    elif model_number == 4:
        if verbose: print('Modeling with an exponential on order %i polynomial background'%(npoly,))  # endif                
        [chi_eff, gvec, info] = model_Exponential(XX, af, npoly)

    elif model_number == 5:
        if verbose: print('Modeling with an order %i polynomial+Heaviside fn'%(npoly,))  # endif                
        [chi_eff, gvec, info] = model_Heaviside(XX, af, npoly)

    elif model_number == 6:
        if verbose: print('Modeling with a %i step profile'%(npoly,))  # endif        
        [chi_eff, gvec, info] = model_StepSeries(XX, af, npoly)

    elif model_number == 7:
        if verbose: print('Modeling with a quasiparabolic profile')  # endif                
        [chi_eff, gvec, info] = model_qparab(XX, af)
        chi_eff = 10.0 - chi_eff
        info.dchidx = -1.0*info.dprofdx
        gvec = -1.0*gvec

    elif model_number == 8:
        if verbose: print('Modeling with an order %i even polynomial'%(npoly,))  # endif                
        [chi_eff, gvec, info] = model_evenpoly(XX, af, npoly)

    elif model_number == 9:
        if verbose: print('Modeling with a 2-power profile')  # endif                
        [chi_eff, gvec, info] = model_2power(XX, af)
        info.dchidx = info.dprofdx
    # end switch-case

    if nargout == 3:
        return chi_eff, gvec, info
    elif nargout == 2:
        return chi_eff, gvec
    elif nargout == 1:
        return chi_eff
# end def model_chieff()

# =========================================================================== #

def normalize_test_prof(xvar, dPdx, dVdrho):
    dPdx = dPdx/_np.trapz(dPdx, x=xvar)     # Test profile shape : dPdroa
    
    dPdx = _np.atleast_2d(dPdx).T
    dVdrho = _np.atleast_2d(dVdrho).T
    
    # Test power density profile : dPdVol
    dPdV = dPdx/dVdrho
    dPdV[_np.where(_np.isnan(dPdV))] = 0.0
    return dPdV
# end def normalize_test_prof
    
def get_test_Pdep(xvar, rloc=0.1, rhalfwidth=0.05, dVdrho=None):
    sh = _np.shape(xvar)
    if dVdrho is None:
        dVdrho = _np.ones_like(xvar)

    # endif

    # Normalized gaussian for the test power deposition shape : dPdroa
    dPdx = (_np.exp(-0.5*((xvar-rloc)/rhalfwidth)**2)
            / (rhalfwidth*_np.sqrt(2*_np.pi)))
    # Test ECRH power density in MW/m3
    dPdV = normalize_test_prof(xvar, dPdx, dVdrho)
    dPdV = dPdV.reshape(sh)
    return dPdV
# end def
    
def sech(x):
    """
    sech(x)
    Uses numpy's cosh(x).
    """
    return 1.0/_np.cosh(x)

# =========================================================================== #
# =========================================================================== #


if __name__ == '__main__':

    XX = _np.linspace(0, 1, 200)
    npoly = 10
    model_number = 1
#    af = [0.1,0.1,0.1]
    af = None
#    af = [2.0,0.1,0.1,0.5]

#    [chi_eff, gvec, info] = \
#        model_chieff(af=af, XX=XX, model_number=model_number, npoly=npoly, nargout=3, verbose=True)

    [chi_eff, gvec, info] = \
        model_profile(af=af, XX=XX, model_number=model_number, npoly=npoly, nargout=3, verbose=True)


    info.dchidx = info.dprofdx

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

# =========================================================================== #
# =========================================================================== #






