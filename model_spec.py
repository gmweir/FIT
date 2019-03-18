# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:59:28 2016

Methods supported:
    non-linear least squares - see fitNL.LMFIT methods
            chi2_function = argmin ||y-X*B||^2
    Stepwise regression - see fitNL.fit_fourier
            Uses non-linear least squares by fitting at lowest order and
            adding terms to the fitting function sequentially

Methods to be added:
    Ridge regression - add a shrinkage parameter to the least-squares residual function
            chi2_function = argmin ||y-X*B||^2 + lambda* ||B||^2

    Lasso regression - argmin ||y- X*B||^2
            chi2_function = argmin ||y-X*B||^2 + lambda* ||B||

    ElasticNet regression
            chi2_function = argmin ||y-X*B||^2 + lambda2* ||B||^2 + lambda1* ||B||

Fits to add:
    Sigmoid function (S-curve) - approximated by an arctangent function or other logistics function
        - logistic regression for binary problems
        ex// odds = p/(1-p)  - probability of event occuring (or not)
             ln|odds| = ln|p/(1-p)|
             logit(p) = ln|p/(1-p)| = b0 + b1*x1+b2*x2+b3*x3 + ... + bk*xk
                 f(x) = L/(1+exp(-k*(x-xo)))
                     xo - Sigmoid mid-point
                     L is the curve's maximum value
                     k is the logistic growth rate or steepness of the curve

    Step series  - This is a series of sigmoid functions that approximate a
            profile by a discrete series of levels

    Supergaussian intensity profile
        I(r) = Ip exp(-2(r/w)^n)  ... order (n) increases steepness towards
                        ideal flat top measured versus radius from axis (r)
    Gaussian beam
        I(r) = 2P/(pi*w_z^2) * exp(-2r^2/w_z^2) - Gaussian beam intensity distribution for quasioptics

        E(r,z) = Eo*(wo/wz)*exp(-r^2/wz^2)*exp(i*[k*z-atan(z/zr)+k*r^2/(2*Rz)])
            monochromatic Gaussian beam propagating in z-direction (Amplitude phasor)
            ... oscillating real electric field is E(r,z)*exp(-i*2pi*c*t/lambda)
            wavenumber = 2pi/lambda,
            Rayleigh length / range = zr= pi*wo^2/lambda (or confocal length b=2*zr)
            beam radius wz = wo*sqrt( 1+ (z/zr)^2),
            Radius of curv. of wavefronts = Rz = z*[1+(zr/z)^2]
            Beam divergence in far field = theta = lambda/(pi*wo)
            Complex beam parameter, qz = z+i*zr = (1/Rz - i*lambda/(pi*wz^2))^-1
            Transforming through optics follows the ABCD matrix formalism

    Spline models with variable knot positions
            curvature, kappa = y''/(1+y'^2)^1.5
                      and qi'(xi) = q'[i+1](xi)
                      and qi''(xi) = q''[i+1](xi) for 1<=i<= n-1
        simple piece-wise polynomial function S: [a,b]
        quadratic
        cubic  -  interpolating cubic spline
            Si(x) = zi*(x-t(i-1))^3/(6hi) + z(i-1)*(ti-x)^3/(6hi)
                  + [f(ti)/hi - zihi/6](x-t(i-1))
                  + [f(ti-1)/hi - z(i-1)hi/6](ti-x)
                  where zi = f''(ti) = second derivative of f at ith knot
                        hi = ti - t(i-1)
                        f(ti) are teh values of the function at the ith knot
        pchip


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
#from pybaseutils.Struct import Struct
from pybaseutils import utils as _ut
#from pybaseutils.utils import properror, sech

from FIT.model import ModelClass

#from cmath import *

# ========================================================================== #
# ========================================================================== #


#class MultiModel(ModelClass):
#    """
#
#    """
#    def __init__(self, XX, af=None, **kwargs):
#        super(MultiModel, self).__init__(XX, af, **kwargs)
#    # end def __init__


#class ModelExample(ModelClass):
#    """
#
#    """
#    _af = _np.asarray([1.0], dtype=_np.float64)
#    _LB = _np.asarray([0.0], dtype=_np.float64)
#    _UB = _np.asarray([_np.inf], dtype=_np.float64)
#    _fixed = _np.zeros( (1,), dtype=int)
#    def __init__(self, XX, af=None, **kwargs):
#        super(ModelExample, self).__init__(XX, af, **kwargs)
#    # end def __init__
#
#    @staticmethod
#    def _model(XX, aa, **kwargs):
#        return NotImplementedError
#
#    @staticmethod
#    def _deriv(XX, aa, **kwargs):
#        return NotImplementedError
#
#    @staticmethod
#    def _partial(XX, aa, **kwargs):
#        return NotImplementedError
#
#    @staticmethod
#    def _partial_deriv(XX, aa, **kwargs):
#        return NotImplementedError
#
#    @staticmethod
#    def _hessian(XX, aa):
#        return NotImplementedError
#
#    # ====================================== #
#
#    def unscaleaf(self, ain):
#        return NotImplementedError
## end def ModelExample


# ========================================================================== #
# ========================================================================== #

def power(x, c, real=True):
    if real:
        return _np.power(_np.asarray(x, dtype=_np.complex128), c).real
    else:
        return _np.power(x, c)
    # end if

def log(x, real=True):
    # fails for
    if real:
        return _np.log(_np.asarray(_np.abs(x), dtype=_np.complex128)).real
    else:
        return _np.log(_np.abs(x))
    # end if

def log1p(x, real=True):
    # fails for
    if real:
        return _np.log1p(_np.asarray(x, dtype=_np.complex128)).real
    else:
        return _np.log1p(x)
    # end if

def exp(x, real=True):
    # fails for
    if real:
        return _np.exp(_np.asarray(x, dtype=_np.complex128)).real
    else:
        return _np.exp(x)
    # end if

# ========================================================================== #
# ========================================================================== #

def line(XX, a):
    return ModelLine._model(XX, a)

def line_gvec(XX, a):
    return ModelLine._partial(XX, a)

def deriv_line(XX, a):
    return ModelLine._deriv(XX, a)

def partial_deriv_line(XX, a):
    return ModelLine._partial_deriv(XX, a)

def hessian_line(XX, a):
    return ModelLine._hessian(XX, a)

def model_line(XX=None, af=None, **kwargs):
    """
     y = a * x + b
    """
    return _model(ModelLine, XX, af, **kwargs)


# ========================================================================== #


class ModelLine(ModelClass):
    """
     y = a * x + b
    """
    _af = _np.asarray([2.0,1.0], dtype=_np.float64)
    _LB = _np.array([-_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.array([0, 0], dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX=None, af=None, **kwargs):
        self._af = _np.random.uniform(low=-10.0, high=10.0, size=2)
        super(ModelLine, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        model of a line
            y = a*x + b
        """
        a, b = tuple(aa)
        return a*XX+b

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        derivative of a line
            y = a*x + b
            dydx = a
        """
        a, b = tuple(aa)
        return a*_np.ones_like(XX)

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        derivative of a line
            y = a*x + b
            dydx = a
            dy2dx2 = 0.0
        """
        a, b = tuple(aa)
        return a*_np.zeros_like(XX)

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        Jacobian of a line
            y = a*x + b
            dyda = x
            dydb = 1
        """
        gvec = _np.zeros( (2,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = XX.copy() # aa[0], a
        gvec[1,:] = 1.0       # aa[1], b
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a line
            y = a*x + b
            dydx = a
            d2ydxda = 1.0
            d2ydxdb = 0.0
        """
        dgdx = _np.zeros( (2,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = _np.ones_like(XX)  # aa[0], a
        dgdx[1,:] = _np.zeros_like(XX) # aa[1], b
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a line
            y = a*x + b
            dydx = a
            d2ydxda = 1.0
            d2ydxdb = 0.0
            d2ydx2 = 0
            d3yda3 = 0
            d3ydb3 = 0
        """
        d2gdx2 = _np.zeros( (2,_np.size(XX)), dtype=_np.float64)
        return d2gdx2

    @staticmethod
    def _hessian(XX, aa, **kwargs):
        """
        Hessian of a line
            y = a*x + b
            d2yda2 = 0.0
            d2ydadb = 0.0
            d2ydbda = 0.0
            d2ydb2 = 0.0
        """
        return _np.zeros( (2, 2, _np.size(XX)), dtype=_np.float64)

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        if the model data is scaled, then unscaling the model goes like this:
             (y-miny)/(maxy-miny) = (y-offset)/slope

        with y-scaling:  y'=(y-yo)/ys
             y = ys*a'*x + ys*b' + yo
             a = ys*a'
             b = ys*b' + yo

        with x-scaling:  x'=x/xs
             y = ys*a'*x/xs + ys*b' + yo
             a = ys*a'/xs
             b = ys*b' + yo

        with x-offset:  x=(x-xo)/xs
             y = ys*a'*x/xs + ys*b' + yo - ys*a'*xo/xs
                 a = ys*a'/xs
                 b = ys*(b'-xo*a'/xs) + yo
                   = ys*b' - xo*a + yo
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = ys*ain[0] / xs
        aout[1] = ys*(ain[1]-xo*ain[0]/xs) + yo
        return aout

    def scaleaf(self, ain, **kwargs):
        """
        Inverse function of unscaleaf
            a' = xs*a/ys
            b' = (b-yo)/ys + xo*a'/xs
               = (b-yo)/ys + xo*xs*a/ys/xs
               = b/ys + xo*a/ys - yo/ys
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)
        aout[0] = xs*ain[0]/ys
        aout[1] = (ain[1] + xo*ain[0] - yo)/ys
        return aout

    def unscalecov(self, covin, **kwargs):
        """
        to scale the covariances, simplify the unscaleaf function to ignore offsets
        Use identitites:
         (1)   cov(X, Y) = cov(Y, X)
         (2)   cov(a+X, b+Y) = cov(X,Y)
         (3)   cov(aX, bY) = ab*cov(X,Y)
         (4)   cov(aX+bY, cW+dV) = ac*cov(X,W) + ad*cov(X,V) + bc*cov(Y,W) + bd*cov(Y,V)
         (5)   cov(aX+bY, aX+bY) = a^2*cov(X,X) + b^2*cov(Y,Y) + 2*ab*cov(X,Y)

        Model:
        y = a*x +  b
        cov' = [vara', covab'; covba', varb']

        y = ys*a'*x/xs + ys*b' + yo - ys*a'*xo/xs
            a = ys*a'/xs
            b = ys*(b'-xo*a'/xs) + yo

        vara = (ys/xs)^2*vara'              by (3)
        varb = cov(ys*(b'-xo*a'/xs) + yo)
             = cov(ys*(b'-xo*a'/xs))        by(2)
             = ys^2*cov(b'-xo*a'/xs, b'-xo*a'/xs) by(1)
             = ys^2*( cov(b',b') +(xo/xs)^2*cov(a',a') - 2*xo/xs*cov(b',a') )  by (5)

        covab = cov(a,b)
              = cov(ys*a'/xs, ys*(b'-xo*a'/xs) + yo)
              = ys^2*cov(a'/xs, b'-xo*a'/xs)       by (1) and (2)
              = (ys)^2*( (1/xs)*cov(a',b') + (1/xs)*(-xo/xs)*cov(a',a') ) by (4)
        """
        covin = _np.copy(covin)
        covout = _np.copy(covin)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset) # analysis:ignore unnec. for cov
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        covout[0,0] = power(ys/xs, 2.0)*covin[0,0]
        covout[0,1] = (ys*ys)*( covin[0,1]/xs - xo*covin[0,0]/(xs*xs) )
        covout[1,0] = _np.copy(covout[0,1])
        covout[1,1] = (ys*ys)*( covin[1,1] + (xo*xo/(xs*xs))*covin[0,0] - 2.0*xo/xs*covin[1,0])
        return covout
# end def ModelLine

# ========================================================================== #
# ========================================================================== #

def sines(XX, aa):
    return ModelSines._model(XX, aa)

def partial_sines(XX, aa):
    return ModelSines._partial(XX, aa)

def deriv_sines(XX, aa):
    return ModelSines._deriv(XX, aa)

def partial_deriv_sines(XX, aa):
    return ModelSines._partial_deriv(XX, aa)

def model_sines(XX=None, af=None, **kwargs):
    return _model(ModelSines, XX, af, **kwargs)

# =========================================== #


class ModelSines(ModelClass):
    """
    Fourier series in the sine-phase form:
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii) ii>=1
    """
    _af = _np.asarray([0.0]+[1.0, 33.0, _np.pi/3], dtype=_np.float64)
    _LB = _np.asarray([-_np.inf]+[-_np.inf,   1e-18, -_np.pi], dtype=_np.float64)
    _UB = _np.asarray([ _np.inf]+[ _np.inf, _np.inf,  _np.pi], dtype=_np.float64)
    _fixed = _np.asarray([0]+[0, 0, 0], dtype=int)
    _params_per_freq = 3
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX=None, af=None, **kwargs):
        # Tile defaults to the number of frequencies requested
        if af is not None:
            self.nfreqs = self.getnfreqs(af)
        else:
            # default to a square wave
            self.nfreqs = kwargs.setdefault('nfreqs', 1)
            self.fmod = kwargs.setdefault('fmod', self._af[2])
            self._af[2] = _np.copy(self.fmod)
        # end if
        self._af = _np.asarray( [0.0]+self.nfreqs*self._af[1:].tolist(), dtype=_np.float64)
        self._LB = _np.asarray( [-_np.inf]+self.nfreqs*self._LB[1:].tolist(), dtype=_np.float64)
        self._UB = _np.asarray( [ _np.inf]+self.nfreqs*self._UB[1:].tolist(), dtype=_np.float64)
        self._fixed = _np.asarray( [0]+self.nfreqs*self._fixed[1:].tolist(), dtype=int)
        # shift the frequencies of each sine by harmonics of a square wave for defaults

        # default shape of expansion
        self._shape(**kwargs)

        # ================================ #

        super(ModelSines, self).__init__(XX, af, **kwargs)
    # end def __init__

    def _shape(self, **kwargs):
        sq = kwargs.setdefault('shape', 'sine')
        duty = kwargs.setdefault('duty', 0.5)
        if sq.lower().find('square')>-1:# and duty!=0.5:
            # duty cycled square wave
            # an = 2A/npi * sin(n*pi*tp/T)
            # an = 2A/npi * sin(n*pi*dutycycle)
            AA = _np.copy(self._af[1])
            self._af[0] = self._af[0] + AA*duty
            for ii in range(self.nfreqs):
                nn = 2*(ii+1)-1
                self._af[3*ii + 1] = 2.0*AA*_np.sin(nn*_np.pi*duty)/(_np.pi*nn)  # amplitudes
                self._af[3*ii + 2] = nn*self._af[2]  # i'th odd harm. of default
                self._af[3*ii + 3] = -0.5*_np.pi  # phase of i'th harm. of default
            # end for
        elif sq.lower().find('random')>-1:
            self._af[0] = 0.0
#            self._af[1::3] = _np.random.normal(0.0, 1.0, _np.size(self.af[1::3]))
            for ii in range(self.nfreqs):
                self._af[3*ii + 1] = self._af[1]*_np.random.normal(0.0, 1.0, 1)  # amplitudes
                self._af[3*ii + 2] = self._af[2]*(ii+1)  # i'th harm. of default
                self._af[3*ii + 3] = _np.random.uniform(-0.5*_np.pi, 0.5*_np.pi, size=1)  # phase of i'th harm. of default
            # end for
        else:  # sq.lower().find('uniform')>-1:
            self._af[0] = 0.0
            for ii in range(self.nfreqs):
                self._af[3*ii + 1] = self._af[1]/(ii+1)  # amplitudes
                self._af[3*ii + 2] = self._af[2]*(ii+1)  # i'th harm. of default
                self._af[3*ii + 3] = 0.5*_np.pi*_np.random.normal(0.0, 0.5, 1)  # phase of i'th harm. of default
                if self._af[3*ii+3]<self._LB[3*ii+3]: self._af[3*ii+3] = self._LB[3*ii+3]+0.1
                if self._af[3*ii+3]>self._UB[3*ii+3]: self._af[3*ii+3] = self._UB[3*ii+3]-0.1
#                self._af[3*ii + 3] = _np.random.choice([-0.5*_np.pi, 0.5*_np.pi])  # phase of i'th harm. of default
                self._af[3*ii + 3] = 0.0
            # end for
        # end if

    def _default_plot(self, XX=None, **kwargs):
#        if XX is None:  XX = _np.copy(self.XX) # end if
        if XX is None:
            XX = _np.linspace(-2.0/self.fmod, 2.0/self.fmod, num=500)
        _plt.figure()
        _plt.plot(XX, self.model(XX, self._af, **kwargs))

    @staticmethod
    def sine(XX, a, f, p):
        return a*_np.sin((2*_np.pi*f)*XX+p)

    @staticmethod
    def cosine(XX, a, f, p):
        return a*_np.cos((2*_np.pi*f)*XX+p)

    @staticmethod
    def getnfreqs(aa):
        if _np.mod(3, len(aa)-1) == 0:
            nfreqs = 1
        else:
            nfreqs = (len(aa)-1)/_np.mod(3, len(aa)-1)
        return int(nfreqs)

    @staticmethod
    def _convert2exp(aa):
        """
        The Amp-phase form can be easily converted to the exponential or the
        sine-cosine form
        exponential: sn_x = sum_-N_+N ( cn * exp(i*2pif*n*x ) )
        Amp-phase:   sn_x = Ao/2 + sum_n( An*sin(2pifnx+p) )
            cn = (An/2i)*exp(i*p_n)    n>0
            cn = 0.5*An                n=0
            cn=(-An)/(2i)*exp(-i*p_n) n<0
        """
        An = aa[1::3]
        # wn = 2*_np.pi*aa[2::3]
        pn = aa[3::3]

        nfreqs = ModelSines.getnfreqs(aa)
        aout = _np.zeros((2*nfreqs+1,), dtype=_np.complex128)
        aout[:nfreqs] = 0.5*1j*(An*_np.exp(-1j*pn))
        aout[nfreqs] = 0.5*aa[0]
        aout[nfreqs:] = -0.5*1j*(An*_np.exp(1j*pn))
        return aout

    @staticmethod
    def _convert2fourier(aa):
        """
        The Amp-phase form can be easily converted to the exponential or the
        sine-cosine form
        sine-cosine: sn_x = ao/2 + sum_n( an*cos(2pifnx) + bn*sin(2pifnx) )
        Amp-phase:   sn_x = Ao/2 + sum_n( An*sin(2pifnx+p) )
            ao = Ao
            an = An*sin(p_n)
            bn = An*cos(p_n)
        """
        An = aa[1::3]
        # wn = 2*_np.pi*aa[2::3]
        pn = aa[3::3]

        nfreqs = ModelSines.getnfreqs(aa)
        aout = _np.zeros((2*nfreqs+1,), dtype=_np.float64)
        aout[0] = _np.copy(aa[0])
        aout[1::2] = An*_np.sin(pn)
        aout[2::2] = An*_np.cos(pn)
        return aout

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii)
        """
        XX = _np.copy(XX)
        XX = _np.atleast_2d(XX)                  # (1,len(XX))
        a = _np.atleast_2d(aa[1::3]).T*_np.ones_like(XX) # amp (nfreq, nx)
        f = _np.atleast_2d(aa[2::3]).T  # freq. (nfreq,1)
        p = _np.atleast_2d(aa[3::3]).T*_np.ones_like(XX) # phase (nfreq, nx)
        return 0.5*aa[0] + _np.sum( ModelSines.sine(XX, a, f, p), axis=0)
#        w = 2.0*_np.pi*f # cyclic freq. (nfreq,1)
#        return 0.5*aa[0] + _np.sum( a*_np.sin(w*XX+p), axis=0)

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii)
        dfdx = sum_ii 2*pi*f_ii*a_ii *cos((2pi*f_ii)*XX+p_ii)
        """
        XX = _np.copy(XX)
        XX = _np.atleast_2d(XX)                  # (1,len(XX))
        tmp = _np.ones_like(XX)  # (1,nx)
        a = _np.atleast_2d(aa[1::3]).T   # amp (nfreq, nx)
        f = _np.atleast_2d(aa[2::3]).T  # freq. (nfreq,1)
        w = 2.0*_np.pi*f # cyclic freq. (nfreq,1)
        p = _np.atleast_2d(aa[3::3]).T # phase (nfreq, nx)
        return _np.sum( ModelSines.cosine(XX, (w*_np.ones_like(XX))*(a*tmp), f, p*tmp), axis=0)
#        return _np.sum( (w*_np.ones_like(XX))*(a*tmp)*_np.cos(w*XX+p*tmp), axis=0)

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii)
        dfdx = sum_ii 2*pi*f_ii*a_ii *cos((2pi*f_ii)*XX+p_ii)

        dfdao = 0.5
        dfdai = sin(2pifii*XX+pii)
        dfdfi = 2piXX*a*cos(2pifii*XX+pii)
        dfdpi = a*cos(2pifii*XX+pii)
        """
        XX = _np.copy(XX)
        XX = _np.atleast_2d(XX)                  # (1,len(XX))
        tmp = _np.ones_like(XX)  # (1,nx)
        a = _np.atleast_2d(aa[1::3]).T # amp (nfreq, nx)
        f = _np.atleast_2d(aa[2::3]).T  # freq. (nfreq,1)
        w = 2.0*_np.pi*f # cyclic freq. (nfreq,1)
        p = _np.atleast_2d(aa[3::3]).T # phase (nfreq, nx)

        gvec = _np.zeros( (len(aa), _np.size(XX)), dtype=_np.float64)
        gvec[0, :] = 0.5
        gvec[1::3, :] = ModelSines.sine(XX, 1.0, f, p*tmp)
        gvec[2::3, :] = ModelSines.cosine(XX, (2.0*_np.pi*_np.ones_like(w)*XX)*(a*tmp), f, p*tmp)
        gvec[3::3, :] = ModelSines.cosine(XX, a*tmp, f, p*tmp)
#        gvec[1::3, :] = _np.sin(w*XX+p*tmp)
#        gvec[2::3, :] = (2.0*_np.pi*_np.ones_like(w)*XX)*(a*tmp)*_np.cos(w*XX+p*tmp)
#        gvec[3::3, :] = (a*tmp)*_np.cos(w*XX+p*tmp)
        return gvec

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii)
        dfdx = sum_ii 2*pi*f_ii*a_ii *cos((2pi*f_ii)*XX+p_ii)

        dfdao = 0.5
        dfdai = sin(2pifii*XX+pii)
        dfdfi = 2piXX*a*cos(2pifii*XX+pii)
        dfdpi = a*cos(2pifii*XX+pii)

        d2fdx2 = sum_ii     (2*pi*f_ii*a_ii)  * d/dx(   cos((2pi*f_ii)*XX+p_ii) )
               =  -1*sum_ii  a_ii * (2*pi*f_ii)^2 * sin((2pi*f_ii)*XX+p_ii)

        d2fdx2 = -1 * sum_ii (2*pi*f_ii)^2*a_ii *sin((2pi*f_ii)*XX+p_ii)
        """
        XX = _np.copy(XX)
        a = _np.copy(aa[1::3])
        f = _np.copy(aa[2::3])
        p = _np.copy(aa[3::3])

        XX = _np.atleast_2d(XX)  # (1,nx)
        tmp = _np.ones_like(XX)  # (1,nx)
        a = _np.atleast_2d(a).T  # (nfreq,1)
        f = _np.atleast_2d(f).T  # (nfreq,1)
        p = _np.atleast_2d(p).T  # (nfreq,1)
        w = 2.0*_np.pi*f
        return _np.sum( ModelSines.sine(XX, -1*(power(w, 2.0)*tmp)* (a*tmp), f, p*tmp)  , axis=0)
#        return -1*_np.sum( (power(w, 2.0)*tmp)* (a*tmp) * _np.sin(w*XX + p*tmp)  , axis=0)

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii)
        dfdx = sum_ii 2*pi*f_ii*a_ii *cos((2pi*f_ii)*XX+p_ii)

        dfdao = 0.5
        dfdai = sin(2pifii*XX+pii)
        dfdfi = 2piXX*a*cos(2pifii*XX+pii)
        dfdpi = a*cos(2pifii*XX+pii)

        d2fdx2 = -1 * sum_ii (2*pi*f_ii)^2*a_ii *sin((2pi*f_ii)*XX+p_ii)

        d2fdxdao = 0.0
        d2fdxdai = 2*pi*f_ii*cos((2pi*f_ii)*XX+p_ii)
        d2fdxdfi = 2*pi*a_ii *cos((2pi*f_ii)*XX+p_ii) - (2*pi)^2*f_ii*XX*a_ii *sin((2pi*f_ii)*XX+p_ii)
        d2fdxdpi = -2*pi*f_ii*a_ii *sin((2pi*f_ii)*XX+p_ii)
        """
        XX = _np.copy(XX)
        nx = len(XX)
        XX = _np.atleast_2d(XX)                  # (1,len(XX))
        a = _np.atleast_2d(aa[1::3]).T # amp (nfreq, nx)
        f = _np.atleast_2d(aa[2::3]).T # freq. (nfreq,1)
        p = _np.atleast_2d(aa[3::3]).T # phase (nfreq, nx)
        tmp = _np.ones_like(XX)  # (1,nx)
        w = 2.0*_np.pi*f  # cyclic freq. (nfreq,1)

        dgdx = _np.zeros( (len(aa), nx), dtype=_np.float64)
        dgdx[0, :] = 0.0
        dgdx[1::3, :] = ModelSines.cosine(XX, w*_np.ones_like(XX), f, p*tmp)
        dgdx[2::3, :] = (ModelSines.cosine(XX, 2.0*_np.pi*(a*tmp), f, p*tmp)
                       - ModelSines.sine(XX, 2.0*_np.pi*(a*tmp)*(w*XX), f, p*tmp))
        dgdx[3::3, :] = ModelSines.sine(XX, (-1*w*_np.ones_like(XX))*(a*tmp), f, p*tmp)
#        dgdx[1::3, :] = (w*_np.ones_like(XX))*_np.cos(w*XX+p*tmp)
#        dgdx[2::3, :] = (2.0*_np.pi)*(a*tmp)*(_np.cos(w*XX+p*tmp) - (w*XX)*_np.sin(w*XX+p*tmp))
#        dgdx[3::3, :] = (-1*w*_np.ones_like(XX))*(a*tmp)*_np.sin(w*XX+p*tmp)
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii)

        dfdao = 0.5
        dfdai = sin(2pifii*XX+pii)
        dfdfi = 2piXX*a*cos(2pifii*XX+pii)
        dfdpi = a*cos(2pifii*XX+pii)

        dfdx = sum_ii 2*pi*f_ii*a_ii *cos((2pi*f_ii)*XX+p_ii)

        d2fdxdao = 0.0
        d2fdxdai = 2*pi*f_ii*cos((2pi*f_ii)*XX+p_ii)
        d2fdxdfi = 2*pi*a_ii *cos((2pi*f_ii)*XX+p_ii) - (2*pi)^2*f_ii*XX*a_ii *sin((2pi*f_ii)*XX+p_ii)
        d2fdxdpi = -2*pi*f_ii*a_ii *sin((2pi*f_ii)*XX+p_ii)

        d2fdx2 = -1 * sum_ii (2*pi*f_ii)^2*a_ii *sin((2pi*f_ii)*XX+p_ii)
        d3fdx2dao = 0.0
        d3fdx2dai = -(2*pi*f_ii)^2*sin((2pi*f_ii)*XX+p_ii)
        d3fdx2dfi = -1*(2*pi)^2*2f_ii*a_ii *sin((2pi*f_ii)*XX+p_ii) - (2*pi*f)^2*(2*pi*XX)*a_ii *cos((2pi*f_ii)*XX+p_ii)
        d3fdx2dpi = -1*(2*pi*f_ii)^2*a_ii *cos((2pi*f_ii)*XX+p_ii)
        """
        XX = _np.copy(XX)
        a = _np.copy(aa[1::3])
        f = _np.copy(aa[2::3])
        p = _np.copy(aa[3::3])


        XX = _np.atleast_2d(XX)  # (1,nx)
        tmp = _np.ones_like(XX)  # (1,nx)
        a = _np.atleast_2d(a).T  # (nfreq,1)
        f = _np.atleast_2d(f).T  # (nfreq,1)
        p = _np.atleast_2d(p).T  # (nfreq,1)
        w = 2.0*_np.pi*f

        d2gdx2 = _np.zeros( (len(aa), _np.size(XX)), dtype=_np.float64)
        d2gdx2[0, :] = 0.0
        d2gdx2[1::3, :] = ModelSines.sine(XX, -1*(power(w, 2.0)*tmp), f, p*tmp)
        d2gdx2[2::3, :] = ( ModelSines.sine(XX, -2.0*power(2.0*_np.pi, 2.0)*((f*a)*tmp), f, p*tmp)
                          - ModelSines.cosine(XX, (power(w, 2.0)*a)*(2.0*_np.pi*XX), f, p*tmp))
        d2gdx2[3::3, :] = ModelSines.cosine(XX, -1*(power(w, 2.0)*a)*tmp, f, p*tmp)
        return d2gdx2

    # ====================================== #

    @staticmethod
    def _hessian(XX, aa, **kwargs):
        """
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii)
        dfdx = sum_ii 2*pi*f_ii*a_ii *cos((2pi*f_ii)*XX+p_ii)

        dfdao = 0.5
        dfdai = sin(2pifii*XX+pii)
        dfdfi = 2piXX*a*cos(2pifii*XX+pii)
        dfdpi = a*cos(2pifii*XX+pii)

        Hessian:
                  [[d2fdao2(x), d2fdaidao(x), d2fdfidao(x), ...]
                  [d2fdaodai(x), d2fdai2(x),  d2fdfidai(x), ...]
                  [d2fdaodfi(x), ...

        d2fdao2 = 0.0
        d2faidao = 0.0
        d2fdfidao = 0.0
        d2fdpidao = 0.0

        d2fdai2 = 0.0
        d2fdfidai = 2piXX*cos(2pifii*XX+pii)
        d2fdpidai = cos(2pifii*XX+pii)

        d2fdfi2 = -(2piXX)^2*f*a*sin(2pifii*XX+pii)
        d2fdfidpi = -2piXX*a*sin(2pifii*XX+pii)

        d2fdpi2 = -a*sin(2pifii*XX+pii)
        """
        numfit = _np.size(aa)
        XX = _np.copy(XX)
        XX = _np.atleast_2d(XX)                  # (1,len(XX))
        f = _np.atleast_2d(aa[2::3]).T    # linear freq. (nfreq,1)
        w = 2.0*_np.pi*f                  # cyclic freq. (nfreq,1)
        a = _np.atleast_2d(aa[1::3]).T # amp (nfreq, nx)
        p = _np.atleast_2d(aa[3::3]).T # phase (nfreq, nx)
        tmp = _np.ones_like(XX)  # (1,nx)

        hess = _np.zeros((numfit, numfit, _np.size(XX)), dtype=_np.float64)
        # diagonal:
        # hess[0::3, 0::3, :] = 0.0                                                    # d2fao2
        # hess[1::3, 1::3, :] = 0.0                                                    # d2fdai2
        hess[2::3, 2::3, :] = ModelSines.sine(XX, -1.0*(f*power(2.0*_np.pi*XX, 2.0))*(a*tmp), f, p*tmp) # d2fdfi2
        hess[3::3, 3::3, :] = ModelSines.sine(XX, -1.0*a*tmp, f, p*tmp)                                   # d2fdpi2

        # Upper triangle
        # hess[0, :, :] = 0.0                                                                   #d2fdaod_
        hess[1::3, 2::3, :] = ModelSines.cosine(XX, 2.0*_np.pi*(_np.ones_like(w)*XX), f, p*tmp)    # d2fdaidfi
        hess[1::3, 3::3, :] = ModelSines.cosine(XX, 1.0, f, p*tmp)                                      # d2fdaidpi
        hess[2::3, 3::3, :] = ModelSines.sine(XX, -2.0*_np.pi*(_np.ones_like(w)*XX)*(a*tmp), f, p*tmp) # d2fdfidpi

        # Lower triangle by symmetry
        # hess[:, 0::3, :] =
#        for ii in range(numfit):    # TODO!:  CHECK THIS!
#            for jj in range(numfit):
#                hess[jj, ii, :] = hess[ii, jj, :]
#            # end for
#        # end for
        hess[2::3, 1::3, :] = hess[1::3, 2::3, :].T
        hess[3::3, 1::3, :] = hess[1::3, 3::3, :].T
        hess[3::3, 2::3, :] = hess[2::3, 3::3, :].T
        return hess

    # ====================================== #

#    def scaledat(self, xdat, ydat, vdat, vxdat=None, **kwargs):
#        super(ModelSines, self).scaledat(xdat, ydat, vdat, vxdat=vxdat, **kwargs)

    def unscaleaf(self, ain, **kwargs):
        """
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii)

        y-shifting: y'=(y-yo)/ys
          y = yo + 0.5*ys*ao' + ys*sum_ii a_ii' *sin((2pi*f_ii')*XX+p_ii')
            0.5*ao = yo + 0.5*ys*ao'
            [a_i, f_i, p_i] = [ys*a_i', f_i', p_i']

        x-shifting: x'=(x-xo)/xs
          y = yo + 0.5*ys*ao' + ys*sum_ii a_ii' * sin(2pif'(x-xo)/xs+p')
            = yo + 0.5*ys*ao' + ys*sum_ii a_ii' * sin(2pif'x/xs-2pif'xo/xs+p')
                ao = 2*yo + ys*ao'
                a_i = ys*a_i'
                f_i = f_i'/xs
                p_i = p_i'- 2pif_i'xo/xs
            [a_i, f_i, p_i] = [ys*a_i', f_i'/xs, p_i' - 2pif_i'*xo/xs]
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = 2*yo + ys*ain[0]
        aout[1::3] = ys*ain[1::3]
        aout[2::3] = ain[2::3]/xs
        aout[3::3] = ain[3::3] - 2.0*_np.pi*aout[2::3]*xo
        # wrap the phase back into the correct bounds
        aout[3::3] = _np.angle(_np.exp(1j*aout[3::3]))
#        aout[3::3] = (aout[3::3]+_np.pi) % (2.0*_np.pi) - 2.0*_np.pi
        return aout

    def scaleaf(self, ain, **kwargs):
        """
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii)
                ao = 2*yo + ys*ao'
                a_i = ys*a_i'
                f_i = f_i'/xs
                p_i = p_i'- 2pif_i'xo/xs

                ao' = (ao-2*yo)/ys
                ai' = ai/ys
                fi' = xs*fi
                pi' = pi + 2pifi'*xo/xs
                    = pi + 2pifi*xo
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = (ain[0] - 2.0*yo)/ys
        aout[1::3] = ain[1::3]/ys
        aout[2::3] = xs*ain[2::3]
        aout[3::3] = ain[3::3] + 2.0*_np.pi*aout[2::3]*xo/xs
        # wrap the phase back into the correct bounds
        aout[3::3] = _np.angle(_np.exp(1j*aout[3::3]))
        return aout

    def unscalecov(self, covin, **kwargs):
        """
        to scale the covariances, simplify the unscaleaf function to ignore offsets
        Use identitites:
         (1)   cov(X, Y) = cov(Y, X)
         (2)   cov(a+X, b+Y) = cov(X,Y)
         (3)   cov(aX, bY) = ab*cov(X,Y)
         (4)   cov(aX+bY, cW+dV) = ac*cov(X,W) + ad*cov(X,V) + bc*cov(Y,W) + bd*cov(Y,V)
         (5)   cov(aX+bY, aX+bY) = a^2*cov(X,X) + b^2*cov(Y,Y) + 2*ab*cov(X,Y)

        Model:
          y = yo + 0.5*ys*ao' + ys*sum_ii a_ii' * sin(2pif'x/xs-2pif'xo/xs+p')
                ao = 2*yo + ys*ao'
                a_i = ys*a_i'
                f_i = f_i'/xs
                p_i = p_i'-2pif_i'xo/xs
        cov' = [  varao', covaoai'; covaofi', covaopi',
                covaiao',   varai', covaifi', covaipi',
                covfiao', covfiai',   varfi', covfipi',
                covpiao', covpiai', covpifi', varpi']


        varao = ys^2*varao'              by (2+3)
        varai = ys^2*varai'
        varfi = varfi/xs^2
        varpi = cov(p'-2pif'xo/xs, p'-2pif'xo/xs)
              = varp' + (2pixo/xs)^2*varf' - 2pixo/xs * cov(p', f')

        cov(ao,ai) = cov(ys*ao', ys*ai')
                   = ys^2*cov(ao', ai')
        cov(ao,fi) = ys/xs*cov(ao', fi')
        cov(ao,pi) = cov(ys*ao', p'-2pixo/xs*f')
                    = ys*(cov(ao', p') - 2pixo/xs*cov(ao', f'))

        cov(ai,fi) = cov(ys*ai', f'/xs)
                   = ys/xs * cov(ai', f')
        cov(ai,pi) = cov(ys*ai', p'-2pixo/xs*f')
                   = ys*( cov(ai', p') - 2pixo/xs * cov(ai, f') )
        cov(fi,pi) = cov(f'/xs, p'-2pixo/xs*f')
                   = 1/xs*( cov(f',p')-2pixo/xs*cov(f',f') )
        """
        covin, covout = _np.copy(covin), _np.copy(covin)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset) # analysis:ignore   this is not necessary in the covariance
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        # diagonal terms
        covout[0,0] = (ys*ys)*covin[0,0]
        covout[1::3, 1::3] = (ys*ys)*covin[1::3, 1::3]
        covout[2::3, 2::3] = power(1.0/xs, 2.0)*covin[2::3, 2::3]
        covout[3::3, 3::3] = (covin[3::3, 3::3]
                        + power(_np.abs(2.0*_np.pi*xo/xs), 2.0)*covin[2::3,2::3]
                        - 2.0*_np.pi*xo/xs*covin[3::3, 2::3])

        # first row
        covout[0,1::3] = (ys*ys)*covin[0,1::3]  # ao-a_i
        covout[0,2::3] = (ys/xs)*covin[0,2::3]             # ao-f_i
        covout[0,3::3] = ys*(covin[0,3::3] - 2.0*_np.pi*xo/xs*covin[0,2::3]) # ao-p_i

        # first column
        covout[1::3,0] = _np.copy(covout[0,1::3])  # a_i-ao
        covout[2::3,0] = _np.copy(covout[0,2::3])  # f_i-ao
        covout[3::3,0] = _np.copy(covout[0,3::3])  # p_i-ao

        # mixed terms from sines:
        covout[1::3,2::3] = (ys/xs)*covin[1::3,2::3] # a_i-f_i
        covout[1::3,3::3] = ys*(covin[1::3,3::3] - 2.0*_np.pi*xo/xs*covin[1::3,2::3])    # a_i-p_i
        covout[2::3,3::3] = (1.0/xs)*( covin[2::3, 3::3] - 2.0*_np.pi*xo/xs*covin[2::3,2::3] )   # f_i-p_i

        covout[2::3,1::3] = _np.copy(covout[1::3,2::3])  # f_i-a_i
        covout[3::3,1::3] = _np.copy(covout[1::3,3::3])  # p_i-a_i
        covout[3::3,2::3] = _np.copy(covout[2::3,3::3])  # p_i-f_i
        return covout

    def scalings(self, xdat, ydat, **kwargs):
        """
        """
#        self.xslope = 1.0
#        self.xoffset = 0.0
        return super(ModelSines, self).scalings(xdat, ydat, **kwargs)

# end def ModelSines



# ========================================================================== #
# ========================================================================== #

def fourier(XX, aa):
    return ModelFourier._model(XX, aa)

def partial_fourier(XX, aa):
    return ModelFourier._partial(XX, aa)

def deriv_fourier(XX, aa):
    return ModelFourier._deriv(XX, aa)

def partial_deriv_fourier(XX, aa):
    return ModelFourier._partial_deriv(XX, aa)

def model_fourier(XX=None, af=None, **kwargs):
    return _model(ModelFourier, XX, af, **kwargs)

# =========================================== #


class ModelFourier(ModelClass):
    """
    Fourier series in the sine-cosine form:
         f = ao/2 + sum_ii( a_ii*cos((2*pi*f*ii)*x) + b_ii*sin((2*pi*f*ii)*x))

      = ao/2 + sum_ii( a_ii*cos((2*pi*f*ii)*x) + b_ii*sin((2*pi*f*ii)*x))
      = ao/2 + a1*cos(2pif*x) + b1*sin(2pif*x) + a2*cos(4pif*x) + b2*sin(4pif*x) + ...
        af - [fundamental frequency, offset,
              a_ii, b_ii, a_ii+1, b_ii+1, ...]
    """
    _af = _np.asarray([  33.0,     0.0], dtype=_np.float64)
    _LB = _np.asarray([  1e-18,-_np.inf], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (2,), dtype=int)
    _params_per_freq = 2
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX=None, af=None, **kwargs):
        # Tile defaults to the number of frequencies requested
        if af is not None:
            self.nfreqs = self.getnfreqs(af)
#            self.fmod = af[0]
        else:
            self.nfreqs = kwargs.setdefault('nfreqs', 1)
            self.fmod = kwargs.setdefault('fmod', self._af[0])
            self._af[0] = _np.copy(self.fmod)
        # end if

        if self.nfreqs>1:
            self._af = _np.asarray(self._af.tolist()+self.nfreqs*[0.0, 0.0], dtype=_np.float64)
            self._LB = _np.asarray(self._LB.tolist()+self.nfreqs*[-_np.inf,-_np.inf], dtype=_np.float64)
            self._UB = _np.asarray(self._UB.tolist()+self.nfreqs*[ _np.inf, _np.inf], dtype=_np.float64)
            self._fixed = _np.asarray(self._fixed.tolist()+self.nfreqs*[ 0, 0], dtype=_np.float64)

        self._shape(**kwargs)
        super(ModelFourier, self).__init__(XX, af, **kwargs)
    # end def __init__

    def _shape(self, **kwargs):
        MS = ModelSines(**kwargs)
        self._af = MS._convert2fourier(MS._af)
#        sq = kwargs.setdefault('shape', 'sine')
#        duty = kwargs.setdefault('duty', 0.5)
#        if sq.lower().find('square')>-1:# and duty!=0.5:
#            # duty cycled square wave
#            # an = 2A/npi * sin(n*pi*tp/T)
#            # an = 2A/npi * sin(n*pi*dutycycle)
#            ff = _np.copy(self._af[0])
#            AA = 1.0
#            self._af = _np.zeros((2+2*self.nfreqs,), dtype=_np.float64)
#            self._af[0] = ff
#            self._af[1] = self._af[1] + AA*duty
#            for ii in range(self.nfreqs):
#                if (ii+1) % 2 == 0:  # if the frequency is even
##                    continue
##                nn = 2*(ii+1)-1
#                nn = ii+1
#                self._af[2*ii + 2 + 0] = 2.0*AA*_np.sin(nn*_np.pi*duty)/(_np.pi*nn)  # amplitudes
#                self._af[2*ii + 2 + 1] = 0.0  # amplitudes of sine
#            # end for
#        else:
#            for ii in range(self.nfreqs):
#                self._af[2*ii + 2 + 0 ] = 0.5*self._af[1]/(ii+1)  # amplitudes
#                self._af[2*ii + 2 + 1 ] = 0.5*self._af[1]/(ii+1)  # i'th harm. of default
##                self._af[2*ii + 2 + 1 ] = 0.5*self._af[1]/(ii+1)  # i'th harm. of default
#            # end for
        # end if

    def _default_plot(self, XX=None):
#        if XX is None:  XX = _np.copy(self.XX) # end if
        if XX is None:
            XX = _np.linspace(-2.0/self.fmod, 2.0/self.fmod, num=500)
        _plt.figure()
        _plt.plot(XX, self.model(XX, self._af))

    @staticmethod
    def cosine(XX, a, f):
        return a*_np.cos((2*_np.pi*f)*XX)

    @staticmethod
    def sine(XX, b, f):
        return b*_np.sin((2*_np.pi*f)*XX)

    @staticmethod
    def getnfreqs(aa):
        if _np.mod(2, len(aa)-2) == 0:
            nfreqs = 1
        else:
            nfreqs = (len(aa)-2)/_np.mod(2, len(aa)-2)
        # end if
        return int(nfreqs)

    @staticmethod
    def _convert2sines(aa):
        """
        The Fourier model can be expressed equivalently in the amplitude-
        phase form (sines above), sin-cos form (here), or exponential form
        (Fourier transform).
        Amp-phase form: sn_x = Ao/2 + sum_1N An sin(2pifnx + p_n) for n>1
        sin-cos form:   sn_x = ao/2 + sum_1N an cos(2pifnx) + bn sin(2pifnx)
            Ao = ao
            An = sqrt( an^2 + bn^2)
                an = An sin(p_n)
                bn = An cos(p_n)
                an/bn = sin/cos = tan(p_n)
            p_n = arctan(an/bn)  (arctan2(an, bn))
        """
        nfreqs = ModelFourier.getnfreqs(aa)
        f  = _np.copy(aa[0])
#        aout = _np.zeros([0.0]+[3*nfreqs], dtype=_np.float64)
        aout = _np.zeros((3*nfreqs+1,), dtype=_np.float64)

        aout[0] = _np.copy(aa[1])  # offset
        aout[1::3] = _np.sqrt( aa[2::2]*aa[2::2] + aa[3::2]*aa[3::2] ) # amplitude
        aout[2::3] = (_np.asarray(range(nfreqs))+1.0)*f # frequency
        aout[3::3] = _np.arctan2(aa[2::2], aa[3::2])    # phase
        return aout

    @staticmethod
    def _convert2exp(aa):
        """
        The sine-cosine form can be easily converted to the exponential or the
        amp-phase form
        exponential:  sn_x = sum_-N_+N ( cn * exp(i*2pif*n*x ) )
        sin-cos form: sn_x = ao/2 + sum_1N an cos(2pifnx) + bn sin(2pifnx)
            cn = 0.5*(an-i*bn)    n>0
            cn = 0.5*ao           n=0
            cn = 0.5*(an+i*bn)    n<0   # this is complex conjugate of cn>0
        """
        an = aa[1::2]
        bn = aa[2::2]

        nfreqs = ModelFourier.getnfreqs(aa)
        aout = _np.zeros((2*nfreqs+1,), dtype=_np.complex128)
        aout[:nfreqs] = 0.5*(an+1j*bn)
        aout[nfreqs]  = 0.5*aa[0]
        aout[nfreqs:] = 0.5*(an-1j*bn)
        return aout

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
         f = ao + sum_ii( a_ii*cos((2*pi*f*ii)*x) + b_ii*sin((2*pi*f*ii)*x))
        """
        a1 = ModelFourier._convert2sines(aa)
        return ModelSines._model(XX, a1)

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
         f = ao + sum_ii( a_ii*cos((2*pi*f*ii)*x) + b_ii*sin((2*pi*f*ii)*x))

         dfdx = (2*pi*f)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
        """
        a1 = ModelFourier._convert2sines(aa)
        return ModelSines._deriv(XX, a1)

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
         f = ao + sum_ii( a_ii*cos((2*pi*f*ii)*x) + b_ii*sin((2*pi*f*ii)*x))

         dfdx = (2*pi*f)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
         d2fdx2 = (2*pi*f)^2*sum_ii( -ii^2*a_ii*cos((2*pi*f*ii)*x) - ii^2*b_ii*sin((2*pi*f*ii)*x))
        """
        a1 = ModelFourier._convert2sines(aa)
        return ModelSines._deriv2(XX, a1)

    @staticmethod
    def _deriv3(XX, aa, **kwargs):
        """
         f = ao + sum_ii( a_ii*cos((2*pi*f*ii)*x) + b_ii*sin((2*pi*f*ii)*x))

         dfdx = (2*pi*f)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
         d2fdx2 = (2*pi*f)^2*sum_ii( -ii^2*a_ii*cos((2*pi*f*ii)*x) - ii^2*b_ii*sin((2*pi*f*ii)*x))
         d3fdx3 = (2*pi*f)^3*sum_ii( ii^3*a_ii*sin((2*pi*f*ii)*x) - ii^3*b_ii*cos((2*pi*f*ii)*x))
        """
        nfreqs = ModelFourier.getnfreqs(aa)
        f = _np.copy(aa[0])
        w = 2.0*_np.pi*f
        ai = _np.atleast_2d(_np.copy(aa[2::2])).T
        bi = _np.atleast_2d(_np.copy(aa[3::2])).T

        tmp = _np.ones_like(_np.atleast_2d(XX))
        ii = _np.atleast_2d(_np.asarray(range(nfreqs), dtype=_np.float64) + 1.0).T
#        return _np.sum(
#          ModelFourier.sine(_np.atleast_2d(XX), power(w, 3.0)*(power(ii, 3.0)*tmp)*(ai*tmp), f)
#                        , axis=0)
        return power(w, 3.0)*_np.sum( (power(ii, 3.0)*tmp)*(
                (ai*tmp)*_np.sin(w*(ii*_np.atleast_2d(XX)))
              - (bi*tmp)*_np.cos(w*(ii*_np.atleast_2d(XX)))
                ), axis=0)

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
         f = 0.5*ao + sum_ii( a_ii*cos((2*pi*f*ii)*x) + b_ii*sin((2*pi*f*ii)*x))

         dfdx = (2*pi*f)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
         d2fdx2 = (2*pi*f)^2*sum_ii( -ii^2*a_ii*cos((2*pi*f*ii)*x) - ii^2*b_ii*sin((2*pi*f*ii)*x))
         d3fdx3 = (2*pi*f)^3*sum_ii( ii^3*a_ii*sin((2*pi*f*ii)*x) - ii^3*b_ii*cos((2*pi*f*ii)*x))

         dfdf = (2*pi*x)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
              = dfdx*x/f
         dfdao = 0.5
         dfdai = cos((2*pi*f*ii)*x)
         dfdbi = sin((2*pi*f*ii)*x)
        """
        nfreqs = ModelFourier.getnfreqs(aa)
        f  = aa[0]
        ao = aa[1]
        gvec = ao*_np.zeros( (len(aa), _np.size(XX)), dtype=_np.float64)
        gvec[0, :] = (XX/f)*ModelFourier._deriv(XX, aa, **kwargs)
        gvec[1, :] = 0.5   # dfdao
        gvec[2::2, :] = _np.cos(2*_np.pi*(f*_np.atleast_2d(_np.asarray(range(nfreqs))+1).T)*_np.atleast_2d(XX))
        gvec[3::2, :] = _np.sin(2*_np.pi*(f*_np.atleast_2d(_np.asarray(range(nfreqs))+1).T)*_np.atleast_2d(XX))
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
          f = ao + sum_ii( a_ii*cos((2*pi*f*ii)*x) + b_ii*sin((2*pi*f*ii)*x))

          dfdx = (2*pi*f)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
          d2fdx2 = (2*pi*f)^2*sum_ii( -ii^2*a_ii*cos((2*pi*f*ii)*x) - ii^2*b_ii*sin((2*pi*f*ii)*x))
          d3fdx3 = (2*pi*f)^3*sum_ii( ii^3*a_ii*sin((2*pi*f*ii)*x) - ii^3*b_ii*cos((2*pi*f*ii)*x))

          dfdf = (2*pi*x)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
               = dfdx*x/f
          dfdao = ones
          dfdai = cos((2*pi*f*ii)*x)
          dfdbi = sin((2*pi*f*ii)*x)

          d2fdxdf = (2*pi)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
                 + (2pix)*(2pif)*sum_ii( -ii^2*a_ii*cos((2*pi*f*ii)*x) - ii^2*b_ii*sin((2*pi*f*ii)*x))
                  = dfdx/f + d2fdx2*x/f
          d2fdxdao = 0.0
          d2fdxdai = -(2*pi*f)*ii*sin(2*pi*f*ii*x) = -2pif*ii*dfdbi
          d2fdxdbi =  (2*pi*f)*ii*cos(2*pi*f*ii*x) = 2pif*ii*dfdai
        """
        nfreqs = ModelFourier.getnfreqs(aa)
        f  = _np.copy(aa[0])
        ao = _np.copy(aa[1])

        XX = _np.copy(XX)
        w = 2*_np.pi*f  # cyclic freq. (1,)
        tmp = _np.ones_like(_np.atleast_2d(XX))
        ii = _np.atleast_2d(_np.asarray(range(nfreqs), dtype=_np.float64)+1).T

        dgdx = ao*_np.zeros( (_np.size(aa), _np.size(XX)), dtype=_np.float64)
        dgdx[0, :] =  ModelFourier._deriv(XX, aa, **kwargs)/f + ModelFourier._deriv2(XX, aa, **kwargs)*XX/f
        dgdx[1, :] = 0.0
        dgdx[2::2, :] = -w*(ii*tmp)*_np.sin(w*ii*_np.atleast_2d(XX))
        dgdx[3::2, :] =  w*(ii*tmp)*_np.cos(w*ii*_np.atleast_2d(XX))
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
          prof = ao + sum_ii( a_ii*cos((2*pi*f*ii)*x) + b_ii*sin((2*pi*f*ii)*x))

          dfdx = (2*pi*f)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
          d2fdx2 = (2*pi*f)^2*sum_ii( -ii^2*a_ii*cos((2*pi*f*ii)*x) - ii^2*b_ii*sin((2*pi*f*ii)*x))
          d3fdx3 = (2*pi*f)^3*sum_ii( ii^3*a_ii*sin((2*pi*f*ii)*x) - ii^3*b_ii*cos((2*pi*f*ii)*x))

          dfdf = (2*pi*x)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
               = dfdx*x/f
          dfdao = ones
          dfdai = cos((2*pi*f*ii)*x)
          dfdbi = sin((2*pi*f*ii)*x)

          d2fdxdf = (2*pi)*sum_ii( -ii*a_ii*sin((2*pi*f*ii)*x) + ii*b_ii*cos((2*pi*f*ii)*x))
                 + (2pix)*(2pif)*sum_ii( -ii^2*a_ii*cos((2*pi*f*ii)*x) - ii^2*b_ii*sin((2*pi*f*ii)*x))
                  = dfdx/f + d2fdx2*x/f
          d2fdxdao = 0.0
          d2fdxdai = -(2*pi*f)*ii*sin(2*pi*f*ii*x) = -2pif*ii*dfdbi
          d2fdxdbi =  (2*pi*f)*ii*cos(2*pi*f*ii*x) = 2pif*ii*dfdai

          d3fdx2df = 2/f*(2*pi*f)^2*sum_ii( -ii^2*a_ii*cos((2*pi*f*ii)*x) - ii^2*b_ii*sin((2*pi*f*ii)*x))
                + (2pix)*(2*pi*f)^2*sum_ii(  ii^3*a_ii*sin((2*pi*f*ii)*x) - ii^3*b_ii*cos((2*pi*f*ii)*x))
                   = 2*d2fdx2/f+ d3fdx3*x/f
          d3fdx2dao = 0.0
          d3fdx2dai = -(2*pi*f)^2*ii^2*cos((2*pi*f*ii)*x)
          d3fdx2dbi = -(2*pi*f)^2*ii^2*sin((2*pi*f*ii)*x)
        """
        nfreqs = ModelFourier.getnfreqs(aa)
        f  = _np.copy(aa[0])
        ao = _np.copy(aa[1])

        XX = _np.copy(XX)
#        XX = _np.atleast_2d(XX)                  # (1,len(XX))
        w = 2*_np.pi*f                  # cyclic freq. (1,)
        ii = _np.atleast_2d(_np.asarray(range(nfreqs))+1).T
        tmp = _np.ones_like(_np.atleast_2d(XX))

        d2gdx2 = ao*_np.zeros( (_np.size(aa), _np.size(XX)), dtype=_np.float64)
        d2gdx2[0, :] = (2.0/f)*ModelFourier._deriv2(XX, aa, **kwargs) + \
                       ModelFourier._deriv3(XX, aa, **kwargs)*XX/f
        d2gdx2[1, :] = 0.0    # d3fddx2ao
        d2gdx2[2::2, :] = -power(w, 2.0)*(power(ii, 2.0)*tmp)*_np.cos(w*ii*_np.atleast_2d(XX))
        d2gdx2[3::2, :] = -power(w, 2.0)*(power(ii, 2.0)*tmp)*_np.sin(w*ii*_np.atleast_2d(XX))
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa, *kwargs):
#        return NotImplementedError

    # ====================================== #

    def unscaleaf(self, ain, **kwargs):
        """
        y-shifting and scaling is easy:        y' = (y-yo)/ys

         y = ao + sum_ii( a_ii*cos((2*pi*f*ii)*x) + b_ii*sin((2*pi*f*ii)*x))

         y = yo + ys*ao' + sum_ii( ys*ai'*cos(2pif'x*ii) + ys*bi'*sin(2pif'x*ii)
           ao = yo+ys*ao'
           ai = ys*ai'
           bi = ys*bi'

        x-shifting requires trigonometry:      x' = (x-xo)/xs
           w' = 2pif'/xs

         y = yo + ys*ao' + sum_ii( ys*ai'*cos(w'(x-xo)*ii) + ys*bi'*sin(w'(x-xo)*ii)
             cos(a-b) = cosx cosy + sinx siny
             sin(a-b) = sinx cosy - sinx siny
         cos(2pif'(x-xo)*ii) = cos(w'ii*(x-xo))
           = cos(w'ii*x) * cos(-w'ii*xo) + sin(w'ii*x) * sin(-w'ii*xo)
           = cos(w'ii*x) * cos(w'ii*xo) - sin(w'ii*x) * sin(w'ii*xo)
         sin(2pif'(x-xo)*ii) = sin(w'ii*(x-xo))
           = sin(w'ii*x) * cos(-w'ii*xo) - sin(w'ii*x) * sin(-w'ii*xo)
           = sin(w'ii*x) * cos(w'ii*xo) + sin(w'ii*x) * sin(w'ii*xo)

         y = yo + ys*ao' + sum_ii( ys*ai'*cos(w'ii*xo)cos(w'ii*x)
                                 - ys*ai'*sin(w'ii*xo)sin(w'ii*x)
                                 + ys*bi'*sin(w'ii*xo)cos(w'ii*x)
                                 + ys*bi'*sin(w'ii*xo)sin(w'ii*x)
           = (yo + ys*ao') + sum_ii(
            (ys*ai'*cos(w'ii*xo) + ys*bi'*sin(w'ii*xo) )* cos(w'ii*x)
          + (ys*bi'*sin(w'ii*xo) - ys*ai'*sin(w'ii*xo) )* sin(w'ii*x) )

        finally:
            f = f'/xs
            ao = yo + ys*ao'
            ai = ys*ai'*cos(2pif'ii/xs*xo) + ys*bi'*sin(2pif'ii/xs*xo)
            bi = ys*bi'*sin(2pif'ii/xs*xo) - ys*ai'*sin(2pif'ii/xs*xo)
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout = ModelFourier._convert2sines(aout)
        aout = ModelSines.unscaleaf(self, aout, xo=xo, xs=xs, yo=yo, ys=ys)
        aout = ModelSines._convert2fourier(aout)

#        nfreqs = ModelFourier.getnfreqs(ain)
#        tmp = _np.asarray(range(nfreqs))+1
#
#        w  = 2*_np.pi*ain[0]/xs
#        aout[0] = ain[0]/xs
#        aout[1] = yo + ys*ain[1]
#        aout[2::2] = ys*(ain[2::2]*_np.cos(tmp*w*xo) + ain[3::2]*_np.sin(w*tmp*xo))
#        aout[3::2] = ys*(ain[3::2]*_np.sin(tmp*w*xo) - ain[2::2]*_np.sin(w*tmp*xo))
        return aout

    def scaleaf(self, ain, **kwargs):
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout = ModelFourier._convert2sines(aout)
        aout = ModelSines.scaleaf(self, aout, xo=xo, xs=xs, yo=yo, ys=ys)
        aout = ModelSines._convert2fourier(aout)
        return aout

    def unscalecov(self, covin, **kwargs):
        """
        to scale the covariances, simplify the unscaleaf function to ignore offsets
        Use identitites:
         (1)   cov(X, Y) = cov(Y, X)
         (2)   cov(a+X, b+Y) = cov(X,Y)
         (3)   cov(aX, bY) = ab*cov(X,Y)
         (4)   cov(aX+bY, cW+dV) = ac*cov(X,W) + ad*cov(X,V) + bc*cov(Y,W) + bd*cov(Y,V)
         (5)   cov(aX+bY, aX+bY) = a^2*cov(X,X) + b^2*cov(Y,Y) + 2*ab*cov(X,Y)
         (6)   cov(aX, cW+d) = ac*cov(X,W)

        Model:
         y = (yo + ys*ao') + sum_ii(
            (ys*ai'*cos(w'ii*xo) + ys*bi'*sin(w'ii*xo) )* cos(w'ii*x)
          + (ys*bi'*sin(w'ii*xo) - ys*ai'*sin(w'ii*xo) )* sin(w'ii*x) )
                f = f'/xs
                ao = yo + ys*ao'
                ai = ys*ai'*cos(2pif'ii/xs*xo) + ys*bi'*sin(2pif'ii/xs*xo)
                bi = ys*bi'*sin(2pif'ii/xs*xo) - ys*ai'*sin(2pif'ii/xs*xo)

        varfi = varfi'/xs^2
        varao = ys^2*varao'              by (2+3)
        varai = ys^2*cov(ai', ai') + (xo/xs)^2*cov(f',f') + 2ys*xo/xs*cov(ai',f')
              + ys^2*cov(bi', bi') + (xo/xs)^2*cov(f',f') + 2ys*xo/xs*cov(bi',f')
              + ys^2*cov(ai', bi') + ys*xo/xs*cov(ai',f')+ ys*xo/xs*cov(fi',bi') + ys^2*cov(fi',fi')
              = ys^2*varai' + (2(xo/xs)^2+ys^2)*varf' + ys^2*varbi'
               + 3ys*xo/xs*cov(ai',f') +  3ys*xo/xs*cov(bi',f') + ys^2*cov(ai', bi')
        varbi = ys^2*cov(bi', bi') + (xo/xs)^2*cov(f',f') + 2ys*xo/xs*cov(bi',f')
              + ys^2*cov(-ai', -ai') + (xo/xs)^2*cov(f',f') + 2ys*xo/xs*cov(-ai',f')
              + ys^2*cov(bi', -ai') + ys*xo/xs*cov(bi',f')+ ys*xo/xs*cov(fi',-ai') + ys^2*cov(fi',fi')
              = ys^2*varbi' + (2(xo/xs)^2+ys^2)*varf' + ys^2*varai'
              + 3ys*xo/xs*cov(bi',f') - 3ys*xo/xs*cov(ai',f') - ys^2*cov(bi', ai')

        cov(ao, fi) = cov(yo+ys*ao', f'/xs)
                    = ys/xs*cov(ao', f')

        cov(ao,ai) = cov(yo+ys*ao', ys*ai'*cos(2pif'ii/xs*xo) + ys*bi'*sin(2pif'ii/xs*xo))
                = ys^2*cov(ao', ai'*cos(2pif'ii/xs*xo) + bi'*sin(2pif'ii/xs*xo))
                = ys^2*cov(ao', ai'*cos(2pif'ii/xs*xo) )+ ys^2*cov(ao', bi'*sin(2pif'ii/xs*xo) )
                = ys^2*( cov(ao', ai') + 2*xo/xs*cov(ao', f') + cov(ao', bi') )

        cov(ao,bi) = cov(yo+ys*ao', ys*bi'*sin(2pif'ii/xs*xo) - ys*ai'*sin(2pif'ii/xs*xo))
                 = ys^2*cov(ao', bi'*sin(2pif'ii/xs*xo) - ai'*sin(2pif'ii/xs*xo))
                 = ys^2*cov(ao', bi'*sin(2pif'ii/xs*xo)) + ys^2*cov(ao', -ai'*sin(2pif'ii/xs*xo))
                 = ys^2*( cov(ao', bi') + 2*xo/xs*cov(ao', f') - cov(ao', ai') )

        cov(ai,fi) = cov(ys*ai'*cos(2pif'ii/xs*xo) + ys*bi'*sin(2pif'ii/xs*xo), f'/xs)
            = ys/xs*( cov(ai'*cos(2pif'ii/xs*xo), f') + cov(bi'*sin(2pif'ii/xs*xo), f') )
            = ys/xs*( cov(ai', f') + cov(bi', f') + 2*xo/xs*varf' )

        cov(bi,fi) = cov(ys*bi'*sin(2pif'ii/xs*xo) - ys*ai'*sin(2pif'ii/xs*xo), f'/xs)
            = ys/xs*( cov(bi'*sin(2pif'ii/xs*xo) - ai'*sin(2pif'ii/xs*xo), f')
            = ys/xs*( cov(bi', f') - cov(ai', f') + 2*xo/xs*varf' )

        cov(ai, bi) = cov( ys*ai'*cos(2pif'ii/xs*xo) + ys*bi'*sin(2pif'ii/xs*xo),
                           ys*bi'*sin(2pif'ii/xs*xo) - ys*ai'*sin(2pif'ii/xs*xo) )
             = ys^2*cov( ai'*cos(2pif'ii/xs*xo) + bi'*sin(2pif'ii/xs*xo),
                         bi'*sin(2pif'ii/xs*xo) - ai'*sin(2pif'ii/xs*xo) )
             = ys^2*( cov( ai'*f'*xo/xs + bi'*f'*xo/xs,  bi'*f'*xo/xs )
                      cov( ai'*f'*xo/xs + bi'*f'*xo/xs, -ai'*f'*xo/xs ) )
             = (ys*xo/xs)^2*( cov( ai'*f', bi'*f' ) + cov( bi'*f', bi'*f' )
                            - cov( ai'*f', ai'*f' ) - cov( bi'*f', ai'*f' ))
             = (ys*xo/xs)^2*( cov( ai', bi' ) + cov( ai', f' )
                            + cov( f', bi' )+ cov( f', f' )
                            + cov( bi', bi' ) + cov( bi', f' )
                            + cov( f', bi' ) + cov( f', f' )
                            - cov( ai', ai' ) - cov( ai', f' )
                            - cov( f', ai' ) - cov( f', f' )
                            - cov( bi', ai' ) - cov( bi', f' )
                            - cov( f', ai' ) - cov( f', f' ))
             = (ys*xo/xs)^2*( varbi' + 2*cov(bi', f') -2*cov(ai', f') - varai')
        """
        covin, covout = _np.copy(covin), _np.copy(covin)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset) # analysis:ignore   this is not necessary in the covariance
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        # diagonal first
        covout[0, 0] = covin[0, 0]/(xs*xs)  # varf
        covout[1, 1] = covin[1, 1]*(ys*ys)  # varao
        covout[2::2, 2::2] = ((ys*ys)*covin[2::2, 2::2]
               + ((ys*ys)+2.0*power(xo/xs, 2.0))*covin[0, 0]
               + (ys*ys)*covin[3::2, 3::2]
               + 3.0*ys*xo/xs*(_np.atleast_2d(covin[2::2,0]).T*_np.ones(1,len(covout[0,2::2])))
               + 3.0*ys*xo/xs*(_np.atleast_2d(covin[3::2,0]).T*_np.ones(1,len(covout[0,2::2])))
               + (ys*ys)*covin[2::2,3::2])  # varai
        covout[3::2, 3::2] = ((ys*ys)*covin[3::2, 3::2]
               + ((ys*ys) + 2.0*power(xo/xs, 2.0))*covin[0, 0]
               + (ys*ys)*covin[2::2, 2::2]
               + 3.0*ys*xo/xs*(_np.atleast_2d(covin[3::2,0]).T*_np.ones(1,len(covout[0,2::2])))
               - 3.0*ys*xo/xs*(_np.atleast_2d(covin[2::2,0]).T*_np.ones(1,len(covout[0,2::2])))
               - (ys*ys)*covin[3::2, 2::2])

        # Constant offset / frequency covariance
        covout[1,0] = ys/xs*covin[1,0]   # cov ao-fi
        covout[0,1] = covout[1,0]        # cov fi-ao

        # ai-fi
        covout[2::2,0] = (ys/xs)*( covin[2::2,0] + covin[3::2,0] + 2.0*xo/xs*covin[0,0] )
        covout[0,2::2] = covout[2::2,0]

        # bi-fi
        covout[3::2,0] = (ys/xs)*( covin[3::2,0] - covin[2::2,0] + 2.0*xo/xs*covin[0,0] )
        covout[0,3::2] = covout[3::2,0]

        # 2nd row - cov ao-ai
        covout[1,2::2] = (ys*ys)*( covin[1,2::2] + 2.0*xo/xs*covin[1,0] + covin[1,3::2] )
        covout[2::2, 1] = covout[1,2::2]

        # 2nd row - cov ao-bi
        covout[1,3::2] = (ys*ys)*( covin[1,3::2] + 2.0*xo/xs*covin[1,0] - covin[1,2::2] )
        covout[3::2, 1] = covout[1,3::2]

        # mixed coefficient covariance
        covout[2::2,3::2] = power(ys*xo/xs, 2.0)*( covin[3::2,3::2] - covin[2::2,2::2]
            + 2.0*(_np.atleast_2d(covin[3::2,0]).T*_np.ones(1,len(covout[0,2::2])))
            - 2.0*(_np.atleast_2d(covin[2::2,0]).T*_np.ones(1,len(covout[0,2::2]))) )
        return covout

    # ====================================== #
# end def ModelFourier


# ========================================================================== #
# ========================================================================== #


def poly(XX, aa):
    return ModelPoly._model(XX, aa)

def partial_poly(XX, aa):
    return ModelPoly._partial(XX, aa)

def deriv_poly(XX, aa):
    return ModelPoly._deriv(XX, aa)

def partial_deriv_poly(XX, aa):
    return ModelPoly._partial_deriv(XX, aa)

def model_poly(XX=None, af=None, **kwargs):
    return _model(ModelPoly, XX, af, **kwargs)

# =========================================== #


class ModelPoly(ModelClass):
    """
    --- Straight Polynomial ---
    Model - y ~ sum( af(ii)*XX^(polyorder-ii))
    af    - estimate of fitting parameters
    XX    - independent variable
    """
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        if af is not None:
            npoly = _np.size(af)  # Number of fitting parameters
        else:
            npoly = kwargs.setdefault('npoly', 4)
        self._af = _np.random.uniform(low=-5.0, high=5.0, size=npoly)
        self._LB = -_np.inf*_np.ones((npoly,), dtype=_np.float64)
        self._UB = _np.inf*_np.ones((npoly,), dtype=_np.float64)
        self._fixed = _np.zeros( _np.shape(self._LB), dtype=int)
        super(ModelPoly, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        pp = _np.poly1d(aa)
        return pp(XX)

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
         The g-vector contains the partial derivatives used for error propagation
         f = a1*x^2+a2*x+a3
         dfda1 = x^2;
         dfda2 = x;
         dfda3 = 1;
         gvec(0,1:nx) = XX**2;
         gvec(1,1:nx) = XX   ;
         gvec(2,1:nx) = 1;
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)

        gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(num_fit):  # ii=1:num_fit
            gvec[ii, :] = power(XX, ii)
        # endfor
        return gvec[::-1, :]
#        for ii in range(num_fit):  # ii=1:num_fit
#            kk = num_fit - (ii + 1)
#            gvec[ii, :] = XX**kk
#        # endfor
#        return gvec

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        pp = _np.poly1d(aa).deriv()
        return pp(XX)

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
         The jacobian for the derivative
         f = a1*x^2+a2*x+a3
         dfdx = 2*a1*x+a2

         d2fdxda1 = 2*x;
         d2fdxda2 = 1;
         d2fdxda3 = 0;
         dgdx(1,1:nx) = 2*XX;
         dgdx(2,1:nx) = 1.0;
         dgdx(3,1:nx) = 0.0;
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)

        dgdx = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(1, num_fit):
            dgdx[ii,:] = ii*power(XX, ii-1.0)
        # end for
        return dgdx[::-1, :]
#        for ii in range(num_fit-1):
#            kk = (num_fit-1) - (ii + 1)
#            dgdx[ii,:] = (kk+1)*(XX**kk)
#        # end for
#        return dgdx

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        pp = _np.poly1d(aa).deriv().deriv()
        return pp(XX)

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
         The jacobian for the 2nd derivative
         f = a1*x^2+a2*x+a3
         dfdx = 2*a1*x+a2

         d2fdxda1 = 2*x;
         d2fdxda2 = 1;
         d2fdxda3 = 0;
         dgdx(1,1:nx) = 2*XX;
         dgdx(2,1:nx) = 1.0;
         dgdx(3,1:nx) = 0.0;

         d2fdx2 = 2*a1
         d3fdx2da1 = 2.0
         d3fdx2da2 = 0.0
         d3fdx2da3 = 0.0
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)

        d2gdx2 = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(2, num_fit):
            d2gdx2[ii,:] = (ii)*(ii-1.0)*power(XX, ii-2.0)
        # end for
        return d2gdx2[::-1, :]

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def nCr(self, n, r):
        return _np.float64(_ut.factorial(n)/(_ut.factorial(r)*_ut.factorial(n-r)))

    def unscaleaf(self, ain, **kwargs):
        """
        if the data is scaled, then unscaling it goes like this:
        y-scaling: y' = y/ys
            y'= y/ys = sum( a_i'*x^i )
                a_i = ys*a_i'
        y-shifting: y' = (y-yo)/ys
            y'= (y-yo)/ys = sum( a_i'*x^i )
                a_o = ys*a_i'+yo
                a_i = ys*a_i'
        x-scaling: x' = x/xs
            y'= (y-yo)/ys = sum( a_i'*(x/xs)^i )
                a_o = ys*a_i'+yo
                a_i = ys*a_i' / xs^i
        x-shifting: x' = (x-xo)/xs
            y'= (y-yo)/ys = sum( a_i'*((x-xo)/xs)^i )
              = sum( a_i'/xs^i*(x-xo)^i )
             possible but complicated ... requires binomial / multinomial theorem
            y = yo + ys*sum( a_i'/xs^i*sum( (i,k)*x^k*(-xo)^(i-k)) )
                     outer summation up to n=order in x
                    inner summation up to i (current order in x)

                      k=0                   k=1                k=2
             i=0 | yo+ys*ao
             i=1 | -ys*a1*xo/xs        ys*a1/xs*x
                 | ys*a2*xo^2/xs^2  -2*ys*a2*xo*x/xs^2     ys*a2*x^2/xs^2
                 |              ...

            The matrix is sorted in powers of x in the columns
                Rewrite the series for the coefficients
                ak' = ys*sum_i (i,k)*(-xo)^(i-k)*xs^(-i)*ai
                ao' = ao' + yo
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        nn = len(aout)    # maximum order of x in the problem#
        aout = _np.zeros_like(ain)
        for kk in range(nn):
            for ii in range(kk, nn):
                tmp = _np.copy(ain[-(ii+1)])
                tmp *= power(1.0/xs, kk)
                tmp *= self.nCr(ii,kk)
                tmp *= power(-1.0*xo/xs, ii-kk)
                aout[-(kk+1)] += _np.copy(tmp)
            # end for
        # end for
        aout *= ys
        aout[-1] += yo
        return aout

    def scaleaf(self, ain, **kwargs):
        """
        Rewrite the series for the coefficients
            ak' = ys*sum_i (i,k)*(-xo)^(i-k)*xs^(-i)*ai
            ao' = ao' + yo

            ys*ao + yo = sum_i=[0ton](i,0)*xo^(i-0)*xs^i*ai'
            ys*ak = sum_i=[kton](i,k)*xo^(i-k)*xs^i*ai'

            ao = sum_i=[1ton](i,1)*xo^(i-1)*xs^i*ai'/ys + (ao'-yo)/ys
            ai = sum_i=[kton](i,k)*xo^(i-k)*xs^(i)*ai' / ys
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        ain[-1] -= yo
        ain /= ys

        nn = len(aout)    # maximum order of x in the problem#
        aout = _np.zeros_like(ain)
        for kk in range(nn):
            for ii in range(kk, nn):
                tmp = _np.copy(ain[-(ii+1)])
                tmp *= power(xs, kk)
                tmp *= self.nCr(ii,kk)
                tmp *= power(xo, ii-kk)
                aout[-(kk+1)] += _np.copy(tmp)
            # end for
        # end for
        return aout

#    def unscalecov(self, covin, **kwargs):
#        """
#        to scale the covariances, simplify the unscaleaf function to ignore offsets
#        Use identitites:
#         (1)   cov(X, Y) = cov(Y, X)
#         (2)   cov(a+X, b+Y) = cov(X,Y)
#         (3)   cov(aX, bY) = ab*cov(X,Y)
#         (4)   cov(aX+bY, cW+dV) = ac*cov(X,W) + ad*cov(X,V) + bc*cov(Y,W) + bd*cov(Y,V)
#         (5)   cov(aX+bY, aX+bY) = a^2*cov(X,X) + b^2*cov(Y,Y) + 2*ab*cov(X,Y)
#
#        Model:
#            y'= (y-yo)/ys = sum( a_i'*((x-xo)/xs)^i )
#
#            y = yo + ys*sum( a_i'*((x-xo)/xs)^i )
#
#        Rewrite the series for the coefficients
#            ak' = ys*sum_i=[kton](i,k)*(-xo)^(i-k)*xs^(-i)*ai
#            ao' = ao'+yo
#
#        varao = varai at i=0   use (2)
#        varak = cov(ys*sum_i=[kton](i,k)*(-xo)^(i-k)*xs^(-i)*ai, ys*sum_i=[kton](i,k)*(-xo)^(i-k)*xs^(-i)*ai)
#              = ys^2*cov(sum_i=[kton](i,k)*(-xo)^(i-k)*xs^(-i)*ai)   use (5)
#              = ys^2*sum_i=[kton](
#                  ((i,k)*(-xo)^(i-k)*xs^(-i))^2* varai'      variance of i
#                + sum_j=[k'ton',j!=i](  ((j,k')*(-xo)^(j-k')*xs^(-j))^2*varaj')    variance of j not equal to i
#                + 2*(i,k)*(-xo)^(i-k)*xs^(-i)*sum_j=[k'ton',j!=i](  ((j,k')*(-xo)^(j-k')*xs^(-j))^2)*cov(ai',aj')    covariance of i with j (not equal to i)
#                )
#
#        cov(ak,aj) =  use (4)
#              = cov(ys*sum_i1=[kton](i1,k)*(-xo)^(i1-k)*xs^(-i1)*ai1, ys*sum_i2=[jton](i2,j)*(-xo)^(i2-j)*xs^(-i2)*ai2)
#              = ys^2* sum_i1[kton]( sum_i2[jton] (i1,k)*(i2,j)*(-xo)^(i1+i2-k-j)*xs^(-i1-i2)*cov(ai1,ai2) )
#        """
#        covin = _np.copy(covin)
#        covout = _np.copy(covin)
#        ys = kwargs.setdefault('ys', self.slope)
#        yo = kwargs.setdefault('yo', self.offset) # analysis:ignore unnec. for cov
#        xs = kwargs.setdefault('xs', self.xslope)
#        xo = kwargs.setdefault('xo', self.xoffset)
#
#        # Cover entire matrix then overwrite the diagonals
#        nn = _np.size(covout, axis=0)    # maximum order of x in the problem#
#        covout = _np.zeros_like(covin)
#        for kk in range(nn):
#            for jj in range(nn):
#                for ii in range(kk,nn):
#                    for ij in range(jj,nn):
#                        tmp = _np.copy(covin[-(ii+1), -(ij+1)])
#                        tmp *= power(1.0/xs, kk+jj)
#                        tmp *= self.nCr(ii,kk)
#                        tmp *= self.nCr(ij,jj)
#                        tmp *= power(-1.0*xo/xs, ii-kk)
#                        tmp *= power(-1.0*xo/xs, ij-jj)
#                        covout[-(ii+1),-(ij+1)] += _np.copy(tmp)
#                    # end for
#                # end for
#            # end for
#        # end for
#
#sum_i=[kton](
#            ((i,k)*(-xo)^(i-k)*xs^(-i))^2* varai'      variance of i
#                + sum_j=[k'ton',j!=i](  ((j,k')*(-xo)^(j-k')*xs^(-j))^2*varaj')    variance of j not equal to i
#                + 2*(i,k)*(-xo)^(i-k)*xs^(-i)*sum_j=[k'ton',j!=i](  ((j,k')*(-xo)^(j-k')*xs^(-j))^2)*cov(ai',aj')    covariance of i with j (not equal to i)
#                )
#        # overwrite the diagonals
#        for kk in range(nn):
#            for ii in range(kk,nn):
#                tmp = _np.copy(covin[-(ii+1), -(ii+1)])
#                tmp *= power(1.0/xs, kk)
#                tmp *= self.nCr(ii,kk)
#                        tmp *= self.nCr(ij,jj)
#                        tmp *= power(-1.0*xo/xs, ii-kk)
#                        tmp *= power(-1.0*xo/xs, ij-jj)
#                        covout[-(ii+1),-(ij+1)] += _np.copy(tmp)
#
#             for jj in range(nn):
#                for ii in range(kk,nn):
#                    for ij in range(jj,nn):
#                        tmp = _np.copy(covin[-(ii+1), -(ij+1)])
#                        tmp *= power(1.0/xs, kk+jj)
#                        tmp *= self.nCr(ii,kk)
#                        tmp *= self.nCr(ij,jj)
#                        tmp *= power(-1.0*xo/xs, ii-kk)
#                        tmp *= power(-1.0*xo/xs, ij-jj)
#                        covout[-(ii+1),-(ij+1)] += _np.copy(tmp)
#                    # end for
#                # end for
#            # end for
#        # end for
#        covout *= (ys*ys)
#
#        return covout
    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelPoly, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelPoly

# ========================================================================== #
# ========================================================================== #

def prodexp(XX, aa):
    return ModelProdExp._model(XX, aa)

def deriv_prodexp(XX, aa):
    return ModelProdExp._deriv(XX, aa)

def partial_prodexp(XX, aa):
    return ModelProdExp._partial(XX, aa)

def partial_deriv_prodexp(XX, aa):
    return ModelProdExp._partial_deriv(XX, aa)

def model_ProdExp(XX=None, af=None, **kwargs):
    """
    --- Product of Exponentials ---
    Model - y ~ prod(af(ii)*XX^(polyorder-ii))
    af    - estimate of fitting parameters
    XX    - independent variable
        npoly is overruled by the shape of af.  It is only used if af is None

    """
    return _model(ModelProdExp, XX, af, **kwargs)

# ================================== #

class ModelProdExp(ModelClass):
    """
    --- Product of Exponentials ---
    Model - y ~ prod(af(ii)*XX^(polyorder-ii))
    af    - estimate of fitting parameters
    XX    - independent variable
        npoly is overruled by the shape of af.  It is only used if af is None

    """
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        if af is not None:
            npoly = _np.size(af)  # Number of fitting parameters
        else:
            npoly = kwargs.setdefault('npoly', 4)
        self._af = _np.random.uniform(low=-5.0, high=5.0, size=npoly)
        self._LB = -_np.inf*_np.ones_like(self._af)
        self._UB = _np.inf*_np.ones_like(self._af)
        self._fixed = _np.zeros(_np.shape(self._LB), dtype=int)
        super(ModelProdExp, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        Product of exponentials
         y = exp( a0*x^n + a1*x^(n-1) + ... a1*x + a0)
        """
        # Polynomial of order len(af)
        return exp(ModelPoly._model(XX, aa, **kwargs))

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
         The g-vector contains the partial derivatives used for error propagation
         f = exp(a1*x^2+a2*x+a3)
         dfda1 = x^2*f;
         dfda2 = x  *f;
         dfda3 = f;
         gvec(0,1:nx) = XX**2.*prof;
         gvec(1,1:nx) = XX   .*prof;
         gvec(2,1:nx) =        prof;
        """
        nx = _np.size(XX)
        npoly = _np.size(aa)  # Number of fitting parameters
        prof = ModelProdExp._model(XX, aa)

        gvec = _np.zeros((npoly, nx), dtype=_np.float64)
        for ii in range(npoly):
            gvec[ii, :] = power(XX, ii)*prof
#            # Formulated this way, there is an analytic jacobian:
#            kk = num_fit - (ii + 1)
#            gvec[ii, :] = (XX**kk)*prof
        # endif
        gvec = gvec[::-1, :]
        return gvec

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        derivative of product of exponentials w.r.t. x is analytic as well:
         y = exp( a0*x^n + a1*x^(n-1) + ... a1*x + a0)

         f = exp(a1x^n+a2x^(n-1)+...a(n+1))
         f = exp(a1x^n)exp(a2x^(n-1))...exp(a(n+1)));
         dfdx = (n*a1*x^(n-1)+(n-1)*a2*x^(n-2)+...a(n))*f
        """
        return ModelPoly._deriv(XX, aa)*ModelProdExp._model(XX, aa)

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
         y = exp( a0*x^n + a1*x^(n-1) + ... a1*x + a0)

         The g-vector (jacobian) for the derivative
         dfdx = (...+2*a1*x + a2)*exp(...+a1*x^2+a2*x+a3)

         d2fdxdai =poly_partial_deriv*model + poly_deriv*model_partial
        """
        npoly = _np.size(aa)  # Number of fitting parameters

        prof = ModelProdExp._model(XX, aa, **kwargs)
        gvec = ModelProdExp._partial(XX, aa, **kwargs)
        dpdx = ModelPoly._deriv(XX, aa, **kwargs)
        dpda = ModelPoly._partial_deriv(XX, aa, **kwargs)

        return dpda*(_np.ones((npoly,1), dtype=_np.float64)*_np.atleast_2d(prof)) \
          + (_np.ones((npoly,1), dtype=_np.float64)*_np.atleast_2d(dpdx))*gvec

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        second derivative of product of exponentials w.r.t. x is analytic as well:
         y = exp( a0 + a1*x + a2*x^2 + ...)

         f = exp(a1x^n+a2x^(n-1)+...a(n+1))
         f = exp(a1x^n)exp(a2x^(n-1))...exp(a(n+1)));
             using dlnfdx = dfdx / f = > dfdx = dlnfdx*f
         dfdx = (n*a1*x^(n-1)+(n-1)*a2*x^(n-2)+...a(n))*f

         d2fdx2 = (n*a1*x^(n-1)+(n-1)*a2*x^(n-2)+...a(n))*dfdx
           + d/dx(n*a1*x^(n-1)+(n-1)*a2*x^(n-2)+...a(n))*f

         d2fdx2 = (dfdx)^2/f + f*d2poly/dx2

         d2fdxdai =poly_partial_deriv*model + poly_deriv*model_partial
        """
        dfdx = ModelProdExp._deriv(XX,aa)
        return (dfdx*dfdx/ModelProdExp._model(XX,aa)
                + ModelPoly._deriv2(XX, aa)*ModelProdExp._model(XX, aa))

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        second derivative of product of exponentials w.r.t. x is analytic as well:
         y = exp( a0 + a1*x + a2*x^2 + ...)

         f = exp(a1x^n+a2x^(n-1)+...a(n+1))
         f = exp(a1x^n)exp(a2x^(n-1))...exp(a(n+1)));
             using dlnfdx = dfdx / f = > dfdx = dlnfdx*f
         dfdx = (n*a1*x^(n-1)+(n-1)*a2*x^(n-2)+...a(n))*f

         d2fdx2 = (n*a1*x^(n-1)+(n-1)*a2*x^(n-2)+...a(n))*dfdx
           + d/dx(n*a1*x^(n-1)+(n-1)*a2*x^(n-2)+...a(n))*f

         d2fdx2 = (dfdx)^2/f + f*d2poly/dx2

         d3fdx2da = 2*dfdx*d2fdxda/f - (dfdx)^2*dfda/f^2
                 + dfda*d2poly/dx2 + f*d3poly/dx2da
                 = (dfdx/f)*(2*d2fdxda - dfdx*dfda/f)
                 + dfda*d2poly/dx2 + f*d3poly/dx2da
        """
#        d2fdx2 = power(ModelProdExp._deriv(XX,aa), 2.0)/ModelProdExp._model(XX,aa) \
#                      + ModelPoly._deriv2(XX, aa)*ModelProdExp._model(XX, aa)

        npoly = _np.size(aa)  # Number of fitting parameters
        tmp = _np.ones((npoly,1), dtype=_np.float64)

        prof = ModelProdExp._model(XX, aa)
        dprofdx = ModelProdExp._deriv(XX, aa)
        gvec = ModelProdExp._partial(XX, aa)
        dgdx = ModelProdExp._partial_deriv(XX, aa)

        d2gdx2 = ((tmp*_np.atleast_2d(dprofdx/prof))*(2.0*dgdx  - (tmp*_np.atleast_2d(dprofdx/prof))*gvec)
               + gvec*(tmp*_np.atleast_2d(ModelPoly._deriv2(XX, aa)))
               + (tmp*_np.atleast_2d(prof))*ModelPoly._partial_deriv2(XX, aa))
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain, **kwargs):
        """
        y-scaling: y' = y/ys
            y'= y/ys = exp( sum( a_i'*x^i ) ) = prod( exp( a_i'*x^i ) )
                ln(y) = ln(ys) + sum(a_i'*x^i)
                a_o = ln(ys) + a_o'
                a_i = a_i'

        y-shifting: y' = (y-yo)/ys
            y'= (y-yo)/ys = exp( sum( a_i'*x^i ) ) = prod( exp( a_i'*x^i ) )
                ln(y-yo) = ln(ys) + sum(a_i'*x^i)
               and ln(y) = sum(a_i * x^i)
                    not be possible with constant coefficients

            translate the coefficients into a polynomial model, then unshift there

        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)   # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        PM = ModelPoly(None)
#        PM.offset = _np.log(_np.abs(ys))
        PM.offset = log(ys)
        PM.slope = 1.0
        PM.xoffset = xo
        PM.xslope = xs
        aout = PM.unscaleaf(aout)
        return aout

    def scaleaf(self, ain, **kwargs):
        """
            translate the coefficients into a polynomial model, then unshift there
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)   # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        PM = ModelPoly(None)
#        PM.offset = _np.log(_np.abs(ys))
        PM.offset = log(ys)
        PM.slope = 1.0
        PM.xoffset = xo
        PM.xslope = xs
        aout = PM.scaleaf(aout)
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
#        self.slope = 1.0
#        self.xoffset = 0.0
#        self.xslope = 1.0
        return super(ModelProdExp, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelProdExp, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelProdExp

# ========================================================================== #
# ========================================================================== #

def evenpoly(XX, aa):
    return ModelEvenPoly._model(XX, aa)

def deriv_evenpoly(XX, aa):
    return ModelEvenPoly._deriv(XX, aa)

def partial_evenpoly(XX, aa):
    return ModelEvenPoly._partial(XX, aa)

def partial_deriv_evenpoly(XX, aa):
    return ModelEvenPoly._partial_deriv(XX, aa)

def model_evenpoly(XX=None, af=None, **kwargs):
    return _model(ModelEvenPoly, XX, af, **kwargs)

# =========================================== #


class ModelEvenPoly(ModelClass):
    """
    --- Polynomial with only even powers ---
    Model - y ~ sum( af(ii)*XX^2*(numfit-ii))
    af    - estimate of fitting parameters (npoly=4, numfit=3, poly= a0*x^4+a1*x^2+a3)
    XX    - independent variable

    I do this by fitting a regular polynomial in sqrt(XX) space
    """
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        if af is not None:
            npoly = _np.size(af)  # Number of fitting parameters
        else:
            npoly = kwargs.setdefault('npoly', 3)
        self._af = _np.random.uniform(low=-5.0, high=5.0, size=npoly)
        self._LB = -_np.inf*_np.ones((npoly,), dtype=_np.float64)
        self._UB = _np.inf*_np.ones((npoly,), dtype=_np.float64)
        self._fixed = _np.zeros( _np.shape(self._LB), dtype=int)
        super(ModelEvenPoly, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        Even Polynomial of order num_fit, Insert zeros for the odd powers
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)
        prof = _np.zeros((nx,), dtype=_np.float64)
        for ii in range(num_fit):
            prof += aa[-(ii+1)]*power(XX, 2.0*ii)
        # end for
        return prof

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
         The g-vector contains the partial derivatives used for error propagation
         f = a1*x^4+a2*x^2+a3
         dfda1 = x^4;
         dfda2 = x^2;
         dfda3 = 1;
         gvec(1,1:nx) = XX**4;
         gvec(2,1:nx) = XX**2;
         gvec(3,1:nx) = 1;
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)
        gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(num_fit):
            gvec[ii, :] = power(XX, 2.0*ii)
        # end for
        return gvec[::-1, :]

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        Derivative of an even polynomial
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)
        dfdx = _np.zeros((nx,), dtype=_np.float64)
        for ii in range(1, num_fit):
            dfdx += (2.0*ii)*aa[-(ii+1)]*power(XX, 2.0*ii-1.0)
        # end for
        return dfdx

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
         The jacobian for the derivative
         f = a1*x^4+a2*x^2+a3
         dfdx = 4*a1*x^3 + 2*a2*x + 0
         dfdxda1 = 4*x^3;
         dfda2 = 2*x^1;
         dfda3 = 0;
         dgdx(1,1:nx) = 4*XX**3;
         dgdx(2,1:nx) = 2*XX;
         dgdx(3,1:nx) = 0.0;
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)
        dgdx = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(1, num_fit):
            dgdx[ii, :] = (2.0*ii)*power(XX, 2.0*ii-1.0)
        # end for
        return dgdx[::-1, :]

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        2nd Derivative of an even polynomial

        f = a1*x^4+a2*x^2+a3
        dfdx = 4*a1*x^3 + 2*a2*x + 0
        dfdxda1 = 4*x^3;
        dfda2 = 2*x^1;
        dfda3 = 0;
        dgdx(1,1:nx) = 4*XX**3;
        dgdx(2,1:nx) = 2*XX;
        dgdx(3,1:nx) = 0.0;

        d2fdx2 = 4*3*a1*x^2 + 2*a2 + 0
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)
        d2fdx2 = _np.zeros((nx,), dtype=_np.float64)
        for ii in range(1,num_fit):
            d2fdx2 += (2.0*ii)*(2.0*ii-1.0)*aa[-(ii+1)]*power(XX, 2.0*ii-2.0)
        # end for
        return d2fdx2

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        Jacobian of the 2nd derivative

        f = a1*x^4+a2*x^2+a3
        dfdx = 4*a1*x^3 + 2*a2*x + 0
        dfdxda1 = 4*x^3;
        dfda2 = 2*x^1;
        dfda3 = 0;
        dgdx(1,1:nx) = 4*XX**3;
        dgdx(2,1:nx) = 2*1*XX**1;
        dgdx(3,1:nx) = 0.0;

        d2fdx2 = 4*3*a1*x^2 + 2*a2 + 0
        d3fdx2da1 = 4*3*x^2
        d3fdx2da2 = 2*1*x^0
        d3fdx2da3 = 0.0
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)
        d2gdx2 = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(1,num_fit):
            d2gdx2[ii, :] = (2.0*ii)*(2.0*ii-1.0)*power(XX, 2.0*ii-2.0)
        # end for
        return d2gdx2[::-1, :]

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain, **kwargs):
        """
        if the data is scaled, then unscaling it goes like this:
        y-scaling: y' = y/ys
            y'= y/ys = sum( a_i'*x^i )
                a_i = ys*a_i'
        y-shifting: y' = (y-yo)/ys
            y'= (y-yo)/ys = sum( a_i'*x^i )
                a_o = ys*a_i'+yo
                a_i = ys*a_i'
        x-scaling: x' = x/xs
            y'= (y-yo)/ys = sum( a_i'*(x/xs)^i )
                a_o = ys*a_i'+yo
                a_i = ys*a_i' / xs^i

        x-shifting: x' = (x-xo)/xs
            possible but complicated ... requires binomial / multinomial theorem
            translate into polynomial model, unscale there, then come back
            ... but then we will end up with odd terms in the expansion
            ... this is reflective of the trick used above
            ( fitting x^2 instead of x to generate an even polynomial in x)
            ( because (x-xo)^2 has an odd term as well:  -2x*xo)


        In the unscaling, we could reset the model that is used internally to
        be that of the polynomial model until unscaled.  Really, this is an
        unnecessary step because the only reason I can think of to use an even
        polynomial is to maintain 'evenness' across the origin
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout = ys*ain/_np.flipud(power(xs, 2.0*_np.asarray(range(len(aout)))))
        aout[-1] += yo
#        nparams = len(aout)
#        if nparams>1:
#            aout = _np.zeros((2*len(ain),), dtype=_np.float64)
#            aout[0::2] = _np.copy(ain)
#            PM = ModelPoly(None)
#            PM.slope = ys
#            PM.offset = yo
#            PM.xslope = xs
#            PM.xoffset = 0.0
#            aout = PM.unscaleaf(aout)
#            aout = aout[0::2]
        # end if
        return aout

    def scaleaf(self, ain, **kwargs):
        """
        unshift and scale as above
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[-1] -= yo
        aout = aout*_np.flipud(power(xs, 2.0*_np.asarray(range(len(aout)))))
        aout = aout/ys
#        if len(ain)>1:
#            aout = _np.zeros((2*len(ain),), dtype=_np.float64)
#            aout[0::2] = _np.copy(ain)
#            PM = ModelPoly(None)
#            PM.slope = ys
#            PM.offset = yo
#            PM.xslope = xs
#            PM.xoffset = 0.0
#            aout = PM.scaleaf(aout)
#            aout = aout[0::2]
#        else:
#            aout[0] = (ain[0]-yo)/ys
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xoffset = 0.0
#        self.xslope = 1.0
        return super(ModelEvenPoly, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelEvenPoly, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelEvenPoly

# ========================================================================== #
# ========================================================================== #


def powerlaw(XX, aa):
    return ModelPowerLaw._model(XX, aa)

def deriv_powerlaw(XX, aa):
    return ModelPowerLaw._deriv(XX, aa)

def partial_powerlaw(XX, aa):
    return ModelPowerLaw._partial(XX, aa)

def partial_deriv_powerlaw(XX, aa):
    return ModelPowerLaw._partial_deriv(XX, aa)

def model_PowerLaw(XX=None, af=None, **kwargs):
    return _model(ModelPowerLaw, XX, af, **kwargs)


# =========================================== #


class ModelPowerLaw(ModelClass):
    """
    --- Power Law w/exponential cut-off ---
    Model - fc = x^( a1*x^(n)+a2*x^(n-1)+...a(n) )
             y = a(n+2)*fc*exp(a(n+1)*x)
    af    - estimate of fitting parameters
    XX    - independent variable

    y-shift:  y'=y/ys
       y = ys*a(n+2) * exp(a(n+1)*x) * x^( a1*x^(n)+a2*x^(n-1)+...a(n) )
    """
    _analytic_xscaling = False
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        if af is not None:
            num_fit = _np.size(af)  # Number of fitting parameters
            npoly = _np.int(num_fit-2)  # Polynomial order from input af
        else:
            npoly = kwargs.setdefault('npoly', 4)
        self._af = 0.1*_np.ones((npoly+2,), dtype=_np.float64)
        self._LB = -_np.inf*_np.ones((npoly+2,), dtype=_np.float64)
        self._UB = _np.inf*_np.ones((npoly+2,), dtype=_np.float64)
        self._fixed = _np.zeros( _np.shape(self._LB), dtype=int)
        super(ModelPowerLaw, self).__init__(XX, af, **kwargs)

        # for numerical derivatives, use a kwargs switch to tell the caller
        # that the model is strictly only valid for x>0
        self.nonnegative = True
    # end def __init__

#    def leftboundary(self, XX, aa, **kwargs):
#        """
#        Create boundary conditions for derivatives at the
#        left boundary.  Note that derivatives of powers laws
#        do not behave well at 0
#        """
#
#        return

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
         Curved power-law:
         fc = x^(a1*x^(n+1)+a2*x^n+...a(n+1))

         With exponential cut-off:
         f  = a(n+2)*exp(a(n+1)*XX)*fc(x);
        """
#        XX = _np.abs(XX)
        polys = ModelPoly._model(XX, aa[:-2])
        exp_factor = exp(aa[-2]*XX)
        return aa[-1]*exp_factor*power(XX, polys)

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
         dfdx = dfcdx*(a(n+2)*e^a(n+1)x) +a(n+1)*f(x);
              = (dfcdx/fc)*f(x) + a(n+1)*f(x)
              =  (dln(fc)dx + a(n+1)) * f(x)

            fc = x^(a1*x^(n+1)+a2*x^n+...a(n+1))
                 dfcdx/fc = d ln(fc)/dx = d/dx poly*ln(x)
                 d ln(fc)/dx = dpolydx*ln(x)+poly/x
                 (dfcdx = fc*( dpolydx*x+poly ))

         dfdx = (  dpolydx*ln(x)+poly/x + a(n+1)) * f(x)

         dfcdx = XX^(-1)*prof*(polys+ddx(poly)*XX*log(XX),
         log is natural logarithm
        """
#        XX = _np.abs(XX)
        prof = ModelPowerLaw._model(XX, aa, **kwargs)
        dlnfdx = ModelPowerLaw._lnderiv(XX, aa, **kwargs)
        return prof*dlnfdx

    @staticmethod
    def _lnderiv(XX, aa, **kwargs):
        """
         dfdx = dfcdx*(a(n+2)*e^a(n+1)x) +a(n+1)*f(x);
              = (dfcdx/fc)*f(x) + a(n+1)*f(x)
              =  (dln(fc)dx + a(n+1)) * f(x)

            fc = x^(a1*x^(n+1)+a2*x^n+...a(n+1))
                 dfcdx/fc = d ln(fc)/dx = d/dx poly*ln(x)
                 d ln(fc)/dx = dpolydx*ln(x)+poly/x
                 (dfcdx = fc*( dpolydx*x+poly ))

         dfdx = (  dpolydx*ln(x)+poly/x + a(n+1)) * f(x)

         dfcdx = XX^(-1)*prof*(polys+ddx(poly)*XX*log(XX),
         log is natural logarithm
        """
#        XX = _np.abs(XX)
        polys = ModelPoly._model(XX, aa[:-2])
        dpolys = ModelPoly._deriv(XX, aa[:-2])
        return ( aa[-1] + polys/XX + _np.log(_np.abs(XX))*dpolys )

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
         dfdx = dfcdx*(a(n+2)*e^a(n+1)x) +a(n+1)*f(x);
              = (dfcdx/fc)*f(x) + a(n+1)*f(x)
              =  (dln(fc)dx + a(n+1)) * f(x)

            fc = x^(a1*x^(n+1)+a2*x^n+...a(n+1))
                 dfcdx/fc = d ln(fc)/dx = d/dx poly*ln(x)
                 d ln(fc)/dx = dpolydx*ln(x)+poly/x
                 (dfcdx = fc*( dpolydx*x+poly ))

         dfdx = (  dpolydx*ln(x)+poly/x + a(n+1)) * f(x)

         dfcdx = XX^(-1)*prof*(polys+ddx(poly)*XX*log(XX),
         log is natural logarithm

        d2fdx2 = dfdx*(dfdx/f) + f*( dpoly2dx2*ln|x| + dpolydx/x
                                    + dpolydx/x - poly/x^2 )
               = dfdx*(dfdx/f) + f*( dpoly2dx2*ln|x| + 2*dpolydx/x - poly/x^2 )
               = dfdx*dlnfdx + f*( dpoly2dx2*ln|x| + 2*dpolydx/x - poly/x^2 )

            Using dfdx = f*dlnfdx
        """
#        XX = _np.abs(XX)
        prof = ModelPowerLaw._model(XX, aa, **kwargs)
        dlnfdx = ModelPowerLaw._lnderiv(XX, aa, **kwargs)
        dprofdx = ModelPowerLaw._deriv(XX, aa, **kwargs)

        polys = ModelPoly._model(XX, aa[:-2], **kwargs)
        dpolys = ModelPoly._deriv(XX, aa[:-2], **kwargs)
        d2poly = ModelPoly._deriv2(XX, aa[:-2], **kwargs)
        return dprofdx*dlnfdx + prof*( d2poly*_np.log(_np.abs(XX))
                + 2.0*dpolys/XX - polys/(XX*XX) )

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
         The g-vector contains the partial derivatives used for error propagation
         f = a(n+2) * exp(a(n+1)*x) * x^( a1*x^(n)+a2*x^(n-1)+...a(n) )
         f = a(n+2) * exp(a(n+1)*XX) * fc(x)

         dfda_1ton = a(n+2) * exp(a(n+1)*XX) * dfc/da_n
                   = dfc/da_n * (f/fc) = dlnfc/da_n * f
                   = f * d/da_n poly*ln(x) = f*ln(x) * d/da_n poly
         dfda_n = x^n*ln(x)*f
         dfda_n+1 = XX*f
         dfda_n+2 = f/a_n+2

         gvec(0,1:nx) = XX**(n)*_np.log(_np.abs(XX))*prof
         gvec(1,1:nx) = XX**(n-1)*_np.log(_np.abs(XX))*prof
         gvec(2,1:nx) = XX**(n-2)*_np.log(_np.abs(XX))*prof
         ...
         gvec(-3,1:nx) = XX**(0)*_np.log(_np.abs(XX))*prof
         gvec(-2, 1:nx) = XX*prof
         gvec(-1, 1:nx) = prof/a[-1]
        """
#        XX = _np.abs(XX)
        nx = _np.size(XX)
        num_fit = _np.size(aa)  # Number of fitting parameters

        gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(2, num_fit):
            gvec[ii, :] = power(XX, ii-2.0)
        # end for
        gvec = gvec[::-1, :]

        prof = ModelPowerLaw._model(XX, aa, **kwargs)
        gvec *= _np.log(_np.abs(XX)) * prof
        gvec[-2, :] = prof*XX
        gvec[-1, :] = prof/aa[-1]
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
         The jacobian of the derivative
            dfda_i = d/da_i dfdx

         f = a(n+2) * exp(a(n+1)*x) * x^( a1*x^(n)+a2*x^(n-1)+...a(n) )

         f = a(n+2) * exp(a(n+1)*XX) * fc(x)

         dfdx = (  dpolydx*ln(x)+poly/x + a(n+1)) * f(x)
              = dpolydx*ln(x)*f(x)  +  poly/x * f(x)  +  a(n+1) * f(x)

        d/da dfdx = d/da( dpolydx*ln(x)+poly/x + a(n+1)) * f(x)
                      + ( dpolydx*ln(x)+poly/x + a(n+1)) * df/da
                                      where dfda_n = x^n*ln(x)*f
                                      and dlnfda_n = x^n*ln(x)

        d/da dfdx = f(x) * [ d/da( dpolydx*ln(x)+poly/x + a(n+1))
                           + ( dpolydx*ln(x)+poly/x + a(n+1)) * dlnf/da ]
                  = f(x) * d/da( dpolydx*ln(x)+poly/x + a(n+1))
                     + dfdx * dlnfda

        d/da dfdx = f(x) * d/da( dpolydx*ln(x)+poly/x + a(n+1)) + dfdx*x^n*ln(x)
                                  (3)          (2)      (1)       (0)
         df/dadx_i=0toN
          0) df/dadx_i  = dfdx * x^n*ln(x) +
          1) df/dadx_i  = f(x) * d/da(a[n+1]) = 0 as the sum only goes up to n +
          2) df/dadx_i  = f(x) * x^(n-1)
          3) df/dadx_i  = f(x) * n*ln(x)*x^(n-1)
        df/dadx_-2  = dfdx * x
        df/dadx_-1  = dfdx / a(-1)

    generator:
         gvec(ii, 1:nx) = XX**(n-ii)*_np.log(_np.abs(XX))*dprofdx + (1.0+(n-ii)*_np.log(_np.abs(XX)))*XX**(n-ii-1)*prof

         gvec(0,1:nx) = XX**(n-0)*_np.log(_np.abs(XX))*dprofdx  + 0
                      + XX**(n-1)*prof + n*_np.log(_np.abs(XX))*XX**(n-1)*prof
                      = XX**(n-0)*_np.log(_np.abs(XX))*dprofdx  + (1+(n-0)*_np.log(_np.abs(XX)))*XX**(n-0-1)*prof

         gvec(1,1:nx) = XX**(n-1)*_np.log(_np.abs(XX))*dprofdx + 0
                      + XX**(n-2)*prof + (n-1)*_np.log(_np.abs(XX))*XX**(n-2)*prof
                      = XX**(n-1)*_np.log(_np.abs(XX))*dprofdx + (1+(n-1)*_np.log(_np.abs(XX)))*XX**(n-2)*prof
         ...
         gvec(n,1:nx) = XX**(0)*_np.log(_np.abs(XX))*dprofdx + (1+(0)*_np.log(_np.abs(XX)))*XX**(-1)*prof
                      = _np.log(_np.abs(XX))*dprofdx + prof/XX
         gvec(-2, 1:nx) = XX*dprofdx
         gvec(-1, 1:nx) = dprofdx/a[-1]
        """
#        XX = _np.abs(XX)
        nx = _np.size(XX)
        num_fit = _np.size(aa)  # Number of fitting parameters
#        npoly = len(aa[:-2])    #    npoly = num_fit-3?

        prof = ModelPowerLaw._model(XX, aa)
        dprofdx = ModelPowerLaw._deriv(XX, aa)

        dgdx = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(2, num_fit):
            dgdx[ii, :] = (ii-2.0)*power(XX, ii-3.0)*_np.log(_np.abs(XX)) * prof
            dgdx[ii, :] += power(XX, ii-2.0)*_np.log(_np.abs(XX)) * dprofdx
            dgdx[ii, :] += power(XX, ii-2.0)*prof/XX
#            dgdx[ii, :] = (power(XX, ii)*_np.log(_np.abs(XX))*dprofdx
#                              + (1.0+ii*_np.log(_np.abs(XX)))*power(XX, ii-1.0)*prof )
        # end for
        dgdx = dgdx[::-1, :]
        dgdx[-2, :] = XX*dprofdx + prof
        dgdx[-1, :] = dprofdx/aa[-1]
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        d2fdx2 = dfdx*(dfdx/f) + f*( dpoly2dx2*ln|x| + 2*dpolydx/x - poly/x^2 )

        d3fdx2da = 2.0*d2fdxda*(dfdx/f) - (dfdx)^2*dfda/f^2
                + dfda*( dpoly2dx2*ln|x| + 2*dpolydx/x - poly/x^2 )
                + f*( d3polydx2da*ln|x| + 2*d2polydxda/x - dpolyda/x^2 )
        d2gdx2  = dfdx/f * (2*dgdx - (dfdx/f)*gvec)
                + gvec*( poly_deriv2*ln|x| + 2*poly_deriv/x - polys/x^2)
                + f*( poly_partial_deriv2*ln|x| + 2*poly_partial_deriv/x
                      - poly_partial/x^2)

        d2gdx2 = d/da d2fdx2
               = 2*dfdx*d2fdxda/f - dfdx^2*dfda/f^2
               + dfda*( d2polydx2*ln|x| + 2*dpolydx/x - poly/x^2 )
               +    f*( d3polydx2da*ln|x| + 2*d2polydxda/x - dpolyda/x^2 )
               = 2*dlnfdx*d2fdxda
               + dfda*( d2polydx2*ln|x| + 2*dpolydx/x - poly/x^2 - dlnfdx^2)
               +    f*( d3polydx2da*ln|x| + 2*d2polydxda/x - dpolyda/x^2 )
       """
#        XX = _np.abs(XX)
        nx = _np.size(XX)
        na = _np.size(aa)
        f = ModelPowerLaw._model(XX, aa)
        dfdx = ModelPowerLaw._deriv(XX, aa)
        gvec = ModelPowerLaw._partial(XX, aa)
        dgdx = ModelPowerLaw._partial_deriv(XX, aa)

        polys = ModelPoly._model(XX, aa[:-2])
        poly_partial = ModelPoly._partial(XX, aa[:-2])
        poly_deriv = ModelPoly._deriv(XX, aa[:-2])
        poly_partial_deriv = ModelPoly._partial_deriv(XX, aa[:-2])
        poly_deriv2 = ModelPoly._deriv2(XX, aa[:-2])
        poly_partial_deriv2 = ModelPoly._partial_deriv2(XX, aa[:-2])

        tmp = _np.ones((na,1), dtype=_np.float64)
        dlnfdx = _np.atleast_2d(dfdx/f)
        poly_partial = _np.concatenate((poly_partial, _np.zeros((2,nx), dtype=_np.float64)), axis=0)
        poly_partial_deriv = _np.concatenate((poly_partial_deriv, _np.zeros((2,nx), dtype=_np.float64)), axis=0)
        poly_partial_deriv2 = _np.concatenate((poly_partial_deriv2, _np.zeros((2,nx), dtype=_np.float64)), axis=0)

        d2gdx2 = ( (2.0*tmp*dlnfdx)*dgdx + gvec*(tmp*_np.atleast_2d(
               poly_deriv2*_np.log(_np.abs(XX)) + 2.0*poly_deriv/XX - polys/(XX*XX) - (dlnfdx*dlnfdx)))
               + (tmp*_np.atleast_2d(f/(XX*XX)))*(
                  poly_partial_deriv2*(tmp*_np.atleast_2d((XX*XX)*_np.log(_np.abs(XX))))
                        + (2.0*tmp*XX)*poly_partial_deriv - poly_partial) )
        return d2gdx2
#        d2gdx2 = (tmp*dlnfdx)*(2.0*dgdx - (tmp*dlnfdx)*gvec)
#        d2gdx2 += gvec*(tmp*_np.atleast_2d(
#                       poly_deriv2*_np.log(_np.abs(XX)) + 2.0*poly_deriv/XX - polys/power(XX,2.0)
#                       ))
#        d2gdx2 += (tmp*_np.atleast_2d(f))*(
#                poly_partial_deriv2*(tmp*_np.atleast_2d(_np.log(_np.abs(XX))))
#              + 2.0*poly_partial_deriv*(tmp*_np.atleast_2d(1.0/XX))
#              - poly_partial*(tmp*_np.atleast_2d(1.0/power(XX, 2.0)))
#                )
#        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain, **kwargs):
        """
        f = a(n+2) * exp(a(n+1)*x) * x^( a1*x^(n)+a2*x^(n-1)+...a(n) )

        f = a(n+2) * exp(a(n+1)*XX) * fc(x)

        y-scaling:   y' = y/ys
         y = ys*an2 * exp(an1*x) * x^( a1*x^(n)+a2*x^(n-1)+...a(n) )
         a(n+2) = ys* an2'

        y-shifting:  y' = (y-yo)/ys
         y = yo + ys*an2 * exp(an1*x) * x^( a1*x^(n)+a2*x^(n-1)+...a(n) )
        ========>  Not possible with non-linearity

        x-scaling:   x' = x/ys
         y = ys*an2 * exp(an1*x/xs) * (x/xs)^( a1*(x/xs)^(n)+a2*(x/xs)^(n-1)+...a(n) )
         ln|y|  = an1*x/xs + ln|ys*an2|
                  + ln|(x/xs)^( a1*(x/xs)^(n)+a2*(x/xs)^(n-1)+...a(n) )|
                = an1*x/xs + ln|ys*an2|
                  + ln|x|*( a1*(x/xs)^(n)+a2*(x/xs)^(n-1)+...a(n) )
                  - ln|xs|*( a1*(x/xs)^(n)+a2*(x/xs)^(n-1)+...a(n) )

            ... this is a polynomial ...
            therefore we can shift it in log-space
            ... gathering terms will be a pain ... leave this for later

        ============

         a(n+2) = ys* an2'
         a(n+1) = an1'/xs
         a(n) = an' - an1'*xo/xs
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)   # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[-1] *= ys
        aout[:-2] /= xs
#        aout[:-3] = ModelPoly.unscaleaf(ain[:-3], xo=xo, xs=xs, ys=ys, yo=yo)
        return aout

    def scaleaf(self, ain, **kwargs):
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)   # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)   # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[-1] /= ys
#        aout[:-2] *= xs
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
        self.xoffset = 0.0
        self.xslope = 1.0
        return super(ModelPowerLaw, self).scalings(xdat, ydat, **kwargs)
    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelPowerLaw, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelPowerLaw

# ========================================================================== #
# ========================================================================== #


def parabolic(XX, aa):
    return ModelParabolic._model(XX, aa)

def deriv_parabolic(XX, aa):
    return ModelParabolic._deriv(XX, aa)

def partial_parabolic(XX, aa):
    return ModelParabolic._partial(XX, aa)

def partial_deriv_parabolic(XX, aa):
    return ModelParabolic._partial_deriv(XX, aa)

def model_parabolic(XX=None, af=None, **kwargs):
    return _model(ModelParabolic, XX, af, **kwargs)

# =========================================== #


class ModelParabolic(ModelClass):
    """
    A parabolic profile with one free parameters:
        f(x) ~ a*(1.0-x^2)
        XX - x - independent variable
        af - a - central value of the plasma parameter

    """
    _af = _np.asarray([1.0], dtype=_np.float64)
    _LB = _np.asarray([1e-18], dtype=_np.float64)
    _UB = _np.asarray([_np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (1,), dtype=int)
    _analytic_xscaling = False
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelParabolic, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        return aa*(1.0 - XX*XX)

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        return -2.0*aa*XX

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        return -2.0*aa*_np.ones_like(XX)

    @staticmethod
    def _partial(XX, aa, **kwargs):
        return _np.atleast_2d(ModelParabolic._model(XX, aa) / aa)

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        return _np.atleast_2d(-2.0*XX)

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        return _np.atleast_2d(-2.0*_np.ones_like(XX))

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain, **kwargs):
        """
        y = a*(1.0-x^2)

        y-scaling: y'=y/ys
            a = ys*a'
        y-shifting: y'=(y-yo)/ys
            y = yo + ys*a'*(1.0-x^2) = yo + ys*a' - ys*a'x^2
                Not possible with constant coefficients
        x-scaling: x'=x/xs
            y = ys*a'*(1.0-(x/xs)^2)
              = ys*a'/xs^2 * (xs^2 - x^2)
              = ys*a'/xs^2 * (xs^2 - x^2) + ys*a'/xs^2 - ys*a'/xs^2
              = ys*a'/xs^2 * (1.0 - x^2) + ys*a'*(1.0-xs^-2)

                Not possible with constant coefficients unless
                we fix yo = -ys*a'*(1.0-xs^-2)

            Then we cannot use the original model for anything
        x-shifting: x'=(x-xo)/xs
            y = ys*a'*(1.0-(x-xo)^2/xs^2)
              = ys*a'/xs^2*(xs^2-x^2-2xo*x+xo^2)
              = ys*a'/xs^2*(xs^2-x^2-2xo*x+xo^2) + ys*a'/xs^2 - ys*a'/xs^2
              = ys*a'/xs^2*(1.0-x^2) - ys*a'*(1.0-(x-xo)^2 - xs^-2)
              Not possible with constant coefficients
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)  # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore
        return aout*ys

    def scaleaf(self, ain, **kwargs):
        """
        y = a*(1.0-x^2)
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)  # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore
        return aout/ys

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
        self.xslope = 1.0
        self.xoffset = 0.0
        return super(ModelParabolic, self).scalings(xdat, ydat, **kwargs)


    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelParabolic, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelParabolic


# ========================================================================== #

# ========================================================================== #


def _exp(XX, aa):
    return ModelExp._model(XX, aa)

def deriv_exp(XX, aa):
    return ModelExp._deriv(XX, aa)

def partial_exp(XX, aa):
    return ModelExp._partial(XX, aa)

def partial_deriv_exp(XX, aa):
    return ModelExp._partial_deriv(XX, aa)

def model_Exp(XX=None, af=None, **kwargs):
    return _model(ModelExp, XX, af, **kwargs)

# =========================================== #


class ModelExp(ModelClass):
    """
    --- Exponential ---
    Model - y = a*exp(b*XX^c)

    af    - estimate of fitting parameters [a, b, c]
    XX    - independent variables
        valid for x>0
        at x = 0


    """
#    _af = _np.asarray([1.0, -1.0/0.4, 1.0], dtype=_np.float64)
    _af = _np.asarray([1.0, -1.0/0.4, 2.5], dtype=_np.float64)
    _LB = _np.array([1e-18, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelExp, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        Model - f = a*exp(b*XX^c)

        if x is complex
        """
        a, b, c = tuple(aa)
        if _np.iscomplex(XX).any():
            return a*exp(b*power(XX, c, real=False), real=False)
        else:
            return a*exp(b* power(XX, c))
#            return a*_np.exp(b* XX**c)
        # end if

#    @staticmethod
#    def xlogxtoc(XX, c=1.0):
#        """
#        Use L'Hospital's rule to get the result as x goes to 0+
#         x*ln(x) = ln(x)/(1/x) limit is (1/x) / -(x^-2) = lim -x = 0
#
#         x^c*ln(x^c) = ln(x^c)/(1/x^c)
#         limit is (c*x^(c-1)/x^c)) / -(x^-2) = -c/x * x^2  lim -c*x = 0
#        """
#        xt = ModelExp.xtoc(XX, c=c)
#        logxt = ModelExp.logxtoc(XX, c=c)
#        xtlogxt = xt*logxt
#        if _np.isnan(xtlogxt).any() and (xt==0).any():
#            xtlogxt[xt==0] = 0.0
#        return xtlogxt

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        f     = a*exp(b*XX^c)

        dfdx  = a*b*c*XX^(c-1)*exp(b*XX^c)
               = b*c*XX^(c-1) * (a*exp(b*XX^c))
               = b*c*XX^(c-1) * prof
        dfdx = b*c*x^(c-1)*prof
        """
        XX =_np.copy(XX)
        a, b, c = tuple(aa)
        prof = ModelExp._model(XX, aa, **kwargs)
#        return b*c*XX**(c-1.0)*prof
        return b*c*power(XX, c-1.0)*prof
#        return b*c*ModelExp.xtoc(XX, c=c-1.0)*prof

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        f     = a*exp(b*XX^c)

        dfdx  = a*b*c*XX^(c-1)*exp(b*XX^c)
               = b*c*XX^(c-1) * (a*exp(b*XX^c))
               = b*c*XX^(c-1) * prof
        dfdx = b*c*x^(c-1)*prof

        d2fdx2 = a*b*c*x^(c-2)*exp(b*x^c)*(b*c*x^c+c-1.0)
               = b*c*x^(c-2)*prof*( b*c*x^c+c-1.0 )
               = b*c*prof*( b*c*x^(2c-2)+(c-1.0)*x^(c-2) )
              lim x-> 0 = 0
        """
        XX =_np.copy(XX)
        a, b, c = tuple(aa)
        prof = ModelExp._model(XX, aa, **kwargs)

#        d2fdx2 = (b*c*XX**(c-2.0)*prof*(b*c*XX**c+c-1.0) )
        d2fdx2 = b*c*prof*power(XX, c-2.0)*(b*c*power(XX, c)+c-1.0)
#        d2fdx2 = b*c*prof*(b*c*ModelExp.xtoc(XX, 2*c-2)+(c-1.0)*ModelExp.xtoc(XX, c-2.0))
#        if _np.isnan(d2fdx2).any() and (XX==0).any():
#            d2fdx2[XX==0] = 0
        return d2fdx2

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
         f     = a*exp(b*XX^c)
         dfdx  = a*b*c*XX^(c-1)*exp(b*XX^c)

         dfda = f/a
         dfdb = XX^c*prof
         dfdc = prof*b*XX^c*log(XX)
              = prof*b*XX^(c-1)*XX*log(XX)
             domain of dfdc is x>0
             L'hospital's rule:   XX^c*log(XX) = log(XX)/XX^-c
                 limit as x goes to 0+ is 0
                 1/x  /  -c*x^(-c-1) = -x^c/c = 0
        """
        XX =_np.copy(XX)
        nx = _np.size(XX)
        num_fit = _np.size(aa)
        a, b, c = tuple(aa)

        prof = ModelExp._model(XX, aa, **kwargs)

        gvec = _np.zeros( (num_fit, nx), dtype=float)
        gvec[0, :] = prof/a
        gvec[1, :] = prof*power(XX, c)
        gvec[2, :] = b*power(XX, c)*log(XX)*prof
#        gvec[2, :] = b*ModelExp.logxtoc(XX, c=1.0)*ModelExp.xtoc(XX, c=c)*prof
#        if _np.isnan(gvec[2,:]).any() and (XX==0).any():
#            gvec[2,XX==0] = 0
        return gvec
#
#        gvec[1, :] = prof*XX**c
#        gvec[2, :] = b*_np.log(_np.abs(XX))*XX**c*prof
#        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
         f     = a*exp(b*XX^c)
         dfdx  = a*b*c*XX^(c-1)*exp(b*XX^c)
               = b*c*prof*XX^(c-1)

         dfda = f/a
         dfdb = XX^c*prof
         dfdc = prof*b*XX^c*log(XX)

         d2fdxda = dprofdx / a
         d2fdxdb = a*c*XX^(c-1)*exp(b*XX^c) + a*b*c*XX^(c-1)*XX^c*exp(b*XX^c)
                 = dprofdx/b + dprofdx * XX^c = dprofdx*( 1/b + XX^c )
         d2fdxdc = a*b*XX^(c-1)*exp(b*XX^c)  * ( b*c*XX^c*ln|x|+c*ln|x| + 1 )
                 = dprofdx * ( b*XX^c*ln|x| + ln|x| + 1.0/c )
                 = dprofdx * ( (b*x^c+1)ln|x| + 1.0/c )

             domain of d2fdc2 is x>0
             L'hospital's rule:   x^c*ln(x) + ln(x) -> 0 - inf
                 dprofdx * ( (b*x^c+1)ln|x| + 1.0/c )
                 b*c*prof*XX^(c-1)* ( b*XX^c*ln|x| + ln|x| + 1.0/c )
             lim XX^(c-1)* ( b*XX^c*ln|x| + ln|x| + 1.0/c )
                 = ( b*XX^c*ln|x| + ln|x| + 1.0/c ) / XX^(1-c)
                 = ( b*x^(c-1)*(1 + c*ln|x|) + 1/x ) / (1-c)XX^(-c) = inf / 0
                 = ( b*(c-1)*x^(2c-2)*(1 + c*ln|x|) + b*c*x^(2c-2) - x^(c-2) ) / (1-c)
                     if c<2, lim = inf
                     if c>2, lim = 0

        """
        XX =_np.copy(XX)
        nx = _np.size(XX)
        num_fit = _np.size(aa)
        a, b, c = tuple(aa)

        dprofdx = ModelExp._deriv(XX, aa, **kwargs)

        dgdx = _np.zeros( (num_fit,nx), dtype=float)
        dgdx[0, :] = dprofdx / a
        dgdx[1, :] = dprofdx*( power(XX, c) + 1.0/b )
#        dgdx[2, :] = dprofdx*( b*_np.log(_np.abs(XX))*XX**c + _np.log(_np.abs(XX)) + 1.0/c )
        dgdx[2, :] = dprofdx*( b*_np.log(_np.abs(XX))*power(XX, c) + _np.log(_np.abs(XX)) + 1.0/c )
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
         f     = a*exp(b*XX^c)
         dfdx  = a*b*c*XX^(c-1)*exp(b*XX^c)

         dfda = f/a
         dfdb = XX^c*prof
         dfdc = prof*b*XX^c*log(XX)

         d2fdxda = dprofdx / a
         d2fdxdb = a*c*XX^(c-1)*exp(b*XX^c) + a*b*c*XX^(c-1)*XX^c*exp(b*XX^c)
                 = dprofdx/b + dprofdx * XX^c = dprofdx*( 1/b + XX^c )
         d2fdxdc = a*b*XX^(c-1)*exp(b*XX^c)  * ( b*c*XX^c*ln|x|+c*ln|x| + 1 )
                 = dprofdx * ( b*XX^c*ln|x| + ln|x| + 1.0/c )

         d3fdx2da = d2profdx2 / a
         d3fdx2db = d2profdx2*(1/b + x^c) + dprofdx*c*x^(c-1)
         d3fdx2dc = d2profdx2*( b*x^c*ln|x| + ln|x| + 1.0/c )
                  + dprofdx * ( b*c*x^(c-1)*ln|x| + b*x^(c-1) + 1/x)
                  = d2profdx2*( (b*x^c + 1)*ln|x| + 1.0/c )
                  + dprofdx * ( b*c*x^(c-1)*ln|x| + b*x^(c-1) + 1/x)
                  = d2profdx2*( (b*x^c + 1)*ln|x| + 1.0/c )
                  + dprofdx * ( (b*c*ln|x| + b)*x^(c-1) + 1/x)

             L'hospital's rule:   XX^c*log(XX) + ln(x)
                 lim x->0 (b*x^c+1)ln|x|
                    ln(x)  / (b*x^c+1)^-1
                    1/x  * -b*c*x^(c-1)*(b*x^c+1)^-2
                    -b*c*x^(c-2)/(b*x^c+1)^2 goes to 0 as x -> 0
                 lim x->0 (b*c*ln|x| + b)*x^(c-1)
                    (b*c*ln(x)+b) / x^-(c-1)
                    b*c/x  * -(c-1)*x^-(-c+1-1)
                    -b*c*(c-1)*x^(c-1) goes to 0 as x -> 0 if c>1
        """
        XX =_np.copy(XX)
        nx = _np.size(XX)
        num_fit = _np.size(aa)
        a, b, c = tuple(aa)

        dprofdx = ModelExp._deriv(XX, aa, **kwargs)
        d2profdx2 = ModelExp._deriv2(XX, aa, **kwargs)

        d2gdx2 = _np.zeros( (num_fit,nx), dtype=float)
        d2gdx2[0, :] = d2profdx2 / a
        d2gdx2[1, :] = (d2profdx2*( power(XX, c) + 1.0/b ) + dprofdx*c*power(XX, c-1.0) )
        d2gdx2[2, :] = d2profdx2*( b*_np.log(_np.abs(XX))*power(XX, c)
                    + _np.log(_np.abs(XX)) + 1.0/c ) \
                    + dprofdx*( b*c*_np.log(_np.abs(XX))*power(XX, c-1.0)
                    + b*power(XX, c-1.0) + 1.0/XX )
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        y = a*exp(b*XX^c)

        y-scaling:  y' = y/ys
            y = ys*a'*exp(b'*XX^c')
                a = ys*a'
        y-shifting:  y' = (y-yo)/ys
            y = yo + ys*a'*(exp(b'*XX^c'))
                Not possible with constant coefficients

        x-scaling:  x' = x/xs
            y = ys*a'*exp(b'*(x/xs)^c')
                a = ys*a'
                b = b'/xs^c'

        x-shifting:  x' = (x-xo)/xs
            y = ys*a'*exp(b'/xs^c'*(x-xo)^c')
              = ys*a'*exp(b'^-c'/xs*(x-xo))^c'
              = ys*a'*exp(b'^-c'*(-xo)/xs)^c'*exp(b'^-c'*x/xs)^c'
              = ys*a'*exp(-b'^-c'*xo/xs)^c'*exp(b'/xs^c'*x^c')
              = ys*a'*exp(b'*(-xo/xs)^c')*exp(b'/xs^c'*x^c')

        y = a*exp(b*XX^c)
                a = ys*a'*exp(b'*(-xo/xs)^c')
                b = b'/xs^c'
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)   # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)
#        aout[0] = ys*ain[0]*exp(ain[1]*power(-1.0*xo/xs, ain[2]))
        aout[0] = ys*ain[0]*exp(ain[1]*power(-1.0*xo/xs, ain[2]))
        aout[1] = ain[1]/power(xs, ain[2])
        return aout

    def scaleaf(self, ain, **kwargs):
        """
        y = a*exp(b*XX^c)
                a = ys*a'*exp(b'*(-xo/xs)^c')
                b = b'/xs^c'

                c' = c
                b' = b*xs^c'
                a' = (a/ys)*exp(-b'*(-xo/xs)^c')
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)   # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)
#        aout[0] = (ain[0]/ys)*exp(-1.0*ain[1]*power(-xo, ain[2]))
        aout[1] = ain[1]*power(xs, aout[2])
        aout[0] = (ain[0]/ys)*exp(-1.0*aout[1]*power(-1.0*xo/xs, aout[2]))
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
        self.xoffset = 0.0   #  this should work, where is it?
#        self.slope = 1.0
#        self.xslope = 1.0
        return super(ModelExp, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelExp, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelExp
# ========================================================================== #


def expcutoff(XX, aa):
    return ModelExponential._model(XX, aa)

def deriv_expcutoff(XX, aa):
    return ModelExponential._deriv(XX, aa)

def partial_expcutoff(XX, aa):
    return ModelExponential._partial(XX, aa)

def partial_deriv_expcutoff(XX, aa):
    return ModelExponential._partial_deriv(XX, aa)

def model_Exponential(XX=None, af=None, **kwargs):
    return _model(ModelExponential, XX, af, **kwargs)

# =========================================== #


class ModelExponential(ModelClass):
    """
    --- Exponential on Background ---
    Model - y = a*(exp(b*XX^c) + XX^d)

    af    - estimate of fitting parameters [a, b, c, d]
    XX    - independent variables

    domain on profile::  x>0
        prof is real for x real and (c>0 and x>0 and d>0)
                                  or (c<=0 and x>0)
                                  or (d<=0 and x>0)

    domain on jacobian of derivative: x>0
        d2fdx2db is real for x real and (c<=1 and x>0) or (c>1 and x>=0)

    domain on jacobian of 2nd derivative:   x>0 generally
        d3fdx2db is real for x real and (c<=2 and x>0) or (c>2 and x>=0)

    """
    _af = 0.1*_np.ones((4,), dtype=_np.float64)
    _LB = _np.array([1e-18, -_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (4,), dtype=int)
#    leftboundary = 0.0
    _analytic_xscaling = False
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelExponential, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def absX(XX):
        XX = _np.copy(XX)
        return _np.abs(XX)

    @staticmethod
    def _separate_model(XX, aa, **kwargs):
        """
         f     = a*(exp(b*XX^c) + XX^d) = f1+f2;
        """
        XX = ModelExponential.absX(XX)
        a, b, c, d = tuple(aa)
        prof1 = a*exp(b* power(XX, c))
        prof2 = a*power(XX, d)
        return prof1, prof2

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
         f     = a*(exp(b*XX^c) + XX^d) = f1+f2;
        """
        prof1, prof2 = ModelExponential._separate_model(XX, aa)
        return prof1 + prof2

    @staticmethod
    def _separate_deriv(XX, aa, **kwargs):
        """
        f     = a*(exp(b*XX^c) + XX^d) = f1+f2;

        dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
               = b*c*XX^(c-1) * (a*exp(b*XX^c))  + a*d*XX^(d-1)
               = b*c*XX^(c-1) * prof1 + dprof2dx

        dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
              = b*c*x^(c-1)*f1 + d*f2/x

        df1dx = b*c*x^(c-1)*f1
        df2dx = d*f2/x
              = a*d*x^(d-1)
        """
        prof1, prof2 = ModelExponential._separate_model(XX, aa)

        XX = ModelExponential.absX(XX)
        a, b, c, d = tuple(aa)
        dprof1dx = b*c*power(XX, c-1.0)*prof1
        dprof2dx = a*d*power(XX, d-1.0)
#        dprof2dx = d*prof2/XX
        return dprof1dx, dprof2dx

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        f     = a*(exp(b*XX^c) + XX^d) = f1+f2;

        dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
               = b*c*XX^(c-1) * (a*exp(b*XX^c))  + a*d*XX^(d-1)
               = b*c*XX^(c-1) * prof1 + dprof2dx

        dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
              = b*c*x^(c-1)*f1 + d*f2/x

        df1dx = b*c*x^(c-1)*f1
        df2dx = d*f2/x
              = a*d*x^(d-1)
        """
        dprof1dx, dprof2dx = ModelExponential._separate_deriv(XX, aa)
        return dprof1dx+dprof2dx

    @staticmethod
    def _separate_deriv2(XX, aa, **kwargs):
        """
        f     = a*(exp(b*XX^c) + XX^d) = f1+f2;

        dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
               = b*c*XX^(c-1) * (a*exp(b*XX^c))  + a*d*XX^(d-1)
               = b*c*XX^(c-1) * prof1 + dprof2dx

        dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
              = b*c*x^(c-1)*f1 + d*f2/x

        df1dx = b*c*x^(c-1)*f1
        df2dx = d*f2/x
              = a*d*x^(d-1)

        d2f1dx2 = a*b*c*x^(c-2)*exp(b*x^c)*(b*c*x^c+c-1.0)
                = b*c*x^(c-2)*f1*( b*c*x^c+c-1.0 )
        d2f2dx2 = a*(d-1)*d*x^(d-2)
        """
        a, b, c, d = tuple(aa)
        prof1, prof2 = ModelExponential._separate_model(XX, aa, **kwargs)
#        dprof1dx, dprof2dx = ModelExponential._separate_deriv(XX, aa, **kwargs)

        XX = ModelExponential.absX(XX)
#        d2prof1dx2 = b*c*power(XX, c-1.0)*( (c-1.0)*prof1/XX + dprof1dx )
#        d2prof2dx2 = (d-1.0)*prof2/XX
        d2prof1dx2 = (b*c*power(XX, c-2.0)*prof1*(b*c*power(XX, c)+c-1.0) )
        d2prof2dx2 = a*(d-1.0)*d*power(XX, d-2.0)
        return d2prof1dx2, d2prof2dx2

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        f     = a*(exp(b*XX^c) + XX^d) = f1+f2;

        dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
               = b*c*XX^(c-1) * (a*exp(b*XX^c))  + a*d*XX^(d-1)
               = b*c*XX^(c-1) * prof1 + dprof2dx

        dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
              = b*c*x^(c-1)*f1 + d*f2/x

        df1dx = b*c*x^(c-1)*f1
        df2dx = d*f2/x
              = a*d*x^(d-1)

        d2f1dx2 = a*b*c*x^(c-2)*exp(b*x^c)*(b*c*x^c+c-1.0)
                = b*c*x^(c-2)*f1*( b*c*x^c+c-1.0 )
        d2f2dx2 = a*(d-1)*d*x^(d-2)

        d2fdx2 = b*c*(c-1)*XX^(c-2)*prof1
               + b*c*XX^(c-1)*dprof1dx
               + a*d*(d-1)*XX^(d-2)
               = b*c*XX^(c-1)*( (c-1)*prof1/XX + dprof1dx )
               + (d-1)*prof2/XX
        """
        d2prof1dx2, d2prof2dx2 = ModelExponential._separate_deriv2(XX, aa, **kwargs)
        return d2prof1dx2 + d2prof2dx2

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
         f     = a*(exp(b*XX^c) + XX^d) = f1+f2;
         dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))

         dfda = f/a
         dfdb = XX^c*f1
         dfdc = f1*b*XX^c*log(XX)
         dfdd = a*XX^d*log(XX) = log(XX)*f2
        """
        nx = _np.size(XX)
        num_fit = _np.size(aa)
        gvec = _np.zeros( (num_fit, nx), dtype=float)

        prof1, prof2 = ModelExponential._separate_model(XX, aa)
        prof = prof1 + prof2

        a, b, c, d = tuple(aa)
        XX = ModelExponential.absX(XX)

#        gvec[0, :] = _np.exp(b* power(XX, c)) + power(XX, d)
        gvec[0, :] = prof/a
        gvec[1, :] = prof1*power(XX, c)
        gvec[2, :] = b*_np.log(_np.abs(XX))*power(XX, c)*prof1
        gvec[3, :] = _np.log(_np.abs(XX))*prof2
#        gvec[2, :] = b*prof1*_np.log(_np.abs(XX))*power(XX, c)
#        gvec[3, :] = a*_np.log(_np.abs(XX))*power(XX, d)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
         f     = a*(exp(b*XX^c) + XX^d) = f1+f2;
         dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))

         dfda = f/a
         dfdb = XX^c*f1;
         dfdc = f1*XX^c*log10(XX)
         dfdd = a*XX^d*log10(XX) = log10(XX)*f2

         d2fdxda = dprofdx / a
         d2fdxdb = a*c*XX^(c-1)*exp(b*XX^c) + a*b*c*XX^(c-1)*XX^c*exp(b*XX^c)
                 = dprof1dx/b + dprof1dx * XX^c = dprof1dx*( 1/b + XX^c )
         d2fdxdc = a*b*XX^(c-1)*exp(b*XX^c)  * ( b*c*XX^c*ln|x|+c*ln|x| + 1 )
                 = dprof1dx * ( b*XX^c*ln|x| + ln|x| + 1.0/c )
         d2fdxdd = a*XX^(d-1) + a*d*XX^(d-1)*ln|x|
                 = dprof2dx/d + dprof2dx * ln|x|
        """
        nx = _np.size(XX)
        num_fit = _np.size(aa)
        dgdx = _np.zeros( (num_fit,nx), dtype=float)

        dprof1dx, dprof2dx = ModelExponential._separate_deriv(XX, aa)
        dprofdx = dprof1dx + dprof2dx

        a, b, c, d = tuple(aa)
        XX = ModelExponential.absX(XX)

        dgdx[0, :] = dprofdx / a
        dgdx[1, :] = dprof1dx*( power(XX, c) + 1.0/b )
        dgdx[2, :] = dprof1dx*( b*_np.log(_np.abs(XX))*power(XX, c)  # TODO:  this has an analytic limit of 0 at x=0
            + _np.log(_np.abs(XX)) + 1.0/c )
        dgdx[3, :] = ( a*power(XX, d-1.0)*( 1.0 + d*_np.log(_np.abs(XX)) ) )
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
         f     = a*(exp(b*XX^c) + XX^d) = f1+f2;
         dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
         d2fdx2 = b*c*(c-1)*XX^(c-2)*prof1
                 + b*c*XX^(c-1)*dprof1dx
                 + a*d*(d-1)*XX^(d-2)
                = b*c*XX^(c-1)*( (c-1)*prof1/XX + dprof1dx )
                  + (d-1)*prof2/XX

         dfda = f/a
         dfdb = XX^c*f1;
         dfdc = f1*XX^c*log10(XX)
         dfdd = a*XX^d*log10(XX) = log10(XX)*f2

         d2fdxda = dprofdx / a
         d2fdxdb = a*c*XX^(c-1)*exp(b*XX^c) + a*b*c*XX^(c-1)*XX^c*exp(b*XX^c)
                 = dprof1dx/b + dprof1dx * XX^c = dprof1dx*( 1/b + XX^c )
         d2fdxdc = a*b*XX^(c-1)*exp(b*XX^c)  * ( b*c*XX^c*ln|x|+c*ln|x| + 1 )
                 = dprof1dx * ( b*XX^c*ln|x| + ln|x| + 1.0/c )
         d2fdxdd = a*XX^(d-1) + a*d*XX^(d-1)*ln|x|
                 = dprof2dx/d + dprof2dx * ln|x|

         d3fdx2da = d2profdx2 / a
         d3fdx2db = d2prof1dx2*(1/b + x^c) + dprof1dx*c*x^(c-1)
         d3fdx2dc = d2prof1dx2*( b*x^c*ln|x| + ln|x| + 1.0/c )
                  + dprof1dx * ( b*c*x^(c-1)*ln|x| + b*x^(c-1) + 1/x)
         d3fdx2dd = d2prof2dx/d + d2prof2dx2*ln|x| + dprof2dx/x

        """
        nx = _np.size(XX)
        num_fit = _np.size(aa)

        dprof1dx, dprof2dx = ModelExponential._separate_deriv(XX, aa, **kwargs)
#        dprofdx = dprof1dx + dprof2dx
        d2prof1dx2, d2prof2dx2 = ModelExponential._separate_deriv2(XX, aa, **kwargs)
        d2profdx2 = d2prof1dx2 + d2prof2dx2

        a, b, c, d = tuple(aa)
        XX = ModelExponential.absX(XX)

        d2gdx2 = _np.zeros( (num_fit,nx), dtype=float)
        d2gdx2[0, :] = d2profdx2 / a
        d2gdx2[1, :] = (d2prof1dx2*( power(XX, c) + 1.0/b ) + dprof1dx*c*power(XX, c-1.0) )
        d2gdx2[2, :] = d2prof1dx2*( b*_np.log(_np.abs(XX))*power(XX, c)  # TODO:  this has an analytic limit of 0 at x=0
                    + _np.log(_np.abs(XX)) + 1.0/c ) \
                    + dprof1dx*( b*c*_np.log(_np.abs(XX))*power(XX, c-1.0)   # TODO:  this has an analytic limit of 0 at x=0
                    + b*power(XX, c-1.0) + 1.0/XX )
        d2gdx2[3, :] = d2prof2dx2/d + d2prof2dx2*_np.log(_np.abs(XX)) + dprof2dx/XX
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        y = a*(exp(b*XX^c) + XX^d) = f1+f2

        y-scaling:  y' = y/ys
            y = ys*a'*(exp(b'*XX^c') + XX^d')
                a = ys*a'
        y-shifting:  y' = (y-yo)/ys
            y = yo + ys*a'*(exp(b'*XX^c') + XX^d')
                Not possible without x-shifting
        x-scaling:  x' = x/xs
            y = ys*a'*(exp(b'*(x/xs)^c') + XX^d'*xs^-d)
              = xs^d*ys*a'*(xs^-d*exp(b'/xs^c'*x^c') + XX^d')
                  xs^-d' = exp(ln(xs^-d')) = exp(-d'*ln|xs|)
              = xs^d*ys*a'*(exp(b'/xs^c'*x^c'-d'*ln|xs|) + XX^d')
                Not possible due to nonlinearity shift in exponent without x-shifting

        x-scaling:  x' = (x-xo)/xs
            y = xs^d*ys*a'*(exp(b'/xs^c'*(x-xo)^c'-d'*ln|xs|) + (x-xo)^d')
                Not possible without compensating y-shift

            y = yo + xs^d*ys*a'*(exp(b'/xs^c'*(x-xo)^c'-d'*ln|xs|) + (x-xo)^d')
                f1 = xs^d*ys*a'*exp(b'/xs^c'*(x-xo)^c'-d'*ln|xs|)
                    use binomial theorem to expand (x-xo)^c
                         the last term will give us -b'/xs^c'*xs^c' = d'ln|xs|
                         or b' = -d'ln|xs|  ... won't work with other terms
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)   # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)   # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore
        aout[0] *= ys
        return aout

    def scaleaf(self, ain, **kwargs):
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)   # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)   # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore
        aout[0] /= ys
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
        self.xslope = 1.0
        self.xoffset = 0.0
        return super(ModelExponential, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelExponential, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelExponential

# ========================================================================== #
# ========================================================================== #


def gaussian(XX, aa):
    return ModelGaussian._model(XX, aa)

def partial_gaussian(XX, aa):
    return ModelGaussian._partial(XX, aa)

def deriv_gaussian(XX, aa):
    return ModelGaussian._deriv(XX, aa)

def partial_deriv_gaussian(XX, aa):
    return ModelGaussian._partial_deriv(XX, aa)

def _model_gaussian(XX=None, af=None, **kwargs):
    """
    """
    return _model(ModelGaussian, XX, af, **kwargs)

# ==================================== #


class ModelGaussian(ModelClass):
    """
    A gaussian with three free parameters:
        XX - x - independent variable
        af - magnitude, shift, width

        af[0]*_np.exp(-(XX-af[1])**2/(2.0*af[2]**2))

    f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
        A = af[0]
        xo = af[1]
        ss = af[2]

        note that if normalized, then the PDF form is used so that the integral
        is equal to af[0]: For that form use ModelNormal
            af[0]*_np.exp(-(XX-af[1])**2/(2.0*af[2]**2))/_np.sqrt(2.0*pi*af[2]**2.0)
    """
    _af = _np.asarray([1.0e-1/0.05, -0.3, 0.1], dtype=_np.float64)
    _LB = _np.array([-_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.array([0, 0, 0], dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelGaussian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        model of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
          = g/nn              where g = Gaussian
                              and   nn = sqrt(2*pi*ss^2.0)
            A = af[0]
            xo = af[1]
            ss = af[2]
        """
        AA, x0, ss = tuple(aa)
        return AA*exp(-(XX-x0)*(XX-x0)/(2.0*(ss*ss)))

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        derivative of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2 *f
        """
        AA, x0, ss = tuple(aa)
        prof = ModelGaussian._model(XX, aa, **kwargs)
        return -1.0*(XX-x0)/(ss*ss) * prof

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        2nd derivative of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2 *f
        d2fdx2 = -1/ss^2*f - (x-xo)/ss^2*dfdx
        """
        AA, x0, ss = tuple(aa)
        prof = ModelGaussian._model(XX, aa, **kwargs)
        dprofdx = ModelGaussian._deriv(XX, aa, **kwargs)
        return -1.0*prof/(ss*ss) - dprofdx*(XX-x0)/(ss*ss)

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        Jacobian of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2 *f
        d2fdx2 = -1/ss^2*f - (x-xo)/ss^2*dfdx

        dfdA = f/A
        dfdxo = A*(x-xo)*exp(-(x-xo)**2.0/(2.0*ss**2.0))/ss^2
              = (x-xo)/ss^2 * f
        dfdss =  (x-xo)^2/ss^3 * f
        """
        AA, x0, ss = tuple(aa)
        prof = ModelGaussian._model(XX, aa, **kwargs)
        gvec = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = prof/AA
        gvec[1,:] = ((XX-x0)/(ss*ss))*prof
        gvec[2,:] = ((XX-x0)*(XX-x0)/(ss*ss*ss)) * prof
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2 *f
        d2fdx2 = -1/ss^2*f - (x-xo)/ss^2*dfdx

        dfdA = f/A
        dfdxo = A*(x-xo)*exp(-(x-xo)**2.0/(2.0*ss**2.0))/ss^2
              = (x-xo)/ss^2 * f
        dfdss =  (x-xo)^2/ss^3 * f

        d2fdxdA = d/dx(f/A) = -(x-xo)/ss^2 * f/A
                = dfdx/A
        d2fdxdxo = f/(ss^2) + (x-xo)/ss^2 * dfdx
                = -d2fdx2
        d2fdxdss = 2*(x-xo)/ss^3 * f + (x-xo)^2/ss^3 * dfdx
                =  -2*dfdx/ss + (x-xo)^2/ss^3 * dfdx
        """
        AA, x0, ss = tuple(aa)

#        prof = ModelGaussian._model(XX, aa, **kwargs)
        dfdx = ModelGaussian._deriv(XX, aa, **kwargs)
        d2fdx2 = ModelGaussian._deriv2(XX, aa, **kwargs)

        dgdx = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = dfdx/AA
        dgdx[1,:] = -d2fdx2
        dgdx[2,:] = -2.0*dfdx/ss + dfdx*(XX-x0)*(XX-x0)/(ss*ss*ss)
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2 *f
        d2fdx2 = -1/ss^2*f - (x-xo)/ss^2*dfdx

        dfdA = f/A
        dfdxo = A*(x-xo)*exp(-(x-xo)**2.0/(2.0*ss**2.0))/ss^2
              = (x-xo)/ss^2 * f
        dfdss =  (x-xo)^2/ss^3 * f

        d2fdxdA = d/dx(f/A) = -(x-xo)/ss^2 * f/A
                = dfdx/A
        d2fdxdxo = f/(ss^2) + (x-xo)/ss^2 * dfdx
                = -d2fdx2
        d2fdxdss = 2*(x-xo)/ss^3 * f + (x-xo)^2/ss^3 * dfdx
                =  -2*dfdx/ss + (x-xo)^2/ss^3 * dfdx

        d3fdx2dA = d2fdx2/A
        d3fdx2dxo = -dfdxo/(ss^2) + dfdx/ss^2 - (x-xo)/ss^2 * d2fdxdxo
                  = -(x-xo)/(ss^4)*f + dfdx/ss^2 + (x-xo)/ss^2 * d2fdx2
        d3fdx2dss = -2*d2fdx2/ss + 2*(x-xo)/ss^3 *dfdx + (x-xo)^2/ss^3 *d2fdx2
        """
        AA, x0, ss = tuple(aa)
        prof = ModelGaussian._model(XX, aa, **kwargs)
        dfdx = ModelGaussian._deriv(XX, aa, **kwargs)
        d2fdx2 = ModelGaussian._deriv2(XX, aa, **kwargs)

#        term1 = (-1.0*(XX-x0)/power(ss,2.0))
#        gvec = ModelGaussian._partial(XX, aa, **kwargs)

        d2gdx = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        d2gdx[0,:] = d2fdx2/AA
        d2gdx[1,:] = -(XX-x0)*prof/(ss*ss*ss*ss) + dfdx/(ss*ss) + (XX-x0)*d2fdx2/(ss*ss)
        d2gdx[2,:] = -2.0*d2fdx2/ss + 2.0*(XX-x0)/(ss*ss*ss)*dfdx + (XX-x0)*(XX-x0)*d2fdx2/(ss*ss*ss)
        return d2gdx

#    @staticmethod
#    def _hessian(XX, aa, **kwargs):
#       """
#        Hessian of a gaussian
#        [d2fda2, d2fdadb, d2fdadc; d2fdadb, d2fdb2, d2fdbdc; d2fdadc, d2fdbdc, d2fdc2]
#       d2fda2 = 0.0
#       d2fdb2 = d/dxo (x-xo)/ss^2 * f
#       """
#       hmat = _np.zeros((3, 3, _np.size(XX)), dtype=_np.float64)
#       return hmat
    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        If fitting with scaling, then the algebra necessary to unscale the problem
        to original units is:

            y-scaling:  y' = (y-yo)/ys
             (y-miny)/(maxy-miny) = (y-yo)/ys
             (y-yo)/ys = a' exp(-1*(x-b')^2/(2*c'^2))

             y-yo = ys * a' * exp(-1*(x-b')^2/(2*c'^2))
                 assume a = ys*a'+yo

             yo + ys*a'*exp(-1*(x-b')^2/(2*c'^2)) = a*exp(-1*(x-b)^2/(2*c^2))
             a = yo*exp((x-b)^2/(2*c^2)) + ys*a'*exp((x-b)^2/(2*c^2)-(x-b')^2/(2*c'^2))

             if b=b' and c=c' then a = yo*exp((x-b)^2/2c^2) + ys*a'
                 not possible unless x==b for all x  OR yo = 0

            x-scaling: x' = (x-xo)/xs
             (y(x') - yo)/ys = a'*exp(-1*(x'-b')^2.0/(2*c'^2))
               y = yo+ys*a'*exp(-1*(x/xs-xo/xs-b')^2.0/(2*c'^2))
                 = yo+ys*a'*exp(-1*(x-xo-xs*b')^2.0/(2*c'^2*xs^2))
             here:
                 a = a'*ys
                 b = b'*xs + xo
                 c = c'*xs
                 iff yo=0.0
                 x-shift and xy-scaling works, but yo shifting does not
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)   # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = ys*ain[0]
        aout[1] = xs*ain[1]+xo
        aout[2] = xs*ain[2]
        return aout

    def scaleaf(self, ain, **kwargs):
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)   # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] =  ain[0]/ys
        aout[1] = (ain[1]-xo)/xs
        aout[2] = ain[2]/xs
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
        return super(ModelGaussian, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

    def checkbounds(self, dat):
        return super(ModelGaussian, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelGaussian


# ========================================================================== #
# ========================================================================== #


def offsetgaussian(XX, aa):
    return ModelOffsetGaussian._model(XX, aa)

def deriv_offsetgaussian(XX, aa):
    return ModelOffsetGaussian._deriv(XX, aa)

def partial_offsetgaussian(XX, aa):
    return ModelOffsetGaussian._partial(XX, aa)

def partial_deriv_offsetgaussian(XX, aa):
    return ModelOffsetGaussian._partial_deriv(XX, aa)

def _model_offsetgaussian(XX=None, af=None, **kwargs):
    """
    """
    return _model(ModelOffsetGaussian, XX, af, **kwargs)

# ==================================== #

class ModelOffsetGaussian(ModelClass):
    """
    A gaussian with four free parameters:
        XX - x - independent variable
        af - y-offset, magnitude, shift, width

    f = A0 + A*exp(-(x-xo)**2.0/(2.0*ss**2.0))

        af[0] + af[1]*_np.exp(-(XX-af[2])**2/(2.0*af[3]**2))

        note that if normalized, then the PDF form is used so that the integral
        is equal to af[1]:
            af[0] + af[1]*exp(-(XX-af[2])**2/(2.0*af[3]**2))/_np.sqrt(2.0*pi*af[3]**2.0)
    """
    _af = _np.asarray([0, 1.0e-1/0.05, -0.3, 0.1], dtype=_np.float64)
    _LB = _np.array([-_np.inf, -_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.array([0, 0, 0, 0], dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelOffsetGaussian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        return ModelGaussian._model(XX, aa[1:]) + aa[0]

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        return ModelGaussian._deriv(XX, aa[1:])

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        return ModelGaussian._deriv2(XX, aa[1:])

    @staticmethod
    def _partial(XX, aa, **kwargs):
        return _np.concatenate(
                (_np.ones((1,_np.size(XX)), dtype=_np.float64),
                 ModelGaussian._partial(XX, aa[1:])), axis=0)

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        return _np.concatenate(
                (_np.zeros((1,_np.size(XX)), dtype=_np.float64),
                 ModelGaussian._partial_deriv(XX, aa[1:])), axis=0)

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        return _np.concatenate(
                (_np.zeros((1,_np.size(XX)), dtype=_np.float64),
                 ModelGaussian._partial_deriv2(XX, aa[1:])), axis=0)

#    @staticmethod
#    def _hessian(XX, aa):
#       """
#        Hessian of an offset gaussian
#        [d2fda2, d2fdadb, d2fdadc; d2fdadb, d2fdb2, d2fdbdc; d2fdadc, d2fdbdc, d2fdc2]
#       d2fda2 = 0.0
#       d2fdb2 = d/dxo (x-xo)/ss^2 * f
#       """
#       hmat = _np.zeros((3, 3, _np.size(XX)), dtype=_np.float64)
#       return hmat
    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        If fitting with scaling, then the algebra necessary to unscale the problem
        to original units is:

            y-scaling:  y' = (y-yo)/ys
             (y-miny)/(maxy-miny) = (y-yo)/ys
             (y-yo)/ys = a0' + a' exp(-1*(x-b')^2/(2*c'^2))

             y = yo + ys*a0' + ys * a' * exp(-1*(x-b')^2/(2*c'^2))

            x-scaling: x' = (x-xo)/xs
             (y(x') - yo)/ys = a'*exp(-1*(x'-b')^2.0/(2*c'^2))   + a0
               y = yo+ys*a'*exp(-1*(x/xs-xo/xs-b')^2.0/(2*c'^2))   + a0*ys
                 = yo+ys*a'*exp(-1*(x-xo-xs*b')^2.0/(2*c'^2*xs^2))  + a0*ys
             here:
                 a0 = a0'*ys + yo
                 a = a'*ys
                 b = b'*xs + xo
                 c = c'*xs
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)
        aout[0] = ys*aout[0] + yo
        aout[1] = ys*aout[1]
        aout[2] = xs*aout[2]+xo
        aout[3] = xs*aout[3]
        return aout

    def scaleaf(self, ain, **kwargs):
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] =  (ain[0]-yo)/ys
        aout[1] = ain[1]/ys
        aout[2] = (ain[2]-xo)/xs
        aout[3] = ain[3]/xs
        return aout

    def scalings(self, xdat, ydat, **kwargs):
#        self.offset = 0.0
        return super(ModelOffsetGaussian, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelOffsetGaussian, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelOffsetGaussian

# =========================================================================== #
# =========================================================================== #


def offsetnormal(XX, aa):
    return ModelOffsetNormal._model(XX, aa)

def partial_offsetnormal(XX, aa):
    return ModelOffsetNormal._partial(XX, aa)

def deriv_offsetnormal(XX, aa):
    return ModelOffsetNormal._deriv(XX, aa)

def partial_deriv_offsetnormal(XX, aa):
    return ModelOffsetNormal._partial_deriv(XX, aa)

def model_offsetnormal(XX=None, af=None, **kwargs):
    norm = kwargs.pop('norm', True)  # analysis:ignore
    return _model(ModelOffsetNormal, XX, af, **kwargs)

# ==================================== #

class ModelOffsetNormal(ModelClass):
    """
    A gaussian with four free parameters:
        XX - x - independent variable
        af - offset, magnitude (AA), shift (xo), width (ss)
            A0 = af[0]
             A = af[1]
            xo = af[2]
            ss = af[3]
    f = A0 + A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/_np.sqrt(2.0*pi*ss**2.0)
    """
    _af = _np.asarray([0.0, 1.0, -0.3, 0.1], dtype=_np.float64)
    _LB = _np.array([-_np.inf, -_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([ _np.inf,  _np.inf,  _np.inf,  _np.inf], dtype=_np.float64)
    _fixed = _np.array([0, 0, 0, 0], dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelOffsetNormal, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        Model of an offset normalized Gaussian (unit area)
        f = A0 + A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
            A0 = af[0]
             A = af[1]
            xo = af[2]
            ss = af[3]
        """
        return aa[0] + ModelNormal._model(XX, aa[1:])

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        derivative of a normalized Gaussian (unit area)
        dfdx = -(x-xo)/ss^2 *f
        """
        return ModelNormal._deriv(XX, aa[1:])

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        derivative of a normalized Gaussian (unit area)
        dfdx = -(x-xo)/ss^2 *f
        """
        return ModelNormal._deriv2(XX, aa[1:])

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        Jacobian of a normalized Gaussian (unit area)
        dfdA0 = 1.0
        dfdA = f/A
        dfdxo = (x-xo)/ss^2 * f
        dfdss =  (x-xo)^2/ss^3 * f + A*exp(-(x-xo)**2.0/(2.0*ss**2.0))*d/ds( sqrt(2*pi*ss^2.0)**-1 )
            d/ds( sqrt(2*pi*ss^2.0)**-1 ) = sqrt(2*pi)^-1 * d/ds( (ss^2.0)**-0.5 ) = sqrt(2*pi)^-1 * d/ds( abs(ss)^-1.0 )
                = sqrt(2*pi)^-1 * (2*ss)*-0.5 * (ss^2.0)^-1.5 = sqrt(2*pi)^-1 *-ss/(ss^3.0) = sqrt(2*pi)^-1 *1.0/ss^2
                = sqrt(2*pi*ss**2.0)^-1 *1.0/abs(ss)
        dfdss =  (x-xo)^2/ss^3 * f + 1.0/abs(ss) * f
        """
        return _np.concatenate((_np.ones((1,_np.size(XX)), dtype=_np.float64), ModelNormal._partial(XX, aa[1:])), axis=0)

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)

        dfdx = -(x-xo)/ss^2 *f

        d2fdxdA0 = 0.0
        d2fdxdA = dfdx/A = -(x-xo)/ss^2 *dfdA
        d2fdxdxo = f/ss^2 -(x-xo)/ss^2 *dfdxo
        d2fdxds = 2*(x-xo)/ss^3 *f -(x-xo)/ss^2 *dfds

        """
        return _np.concatenate((_np.zeros((1,_np.size(XX)), dtype=_np.float64), ModelNormal._partial_deriv(XX, aa[1:])), axis=0)

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)

        dfdx = -(x-xo)/ss^2 *f

        d2fdxdA0 = 0.0
        d2fdxdA = dfdx/A = -(x-xo)/ss^2 *dfdA
        d2fdxdxo = f/ss^2 -(x-xo)/ss^2 *dfdxo
        d2fdxds = 2*(x-xo)/ss^3 *f -(x-xo)/ss^2 *dfds

        """
        return _np.concatenate((_np.zeros((1,_np.size(XX)), dtype=_np.float64), ModelNormal._partial_deriv2(XX, aa[1:])), axis=0)

#    @staticmethod
#    def _hessian(XX, aa):
#       """
#        Hessian of a normalized gaussian
#        [d2fda2, d2fdadb, d2fdadc; d2fdadb, d2fdb2, d2fdbdc; d2fdadc, d2fdbdc, d2fdc2]
#       d2fda2 = 0.0
#       d2fdb2 = d/dxo (x-xo)/ss^2 * f
#       """
#       hmat = _np.zeros((3, 3, _np.size(XX)), dtype=_np.float64)
#       return hmat
    # ====================================== #

    def unscaleaf(self, ain, **kwargs):
        """
        If fitting with scaling, then the algebra necessary to unscale the problem
        to original units is:

            y-scaling:  y' = (y-yo)/ys
             (y-miny)/(maxy-miny) = (y-yo)/ys

            x-scaling: x' = (x-xo)/xs
             (y(x') - yo)/ys = a'*exp(-1*(x'-b')^2.0/(2*c'^2))/sqrt(2*pi*c'^2.0)
               y = yo+ys*a0'+ys*a'*exp(-1*(x/xs-xo/xs-b')^2.0/(2*c'^2))/sqrt(2*pi*c'^2.0)
                 = yo+ys*a0'+ys*a'*exp(-1*(x-xo-xs*b')^2.0/(2*c'^2*xs^2))/sqrt(2*pi*c'^2.0)
             here:
                 a0 = ys*a0'+yo
                 a = a'*ys * xs
                 b = b'*xs + xo
                 c = c'*xs
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = ys*ain[0] + yo
        aout[1] = ys*xs*ain[1]
        aout[2] = xs*ain[2]+xo
        aout[3] = xs*aout[3]
        return aout

    def scaleaf(self, ain, **kwargs):
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = (ain[0]-yo)/ys
        aout[1] = (ain[1]/xs)/ys
        aout[2] = (ain[2]-xo)/xs
        aout[3] = ain[3]/xs
        return aout

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelOffsetNormal, self).checkbounds(dat, self.aa, mag=None)

    # ========================#============= #
# end def ModelOffsetNormal

# =========================================================================== #
# =========================================================================== #


def normal(XX, aa):
    return ModelNormal._model(XX, aa)

def partial_normal(XX, aa):
    return ModelNormal._partial(XX, aa)

def deriv_normal(XX, aa):
    return ModelNormal._deriv(XX, aa)

def partial_deriv_normal(XX, aa):
    return ModelNormal._partial_deriv(XX, aa)

def model_normal(XX=None, af=None, **kwargs):
    norm = kwargs.pop('norm', True)  # analysis:ignore
    return model_gaussian(XX, af, norm=True, **kwargs)

# ==================================== #

class ModelNormal(ModelClass):
    """
    A gaussian with three free parameters:
        XX - x - independent variable
        af - magnitude (AA), shift (xo), width (ss)
            A = af[0]
            xo = af[1]
            ss = af[2]
    f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/_np.sqrt(2.0*pi*ss**2.0)
    """
    _af = _np.asarray([1.0, -0.3, 0.1], dtype=_np.float64)
    _LB = _np.array([-_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.array([0, 0, 0], dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelNormal, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        Model of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
            A = af[0]
            xo = af[1]
            ss = af[2]
        """
        AA, x0, ss = tuple(aa)
        nn = _np.sqrt(2.0*_np.pi*(ss*ss))
        return ModelGaussian._model(XX, aa, **kwargs)/nn

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
          = g/nn              where g = Gaussian
                              and   nn = sqrt(2*pi*ss^2.0)
        dfdx = -(x-xo)/ss^2 *f  = dgdx/nn
        """
        AA, x0, ss = tuple(aa)
        nn = _np.sqrt(2.0*_np.pi*ss*ss)
        return ModelGaussian._deriv(XX, aa, **kwargs)/nn

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
          = g/nn              where g = Gaussian
                              and   nn = sqrt(2*pi*ss^2.0)
        dfdx = -(x-xo)/ss^2 *f = dgdx/nn
        d2fdx2 = d2gdx2/nn
        """
        AA, x0, ss = tuple(aa)
        nn = _np.sqrt(2.0*_np.pi*ss*ss)
        return ModelGaussian._deriv2(XX, aa, **kwargs)/nn

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        Jacobian of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
          = g/nn              where g = Gaussian
                              and   nn = sqrt(2*pi*ss^2.0)
        dfdx = -(x-xo)/ss^2 *f = dgdx/nn
        d2fdx2 = d2gdx2/nn

        dfdA = f/A                      = dgdA/nn
        dfdxo = (x-xo)/ss^2 * f         = dgdxo/nn
        dfdss = dgdss/nn - dnds*g/nn^2  = (dgdss - dnds*f)/nn
          where dnds = 0.5*2*2pi*ss/nn = 2pi*ss/nn = sqrt(2pi)*sign(ss)
        """
        AA, x0, ss = tuple(aa)
        nn = _np.sqrt(2.0*_np.pi*ss*ss)
        dnds = _np.sqrt(2.0*_np.pi)*_np.sign(ss)

        prof = ModelNormal._model(XX, aa, **kwargs)
        g1 = ModelGaussian._partial(XX, aa, **kwargs)

        gvec = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = g1[0,:]/nn
        gvec[1,:] = g1[1,:]/nn
        gvec[2,:] = (g1[2,:]-dnds*prof)/nn
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
          = g/nn              where g = Gaussian
                              and   nn = sqrt(2*pi*ss^2.0)
        dfdx = -(x-xo)/ss^2 *f = dgdx/nn
        d2fdx2 = d2gdx2/nn

        dfdA = f/A                      = dgdA/nn
        dfdxo = (x-xo)/ss^2 * f         = dgdxo/nn
        dfdss = dgdss/nn - dnds*g/nn^2  = (dgdss - dnds*f)/nn
          where dnds = 0.5*2*2pi*ss/nn = 2pi*ss/nn = sqrt(2pi)*sign(ss)

        d2fdxdA = dfdx/A = d2gdxdA/nn
        d2fdxdxo = d2gdxdxo/nn
        d2fdxdss = (d2gdxdss - dnds*dfdx)/nn
        """
        AA, x0, ss = tuple(aa)
        nn = _np.sqrt(2.0*_np.pi*ss*ss)
        dnds = _np.sqrt(2.0*_np.pi)*_np.sign(ss)

        dfdx = ModelNormal._deriv(XX, aa, **kwargs)
        dg = ModelGaussian._partial_deriv(XX, aa, **kwargs)

        dgdx = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = dg[0,:]/nn
        dgdx[1,:] = dg[1,:]/nn
        dgdx[2,:] = (dg[2,:]-dnds*dfdx)/nn
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
          = g/nn              where g = Gaussian
                              and   nn = sqrt(2*pi*ss^2.0)
        dfdx = -(x-xo)/ss^2 *f = dgdx/nn
        d2fdx2 = d2gdx2/nn

        dfdA = f/A                      = dgdA/nn
        dfdxo = (x-xo)/ss^2 * f         = dgdxo/nn
        dfdss = dgdss/nn - dnds*g/nn^2  = (dgdss - dnds*f)/nn
          where dnds = 0.5*2*2pi*ss/nn = 2pi*ss/nn = sqrt(2pi)*sign(ss)

        d2fdxdA = dfdx/A = d2gdxdA/nn
        d2fdxdxo = d2gdxdxo/nn
        d2fdxdss = (d2gdxdss - dnds*dfdx)/nn

        d3fdx2dA = d3gdx2dA/nn
        d3fdx2dxo = d3gdx2dxo/nn
        d3fdx2dss = (d3gdx2dss - dnds*d2fdx2)/nn
        """
        AA, x0, ss = tuple(aa)
        nn = _np.sqrt(2.0*_np.pi*ss*ss)
        dnds = _np.sqrt(2.0*_np.pi)*_np.sign(ss)

        d2fdx2 = ModelNormal._deriv2(XX, aa, **kwargs)
        dg2 = ModelGaussian._partial_deriv2(XX, aa, **kwargs)

        d2gdx2 = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
        d2gdx2[0,:] = dg2[0,:]/nn
        d2gdx2[1,:] = dg2[1,:]/nn
        d2gdx2[2,:] = (dg2[2,:]-dnds*d2fdx2)/nn
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):
#       """
#        Hessian of a normalized gaussian
#        [d2fda2, d2fdadb, d2fdadc; d2fdadb, d2fdb2, d2fdbdc; d2fdadc, d2fdbdc, d2fdc2]
#       d2fda2 = 0.0
#       d2fdb2 = d/dxo (x-xo)/ss^2 * f
#       """
#       hmat = _np.zeros((3, 3, _np.size(XX)), dtype=_np.float64)
#       return hmat
    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        If fitting with scaling, then the algebra necessary to unscale the problem
        to original units is:

            y-scaling:  y' = (y-yo)/ys
             (y-miny)/(maxy-miny) = (y-yo)/ys
             (y-yo)/ys = a' exp(-1*(x-b')^2/(2*c'^2))/sqrt(2*pi*c'^2.0)

             y = yo + ys * a' * exp(-1*(x-b')^2/(2*c'^2))/sqrt(2*pi*c'^2.0)
                 assume a = ys*a'+yo as before
                 a = yo*exp((x-b)^2/(2*c^2)) + ys*a'*exp((x-b)^2/(2*c^2)-(x-b')^2/(2*c'^2))
             if b=b' and c=c' then a = yo*exp((x-b)^2/2c^2) + ys*a'
                 not possible unless x==b for all x  OR yo = 0

            x-scaling: x' = (x-xo)/xs
             (y(x') - yo)/ys = a'*exp(-1*(x'-b')^2.0/(2*c'^2))/sqrt(2*pi*c'^2.0)
               y = yo+ys*a'*exp(-1*(x/xs-xo/xs-b')^2.0/(2*c'^2))/sqrt(2*pi*c'^2.0)
                 = yo+ys*a'*exp(-1*(x-xo-xs*b')^2.0/(2*c'^2*xs^2))/sqrt(2*pi*c'^2.0)
             here:
                 a = a'*ys * xs
                 b = b'*xs + xo
                 c = c'*xs
                 iff yo=0.0
                 x-shift and xy-scaling works, but yo shifting does not
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset) # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = ys*xs*aout[0]
        aout[1] = xs*aout[1]+xo
        aout[2] = xs*aout[2]
        return aout

    def scaleaf(self, ain, **kwargs):
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = (ain[0]/xs)/ys
        aout[1] = (ain[1]-xo)/xs
        aout[2] = (ain[2])/xs
        return aout


    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
        return super(ModelNormal, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelNormal, self).checkbounds(dat, self.aa, mag=None)

    # ========================#============= #
# end def ModelNormal

# =========================================================================== #
# =========================================================================== #


def model_gaussian(XX=None, af=None, **kwargs):
    """
    """
    normalized = kwargs.setdefault('norm', False)
    offsetgaussian = kwargs.pop('offsetgaussian', False)
    if normalized:
        return _model(ModelNormal, XX, af, **kwargs)
    else:
        if offsetgaussian:
            return _model(ModelOffsetGaussian, XX, af, **kwargs)   # only scalable form
        else:
            return _model(ModelGaussian, XX, af, **kwargs)
        # end if
    # end if
# end def


# =========================================================================== #
# =========================================================================== #


def _parse_gaussian_inputs(af, **kwargs):
    noshift = kwargs.setdefault('noshift',False)
    if noshift or len(af)==2:
        noshift = True
        x0 = 0.0
        if len(af) == 3:
            af = _np.delete(_np.copy(af), (1), axis=0)
        AA, ss = tuple(af)
    else:
        AA, x0, ss = tuple(af)
    # end if
    return af

# ========================================================================== #



def loggaussian(XX, aa):
    return ModelLogGaussian._model(XX, aa)

def partial_loggaussian(XX, aa):
    return ModelLogGaussian._partial(XX, aa)

def deriv_loggaussian(XX, aa):
    return ModelLogGaussian._deriv(XX, aa)

def partial_deriv_loggaussian(XX, aa):
    return ModelLogGaussian._partial_deriv(XX, aa)

def model_loggaussian(XX=None, af=None, **kwargs):
    return _model(ModelLogGaussian, XX, af, **kwargs)

# =========================================== #


class ModelLogGaussian(ModelClass):
    """
        A lognormal with three free parameters:
            XX - x - independent variable
            af - magnitude, shift, width

        f = 10*_np.log10( A*exp(-(x-xo)**2.0/(2.0*ss**2.0)) )
          = 10/ln(10)*( ln(A) -(x-xo)**2.0/(2.0*ss**2.0)  )
     """
    _af = _np.asarray([1.0, 0.3, 0.1], dtype=_np.float64)
    _LB = _np.asarray([ 1e-18, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf,  _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelLogGaussian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        f = 10*log10( A*exp(-(x-xo)**2.0/(2.0*ss**2.0)) )
          = 10/ln(10)*ln( exp(ln(A))*exp(-(x-xo)**2.0/(2.0*ss**2.0)) )
          = 10/ln(10)*( ln(A) -(x-xo)**2.0/(2.0*ss**2.0)  )
            A = af[0]     and AA>0
            xo = af[1]
            ss = af[2]

        y = a*x^2 + b*x + c
            a = 10/ln(10) / (2.0*ss**2.0)
            b = 10/ln(10) * 2*xo / (2.0*ss**2.0)
            c = 10/ln(10)*( ln(A) - xo**2.0 / (2.0*ss**2.0) )
        """
        AA, x0, ss = tuple(aa)
#        return 10.0*_np.log10(ModelGaussian._model(XX, aa, **kwargs))
        return (10.0/_np.log(10.0))*(log(AA)-(XX-x0)*(XX-x0)/(2.0*ss*ss))

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        f = 10*_np.log10( A*exp(-(x-xo)**2.0/(2.0*ss**2.0)) )
          = 10/ln(10)*( ln(A) -(x-xo)**2.0/(2.0*ss**2.0)  )
            A = af[0]
            xo = af[1]
            ss = af[2]


        f = 10/ln(10)*ln(g)   where g is gaussian

        dfdx = 10/ln(10) * dln(g)/dx
             = 10/ln(10) * dgdx/g
             or with less function calls

        dfdx = -10.0*(x-xo)/(ss**2.0 * _np.log(10))
        """
        AA, xo, ss = tuple(aa)
        return -10.0*(XX-xo)/(_np.log(10.0)*ss*ss)
#        return (10.0*ModelGaussian._deriv(XX, aa, **kwargs)/
#               (_np.log(10)*ModelGaussian._model(XX, aa, **kwargs)))

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        f = 10*_np.log10( A*exp(-(x-xo)**2.0/(2.0*ss**2.0)) )
          = 10/ln(10)*( ln(A) -(x-xo)**2.0/(2.0*ss**2.0)  )
            A = af[0]
            xo = af[1]
            ss = af[2]


        f = 10/ln(10)*ln(g)   where g is gaussian

        dfdx = 10/ln(10) * dln(g)/dx
             = 10/ln(10) * dgdx/g
        or with less function calls
        dfdx = -10.0*(x-xo)/(ss**2.0 * _np.log(10))

        d2fdx2 = d/dx dlngdx = d/dx dgdx/g = d2gdx2/g-dgdx^2/g^2
               = d2gdx2/g - (dlngdx)^2
        or with less function calls:
        d2fdx2 = -10.0/(ss**2.0 * _np.log(10))
        """
        AA, xo, ss = tuple(aa)
        return (-10.0/(ss*ss * _np.log(10.0)))*_np.ones_like(XX)

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        f = 10*_np.log10( A*exp(-(x-xo)**2.0/(2.0*ss**2.0)) )
          = 10/ln(10)*( ln(A) -(x-xo)**2.0/(2.0*ss**2.0)  )

        dfdA = 10/(A*ln(10))
        dfdxo = 10*(x-xo)/(ss^2 * ln(10))
        dfdss =  10*(x-xo)^2/(ss^3 * ln(10))
        """
        AA, x0, ss = tuple(aa)
        gvec = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = 10.0/(AA*_np.log(10.0))
        gvec[1,:] = 10.0*((XX-x0)/(ss*ss*_np.log(10.0)))
        gvec[2,:] = 10.0*(XX-x0)*(XX-x0)/(ss*ss*ss*_np.log(10.0))
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        f = 10*_np.log10( A*exp(-(x-xo)**2.0/(2.0*ss**2.0)) )
          = 10/ln(10)*( ln(A) -(x-xo)**2.0/(2.0*ss**2.0)  )

        dfdA = 10/(A*ln(10))
        dfdxo = 10*(x-xo)/(ss^2 * ln(10))
        dfdss =  10*(x-xo)^2/(ss^3 * ln(10))

        dfdx = -10.0*(x-xo)/(ss**2.0 * _np.log(10))
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2  * 10.0/ln(10)

        d^2f/dxdA = 0.0
        d^2f/dxdxo = 10.0/(_np.log(10.0)*ss^2)
        d^2f/dxdss = 20.0*(x-xo)/(_np.log(10.0)*ss^3)
        """
        AA, x0, ss = tuple(aa)

        dgdx = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        dgdx[1,:] = 10.0/(_np.log(10.0)*ss*ss)
        dgdx[2,:] = 20.0*(XX-x0)/(_np.log(10.0)*ss*ss*ss)
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        f = 10*_np.log10( A*exp(-(x-xo)**2.0/(2.0*ss**2.0)) )
          = 10/ln(10)*( ln(A) -(x-xo)**2.0/(2.0*ss**2.0)  )

        dfdx = -10.0*(x-xo)/(ss**2.0 * _np.log(10))
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2  * 10.0/ln(10)
        d2fdx2 = -10.0/(ss**2.0 * _np.log(10)

        d2f/dxdA = 0.0
        d2f/dxdxo = 10.0/(_np.log(10.0)*ss^2)
        d2f/dxdss = 20.0*(x-xo)/(_np.log(10.0)*ss^3)

        d3f/dx2dA = 0.0
        d3f/dx2dxo = 0.0
        d3f/dx2dss = 20.0/(_np.log(10.0)*ss^3)
        """
        AA, x0, ss = tuple(aa)
        d2gdx2 = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        d2gdx2[2,:] = 20.0/(_np.log(10.0)*ss*ss*ss)
        return d2gdx2


#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        If fitting with scaling, then the algebra necessary to unscale the problem
        to original units is much easier than the gaussian case:

            y-scaling:  y' = (y-yo)/ys
             (y-miny)/(maxy-miny) = (y-yo)/ys
             (y-yo)/ys = 10/ln(10)*( ln(a') -(x-b')**2.0/(2.0*(c')**2.0)  )

             y = yo + 10/ln(10)* ys * ( ln(a') -(x-b')**2.0/(2.0*(c')**2.0)  )
                 10.0/ln(10)*ln(a) = yo+10.0*ys*ln(a')/ln(10)
                       1/(c**2.0) = 10.0*ys/( ln(10)*(c')**2.0)

                 a = exp( ln(10)/10.0*yo+ys*ln(a') )
                 b = b'
                 c = c'*_np.sqrt( ln(10.0)/ (10.0*ys) )
                     possible to shift and scale y-data (linear problem)

            x-scaling: x' = (x-xo)/xs
             y = yo + 10/ln(10)* ys * ( ln(a') -(x'-b')**2.0/(2.0*(c')**2.0)  )
               = yo + 10/ln(10)* ys * ( ln(a') -(x-xo-xs*b')**2.0/(2.0*(xs*c')**2.0)  )

             here:
                 a = exp( ln(10)*yo/10.0+ys*ln(a') )
                 b = b'*xs + xo
                 c = c'*xs*_np.sqrt( ln(10.0)/ (10.0*ys) )

                    possible to shift and scale x- and y-data
         found in the info Structure
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = exp( _np.log(10.0)*yo/10.0 + ys*log(ain[0]))
        aout[1] = xs*ain[1]+xo
        aout[2] = xs*aout[2]*power( _np.log(10.0)/(10.0*ys), 0.5)
        return aout

    def scaleaf(self, ain, **kwargs):
        """
         a = exp( ln(10)*yo/10.0+ys*ln(a') )
         b = b'*xs + xo
         c = c'*xs*_np.sqrt( ln(10.0)/ (10.0*ys) )

         a' = exp(( ln(a) - ln(10)*yo/10.0 )/ys)
         b' = (b-xo)/xs
         c' = c/(xs*sqrt(ln(10.0)/ (10.0*ys)))
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = exp( ( log(ain[0]) - _np.log(10.0)*yo/10.0 )/ys )
        aout[1] = (ain[1]-xo)/xs
        aout[2] = (ain[2])/(xs*power( _np.log(10.0)/ (10.0*ys), 0.5))
        return aout

    # ====================================== #

    def scalings(self, xdat, ydat, **kwargs):
        """
        """
        self.xslope = 1.0
        self.xoffset = 0.0
        self.slope = 1.0
        self.offset = 0.0
        return super(ModelLogGaussian, self).scalings(xdat, ydat, **kwargs)

#    def checkbounds(self, dat, ain):
#        self.aa = super(ModelLogGaussian, self).checkbounds(dat, self.aa, mag=None)
#        dat = _np.copy(dat)
#        if (dat<0.0).any():  dat[dat<0] = min((1e-10, 1e-3*_np.min(dat[dat>0]))) # end if
#        return dat, self.aa

    # ====================================== #
# end def ModelLogGaussian


# ========================================================================== #
# ========================================================================== #


def lorentzian(XX, aa):
    return ModelLorentzian._model(XX, aa)

def deriv_lorentzian(XX, aa):
    return ModelLorentzian._deriv(XX, aa)

def partial_lorentzian(XX, aa):
    return ModelLorentzian._partial(XX, aa)

def partial_deriv_lorentzian(XX, aa):
    return ModelLine._partial_deriv(XX, aa)

def model_lorentzian(XX=None, af=None, **kwargs):
    return _model(ModelLorentzian, XX, af, **kwargs)

# =========================================== #


class ModelLorentzian(ModelClass):
    """
    A lorentzian with three free parameters:
        XX - x - independent variable
        af - magnitude, shift, width

    Lorentzian normalization such that integration equals AA (af[0])
        f = 0.5*A*s / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi
    """
    _af = _np.asarray([1.0, 0.4, 0.05], dtype=_np.float64)
    _LB = _np.asarray([-_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelLorentzian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        Lorentzian normalization such that integration equals AA (af[0])
            f = 0.5*A*ss / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi
        """
        AA, x0, ss = tuple(aa)
        return AA*0.5*ss/((XX-x0)*(XX-x0)+0.25*ss*ss)/_np.pi

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        ´Derivative of a Lorentzian normalization such that integration equals AA (af[0])
            f = 0.5*A*ss / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi

        dfdx = -2*(x-xo)*f/( ss^2/4 +(x-xo)^2)
            or
        dfdx = -16*A*ss*(x-xo)/( ss^2 +4*(xo-x)^2)^2 / pi
        """
        AA, x0, ss = tuple(aa)
        prof = ModelLorentzian._model(XX, aa, **kwargs)
        denom = ((XX-x0)*(XX-x0)+0.25*ss*ss)
        dx = 2*(XX-x0)

        dfdx = -1*prof*dx/denom
        return dfdx
#        return -1.0*AA*16.0*(XX-x0)*ss/(_np.pi*power(4.0*(XX-x0)*(XX-x0)+ss*ss, 2.0))

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        ´Derivative of a Lorentzian normalization such that integration equals AA (af[0])
            f = 0.5*A*ss / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi

        dfdx = -2*(x-xo)*f/( ss^2/4 +(x-xo)^2)
        d2fdx2 = -2*(f+(x-xo)*dfdx)/( ss^2/4 +(x-xo)^2)
                 + 4*(x-xo)^2*f/( ss^2/4 +(x-xo)^2)^2
               = -2*(f+(x-xo)*dfdx)/( ss^2/4 +(x-xo)^2)
                -2*(x-xo)*dfdx/( ss^2/4 +(x-xo)^2)
               = -2*(f+2*(x-xo)*dfdx)/( ss^2/4 +(x-xo)^2)
            or
        dfdx = -16*A*ss*(x-xo)/( ss^2 +4*(xo-x)^2)^2 / pi
        d2fdx2 = -16*A*ss*( ss^2-12*(x-xo)^2 )/( ss^2 +4*(x-xo)^2)^3 / pi
        """
        AA, x0, ss = tuple(aa)
        prof = ModelLorentzian._model(XX, aa, **kwargs)
        dfdx = ModelLorentzian._deriv(XX, aa, **kwargs)
        denom = ((XX-x0)*(XX-x0)+0.25*ss*ss)
        dx = 2*(XX-x0)
        d2fdx2 = -2*(prof+dx*dfdx)/denom
        return d2fdx2
#        return -1.0*AA*16.0*ss*(ss*ss-12.0*(XX-x0)*(XX-x0))/(_np.pi*power(4.0*(XX-x0)*(XX-x0)+ss*ss, 3.0))

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        Jacobian of a Lorentzian normalization such that integration equals AA (af[0])
            f = 0.5*A*ss / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi

        #  These forms work, but they are really hard to debug
        dfdx = -16*A*ss*(x-xo)/( ss^2 +4*(xo-x)^2)^2 / pi
        d2fdx2 = -16*A*ss*( ss^2-12*(x-xo)^2 )/( ss^2 +4*(x-xo)^2)^3 / pi

        dfdA = f/A
        dfdxo = A*ss*(x-xo)/( ss^2/4.0 + (x-xo)^2 )^2  / pi
        dfds = -2*A*(ss^2-4*(x-xo)^2)/( b^2 + 4*(x-xo)^2 )^2 / pi

         ----------------

        dfdx = -2*(x-xo)*f/( ss^2/4 +(x-xo)^2)
        d2fdx2 = -2*(f+2*(x-xo)*dfdx)/( ss^2/4 +(x-xo)^2)

        dfdA = f/A
        dfdxo = -0.5*A*ss*2*-1*(x-xo) / ( (x-xo)**2.0 + 0.25*ss**2.0 )^2 / pi
              = 2*f*(x-xo) / ( (x-xo)**2.0 + 0.25*ss**2.0 )
        dfds = -0.5*A*ss*( 0.25*2*ss ) / ( (x-xo)**2.0 + 0.25*ss**2.0 )^2 / pi
               + f/s
             = f/s-f*( 0.5*ss ) / ( (x-xo)**2.0 + 0.25*ss**2.0 )
        """
        AA, x0, ss = tuple(aa)
        prof = ModelLorentzian._model(XX, aa, **kwargs)
        denom = ((XX-x0)*(XX-x0)+0.25*ss*ss)
        dx0 = -2*(XX-x0)

        gvec = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = prof/AA
        gvec[1,:] = -prof*dx0/denom
        gvec[2,:] = prof/ss - 0.5*prof*ss/denom

#        gvec[0,:] = 0.5*ss/((XX-x0)*(XX-x0)+0.25*ss*ss)/_np.pi
#        gvec[1,:] = AA*ss*(XX-x0)/power((XX-x0)*(XX-x0)+0.25*ss*ss, 2.0 )/_np.pi
#        gvec[2,:] = -2.0*AA*(ss*ss-4.0*(XX-x0)*(XX-x0))/power(ss*ss+4.0*(XX-x0)*(XX-x0), 2.0)/_np.pi
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a Lorentzian normalization such that
        integration equals AA (af[0])
            f = 0.5*A*ss / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi

        ------------------------

        dfdx = -2*(x-xo)*f/( ss^2/4 +(x-xo)^2)
        d2fdx2 = -2*(f+2*(x-xo)*dfdx)/( ss^2/4 +(x-xo)^2)

        dfdA = f/A
        dfdxo = 2*f*(x-xo) / ( (x-xo)**2.0 + 0.25*ss**2.0 )
        dfds = f/s-f*( 0.5*ss ) / ( (x-xo)**2.0 + 0.25*ss**2.0 )

        d2fdxdA = dfdx/A
        d2fdxdxo = 2*dfdx*(x-xo) / ( (x-xo)^2.0 + 0.25*ss^2.0 )
                 + 2*f / ( (x-xo)^2.0 + 0.25*ss^2.0 )
                 - 4*f*(x-xo)^2 / ( (x-xo)^2.0 + 0.25*ss^2.0 )^2
                = (2*dfdx*(x-xo)+2f) / ( (x-xo)^2.0 + 0.25*ss^2.0 )
                 + 2*dfdx*(x-xo)/ ( (x-xo)^2.0 + 0.25*ss^2.0 )
                = 2*(2*dfdx*(x-xo)+f) / ( (x-xo)^2.0 + 0.25*ss^2.0 )
        d2fdxds = dfdx/s
                - dfdx*( 0.5*ss ) / ( (x-xo)**2.0 + 0.25*ss**2.0 )
                + f*( 0.5*ss )*2*(x-xo) / ( (x-xo)**2.0 + 0.25*ss**2.0 )^2
                = dfdx/s - ss*dfdx/( (x-xo)**2.0 + 0.25*ss**2.0 )
        """
        AA, x0, ss = tuple(aa)
        prof = ModelLorentzian._model(XX, aa, **kwargs)
        dfdx = ModelLorentzian._deriv(XX, aa, **kwargs)
        denom = ((XX-x0)*(XX-x0)+0.25*ss*ss)
        dx = 2*(XX-x0)

        dgdx = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = dfdx/AA
        dgdx[1,:] = 2*(prof+dx*dfdx)/denom
        dgdx[2,:] = dfdx/ss - ss*dfdx/denom

#        dgdx[0,:] = -1.0*16.0*(XX-x0)*ss/(_np.pi*power(4.0*(XX-x0)*(XX-x0)+ss*ss, 2.0))
#        dgdx[1,:] = 16.0*AA*ss*(ss*ss-12.0*(XX-x0)*(XX-x0))/power(ss*ss+4*power(XX-x0, 2.0),3.0)/_np.pi
#        dgdx[2,:] = 16.0*AA*(XX-x0)*(3.0*ss*ss-4.0*(XX*XX-2.0*XX*x0+x0*x0))/(_np.pi*power(ss*ss+4.0*(XX*XX-2.0*XX*x0+x0*x0), 3.0))
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        Jacobian of the 2nd derivative of a Lorentzian normalization such that
        integration equals AA (af[0])
            f = 0.5*A*ss / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi

        # these generally work, but I cannot debug d2fdx2dxo.  rewriting!
        dfdx = -16*A*ss*(x-xo)/( ss^2 +4*(xo-x)^2)^2 / pi
        d2fdx2 = -16*A*ss*( ss^2-12*(x-xo)^2 )/( ss^2 +4*(x-xo)^2)^3 / pi

        dfdA = f/A
        dfdxo = A*ss*(x-xo)/( ss^2/4.0 + (x-xo)^2 )^2  / pi
        dfds = -2*A*(ss^2-4*(x-xo)^2)/( b^2 + 4*(x-xo)^2 )^2 / pi

        d2fdxdA = -16*ss*(x-xo)/( ss^2 +4*(xo-x)^2)^2 / pi
        d2fdxdxo = 16*AA*ss*(ss^2-12*(XX-x0)^2)/(ss^2+4*(XX-x0)^2)^3/pi
        d2fdxdss = 16*AA*(XX-x0)*(3*ss^2-4(XX^2-2XX*x0+x0^2))/(pi*(ss^2+4(XX^2-2XX*x0+x0^2))^3)

        d3fdx2dA = -16*ss*( ss^2-12*(x-xo)^2 )/( ss^2 +4*(x-xo)^2)^3 / pi
        d3fdx2dxo = 768 AA*ss*(x0-x)*(b^2-4*(c-x)^2)/(b^2+4*(x0-XX)^2)^4/pi
        d3fdx2dss = 48*AA*(ss^4-24*ss^2*(x-xo)^2+16*(c-x)^4)/(b^2+4*(x0-XX)^2)^4/pi

        ---------------------------
        Reformulated:
        dfdx = -2*(x-xo)*f/( ss^2/4 +(x-xo)^2)
        d2fdx2 = -2*(f+2*(x-xo)*dfdx)/( ss^2/4 +(x-xo)^2)

        dfdA = f/A
        dfdxo = 2*f*(x-xo) / ( (x-xo)**2.0 + 0.25*ss**2.0 )
        dfds = f/s-f*( 0.5*ss ) / ( (x-xo)**2.0 + 0.25*ss**2.0 )

        d2fdxdA = dfdx/A
        d2fdxdxo = 2*(2*dfdx*(x-xo)+f) / ( (x-xo)^2.0 + 0.25*ss^2.0 )
        d2fdxds = dfdx/s - ss*dfdx/( (x-xo)**2.0 + 0.25*ss**2.0 )

        d3fdx2dA = d2fdx2/A
        d3fdx2dxo = 2*(2*d2fdx2*(x-xo)+2*dfdx+dfdx) / ( (x-xo)^2.0 + 0.25*ss^2.0 )
                  - 2*(x-xo)*2*(2*dfdx*(x-xo)+f) / ( (x-xo)^2.0 + 0.25*ss^2.0 )^2
                  = 2*(2*d2fdx2*(x-xo)+3*dfdx - 2*(x-xo)*d2fdxdxo) / ( (x-xo)^2.0 + 0.25*ss^2.0 )
        d3fdx2ds = d2fdx2/s - ss*d2fdx2/( (x-xo)**2.0 + 0.25*ss**2.0 )
                + 2*(x-xo)* ss*dfdx/( (x-xo)**2.0 + 0.25*ss**2.0 )^2
                 = d2fdx2/s - ss*d2fdx2/( (x-xo)**2.0 + 0.25*ss**2.0 )
                + 2*(x-xo)* ss*dfdx/( (x-xo)**2.0 + 0.25*ss**2.0 )^2
        """
        AA, x0, ss = tuple(aa)
#        prof = ModelLorentzian._model(XX, aa, **kwargs)
        dfdx = ModelLorentzian._deriv(XX, aa, **kwargs)
        d2fdx2 = ModelLorentzian._deriv2(XX, aa, **kwargs)
        d2fdxdxo = ModelLorentzian._partial_deriv(XX, aa, **kwargs)[1,:]
        denom = ((XX-x0)*(XX-x0)+0.25*ss*ss)
        dx = 2*(XX-x0)

        d2gdx2 = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        d2gdx2[0,:] = d2fdx2/AA
        d2gdx2[1,:] = (2.0*d2fdx2*dx+6.0*dfdx-dx*d2fdxdxo)/denom
        d2gdx2[2,:] = d2fdx2/ss - ss*d2fdx2/denom + dx*ss*dfdx/(denom*denom)
#        d2gdx2[0,:] = -1.0*16.0*ss*( ss*ss-12.0*(XX-x0)*(XX-x0) )/power( ss*ss +4.0*(XX-x0)*(XX-x0), 3.0) / _np.pi
#        d2gdx2[1,:] =768.0*AA*ss*(x0-XX)*(ss*ss-4.0*power(x0-XX,4.0))/power(ss*ss+4.0*power(x0-XX,2.0), 4.0)/_np.pi
#        d2gdx2[2,:] = 48.0*AA*(power(ss,4.0)-24.0*ss*ss*power(XX-x0,2.0)+16.0*power(XX-x0,4.0))/power(ss*ss+4.0*power(XX-x0,2.0), 4.0)/_np.pi
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        If fitting with scaling, then the algebra necessary to unscale the problem
        to original units is:

           y-scaling: y' = (y-yo)/ys
            (y-yo) / ys = 0.5*a'*c' / ( (x-b')^2 + 0.25*c'^2 ) / pi

            y-yo = 0.5*ys*a'*c' / ( (x-b')^2 + 0.25*c'^2 ) / pi
            a*c/((x-b)^2 + 0.25*c^2) = 2*pi*yo + ys*a'*c'/((x-b')^2 + 0.25*c'^2)

            we could expand and gather polynomial terms, but the result would be yo=0.0
            for now assume b = b' and c=c'
            a = ys*a' + 2*pi*yo/c*(x^2-2xb+b^2 + 0.25*c^2)
            a = ys*a'  iff yo=0

           x-scaling: x' = (x-xo)/xs
            y-yo = 0.5*ys*a'*c' / ( (x-xo-xs*b')^2/xs^2 + 0.25*c'^2 ) / pi
                 = 0.5*ys*xs*a'*xs*c' /( (x-xo-xs*b')^2 + 0.25*xs^2*c'^2 )/pi
            a = ys*xs*a'
            b = xs*b'+xo
            c = xs*c'
            and yo = 0.0
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = ys*xs*aout[0]
        aout[1] = xs*aout[1]+xo
        aout[2] = xs*aout[2]
        return aout

    def scaleaf(self, ain, **kwargs):
        """
            a = ys*xs*a'
            b = xs*b'+xo
            c = xs*c'
            and yo = 0.0
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = ain[0]/(xs*ys)
        aout[1] = (ain[1]-xo)/xs
        aout[2] = ain[2]/xs
        return aout

    def scalings(self, xdat, ydat, **kwargs):
#        self.xslope = 1.0
#        self.xoffset = 0.0
#        self.slope = 1.0
        self.offset = 0.0
        return super(ModelLorentzian, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelLorentzian, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelLorentzian

# =========================================================================== #
# =========================================================================== #


def pyseudovoigt(XX, aa):
    return ModelPseudoVoigt._model(XX, aa)

def deriv_pyseudovoigt(XX, aa):
    return ModelPseudoVoigt._deriv(XX, aa)

def partial_pyseudovoigt(XX, aa):
    return ModelPseudoVoigt._partial(XX, aa)

def partial_deriv_pyseudovoigt(XX, aa):
    return ModelPseudoVoigt._partial_deriv(XX, aa)

def model_pseudovoigt(XX=None, af=None, **kwargs):
    return _model(ModelPseudoVoigt, XX, af, **kwargs)

class ModelPseudoVoigt(ModelClass):
    """
    A Voigt pfunction is the convolution of a lorentzian / gaussian

    The pseudo-Voigt function is a numerical approximation:
        https://en.wikipedia.org/wiki/Voigt_profile
    """
    _af = _np.concatenate(([0.5], ModelLorentzian._af, ModelNormal._af), axis=0)
    _LB = _np.concatenate(([0.0], ModelLorentzian._LB, ModelNormal._LB), axis=0)
    _UB = _np.concatenate(([1.0], ModelLorentzian._UB, ModelNormal._UB), axis=0)
    _fixed = _np.zeros( (7,), dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        # initialize with the two profiles overlapping in position
        self._af[5] = self._af[2]

        super(ModelPseudoVoigt, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        return (aa[0]*ModelLorentzian._model(XX, aa[1:4], **kwargs)
               + (1.0-aa[0])*ModelNormal._model(XX, aa[4:], **kwargs))

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        return (aa[0]*ModelLorentzian._deriv(XX, aa[1:4], **kwargs)
               + (1.0-aa[0])*ModelNormal._deriv(XX, aa[4:], **kwargs))

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        return (aa[0]*ModelLorentzian._deriv2(XX, aa[1:4], **kwargs)
               + (1.0-aa[0])*ModelNormal._deriv2(XX, aa[4:], **kwargs))

    @staticmethod
    def _partial(XX, aa, **kwargs):
        gvec = _np.concatenate( (_np.atleast_2d(ModelLorentzian._model(XX, aa[1:4], **kwargs)
                                - ModelNormal._model(XX, aa[4:], **kwargs)),
                            aa[0]*ModelLorentzian._partial(XX, aa[1:4], **kwargs),
                           (1.0-aa[0])*ModelNormal._partial(XX, aa[4:], **kwargs)), axis=0)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        dgdx = _np.concatenate( (_np.atleast_2d(ModelLorentzian._deriv(XX, aa[1:4], **kwargs)
                                - ModelNormal._deriv(XX, aa[4:], **kwargs)),
                                 aa[0]*ModelLorentzian._partial_deriv(XX, aa[1:4], **kwargs),
                           (1.0-aa[0])*ModelNormal._partial_deriv(XX, aa[4:], **kwargs)), axis=0)
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        d2gdx2 = _np.concatenate( (_np.atleast_2d(ModelLorentzian._deriv2(XX, aa[1:4], **kwargs)
                                - ModelNormal._deriv2(XX, aa[4:], **kwargs)),
                                 aa[0]*ModelLorentzian._partial_deriv2(XX, aa[1:4], **kwargs),
                           (1.0-aa[0])*ModelNormal._partial_deriv2(XX, aa[4:], **kwargs)), axis=0)
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain, **kwargs):
        """
        A combination of lorentzian and normal models
        y = a * L + (1-a)*N
            Note that x-shift and xy-scaling works for Normal/Lorentzian models
            but that y-shifting does not
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        LM = ModelLorentzian(None)
        LM.slope = ys
        LM.offset = yo
        LM.xslope = xs
        LM.xoffset = xo
        aout[1:4] = LM.unscaleaf(aout[1:4])

        NM = ModelNormal(None)
        NM.slope = ys
        NM.offset = yo
        NM.xslope = xs
        NM.xoffset = xo
        aout[4:] = NM.unscaleaf(aout[4:])
        return aout

    def scaleaf(self, ain, **kwargs):
        """
            a = ys*xs*a'
            b = xs*b'+xo
            c = xs*c'
            and yo = 0.0
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        LM = ModelLorentzian(None)
        LM.slope = ys
        LM.offset = yo
        LM.xslope = xs
        LM.xoffset = xo
        NM = ModelNormal(None)
        NM.slope = ys
        NM.offset = yo
        NM.xslope = xs
        NM.xoffset = xo
        aout[1:4] = LM.scaleaf(aout[1:4])
        aout[4:] = NM.scaleaf(aout[4:])
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
        return super(ModelPseudoVoigt, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelPseudoVoigt, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelPseudoVoigt

# ========================================================================== #
# ========================================================================== #


def loglorentzian(XX, aa):
    return ModelLogLorentzian._model(XX, aa)

def deriv_loglorentzian(XX, aa):
    return ModelLogLorentzian._deriv(XX, aa)

def partial_loglorentzian(XX, aa):
    return ModelLogLorentzian._partial(XX, aa)

def partial_deriv_loglorentzian(XX, aa):
    return ModelLogLorentzian._partial_deriv(XX, aa)

def model_loglorentzian(XX=None, af=None, **kwargs):
    return _model(ModelLogLorentzian, XX, af, **kwargs)

# =========================================== #


class ModelLogLorentzian(ModelClass):
    """
    A log-lorentzian with three free parameters:
        XX - x - independent variable
        af - magnitude, shift, width

      f =  10.0*_np.log10( AA*0.5*ss/((XX-x0)^2+0.25*ss^2)/_np.pi )

      f = 10*log10(A) + 10*log10(s) + 10*log10(0.5) - 10*log10(pi)
        - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )

    A log-lorentzian is a shifted log-parabola in x-s space that is centered at (xo,0)
    """
    _af = _np.asarray([    1.0,      0.4,    0.05], dtype=_np.float64)
    _LB = _np.asarray([  1e-18, -_np.inf,   1e-18], dtype=_np.float64)
    _UB = _np.asarray([_np.inf,  _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelLogLorentzian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        log of a Lorentzian
            f = 10*log10(  0.5*A*s / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi  )
              = 10*log10( 0.5*A*s )
                - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )
                - 10*log10(pi)
            f = 10*log10( 0.5*A*s ) - 10*log10(pi)
              - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )
              shifted log of a quadratic
        """
#        return 10.0*log(lorentzian(XX, aa))/_np.log(10.0)
        AA, x0, ss = tuple(aa)
        return ( 10.0*_np.log10(0.5) + 10.0*_np.log10(AA)  + 10.0*log(ss)/_np.log(10.0) - 10.0*_np.log10(_np.pi)
               - 10.0*log((XX-x0)*(XX-x0) + 0.25*ss*ss)/_np.log(10.0) )

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        derivative of the log of a Lorentzian
            f = 10*log10( 0.5*A*s ) - 10*log10(pi)
              - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )

            dfdx = d/dx ( - 10*_np.log10((x-xo)**2.0 + 0.25*ss**2.0 ) )
                 = -2.0*(x-xo)*10.0/((x-xo)**2.0 + 0.25*ss**2.0 )/_np.log(10)
        """
        AA, x0, ss = tuple(aa)
        return -20.0*(XX-x0)/( _np.log(10.0) *((XX-x0)*(XX-x0) + 0.25*ss*ss) )

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        derivative of the log of a Lorentzian
            f = 10*log10( 0.5*A*s ) - 10*log10(pi)
              - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )

            dfdx = d/dx ( - 10*_np.log10((x-xo)**2.0 + 0.25*ss**2.0 ) )
                 = -2.0*(x-xo)*10.0/((x-xo)**2.0 + 0.25*ss**2.0 )/_np.log(10)
        """
        AA, x0, ss = tuple(aa)
        denom = 4.0*(XX-x0)*(XX-x0)+ss*ss
        return -80.0*(ss*ss-4.0*(XX-x0)*(XX-x0))/( _np.log(10.0) *(denom*denom) )

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        jacobian of the log of a Lorentzian
        f = 10*log10( 0.5*A*s ) - 10*log10(pi)
          - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )


        dfdai = d/dai ( ... )

        dfdA = d/dA 10*log10( 0.5*A*s )
             = d/dA 10*log10( 0.5)+10*log10(A)+10*log10(s )
             = 10.0/( _np.log(10.0)*A )
        dfdxo = -d/dxo 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )
              = 2.0*(x-xo)*10.0/((x-xo)**2.0 + 0.25*ss**2.0 )/_np.log(10)
        dfdss = d/dss 10*log10( 0.5*A*s ) - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )
              = 10/(s*log(10)) -0.25*2.0*10*ss / ( _np.log(10)*((x-xo)**2.0 + 0.25*ss**2.0) )

       """
        AA, x0, ss = tuple(aa)
        denom = (XX-x0)*(XX-x0)+0.25*ss*ss
        gvec = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = 10.0/(_np.log(10.0)*AA)
        gvec[1,:] = 20.0*(XX-x0)/( _np.log(10.0)*denom)
        gvec[2,:] = 10.0/(_np.log(10.0)*ss) - 5.0*ss/( _np.log(10.0)*denom)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        derivative of the log of a Lorentzian
        f = 10*log10( 0.5*A*s ) - 10*log10(pi)
          - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )

        dfdx = d/dx ( - 10*_np.log10((x-xo)**2.0 + 0.25*ss**2.0 ) )
             = -20.0*(x-xo)/((x-xo)**2.0 + 0.25*ss**2.0 )/_np.log(10)

        d2f/dxdAA = 0.0
        d2f/dxdx0 = 80.0*(s-2.0*(x-xo))*(s+2.0*(x-xo))/(ln|10|*(4.0*(x-xo)**2.0 + ss**2.0)**2.0 )
        d2f/dxds = 10*s*(x-xo)/( ln|10|*( (x-xo)^2+0.25s^2 )^2 )
        """
        AA, x0, ss = tuple(aa)
        denom1 = 4.0*(XX-x0)*(XX-x0)+ss*ss
        denom2 = (XX-x0)*(XX-x0)+0.25*ss*ss
        dgdx = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        dgdx[1,:] = 80.0*(ss-2.0*(XX-x0))*(ss+2.0*(XX-x0))/( _np.log(10)*denom1*denom1 )
        dgdx[2,:] = 10.0*ss*(XX-x0)/(_np.log(10.0)*denom2*denom2 )
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        2nd derivative of the log of a Lorentzian
        f = 10*log10( 0.5*A*s ) - 10*log10(pi)
          - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )

        dfdx = d/dx ( - 10*_np.log10((x-xo)**2.0 + 0.25*ss**2.0 ) )
             = -20.0*(x-xo)/((x-xo)**2.0 + 0.25*ss**2.0 )/_np.log(10)

        d2fdxdA = 0.0
        d2fdxdx0 = 80.0*(s-2.0*(x-xo))*(s+2.0*(x-xo))/(ln|10|*(4.0*(x-xo)**2.0 + ss**2.0)**2.0 )
        d2fdxds = 10*s*(x-xo)/( ln|10|*( (x-xo)^2+0.25s^2 )^2 )

        d2fdx2dA = 0.0
        d2fdx2dxo = 10*(x-xo)*(3s^2-4*(x-xo)^2)/( ln|10|*( (x-xo)^2+0.25s^2 )^3 )
        d2fdx2ds = 2.5*s*(s^2-12*(x-xo)^2)/(ln|10|*(s^2/4+(x-xo)^2)^3)
        """
        AA, x0, ss = tuple(aa)
        denom1 = 4.0*(XX-x0)*(XX-x0)+ss*ss
        d2gdx2 = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        d2gdx2[1,:] = -640.0*(XX-x0)*(3.0*ss*ss-4.0*(XX-x0)*(XX-x0))/(_np.log(10.0)*denom1*denom1*denom1)
        d2gdx2[2,:] = 160.0*ss*(ss*ss-12.0*(XX-x0)*(XX-x0))/(_np.log(10.0)*denom1*denom1*denom1)
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        If fitting with scaling, then the algebra necessary to unscale the problem
        to original units is:

           y-scaling: y' = (y-yo)/ys
            (y-yo) / ys = 10*log10(a')-10*log10(c')+10*log10(0.5)-10*log10(pi)
                        - 10*log10((x-b')**2.0 + 0.25*(c')**2.0 )

            y = yo + ys*(10*log10(a')-10*log10(c')+10*log10(0.5)-10*log10(pi)
                        - 10*log10((x-b')**2.0 + 0.25*(c')**2.0 ))
                       group terms to make it easier
                   10*ln(a)/ln(10) = yo+10.0*ys*ln(a')/ln(10)
                      ln(a) = ln(10)/10*yo+ys*ln(a')
                         a = exp(ln(10)/10*yo+ys*ln(a') )
                         b = b'
                         c = c'

           x-scaling: x' = (x-xo)/xs
            y = yo + ys*(10*log10(a')-10*log10(c')+10*log10(0.5)-10*log10(pi)
                 - 10*log10(((x-xo-xs*b')**2.0 + 0.25*(xs*c')**2.0 ))/xs**2.0 )
              = yo + ys*(10*log10(a')-10*log10(c')+10*log10(0.5)-10*log10(pi)
                 - 10*log10((x-xo-xs*b')**2.0 + 0.25*(xs*c')**2.0)
                 - 10*log10( xs**2.0 ) )

                         a = exp(ln(10)/10*yo-2.0*ys*ln(xs)+ys*ln(a'))
                         b = xs*b'+xo
                         c = xs*c'
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        aout[0] = exp( _np.log(10.0)*yo/10.0 - 2.0*ys*log(xs)+ys*log(aout[0]))
        aout[1] = xs*aout[1]+xo
        aout[2] = xs*aout[2]
        return aout

    def scaleaf(self, ain, **kwargs):
        """
            a = ys*xs*a'
            b = xs*b'+xo
            c = xs*c'
            and yo = 0.0
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)


        aout[0] = exp( (log(ain[0])-_np.log(10.0)/10.0*yo-2.0*ys*log(xs))/ys )
        aout[1] = (ain[1]-xo)/xs
        aout[2] = ain[2]/xs
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xslope = 1.0  # problem
#        self.xoffset = 0.0
        self.slope = 1.0   # problem
        self.offset = 0.0
        return super(ModelLogLorentzian, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        self.aa = super(ModelLogLorentzian, self).checkbounds(dat, self.aa, mag=None)
#        dat = _np.copy(dat)
#        if (dat<0.0).any(): dat[dat<0] = min((1e-10, 1e-3*_np.min(dat[dat>0]))) # end if
#        return dat, self.aa

    # ====================================== #
# end def ModelLogLorentzian


# ========================================================================== #
# ========================================================================== #

def _parse_noshift(ain, model_order=2):
    if model_order>1:
        return ain[0:3], ain[3:6], ain[6:]
    elif model_order>0:
        return ain[0:3], ain[3:6], []
    else:
        return ain[0:3], [], []
    # end if
# end def

def _doppler_logdata(xdat, ain, **kwargs):
    return ModelLogDoppler._model(xdat, ain, **kwargs)

def _deriv_doppler_logdata(xdat, ain, **kwargs):
    return ModelLogDoppler._deriv(xdat, ain, **kwargs)

def _partial_doppler_logdata(xdat, ain, **kwargs):
    return ModelLogDoppler._partial(xdat, ain, **kwargs)

def _partial_deriv_doppler_logdata(xdat, ain, **kwargs):
    return ModelLogDoppler._partial_deriv(xdat, ain **kwargs)

# fit model to the data
def model_doppler(XX=None, af=None, **kwargs):
    logdata = kwargs.pop('logdata', False)
#    noshift = kwargs.pop('noshift')
    if logdata:
        return _model(ModelLogDoppler, XX, af, **kwargs)
    else:
        return _model(ModelDoppler, XX, af, **kwargs)
    # end if
# end def

class ModelDoppler(ModelClass):
    """
    Model of a Doppler spectrum.  Common situations:
        1) Single peak - 3 parameter fit
            - Gaussian (Doppler shift, integrated reflected power, width)
        2) Zero frequency peak - additional 2 parameter fit
            - Lorentzian to model window reflection or device cross-talk
        3) Double peak - additional 3 parameter fit
            - Gaussian (2nd Doppler shift, integrated reflected power, width)
    """
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        model_order = kwargs.setdefault('model_order', 2)
        noshift = kwargs.setdefault('noshift', True)
        Fs = kwargs.setdefault('Fs', 1.0)
        self._af = ModelNormal._af
        self._LB = ModelNormal._LB
        self._UB = ModelNormal._UB
        self._LB[0] = 1e-18
        self._LB[1:] = -0.5*Fs
        self._UB[1:] =  0.5*Fs
        self._fixed = ModelNormal._fixed
        if model_order>0:
            self._af = _np.asarray(self._af.tolist()+ModelLorentzian._af.tolist())
            self._LB = _np.asarray(self._LB.tolist()+ModelLorentzian._LB.tolist())
            self._UB = _np.asarray(self._UB.tolist()+ModelLorentzian._UB.tolist())
            self._LB[3] = 1e-18
            self._LB[4:] = -0.5*Fs
            self._UB[4:] =  0.5*Fs
            self._fixed = _np.asarray(self._fixed.tolist()+ModelLorentzian._fixed.tolist())
            if noshift:    self._fixed[4] = 1      # end if
        if model_order>1:
            self._af = _np.asarray(self._af.tolist()+ModelNormal._af.tolist())
            self._LB = _np.asarray(self._LB.tolist()+ModelNormal._LB.tolist())
            self._UB = _np.asarray(self._UB.tolist()+ModelNormal._UB.tolist())
            self._LB[6] = 1e-18
            self._LB[7:] = -0.5*Fs
            self._UB[7:] =  0.5*Fs
            self._fixed = _np.asarray(self._fixed.tolist()+ModelNormal._fixed.tolist())
            self._af[7] *= -1.0
        # end if
        super(ModelDoppler, self).__init__(XX, af, **kwargs)
        if noshift and model_order>0:
            self.fixed[4] = 1
        # end if
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        Doppler model
        f = NormalizedGaussian1
          + Lorentzian
          + Normalized Gaussain2
        """
        model_order = kwargs.setdefault('model_order', 2)
        a0, a1, a2 = _parse_noshift(aa, model_order=model_order)
        model = ModelNormal._model(XX, a0)
        if model_order>0:            model += ModelLorentzian._model(XX, a1)
        if model_order>1:            model += ModelNormal._model(XX, a2)
        return model

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        Derivative of the Doppler model
        dfdx = d/dx (NormalizedGaussian1
             + Lorentzian
             + Normalized Gaussain2 )
        """
        model_order = kwargs.setdefault('model_order', 2)
        a0, a1, a2 = _parse_noshift(aa, model_order=model_order)
        dfdx = ModelNormal._deriv(XX, a0)
        if model_order>0:            dfdx += ModelLorentzian._deriv(XX, a1)
        if model_order>1:            dfdx += ModelNormal._deriv(XX, a2)
        return dfdx

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        2nd Derivative of the Doppler model
        dfdx = d/dx (NormalizedGaussian1
             + Lorentzian
             + Normalized Gaussain2 )
        """
        model_order = kwargs.setdefault('model_order', 2)
        a0, a1, a2 = _parse_noshift(aa, model_order=model_order)
        d2fdx2 = ModelNormal._deriv2(XX, a0)
        if model_order>0:            d2fdx2 += ModelLorentzian._deriv2(XX, a1)
        if model_order>1:            d2fdx2 += ModelNormal._deriv2(XX, a2)
        return d2fdx2

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        Jacobian of the Doppler model
        dfdai = d/dai (NormalizedGaussian1
             + Lorentzian
             + Normalized Gaussain2 )
        """
        model_order = kwargs.setdefault('model_order', 2)
        a0, a1, a2 = _parse_noshift(aa, model_order=model_order)
        gvec = ModelNormal._partial(XX, a0)
        if model_order>0: gvec = _np.concatenate( (gvec, ModelLorentzian._partial(XX, a1)), axis=0)
        if model_order>1: gvec = _np.concatenate( (gvec, ModelNormal._partial(XX, a2)), axis=0)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        Derivative of the Jacobian of the Doppler model
        d2fdxdai = d2/dxdai(NormalizedGaussian1 + Lorentzian + NormalizedGaussain2 )
        """
        model_order = kwargs.setdefault('model_order', 2)
        a0, a1, a2 = _parse_noshift(aa, model_order=model_order)
        dgdx = ModelNormal._partial_deriv(XX, a0)
        if model_order>0: dgdx = _np.concatenate( (dgdx, ModelLorentzian._partial_deriv(XX, a1)), axis=0)
        if model_order>1: dgdx = _np.concatenate( (dgdx, ModelNormal._partial_deriv(XX, a2)), axis=0)
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        Derivative of the Jacobian of the Doppler model
        d2fdxdai = d2/dxdai(NormalizedGaussian1 + Lorentzian + NormalizedGaussain2 )
        """
        model_order = kwargs.setdefault('model_order', 2)
        a0, a1, a2 = _parse_noshift(aa, model_order=model_order)
        d2gdx2 = ModelNormal._partial_deriv2(XX, a0)
        if model_order>0: d2gdx2 = _np.concatenate( (d2gdx2, ModelLorentzian._partial_deriv2(XX, a1)), axis=0)
        if model_order>1: d2gdx2 = _np.concatenate( (d2gdx2, ModelNormal._partial_deriv2(XX, a2)), axis=0)
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain, **kwargs):
        """
        A combination of Lorentzian and Normal models.
        Both can be xy-scaled, and x-shifted but not y-shifted
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)
        a0, a1, a2 = _parse_noshift(aout, model_order=self.model_order)

        NM = ModelNormal(None)
        NM.slope = ys
        NM.xoffset = xo
        NM.xslope = xs
        NM.offset = yo
        aout = NM.unscaleaf(a0)
        if self.model_order>0:
            LM = ModelLorentzian(None)
            LM.slope = ys
            LM.xoffset = xo
            LM.xslope = xs
            LM.offset = yo
            aout = _np.hstack((aout, LM.unscaleaf(a1)))
        if self.model_order>1:
            aout = _np.hstack((aout, NM.unscaleaf(a2)))
        return aout

    def scaleaf(self, ain, **kwargs):
        """
            a = ys*xs*a'
            b = xs*b'+xo
            c = xs*c'
            and yo = 0.0
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)
        a0, a1, a2 = _parse_noshift(aout, model_order=self.model_order)

        NM = ModelNormal(None)
        NM.slope = ys
        NM.xoffset = xo
        NM.xslope = xs
        NM.offset = yo
        aout = NM.scaleaf(a0)
        if self.model_order>0:
            LM = ModelLorentzian(None)
            LM.slope = ys
            LM.xoffset = xo
            LM.xslope = xs
            LM.offset = yo
            aout = _np.hstack((aout, LM.scaleaf(a1)))
        if self.model_order>1:
            aout = _np.hstack((aout, NM.scaleaf(a2)))
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
        return super(ModelDoppler, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelDoppler, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelDoppler


class ModelLogDoppler(ModelDoppler):
    """
    Log of the Doppler spectrum.  Common situations:
        1) Single peak - 3 parameter fit
            - Gaussian (Doppler shift, integrated reflected power, width)
        2) Zero frequency peak - additional 2 parameter fit
            - Lorentzian to model window reflection or device cross-talk
        3) Double peak - additional 3 parameter fit
            - Gaussian (2nd Doppler shift, integrated reflected power, width)
    """
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        model_order = kwargs.setdefault('model_order', 2)
        noshift = kwargs.setdefault('noshift', True)
#        Fs = kwargs.setdefault('Fs', 1.0)
        self._af = ModelNormal._af
        self._LB = ModelNormal._LB
        self._UB = ModelNormal._UB
        self._fixed = ModelNormal._fixed

        self._af[0] = 1e-3
        self._LB[0] = -_np.inf
        self._UB[0] = 10.0
        if model_order>0:
            self._af = _np.asarray(self._af.tolist()+ModelLorentzian._af.tolist())
            self._LB = _np.asarray(self._LB.tolist()+ModelLorentzian._LB.tolist())
            self._UB = _np.asarray(self._UB.tolist()+ModelLorentzian._UB.tolist())

            self._af[3] = 1
            self._LB[3] = -_np.inf
            self._UB[3] = 10.0
            self._fixed = _np.asarray(self._fixed.tolist()+ModelLorentzian._fixed.tolist())
            if noshift: self._fixed[4] = 1 # end if
        if model_order>1:
            self._af = _np.asarray(self._af.tolist()+ModelNormal._af.tolist())
            self._LB = _np.asarray(self._LB.tolist()+ModelNormal._LB.tolist())
            self._UB = _np.asarray(self._UB.tolist()+ModelNormal._UB.tolist())
            self._af[6] = 1e-3
            self._LB[6] = -_np.inf
            self._UB[6] = 10.0
            self._fixed = _np.asarray(self._fixed.tolist()+ModelNormal._fixed.tolist())
            self._af[7] *= -1.0
        # end if
        super(ModelDoppler, self).__init__(XX, af, **kwargs)
        if noshift and model_order>0:
            self.fixed[4] = 1
        # end if
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        Doppler model
        f = NormalizedGaussian1
          + Lorentzian
          + Normalized Gaussain2
        """
        kwargs.setdefault('model_order', 2)
        return 10.0*log(ModelDoppler._model(XX, aa, **kwargs))/_np.log(10.0)

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        Derivative of the Doppler model
        dfdx = d/dx (NormalizedGaussian1
             + Lorentzian
             + Normalized Gaussain2 )

        f = 10*log10(y)
        dfdx = 10.0/ln(10) d ln(y)/dx
            d ln(y)/dx = 1/y * dydx
        """
        kwargs.setdefault('model_order', 2)
        prof = ModelDoppler._model(XX, aa, **kwargs)
        deriv = ModelDoppler._deriv(XX, aa, **kwargs)
        return 10.0*deriv/(prof*_np.log(10.0))

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        2nd Derivative of the Doppler model
        dfdx = d/dx (NormalizedGaussian1
             + Lorentzian
             + Normalized Gaussain2 )

        f = 10*log10(y)
        dfdx = 10.0/ln(10) d ln(y)/dx
            d ln(y)/dx = 1/y * dydx
        d2fdx2 = 10/ln|10| d/dx (dydx/y)
               = 10/ln|10|(d2ydx2/y - dydx^2/y2)
        """
        kwargs.setdefault('model_order', 2)
        prof = ModelDoppler._model(XX, aa, **kwargs)
        deriv = ModelDoppler._deriv(XX, aa, **kwargs)
        deriv2 = ModelDoppler._deriv2(XX, aa, **kwargs)
        return (10.0/_np.log(10.0))*(deriv2/prof-(deriv/prof)*(deriv/prof))

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        Jacobian of the Doppler model
        dfdai = d/dai (NormalizedGaussian1
             + Lorentzian
             + Normalized Gaussain2 )

        f = 10*log10(y)
        dfdx = 10.0/ln(10) d ln(y)/dx
            d ln(y)/dx = 1/y * dydx
        dfda = 10.0 / ln(10) * d ln(y)/da
             = 10.0/ ln(10) *dyda/y
        """
        kwargs.setdefault('model_order', 2)
        prof = ModelDoppler._model(XX, aa, **kwargs)
        gvec = ModelDoppler._partial(XX, aa, **kwargs)
        for ii in range(len(aa)):
            gvec[ii,:] /= prof
        # end for
        return 10.0*gvec/_np.log(10.0)

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        Derivative of the Jacobian of the Doppler model
        d2fdxdai = d2/dxdai(NormalizedGaussian1 + Lorentzian + NormalizedGaussain2 )

        f = 10*log10(y)
        dfdx = 10.0/ln(10) d ln(y)/dx
            d ln(y)/dx = 1/y * dydx
        d2fdxda = d(df/dx)/da
                = 10.0/ln(10) d(dy/dx/y)/da
                = 10.0/ln(10) * ( 1/y*d2ydxda - dyda/y^2 )

        dfda = d/da lny = dyda/y
        d2fdxda = d/x (dyda/y) = d2ydxda/y - dyda*dydx/y^2
        d3fdx2da = d3ydx2da/y - dydx*d2ydxda/y^2
                 - d2ydxda*dydx/y^2 - dyda*d2ydx2/y2 + 2*dyda*dydx^2/y^3
                 = (d3ydx2da*y^-1 -2*d2ydxda*dydx*y^-2 - dyda*d2ydx2*y-2 + 2*dyda*dydx^2*y-3)
        """
        kwargs.setdefault('model_order', 2)
        y = ModelDoppler._model(XX, aa, **kwargs)
        dydx = ModelDoppler._deriv(XX, aa, **kwargs)
        dyda = ModelDoppler._partial(XX, aa, **kwargs)
        d2ydxda = ModelDoppler._partial_deriv(XX, aa, **kwargs)
        dlngdx = _np.zeros_like(dyda)
        for ii in range(len(aa)):
            dlngdx[ii,:] = d2ydxda[ii,:]/y - dyda[ii,:]*dydx/(y*y)
        # end for
        return 10.0*dlngdx/_np.log(10.0)

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        2nd Derivative of the Jacobian of the Doppler model
        d2fdxdai = d2/dxdai(NormalizedGaussian1 + Lorentzian + NormalizedGaussain2 )

        d3fdx2da = d/dx d(df/dx)/da
                = 10.0/ln(10) * ( d3ydx2da/y - d2ydxda*dydx/y2 + 2*dydx*dyda/y^3 - d2ydxda/y^2 )

        dfda = d/da lny = dyda/y
        d2fdxda = d/x (dyda/y) = d2ydxda/y - dyda*dydx/y^2
        d3fdx2da = d3ydx2da/y - dydx*d2ydxda/y^2
                 - d2ydxda*dydx/y^2 - dyda*d2ydx2/y2 + 2*dyda*dydx^2/y^3
                 = (d3ydx2da*y^-1 -2*d2ydxda*dydx*y^-2 - dyda*d2ydx2*y-2 + 2*dyda*dydx^2*y-3)
        """
        kwargs.setdefault('model_order', 2)
        y = ModelDoppler._model(XX, aa, **kwargs)
        dydx = ModelDoppler._deriv(XX, aa, **kwargs)
        d2ydx2 = ModelDoppler._deriv2(XX, aa, **kwargs)

        y = _np.atleast_2d(y)
        dydx = _np.atleast_2d(dydx)
        d2ydx2 = _np.atleast_2d(d2ydx2)

        dyda = ModelDoppler._partial(XX, aa, **kwargs)
        d2ydxda = ModelDoppler._partial_deriv(XX, aa, **kwargs)
        d3ydx2da = ModelDoppler._partial_deriv2(XX, aa, **kwargs)

        d2lngdx2 = _np.zeros_like(dyda)
        for ii in range(len(aa)):
            d2lngdx2[ii,:] = ( d3ydx2da[ii,:]/y -2.0*d2ydxda[ii,:]*dydx/(y*y)
            - dyda[ii,:]*d2ydx2/(y*y) + 2.0*dyda[ii,:]*(dydx*dydx)/(y*y*y) )
        # end for
        return 10.0*d2lngdx2/_np.log(10.0)
#
#        d2gdx2 = _np.zeros_like(dyda)
#        for ii in range(len(aa)):
#            d2gdx2[ii,:] = (d3ydx2da[ii,:]/y - d2ydxda[ii,:]*dydx/power(y, 2.0)
#                    + 2.0*dydx*dyda[ii,:]/power(y, 3.0) - d2ydxda[ii,:]/power(y,2.0))
#        # end for

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain, **kwargs):
        """
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        a0, a1, a2 = _parse_noshift(aout, model_order=self.model_order)

        NM = ModelNormal(None)
        NM.slope = ys
        NM.xoffset = xo
        NM.xslope = xs
        NM.offset = yo
        aout = NM.unscaleaf(a0)
        if self.model_order>0:
            LM = ModelLorentzian(None)
            LM.slope = ys
            LM.xoffset = xo
            LM.xslope = xs
            LM.offset = yo
            aout = _np.hstack((aout, LM.unscaleaf(a1)))
        if self.model_order>1:
            aout = _np.hstack((aout, NM.unscaleaf(a2)))
        return aout

    def scaleaf(self, ain, **kwargs):
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)

        a0, a1, a2 = _parse_noshift(aout, model_order=self.model_order)

        NM = ModelNormal(None)
        NM.slope = ys
        NM.xoffset = xo
        NM.xslope = xs
        NM.offset = yo
        aout = NM.scaleaf(a0)
        if self.model_order>0:
            LM = ModelLorentzian(None)
            LM.slope = ys
            LM.xoffset = xo
            LM.xslope = xs
            LM.offset = yo
            aout = _np.hstack((aout, LM.scaleaf(a1)))
        if self.model_order>1:
            aout = _np.hstack((aout, NM.scaleaf(a2)))
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
#        self.xoffset = 0.0
        self.slope = 1.0 # problem
#        self.xslope = 1.0
        return super(ModelLogDoppler, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelLogDoppler, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelLogDoppler

# ========================================================================== #
# ========================================================================== #


def _twopower(XX, aa):
    return _ModelTwoPower._model(XX, aa)

def _deriv_twopower(XX, aa):
    return _ModelTwoPower._deriv(XX, aa)

def _partial_twopower(XX, aa):
    return _ModelTwoPower._partial(XX, aa)

def _partial_deriv_twopower(XX, aa):
    return _ModelTwoPower._partial_deriv(XX, aa)


def _model_twopower(XX=None, af=None, **kwargs):
    return _model(_ModelTwoPower, XX, af, **kwargs)

class _ModelTwoPower(ModelClass):
    """
        model a two-power fit with three free parameters
            f = a*(1.0 - x**c)**d

            a = amplitude of core
            c = power scaling factor 1
            d = power scaling factor 2

        non-trival domain:  c>0 and 0<=x<1  or   c<0 and x>1
    """
    _analytic_xscaling = False
    _analytic_yscaling = True
#    _af = _np.asarray([1.0, 2.0, 1.0], dtype=_np.float64)
    _af = _np.asarray([5.0, 6.0, 10.0], dtype=_np.float64)
    _LB = _np.asarray([  1e-18,   1e-18, -_np.inf], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)

    # Neumann boundary condition on left boundary
    #        dydx(xlim) = 0.0
#    leftboundary = 0.0
#    leftboundary = {'x':0.0, 'neumann': True, 'value': 0.0}

    # Dirichlet boundary condition on right boundary
    # this model requires dirichlet boundary condition on right
    #        y(xlim) = 0.0
#    rightboundary = 1.0
#    rightboundary = {'x':1.0, 'dirichlet': False, 'value': 0.0}
    # _rzero = 1.0
    def __init__(self, XX, af=None, **kwargs):
        # kwargs.setdefault('rzero', self._rzero)
        super(_ModelTwoPower, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model_base(XX, aa):
        """
            f = (1.0 - x**c)**d
        non-trival domain:  c>0 and 0<=x<1  or   c<0 and x>1

        Note that by definition, f = 0 at x=1
            if f is modified to (r-x**c)**d, then the boundary changes to r
        """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        c, d = tuple(aa)

        arg1 = power(_np.abs(XX), c)
        prof = power(1.0-arg1, d)
#        prof = power(_np.abs(1.0-power(_np.abs(XX), c)), d)

#        if (prof<0).any():
#            prof -= prof.min()
        return prof

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        model a two-power fit
            f = a*(1.0 - x**c)**d

            a = amplitude of core
            c = power scaling factor 1
            d = power scaling factor 2
        """
        a, c, d = tuple(aa)
        return a*_ModelTwoPower._model_base(XX, [c,d])

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
            f = a*(1.0 - x^c)^d

         dfda = f/a = _twopower/a
         dfdc = -a*d*(1.0 - x^c)^(d-1)*x^c*ln|x|
              = -d*x^c*ln(x)*_twopower(x, [c, d-1])
         dfdd = a*(1.0 - x^c)^d * ln|1.0 - x^c|
              = a*(1.0 - x^c)^d * ln1p|-x^c|
       """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, c, d = tuple(aa)

        cs = _np.copy(c)
#        cs = cs*_np.ones_like(XX)
#        cs[XX>1.0] *= -1.0

        gvec = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = _ModelTwoPower._model_base(XX, [c, d])
        gvec[1,:] = (-1.0*d*power(XX, cs)*_np.log(_np.abs(XX))*_ModelTwoPower._model(XX, [a, c, d-1.0]))

        # Error in log1p for 1-X^c at x>1 - returns nan
#        gvec[2,:] = _ModelTwoPower._model(XX, aa)*_np.log1p(-_np.abs(power(XX, cs)))
        gvec[2,:] = _ModelTwoPower._model(XX, aa)
        gvec[2,:] *= log1p(-power(XX, cs))
#        gvec[2,:] *= log(1.0 - power(XX, cs))

#        gvec[0,:] = _ModelTwoPower._model_base(XX, [c, d])
#        gvec[1,XX<=1] = -1.0*d*power(XX[XX<=1], c)*_np.log(_np.abs(XX[XX<=1]))
#        gvec[1,XX>1] = 1.0*d*power(XX[XX>1], -c)*_np.log(_np.abs(XX[XX>1]))
#        gvec[2,XX<=1] = log1p(-_np.abs(power(XX[XX<=1], c)))
#        gvec[2,XX>1] = log1p(-_np.abs(power(XX[XX>1], -c)))
#        gvec[1,:] *= _ModelTwoPower._model(XX, [a, c, d-1.0])
#        gvec[2,:] *= _ModelTwoPower._model(XX, aa)
        return gvec

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
            f = a*(1.0 - x^c)^d
         dfdx = -a*c*d*x^(c-1)*(1.0 - x**c)**(d-1)
              = -c*d*x^(c-1)*_twopower(x, [c, d-1])

         dfda = f/a = _twopower/a
         dfdc = -a*d*(1.0 - x^c)^(d-1)*x^c*ln|x|
              = -d*x^c*ln(x)*_twopower(x, [c, d-1])
         dfdd = a*(1.0 - x^c)^d * ln|1.0 - x^c|
              = a*(1.0 - x^c)^d * ln1p|-x^c|
       """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, c, d = tuple(aa)

        cs = _np.copy(c)
#        cs = cs*_np.ones_like(XX)
#        cs[XX>1.0] *= -1.0

        dfdx = -1.0*cs*d*power(XX, cs-1.0)
        dfdx *= _ModelTwoPower._model(XX, [a, c, d-1.0])

#        dfdx = _np.zeros_like(XX)
#        dfdx[XX<=1] = -1.0*c*d*power(XX[XX<=1], c-1.0)
#        dfdx[XX>1] = 1.0*c*d*power(XX[XX>1], -c-1.0)
#        dfdx *= _ModelTwoPower._model(XX, [a, c, d-1.0])
    #    return -1.0*a*c*d*power(XX, c-1.0)*power(1.0-power(XX,c), d-1.0)
        return dfdx

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
            f = a*(1.0 - x^c)^d
         dfdx = -a*c*d*x^(c-1)*(1.0 - x**c)**(d-1)
              = -c*d*x^(c-1)*_twopower(x, [c, d-1])

         dfda = f/a = _twopower/a
         dfdc = -a*d*(1.0 - x^c)^(d-1)*x^c*ln|x|
              = -d*x^c*ln(x)*_twopower(x, [c, d-1])
         dfdd = a*(1.0 - x^c)^d * ln|1.0 - x^c|
              = a*(1.0 - x^c)^d * ln1p|-x^c|

        d2fdxda = dfdx/a
        d2fdxdc = a*d*x^(c-1)*(1-x^c)^(d-2)*(c*ln|x|*(d*x^c-1)+x^c-1)
                = d*x^(c-1)*_twopower(x, [a, c, d-2.0])*(c*ln|x|*(d*x^c-1)+x^c-1)
        d2fdxdd = -a*c*x^(c-1)*(1-x^c)^(d-1)*( -d*ln|1-x^c| - 1)
                = -1*c*x^(c-1)*_twopower(x, [a, c, d-1.0])*( d*ln1p(-x^c) + 1)

#        d2fdxdc = dfdx/c + ln|x|*dfdx - a*c*d*x^(c-1)*(d-1)*(1.0 - x^c)^(d-2)*-1*x^c*ln|x|
#                = dfdx/c + ln|x|*dfdx - a*c*d*x^(c-1)*(d-1)*(1.0 - x^c)^(d-2)*-1*x^c*ln|x|
#                = dfdx/c + ln|x|*dfdx - dfdx*(d-1)*x^c*ln|x|*(1.0 - x^c)^-1
#                = dfdx*(1.0/c + ln|x| - (d-1)*x^c*ln|x|*_twopower(x, [1.0, c, -1.0]))
#        d2fdxdd = dfdx/d + dfdx*ln|(1.0 - x^c)^(d-1)|
       """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, c, d = tuple(aa)
        dfdx = _ModelTwoPower._deriv(XX, aa)

        cs = _np.copy(c)
#        cs = cs*_np.ones_like(XX)
#        cs[XX>1.0] *= -1.0

        Xcs = power(XX, cs)
        Xcs1 = power(XX, cs-1.0)

        dgdx = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = dfdx/a
#        dgdx[1,XX<=1.0] = d*power(XX[XX<=1.0],c-1.0)*_ModelTwoPower._model(XX[XX<=1.0], [a, c, d-2.0])*(
#                c*_np.log(_np.abs(XX[XX<=1.0]))*(d*power(XX[XX<=1.0],c)-1.0)+power(XX[XX<=1.0],c)-1.0)
#        dgdx[1,XX>1.0] = -d*power(XX[XX>1.0],-c-1.0)*_ModelTwoPower._model(XX[XX>1.0], [a, c, d-2.0])*(
#                -1.0*c*_np.log(_np.abs(XX[XX>1.0]))*(d*power(XX[XX>1.0],-c)-1.0)+power(XX[XX>1.0],-c)-1.0)
#        dgdx[2,XX<=1.0] = -1.0*c*power(XX[XX<=1.0],c-1.0)*_ModelTwoPower._model(XX[XX<=1.0], [a, c, d-1.0])*( d*log1p(-power(XX[XX<=1.0],c)) + 1.0)
#        dgdx[2,XX>1.0] = 1.0*c*power(XX[XX>1.0],-c-1.0)*_ModelTwoPower._model(XX[XX>1.0], [a, c, d-1.0])*( d*log1p(-power(XX[XX>1.0],-c)) + 1.0)

        dgdx[1,:] = d*power(XX,cs-1.0)
        dgdx[1,:] *= _ModelTwoPower._model(XX, [a, c, d-2.0])*(
                cs*log(_np.abs(XX))*(d*power(XX,cs)-1.0)+power(XX, cs)-1.0)
#        dgdx[1,:] *= _ModelTwoPower._model(XX, [a, c, d-2.0])*(
#                cs*_np.log(_np.abs(XX))*(d*power(XX,cs)-1.0)+power(XX, cs)-1.0)
        # Error in log1p at x>1 ... this is solved using complex numbers
#        dgdx[2,:] = -1.0*cs*power(XX,cs-1.0)*_ModelTwoPower._model(XX, [a, c, d-1.0])*( d*log1p(-power(XX,cs)) + 1.0)
        dgdx[2,:] = -1.0*cs*Xcs1*_ModelTwoPower._model(XX, [a, c, d-1.0])*(d*log1p(-Xcs) + 1.0)
#            d*log(1.0-Xcs) + 1.0)
#        if (dgdx>40).any():
#            print('pause')
#        dgdx[1,:] = dfdx*(power(c, -1.0) + _np.log(_np.abs(XX))
#            - (d-1.0)*power(XX, c)*_np.log(_np.abs(XX))*_ModelTwoPower._model(XX, [1.0, c, -1.0]))
#        dgdx[2,:] = dfdx*(power(d, -1.0) + _np.log(_np.abs(_ModelTwoPower._model_base(XX, [c, d-1.0]))))
        return dgdx

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
            f = a*(1.0 - x^c)^d
         dfdx = -a*c*d*x^(c-1)*(1.0 - x**c)**(d-1)
              = -c*d*x^(c-1)*_twopower(x, [c, d-1])

         d2fdx2 = -a*c*d*(c-1)*x^(c-2)*(1.0 - x**c)**(d-1)
                  +a*c^2*d*(d-1)*x^(2c-2)*(1.0 - x**c)**(d-2)

         dfda = f/a = _twopower/a
         dfdc = -a*d*(1.0 - x^c)^(d-1)*x^c*ln|x|
              = -d*x^c*ln(x)*_twopower(x, [c, d-1])
         dfdd = a*(1.0 - x^c)^d * ln|1.0 - x^c|
              = a*(1.0 - x^c)^d * ln1p|-x^c|

        d2fdxda = dfdx/a
        d2fdxdc = a*d*x^(c-1)*(1-x^c)^(d-2)*(c*ln|x|*(d*x^c-1)+x^c-1)
                = d*x^(c-1)*_twopower(x, [a, c, d-2.0])*(c*ln|x|*(d*x^c-1)+x^c-1)
        d2fdxdc = -a*c*x^(c-1)*(1-x^c)^(d-1)*( -d*ln|1-x^c| - 1)
                = -1*c*x^(c-1)*_twopower(x, [a, c, d-1.0])*( d*ln1p(-x^c) + 1)

       """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, c, d = tuple(aa)

        cs = _np.copy(c)
#        cs = cs*_np.ones_like(XX)
#        cs[XX>1.0] *= -1.0

        d2fdx2 = cs*cs*d*(d-1.0)*power(XX, 2.0*cs-2.0)
        d2fdx2 *=_ModelTwoPower._model(XX, [a, c, d-2.0])
        d2fdx2 -= cs*d*(cs-1.0)*(power(XX, cs-2.0)
            *_ModelTwoPower._model(XX, [a, c, d-1.0]))

#        d2fdx2 = cs*cs*d*(d-1.0)*power(XX, 2.0*cs-2.0)*_ModelTwoPower._model(XX, [a, c, d-2.0])
#        d2fdx2 -= cs*d*(cs-1.0)*power(XX, cs-2.0)*_ModelTwoPower._model(XX, [a, c, d-1.0])
        return d2fdx2

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
            f = a*(1.0 - x^c)^d
         dfdx = -a*c*d*x^(c-1)*(1.0 - x**c)**(d-1)
              = -c*d*x^(c-1)*_twopower(x, [c, d-1])

         d2fdx2 = -a*c*d*(c-1)*x^(c-2)*(1.0 - x**c)**(d-1)
                  +a*c^2*d*(d-1)*x^(2c-2)*(1.0 - x**c)**(d-2)

         dfda = f/a = _twopower/a
         dfdc = -a*d*(1.0 - x^c)^(d-1)*x^c*ln|x|
              = -d*x^c*ln(x)*_twopower(x, [c, d-1])
         dfdd = a*(1.0 - x^c)^d * ln|1.0 - x^c|
              = a*(1.0 - x^c)^d * ln1p|-x^c|

        d2fdxda = dfdx/a
        d2fdxdc = a*d*x^(c-1)*(1-x^c)^(d-2)*(c*ln|x|*(d*x^c-1)+x^c-1)
                = d*x^(c-1)*_twopower(x, [a, c, d-2.0])*(c*ln|x|*(d*x^c-1)+x^c-1)
        d2fdxdc = -a*c*x^(c-1)*(1-x^c)^(d-1)*( -d*ln|1-x^c| - 1)
                = -1*c*x^(c-1)*_twopower(x, [a, c, d-1.0])*( d*ln1p(-x^c) + 1)

        d3fdx2da = c*d*x^(c-2)*(1-x^c)^(d-2)*( c*(d*x^c-1)-x^c+1 )
        d3fdx2dc = -a*d*x^(c-2)*(1-x^c)^(d-3)*( c*ln|x|*(
                c*( d^2*x^(2c) +(1-3*d)*x^c+1)-d*x^(2c)+(d+1)*x^c-1 )
              + c*( 2*d*x^(2*c)+(-2*d-2)*x^c +2 ) - (1-x^c)^2 )
        d3fdx2dc = a*c*x^(c-2)*(1-x^c)^(d-2)*(
            c*(2*d*x^c-1) + d*(c*(d*x^c-1)-x^c+1)*ln|1-x^c|-x^c+1)
                 = c*x^(c-2)*_ModelTwoPower._model(XX, [a, c, d-2.0])*(
            c*(2*d*x^c-1) + d*(c*(d*x^c-1)-x^c+1)*ln1p|-x^c|-x^c+1)


       """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, c, d = tuple(aa)
        cs = _np.copy(c)
#        cs = cs*_np.ones_like(XX)
#        cs[XX>1.0] *= -1.0

#        dfdx = _ModelTwoPower._deriv(XX, aa)
#        d2fdx2 = _ModelTwoPower._deriv2(XX, aa)
        Xcs = power(XX, cs)
        Xcs2 = power(XX, cs-2.0)
        X2cs = power(XX, 2.0*cs)
        d2 = power(d, 2.0)

        d2gdx2 = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
        d2gdx2[0, :] = cs*d*Xcs2*_ModelTwoPower._model_base(XX, [c, d-2.0])*(
                cs*(d*Xcs-1.0)-Xcs+1.0 )
        d2gdx2[1, :] = -d*Xcs2*_ModelTwoPower._model(XX, [a, c, d-3.0])*(
                cs*_np.log(_np.abs(XX))*( cs*( d2*X2cs + (1.0-3.0*d)*Xcs+1.0)
              - d*X2cs+(d+1.0)*Xcs-1.0) + cs*( 2.0*d*X2cs+(-2.0*d-2.0)*Xcs + 2.0 )
              - _ModelTwoPower._model(XX, [1.0, c, 2.0]) )
        # Error in log1p ... returns nan for x>1.  Solve this with complex numbers
#        d2gdx2[2, :] = cs*power(XX, cs-2.0)*_ModelTwoPower._model(XX, [a, c, d-2.0])*(
#            cs*(2.0*d*power(XX, cs)-1.0)
#          + d*(cs*(d*power(XX, cs)-1.0)-power(XX, cs)+1.0)*log1p(-1*power(XX, cs))
#          - power(XX, cs)+1.0)
        d2gdx2[2, :] = cs*Xcs2*_ModelTwoPower._model(XX, [a, c, d-2.0])*(
            cs*(2.0*d*Xcs-1.0) + d*(cs*(d*Xcs-1.0)-Xcs+1.0)*( log1p(-Xcs) )- Xcs+1.0)
#          log(1-Xcs) )- Xcs+1.0)

#        d2gdx2[0,:] = d2fdx2/a
#        d2gdx2[1,:] = d2fdx2*(power(c, -1.0) + _np.log(_np.abs(XX))
#            - (d-1.0)*power(XX, c)*_np.log(_np.abs(XX))*_ModelTwoPower._model(XX, [1.0, c, -1.0]))
#        d2gdx2[1,:] += dfdx*(1.0/XX
#            - (d-1.0)*c*power(XX, c-1.0)*_np.log(_np.abs(XX))*_ModelTwoPower._model(XX, [1.0, c, -1.0])
#            - (d-1.0)*power(XX, c)*1.0/XX*_ModelTwoPower._model(XX, [1.0, c, -1.0])
#            - (d-1.0)*power(XX, c)*_np.log(_np.abs(XX))*_ModelTwoPower._deriv(XX, [1.0, c, -1.0]))

#        d2gdx2[2,:] = d2fdx2*(power(d, -1.0) + _np.log(_np.abs(_ModelTwoPower._model_base(XX, [c, d-1.0]))))
#        d2gdx2[2,:] += dfdx*(_ModelTwoPower._deriv(XX, [a, c, d-1.0])/(a*_ModelTwoPower._model_base(XX, [c, d-1.0])))
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
         y-scaling is possible, but everything else is non-linear
                     f = a*(1.0 - x^c)^d
        y' = y/ys
        y = a*ys*(1-x^c)^d
        a = ys*a'
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)  # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[0] = ys*ain[0]
        return aout

    def scaleaf(self, ain, **kwargs):
        """
        a = ys*a'   0> a'=a/ys
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)   # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[0] = ain[0]/ys
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
        self.xoffset = 0.0
        self.xslope = 1.0
        return super(_ModelTwoPower, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(_ModelTwoPower, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def _ModelTwoPower

# ========================================================================== #
# ========================================================================== #


def twopower(XX, aa):
    return ModelTwoPower._model(XX, aa)

def deriv_twopower(XX, aa):
    return ModelTwoPower._deriv(XX, aa)

def partial_twopower(XX, aa):
    return ModelTwoPower._partial_deriv(XX, aa)

def model_twopower(XX=None, af=None, **kwargs):
    return _model(ModelTwoPower, XX, af, **kwargs)

class ModelTwoPower(ModelClass):
    """
    model a two-power fit
        y = a*( (1-b)*(1.0 - x^c)^d + b )

        a = amplitude of core
        b = edge / core
        c = power scaling factor 1
        d = power scaling factor 2

       non-trival domain:  c>0 and 0<=x<1  or   c<0 and x>1
    """
#    _af = _np.asarray([1.0, 0.0, 12.0, 3.0], dtype=_np.float64)
    _af = _np.asarray([1.0, 0.10, 12.0, 3.0], dtype=_np.float64)
    _LB = _np.asarray([1e-18, 0.0, 1e-18, -20.0], dtype=_np.float64)
    _UB = _np.asarray([   20, 1.0,  20.0,  20.0], dtype=_np.float64)
    _fixed = _np.zeros( (4,), dtype=int)
    _analytic_xscaling = False
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelTwoPower, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
         one option: f/a = (1-b)*(1-x^c)^d + b
         2nd option: f = (a-b)*(1- x^c)^d + b
             we'll take the first to build with it easier

        non-trival domain:  c>0 and 0<=x<1  or   c<0 and x>1
        """
        a, b, c, d = tuple(aa)
        return (1.0-b)*_ModelTwoPower._model(XX, [a, c, d]) + a*b

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        f/a = (1-b)*(1-x^c)^d + b
        dfdx = a*(1-b)*c*d*x^(c-1)*(1-x^c)^(d-1)
             = (1-b)*_deriv_twopower

        """
        a, b, c, d = tuple(aa)
        return (1.0-b)*_ModelTwoPower._deriv(XX, [a, c, d])

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        f/a = (1-b)*(1-x^c)^d + b
        dfdx = a*(1-b)*c*d*x^(c-1)*(1-x^c)^(d-1)
             = (1-b)*_deriv_twopower
        """
        a, b, c, d = tuple(aa)
        return (1.0-b)*_ModelTwoPower._deriv2(XX, [a, c, d])

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        f/a = (1-b)*(1-x^c)^d + b
        dfdx = a*b*c*d*x^(c-1)*(1-x^c)^(d-1)

         dfda = b+(1-b)*(1-x^c)^d
         dfdb = a-a*(1-x^c)^d = a-_twopower(XX, [a, c, d])
         dfdc = -a*(1-b)*d*x^c*ln|x|*(1-x^c)^(d-1)
              = -d*(1-b)*ln|x|*x^c*_twopower(x, [a,c,d-1])
         dfdd = a*(1.0-b)(1.0 - x^c)^d * ln|1.0 - x^c|
              = (1.0-b)*_twopower() * ln|_twopower(x, [1.0, c, 1.0])|
        """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, b, c, d = tuple(aa)

        nx = _np.size(XX)
        gvec = _np.zeros((4, nx), dtype=_np.float64)
        gvec[0, :] = ModelTwoPower._model(XX, [1.0, b, c, d])
        gvec[1, :] = a-_ModelTwoPower._model(XX, [a, c, d])
        gvec[2, :] = (-d*(1.0-b)*log(_np.abs(XX))*power(_np.abs(XX), c)
            *_ModelTwoPower._model(XX, [a, c, d-1.0]))
        gvec[3, :] = ((1.0-b)*_ModelTwoPower._model(XX, [a, c, d])
            *log(_np.abs(_ModelTwoPower._model(XX, [1.0, c, 1.0]))) )
#        if (_np.isnan(gvec)).any():
#            print('debugging')
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        f/a = (1-b)*(1-x^c)^d + b
        dfdx = -a*(1-b)*c*d*x^(c-1)*(1-x^c)^(d-1)

         dfda = b+(1-b)*(1-x^c)^d
         dfdb = a-a*(1-x^c)^d = a-_twopower(XX, [a, c, d])
         dfdc = -a*(1-b)*d*x^c*ln|x|*(1-x^c)^(d-1)
              = -d*ln|x|*x^c*twopower(x, [a,b,c,d-1])
         dfdd = a*(1.0-b)(1.0 - x^c)^d * ln|1.0 - x^c|
              = (1.0-b)*_twopower() * ln|_twopower(x, [1.0, c, 1.0])|

        d2fdxda = -(1-b)*c*d*x^(c-1)*(1-x^c)^(d-1)  = dfdx/a
        d2fdxdb = a*c*d*x^(c-1)*(1-x^c)^(d-1)       = -dfdx/(1-b)
        d2fdxdc = -a*(1-b)*d*x^(c-1)*(1-x^c)^(d-1)
                - a*(1-b)*c*d*x^(c-1)*ln|x|*(1-x^c)^(d-1)
                + (d-1)*x^c*ln|x|/(1-x^c) *a*(1-b)*c*d*x^(c-1)*(1-x^c)^(d-1)
                            = dfdx*(c^-1 + ln|x| - (d-1)*x^c*ln|x|/(1-x^c) )
        d2fdxdd = dfdx*( 1/d + ln|1-x^c|) = dfdx*(1/d + ln|_twopower_base(x, [c, 1.0]|)

        """
        a, b, c, d = tuple(aa)
#        XX = _np.abs(XX)
        dfdx = ModelTwoPower._deriv(XX, aa)

        nx = _np.size(XX)
        dgdx = _np.zeros((4, nx), dtype=_np.float64)
        dgdx[0, :] = dfdx/a
        dgdx[1, :] = -dfdx/(1.0-b)
        dgdx[2, :] = dfdx*( 1.0/c + _np.log(_np.abs(XX)) - (d-1.0)*power(XX, c)*_np.log(_np.abs(XX) )
                /_ModelTwoPower._model_base(XX,[c, 1.0]))
        dgdx[3, :] = dfdx*( 1.0/d + log(_np.abs(_ModelTwoPower._model_base(XX, [c, 1.0])) ) )
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        a, b, c, d = tuple(aa)
        dfdx = ModelTwoPower._deriv(XX, aa)
        d2fdx2 = ModelTwoPower._deriv2(XX, aa)
#        XX = _np.abs(XX)

        xc = power(XX, c)
        xc1 = power(XX, c-1.0)

        nx = _np.size(XX)
        dgdx = _np.zeros((4, nx), dtype=_np.float64)
        dgdx[0, :] = d2fdx2/a
        dgdx[1, :] = -d2fdx2/(1.0-b)
        dgdx[2, :] = d2fdx2*( 1.0/c + _np.log(_np.abs(XX))
                - (d-1.0)*xc*_np.log(_np.abs(XX))/_ModelTwoPower._model_base(XX,[c, 1.0]))
        dgdx[2, :] += dfdx*( 1.0/XX - (d-1.0)*c*xc1*_np.log(_np.abs(XX))/_ModelTwoPower._model_base(XX,[c, 1.0])
                - (d-1.0)*xc/(XX*_ModelTwoPower._model_base(XX,[c, 1.0]))
                + (d-1.0)*xc*_np.log(_np.abs(XX))*a*_ModelTwoPower._deriv(XX,[a, c, 1.0])
                /power(_ModelTwoPower._model(XX,[a, c, 1.0]), 2.0) )
        dgdx[3, :] = d2fdx2*( 1.0/d + log(_np.abs(_ModelTwoPower._model_base(XX, [c, 1.0])) ) )
        dgdx[3, :] += dfdx*( _ModelTwoPower._deriv(XX, [a, c, 1.0])/_ModelTwoPower._model(XX, [a, c, 1.0]))
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        y = a*(1-b)*(1-x^c)^d + a*b

        To unscale the problem
        y-scaling: y'= (y-yo)/ys
        y = yo+ys*a'*(1-b')(1-x^c')^d' + ys*a'b'

         1) Scale:       a*(1-b) = ys*a'*(1-b')
         2) Constant:    ab = yo + ys*a'*b'

             ab = a-ys*a'*(1-b')    from (1)
             b = 1-ys*(1-b')*a'/a   from (1)

             ab = yo + ys*a'*b'     from (2)
                a*(1) = (2)
       a-ys*(1-b')*a' = yo + ys*a'*b'
                    a = ys*(1-b')*a' + yo + ys*a'*b'
             a = ys*a' + yo
             b = 1-ys*(1-b')*a'/a
             c= c'
             d= d'

            =======
        x-scaling:  x' = (x-xo)/xs
            y = yo+ys*a'*(1-b')(1-(x-xo)^c/xs^c')^d' + ys*a'b'
                doesn't work due to the nonlinearity, xo=0, xs=1
            ========
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)  # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[0] = yo + ys*ain[0]
        aout[1] = 1.0 - ys*(1-ain[1])*ain[0]/aout[0]
        return aout

    def scaleaf(self, ain, **kwargs):
        """
        y = a*(1-b)*(1-x^c)^d + a*b

             a = ys*a' + yo
             b = 1-ys*(1-b')*a'/a
               = 1-ys*(1-b')*a'/(ys*a' + yo)
             c= c'
             d= d'

             a' = (a-yo)/ys
                 1-b' = -(b-1)*a/(ys*a')
             b' = 1.0-(1-b)*a/(ys*a')
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)  # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[0] = (ain[0] - yo)/ys
        aout[1] = 1.0 - (1-ain[1])*aout[0]/(ys*ain[0])
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xoffset = 0.0
        self.xslope = 1.0
        return super(ModelTwoPower, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelTwoPower, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelTwoPower

# ========================================================================== #


def expedge(XX, aa):
    return ModelExpEdge._model(XX, aa)

def partial_expedge(XX, aa):
    return ModelExpEdge._partial(XX, aa)

def deriv_expedge(XX, aa):
    return ModelExpEdge._deriv(XX, aa)

def partial_deriv_expedge(XX, aa):
    return ModelExpEdge._partial_deriv(XX, aa)

def model_expedge(XX=None, af=None, **kwargs):
    """
    """
    return _model(ModelExpEdge, XX, af, **kwargs)

# ============================================== #

class ModelExpEdge(ModelClass):
    """
    model an exponential edge
        y = e*(1-exp(-x^2/h^2))
        second-half of a quasi-parabolic (no edge, or power factors)
        e = hole width
        h = hole depth

    """
    _af = _np.asarray([    2.0,    1.0], dtype=_np.float64)
    _LB = _np.asarray([   1e-18, 1e-18], dtype=_np.float64)
    _UB = _np.asarray([ _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (2,), dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelExpEdge, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _exp(XX, aa, **kwargs):
        e, h = tuple(aa)
        x2 = power(XX, 2.0)
        h2 = power(h, 2.0)
        return exp(-x2/h2)

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        model an exponential edge
            e = hole width
            h = hole depth
            f = e*(1-exp(-x^2/h^2))
              = e *( 1 - gaussian(XX, [1.0, 0.0, _np.sqrt(0.5)*h]) )
        """
        e, h = tuple(aa)
        return e*(1.0-ModelExpEdge._exp(XX, aa, **kwargs))
#        return e*(1.0-exp(-power(XX, 2.0)/power(h, 2.0)))

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        model an exponential edge
            e = hole width
            h = hole depth
        f = e*(1-exp(-x^2/h^2))
          = e *( 1 - gaussian(XX, [1.0, 0.0, _np.sqrt(0.5)*h]) )

        dfdx = 2ex/h^2 exp(-x^2/h^2)
        """
        e, h = tuple(aa)
        h2 = power(h, 2.0)
        return (2.0*XX*e/h2)*ModelExpEdge._exp(XX, aa, **kwargs)

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        model an exponential edge
            e = hole width
            h = hole depth
        f = e*(1-exp(-x^2/h^2))
          = e *( 1 - gaussian(XX, [1.0, 0.0, _np.sqrt(0.5)*h]) )

        dfdx = 2ex/h^2 exp(-x^2/h^2)
        d2fdx2
        """
        e, h = tuple(aa)
        h2 = power(h, 2.0)
        h4 = power(h, 4.0)
#        dfdx = (2.0*XX*e/power(h, 2.0))*ModelExpEdge._exp(XX, aa, **kwargs)
        d2fdx2 = (2.0*e/h2)*ModelExpEdge._exp(XX, aa, **kwargs)
        d2fdx2 += -(4.0*XX*XX*e/h4)*ModelExpEdge._exp(XX, aa, **kwargs)
        return d2fdx2

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        model an exponential edge
            e = hole width
            h = hole depth
        f = e*(1-exp(-x^2/h^2))
          = e *( 1 - gaussian(XX, [1.0, 0.0, _np.sqrt(0.5)*h]) )

        dfdx = 2ex/h^2 exp(-x^2/h^2)

        dfde = 1-exp(-x^2/h^2)
        dfdh = -e*exp(-x^2/h^2)*(-1*-2*x^2/h^3) = (-2ex^2/h^3)*exp(-x^2/h^2)

        dfde = f/e
        dfdh = -x/h*dfdx
        """
        e, h = tuple(aa)
        x2 = power(XX, 2.0)
        h3 = power(h, 3.0)

        gvec = _np.zeros((2,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] =  1.0 - ModelExpEdge._exp(XX, aa, **kwargs)
        gvec[1,:] = (-2.0*e*x2/h3)*ModelExpEdge._exp(XX, aa, **kwargs)
#        gvec[0,:] =  ModelExpEdge._model(XX, aa, **kwargs)/e
#        gvec[1,:] = (-XX/h)*ModelExpEdge._deriv(XX, aa, **kwargs)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        model an exponential edge
            e = hole width
            h = hole depth
        f = e*(1-exp(-x^2/h^2))
          = e *( 1 - gaussian(XX, [1.0, 0.0, _np.sqrt(0.5)*h]) )

        dfdx = 2ex/h^2 exp(-x^2/h^2)

        dfde = 1-exp(-x^2/h^2)
        dfdh = -e*exp(-x^2/h^2)*(-1*-2*x^2/h^3) = (-2ex^2/h^3)*exp(-x^2/h^2)

        dfde = f/e
        dfdh = -x/h*dfdx

        d2fdxde = 2x/h^2 exp(-x^2/h^2)
        d2fdxdh = -2*2ex/h^3 exp(-x^2/h^2) + 2ex/h^2 (-2x^2/h^3) exp(-x^2/h^2)
                = (-4ex/h^3 - 4ex^3/h^5)exp(-x^2/h^2)
                = -4ex/h^3 (1.0 + x^2/h^2) exp(-x^2/h^2)
                = -4ex/h^5 (h+x)(h-x)exp(-x^2/h^2)
        """
        e, h = tuple(aa)
        h2 = power(h, 2.0)
        h5 = power(h, 5.0)
        dgdx = _np.zeros((2,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = (2.0*XX/h2)*ModelExpEdge._exp(XX, aa, **kwargs)
        dgdx[1,:] = (-4.0*e*XX/h5)*(h-XX)*(h+XX)*ModelExpEdge._exp(XX, aa, **kwargs)
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        model an exponential edge
            e = hole width
            h = hole depth
        f = e*(1-exp(-x^2/h^2))
          = e *( 1 - gaussian(XX, [1.0, 0.0, _np.sqrt(0.5)*h]) )

        dfdx = 2ex/h^2 exp(-x^2/h^2)

        dfde = 1-exp(-x^2/h^2)
        dfdh = -e*exp(-x^2/h^2)*(-1*-2*x^2/h^3) = (-2ex^2/h^3)*exp(-x^2/h^2)

        dfde = f/e
        dfdh = -x/h*dfdx

        d2fdxde = 2x/h^2 exp(-x^2/h^2)
        d2fdxdh = -2*2ex/h^3 exp(-x^2/h^2) + 2ex/h^2 (-2x^2/h^3) exp(-x^2/h^2)
                = (-4ex/h^3 - 4ex^3/h^5)exp(-x^2/h^2)
                = -4ex/h^3 (1.0 + x^2/h^2) exp(-x^2/h^2)
                = -4ex/h^5 (h+x)(h-x)exp(-x^2/h^2)
        """
        e, h = tuple(aa)

        h2 = power(h, 2.0)
        h4 = power(h, 4.0)
        h5 = power(h, 5.0)
        h7 = power(h, 7.0)

        d2gdx2 = _np.zeros((2,_np.size(XX)), dtype=_np.float64)
        d2gdx2[0,:] = (2.0/h2)*ModelExpEdge._exp(XX, aa, **kwargs)
        d2gdx2[0,:] += -(4.0*XX*XX/h4)*ModelExpEdge._exp(XX, aa, **kwargs)


        d2gdx2[1,:] = (-4.0*e/h5)*(h-XX)*(h+XX)*ModelExpEdge._exp(XX, aa, **kwargs)
        d2gdx2[1,:] += (-4.0*e*XX/h5)*(-1.0)*(h+XX)*ModelExpEdge._exp(XX, aa, **kwargs)
        d2gdx2[1,:] += (-4.0*e*XX/h5)*(h-XX)*(1.0)*ModelExpEdge._exp(XX, aa, **kwargs)
        d2gdx2[1,:] += (8.0*e*XX*XX/h7)*(h-XX)*(h+XX)*ModelExpEdge._exp(XX, aa, **kwargs)
        return d2gdx2


#    @staticmethod
#    def _hessian(XX, aa, **kwargs):
#        return None
#        raise NotImplementedError

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
            y-scaling: y' = y/ys
            y' = y/ys = e'*(1-exp(-x^2/h'^2))
                e = e'*ys
                h = h'
                possible with constant coefficients

          y-shifting: y' = (y-yo)/ys  # NOT POSSIBLE
            y' = (y-yo)/ys = e'*(1-exp(-x^2/h'^2))

            y = yo + ys*e' * (1-exp(-x^2/h'^2))
              = (yo-yo*exp(-x^2/h'^2) + ys*e')*(1-exp(-x^2/h'^2))
                   at x=0,   e = ys*e'
                   at x=inf, e = yo+ys*e'
               not possible with this fit and constant coefficients (unless yo=0)

          x-scaling:  x' = x/xs
            y' = y/ys = e'*(1-exp(-x^2/(xs*h')^2))
                e = e'*ys
                h = h'*xs

          x-shifting: x'=(x-xo)/xs  # NOT POSSIBLE
            y' = y/ys = e'*(1-exp(-(x-xo)^2/(xs*h')^2))
               = e'*ys*( 1-exp(-x^2/(xs*h')^2)exp(-2xox/(xs*h')^2)exp(-xo^2/(xs*h')^2) )

               = e'*ys*exp(-xo^2/(xs*h')^2)( exp(xo^2/(xs*h')^2)-exp(-x^2/(xs*h')^2)exp(-2xox/(xs*h')^2) )
                  still not possible

                e = e' * ys
                h = h' * xs
                iff xo, yo = 0
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset) # analysis:ignore

        aout[0] = ys*aout[0]
        aout[1] = xs*aout[1]
        return aout

    def scaleaf(self, ain, **kwargs):
        """
        Undo above:
            y' = y/ys = e'*(1-exp(-x^2/(xs*h')^2))
                e = e'*ys
                h = h'*xs

                e' = e / ys
                h' = h / xs
                iff xo, yo = 0
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset) # analysis:ignore

        aout[0] = aout[0]/ys
        aout[1] = aout[1]/xs
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xoffset = 0.0
        self.offset = 0.0
        return super(ModelExpEdge, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelExpEdge, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelExpEdge


# ========================================================================== #
# ========================================================================== #

def qparab(XX, aa, **kwargs):
    return ModelQuasiParabolic._model(XX, aa, **kwargs)

def deriv_qparab(XX, aa, **kwargs):
    return ModelQuasiParabolic._deriv(XX, aa, **kwargs)

def partial_qparab(XX, aa, **kwargs):
    return ModelQuasiParabolic._partial(XX, aa, **kwargs)

def partial_deriv_qparab(XX, aa, **kwargs):
    return ModelQuasiParabolic._partial_deriv(XX, aa, **kwargs)

def deriv2_qparab(XX, aa, **kwargs):
#    """
#    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
#    This subfunction calculates the second derivative of a quasi-parabolic fit
#    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
#        XX - r/a
#    aa[0] - Y0 - function value on-axis
#    aa[1] - gg - Y1/Y0 - function value at edge over core
#    aa[2],aa[3]-  pp, qq - power scaling parameters
#    aa[4],aa[5]-  hh, ww - hole depth and width
#    """
#    aa = _np.asarray(aa,dtype=_np.float64)
#    a, b, c, d, e, f = tuple(aa)
#
#    d2pdx2 = aa[3]*(aa[2]**2.0)*(aa[3]-1.0)*(1.0+aa[4]-aa[1])*power(_np.abs(XX), 2.*aa[2]-2.0)*(1-power(_np.abs(XX),aa[2]))**(aa[3]-2.0)
#    d2pdx2 -= (aa[2]-1.0)*aa[2]*aa[3]*(1.0+aa[4]-aa[1])*power(_np.abs(XX), aa[2]-2.0)*power(1-power(_np.abs(XX), aa[2]), aa[3]-1.0)
#    d2pdx2 += (2.0*aa[4]*exp(-power(XX,2.0)/power(aa[5], 2.0)))/power(aa[5], 2.0)
#    d2pdx2 -= (4*aa[4]*power(XX, 2.0)*exp(-power(XX, 2.0)/power(aa[5], 2.0)))/power(aa[5], 4.0)
#    d2pdx2 *= a
#    return d2pdx2
    return ModelQuasiParabolic._deriv2(XX, aa, **kwargs)
# end def derive_qparab

def partial_deriv2_qparab(XX, aa, **kwargs):
#    """
#    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
#    This subfunction calculates the jacobian of the second derivative of a
#    quasi-parabolic fit (partial derivatives of the second derivative of a quasi-parabolic fit)
#
#    quasi-parabolic fit:
#    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
#        XX - r/a
#
#    aa[0] - Y0 - function value on-axis
#    aa[1] - gg - Y1/Y0 - function value at edge over core
#    aa[2],aa[3]-  pp, qq - power scaling parameters
#    aa[4],aa[5]-  hh, ww - hole depth and width
#    """
#    aa = _np.asarray(aa,dtype=_np.float64)
#    Y0 = aa[0]
#    g = aa[1]
#    p = aa[2]
#    q = aa[3]
#    h = aa[4]
#    w = aa[5]
#
#    gvec = _np.zeros( (6,_np.size(XX)), dtype=_np.float64)
#    gvec[0,:] = deriv2_qparab(XX, aa) / Y0
#    gvec[1,:] = -p*q*Y0*power(_np.abs(XX), p-2.0)*power(1.0-power(_np.abs(XX), p), q-2.0)*(p*(q*power(_np.abs(XX), p)-1.0)-power(XX, p)+1.0)
#    gvec[2,:] = p*_np.log(_np.abs(XX))*(p*(power(q, 2.0)*power(_np.abs(XX), 2.0*p)-3.0*q*power(_np.abs(XX), p)+power(_np.abs(XX), p)+1.0)-(power(XX, p)-1.0)*(q*power(_np.abs(XX), p)-1.0))
#    gvec[2,:] += (power(_np.abs(XX), p)-1.0)*(2.0*p*(q*power(_np.abs(XX), p)-1.0)-power(_np.abs(XX), p)+1.0)
#    gvec[2,:] *= q*Y0*(g-h-1.0)*power(_np.abs(XX), p-2.0)*(power(1.0-power(_np.abs(XX), p), q-3.0))
#    gvec[3,:] = p*Y0*(-(g-h-1.0))*power(_np.abs(XX), p-2.0)*power(1.0-power(_np.abs(XX), p), q-2.0)*(p*(2.0*q*power(_np.abs(XX), p)-1.0)+q*(p*(q*power(_np.abs(XX), p)-1.0)-power(_np.abs(XX), p)+1.0)*_np.log(_np.abs(1.0-power(_np.abs(XX)**p)))-power(_np.abs(XX), p)+1.0)
#    gvec[4,:] = Y0*(p*q*power(_np.abs(XX), p-2.0)*power(1.0-power(_np.abs(XX), p), q-2.0)*(p*(q*power(_np.abs(XX), p)-1.0)-power(_np.abs(XX), p)+1.0)+(2.0*exp(-power(XX, 2.0)/power(w, 2.0))*(power(w, 2.0)-2.0*power(_np.abs(XX), 2.0)))/power(w, 4.0))
#    gvec[5,:] = -(4.0*h*Y0*exp(-power(XX, 2.0)/power(w, 2.0))*(power(w, 4.0)-5*power(w, 2.0)*power(_np.abs(XX), 2.0)+2.0*power(_np.abs(XX), 4.0)))/power(w, 7.0)
#
#    return gvec
    return ModelQuasiParabolic._partial_deriv2(XX, aa, **kwargs)
# end def partial_deriv2_qparab


def model_qparab(XX=None, af=None, **kwargs):
    """

    """
    kwargs.setdefault('nohollow', False)
    return _model(ModelQuasiParabolic, XX, af, **kwargs)

# ======================================= #


class ModelQuasiParabolic(ModelClass):
    """
    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
        XX - r/a

    a - aa[0] - Y0 - function value on-axis
    e - aa[1] - gg - Y1/Y0 - function value at edge over core
    c, d - aa[2],aa[3]-  pp, qq - power scaling parameters
    h, w - aa[4],aa[5]-  hh, ww - hole depth and width

    f/a = prof1/a + prof2

        prof1 = a*( e+(1-e)*(1-XX^c)^d )    # ModelTwoPower
    {x element R: (d>0 and c=0 and x>0) or (d>0 and x=1)
            or (c>0 and 0<=x<1) or (c<0 and x>1) }

        prof2 = h*(1-exp(-XX^2/w^2))        # EdgePower
   {x element R}


    NOTE:
    The formulation with definition
        prof1 = a*( b+(1-b)*(1-XX^c)^d )   where b = e-h
            doesn't work for the jacobians, where we need the term separated.

        nohollow = True
        af = _np.hstack((af,0.0))
        af = _np.hstack((af,1.0))

    """
    _af = _np.asarray([1.0, 0.05, 2.0, 0.7, -0.24, 0.30], dtype=_np.float64)
#    _af = _np.asarray([1.0, 0.52, 2.0, 0.7, -0.24, 0.30], dtype=_np.float64)
    _LB = _np.asarray([1e-18, 1e-18,  1e-18,-10,-1, 0], dtype=_np.float64)
    _UB = _np.asarray([ 20.0, 1.0, 10, 10, 1, 1], dtype=_np.float64)
    _fixed = _np.zeros( (6,), dtype=int)
    _analytic_xscaling = False
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        nohollow = kwargs.setdefault('nohollow', False)
        if nohollow:
            self._af[4] = 0.0
            self._af[5] = 1.0
            self._fixed[4:] = int(1)
        super(ModelQuasiParabolic, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        f/a = prof1/a + prof2

            prof1 = a*( b+(1-b)*(1-XX^c)^d )    # ModelTwoPower
        {x element R: (d>0 and c=0 and x>0) or (d>0 and x=1)
                or (c>0 and 0<=x<1) or (c<0 and x>1) }
                where b = e-h (edge-hole)

            prof2 = h*(1-exp(-XX^2/w^2))        # EdgePower
       {x element R}


        NOTE:
        The formulation with definition
            prof1 = a*( b+(1-b)*(1-XX^c)^d )   where b = e-h (edge-hole)
                doesn't work for the jacobians, where we need the term separated.
        """
        a, e, c, d, h, w = tuple(aa)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        prof = ( ModelTwoPower._model(XX, [a, e-h, c, d])
                 + a*ModelExpEdge._model(XX, [h, w]) )
        if _np.isnan(prof).any():
            print('Nan')
        return prof

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        f/a = prof1/a + prof2

            prof1 = a*( b+(1-b)*(1-XX^c)^d )    # ModelTwoPower
        {x element R: (d>0 and c=0 and x>0) or (d>0 and x=1)
                or (c>0 and 0<=x<1) or (c<0 and x>1) }
                where b = e-h (edge-hole)

            prof2 = e*(1-exp(-XX^2/h^2))        # EdgePower
       {x element R}

        dfdx = dprof1dx + a*dprof2dx
        """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, e, c, d, h, w = tuple(aa)
        return ( ModelTwoPower._deriv(XX, [a, e-h, c, d])
                 + a*ModelExpEdge._deriv(XX, [h, w]) )

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        f/a = prof1/a + prof2

            prof1 = a*( b+(1-b)*(1-XX^c)^d )    # ModelTwoPower
        {x element R: (d>0 and c=0 and x>0) or (d>0 and x=1)
                or (c>0 and 0<=x<1) or (c<0 and x>1) }
                where b = e-h (edge-hole)

            prof2 = e*(1-exp(-XX^2/h^2))        # EdgePower
       {x element R}

        dfdx = dprof1dx + a*dprof2dx
        d2fdx2 = d2prof1dx2 + a*d2prof2dx2
        """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, e, c, d, h, w = tuple(aa)
        return ( ModelTwoPower._deriv2(XX, [a, e-h, c, d])
                 + a*ModelExpEdge._deriv2(XX, [h, w]) )

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        f/a = prof1/a + prof2

            prof1 = a*( b+(1-b)*(1-XX^c)^d )    # ModelTwoPower
        {x element R: (d>0 and c=0 and x>0) or (d>0 and x=1)
                or (c>0 and 0<=x<1) or (c<0 and x>1) }
                where b = e-h (edge-hole)

            prof2 = e*(1-exp(-XX^2/h^2))        # EdgePower
       {x element R}

        dfdx = dprof1dx + a*dprof2dx
        d2fdx2 = d2prof1dx2 + a*d2prof2dx2

        NOTE:
        The formulation with definition
            f1 = a*( b+(1-b)*(1-XX^c)^d )   where b = e-h (edge-hole)
            df1de = df1db * dbde = +1*df1db
            df1dh = df1db * dbdh = -1*df1db

        dfda = df1da + f2
        dfde = df1db        where b = e-h
             = a -a*(1-x^^c)^d
        dfdc = df1dc
        dfdd = df1dd
        dfdh = df1dh + a*df2dh = -df1db + a*df2dh where b = e-h
        dfdw = a*df2dw
        """
        a, e, c, d, h, w = tuple(aa)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)

        # Note: b = e-h
        g1 = ModelTwoPower._partial(XX, [a, e-h, c, d])
#        df1de = ModelTwoPower._partial(XX, [a, e, c, d])[1,:]
#        df1dh = -1.0*ModelTwoPower._partial(XX, [a, h, c, d])[1,:]

        g2 = a*ModelExpEdge._partial(XX, [h, w])
#        dfda = ModelQuasiParabolic._model(XX, aa)/a
#        dfde = a-_ModelTwoPower._model(XX, [a, c, d])
        dfda = g1[0,:] + ModelExpEdge._model(XX, [h, w])
        dfde = +1.0*g1[1,:]#
        dfdc = g1[2,:]
        dfdd = g1[3,:]
#        dfdh = -1.0*a+_ModelTwoPower._model(XX, [a, c, d])+ g2[0,:]
        dfdh = -1.0*_np.copy(g1[1,:]) + g2[0,:]
        dfdw = g2[1,:]

        gvec = _np.zeros((6,_np.size(XX)), dtype=_np.float64)
        gvec[0, :] = _np.copy(dfda)
        gvec[1, :] = _np.copy(dfde)
        gvec[2, :] = _np.copy(dfdc)
        gvec[3, :] = _np.copy(dfdd)
        gvec[4, :] = _np.copy(dfdh)
        gvec[5, :] = _np.copy(dfdw)
#        gvec[0, :] = ModelQuasiParabolic._model(XX, aa)/a
#        gvec[1:4,:] = ModelTwoPower._partial(XX, [a, e-h, c, d])[1:,:]
#        gvec[4:, :] = a*ModelExpEdge._partial(XX, [h, w])
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        f/a = prof1/a + prof2

            prof1 = a*( b+(1-b)*(1-XX^c)^d )    # ModelTwoPower
        {x element R: (d>0 and c=0 and x>0) or (d>0 and x=1)
                or (c>0 and 0<=x<1) or (c<0 and x>1) }
                where b = e-h (edge-hole)

            prof2 = e*(1-exp(-XX^2/h^2))        # EdgePower
       {x element R}

        dfdx = dprof1dx + a*dprof2dx
        d2fdx2 = d2prof1dx2 + a*d2prof2dx2

        NOTE:
        The formulation with definition
            f1 = a*( b+(1-b)*(1-XX^c)^d )   where b = e-h (edge-hole)
            df1de = df1db * dbde = +1*df1db
            df1dh = df1db * dbdh = -1*df1db

        dfda = df1da + f2
        dfde = df1db        where b = e-h
        dfdc = df1dc
        dfdd = df1dd
        dfdh = df1dh + a*df2dh = -df1db + a*df2dh where b = e-h
        dfdw = a*df2dw

        d2fdadx = df1dadx + df2dx
        d2fdedx = df1dbdx   where b = e-h
        d2fdcdx = df1dcdx
        d2fdddx = df1dddx
        d2fdhdx = -1.0*df1dbdx + df2dhdx   where b = e-h
        d2fdwdx = df2dwdx
        """
        a, e, c, d, h, w = tuple(aa)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)

        # Note: b = e-h
        dg1dx = ModelTwoPower._partial_deriv(XX, [a, e-h, c, d])
        dg2dx = a*ModelExpEdge._partial_deriv(XX, [h, w])
#        d2fdxda = ModelQuasiParabolic._deriv(XX, aa)/a
#        d2fdxde = a-_ModelTwoPower._model(XX, [a, c, d])
        d2fdxda = dg1dx[0,:] + ModelExpEdge._deriv(XX, [h, w])
        d2fdxde = +1.0*dg1dx[1,:]
        d2fdxdc = dg1dx[2,:]
        d2fdxdd = dg1dx[3,:]
        d2fdxdh = -1.0*dg1dx[1,:] + dg2dx[0,:]
        d2fdxdw = dg2dx[1,:]

        dgdx = _np.zeros((6,_np.size(XX)), dtype=_np.float64)
        dgdx[0, :] = _np.copy(d2fdxda)
        dgdx[1, :] = _np.copy(d2fdxde)
        dgdx[2, :] = _np.copy(d2fdxdc)
        dgdx[3, :] = _np.copy(d2fdxdd)
        dgdx[4, :] = _np.copy(d2fdxdh)
        dgdx[5, :] = _np.copy(d2fdxdw)
#        dgdx[1:4,:] = ModelTwoPower._partial_deriv(XX, [a, e-h, c, d])[1:,:]
#        dgdx[4:,:] = a*ModelExpEdge._partial_deriv(XX, [h, w])
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        a, e, c, d, h, w = tuple(aa)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)

        # Note: b = e-h
        d2g1dx2 = ModelTwoPower._partial_deriv2(XX, [a, e-h, c, d])
        d2g2dx2 = a*ModelExpEdge._partial_deriv2(XX, [h, w])
        d3fdx2da = d2g1dx2[0,:] + ModelExpEdge._deriv2(XX, [h, w])
        d3fdx2de = +1.0*d2g1dx2[1,:]
        d3fdx2dc = d2g1dx2[2,:]
        d3fdx2dd = d2g1dx2[3,:]
        d3fdx2dh = -1.0*d2g1dx2[1,:] + d2g2dx2[0,:]
        d3fdx2dw = d2g2dx2[1,:]

        d2gdx2 = _np.zeros((6,_np.size(XX)), dtype=_np.float64)
        d2gdx2[0, :] = _np.copy(d3fdx2da)
        d2gdx2[1, :] = _np.copy(d3fdx2de)
        d2gdx2[2, :] = _np.copy(d3fdx2dc)
        d2gdx2[3, :] = _np.copy(d3fdx2dd)
        d2gdx2[4, :] = _np.copy(d3fdx2dh)
        d2gdx2[5, :] = _np.copy(d3fdx2dw)

#        d2gdx2 = _np.zeros((6,_np.size(XX)), dtype=_np.float64)
#        d2gdx2[0, :] = ModelQuasiParabolic._deriv2(XX, aa)/a
#        d2gdx2[1:4,:] = ModelTwoPower._partial_deriv2(XX, [a, e-h, c, d])[1:,:]
#        d2gdx2[4:,:] = a*ModelExpEdge._partial_deriv2(XX, [h, w])
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        rescaling: y'=(y-yo)/ys
            y' = a'*( b'+(1-b')*(1-x^c')^d' + h'*(1-exp(x^2/w'^2)))
               = a'*b'+a'*h' + a'*(1-b')*(...) + a'*h'*(...)

            y = yo+ys*a'*(...)
                by inspection   ... wrap a' = ys*a'
                (1)
            constants :   (1)  a*b+a*h = yo+ys*(a'*b'+a'h')
            twopower :    (2)  a*(1-b) = ys*a'*(1-b')
            exp term :    (3)   a*h = ys*a'*h'

            a = yo+ys*a'         or            a' = (a-yo)/ys
            h = ys*h'*a'/a                     h' = h*a/(ys*a')
            e = h + (yo+ys*a'*(e'-h'))/a    e' = h'+1/ys *(a*(e-h)-yo)/a'
                  c = c'
                  d = d'
                  w = w'
                                iff xs = 1.0, xo = 0.0

        rescaling: x'=(x-xo)/xs
            is not possible because of the non-linearities in the x-functions
                (1-x^a2)^a3 = (1-(x-xo)^a2'/xs^a2')^a3'
          and   exp(x^2/a5^2) = exp((x-xo)^2/(xs*a5')^2)

        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)   # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[0] = ys*ain[0]+yo
        aout[4] = ys*ain[0]*ain[4]/aout[0]
        aout[1] = aout[4] + (yo+ys*ain[0]*(ain[1]-ain[4]))/aout[0]
        return aout

    def scaleaf(self, ain, **kwargs):
        """
            a = yo+ys*a'         or            a' = (a-yo)/ys
            h = ys*h'*a'/a                     h' = h*a/(ys*a')
            e = h + (yo+ys*a'*(e'-h'))/a    e' = h'+1/ys *(a*(e-h)-yo)/a'
                  c = c'
                  d = d'
                  w = w'
                                iff xs = 1.0, xo = 0.0
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)
        xs = kwargs.setdefault('xs', self.xslope)   # analysis:ignore
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[0] = (ain[0]-yo)/ys
        aout[4] = ain[0]*ain[4]/(ys*aout[0])
        aout[1] = (ain[0]*(ain[1]-ain[4])-yo)/(ys*aout[0]) + aout[4]
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xoffset = 0.0
        self.xslope = 1.0
#        self.slope = 1.0
#        self.offset = 0.0
        return super(ModelQuasiParabolic, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelQuasiParabolic, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelQuasiParabolic

# ========================================================================== #
# ========================================================================== #


def flattop(XX, aa):
    return ModelFlattop._model(XX, aa)

def deriv_flattop(XX, aa):
    return ModelFlattop._deriv(XX, aa)

def partial_flattop(XX, aa):
    return ModelFlattop._partial(XX, aa)

def partial_deriv_flattop(XX, aa):
    return ModelFlattop._partial_deriv(XX, aa)

def model_flattop(XX=None, af=None, **kwargs):
    """
    """
    return _model(ModelFlattop, XX, af, **kwargs)

class ModelFlattop(ModelClass):
    """
    A flat-top plasma parameter profile with three free parameters:
        a, b, c
    prof ~ f(x) = a / (1 + (x/b)^c)
        af[0] - a - central value of the plasma parameter
        af[1] - b - determines the gradient location, (sqrt(b) is the width)
        af[2] - c - the gradient steepness
    The profile is constant near the plasma center, smoothly descends into a
    gradient near (x/b)=1 and tends to zero for (x/b)>>1

    domain: (b<0 and x<0) or (b>0 and x>0)
    """
    _af = _np.asarray([1.0, 0.4, 5.0], dtype=_np.float64)
    _LB = _np.asarray([1e-18, 1e-18, 1.0], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, 1.0, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelFlattop, self).__init__(XX, af, **kwargs)
    # end def __init__

    def __str__(self):
        return "Model: f(x)= %3.1f/(1+(x/%3.1f)^%3.1f)"%(self.af[0], self.af[1], self.af[2])

    def __repr__(self):
        return "Flattop Model(a=%3.1f, b=%3.1f, c=%3.1f)"%(self.af[0], self.af[1], self.af[2])

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        A flat-top plasma parameter profile with three free parameters:
            a, b, c
        prof ~ f(x) = a / (1 + (x/b)^c)

        """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, b, c = tuple(aa)
        temp = power(XX/b, c)
        return a / (1.0 + temp)

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)
        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)
        """
        a, b, c = tuple(aa)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        temp = power(XX/b, c)
        return -1.0*a*c*temp/(XX*power(1.0+temp,2.0))

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)
        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)

        dfda = 1.0/(1+power(x/b, c))
        dfdb = a*c*(x/b)^c/(b*(1+(x/b)^c)^2)
             = prof^2*(c/ab)*(x/b)^c
        dfdc = -1.0*a*(x/b)^c*ln|x/b|/ (1+(x/b)^c)^2
             = -1*(prof^2/a)*(x/b)^c*ln|x/b|
        """
        nx = _np.size(XX)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, b, c = tuple(aa)
        temp = power(XX/b, c)
        prof = flattop(XX, aa)

        gvec = _np.zeros((3, nx), dtype=_np.float64)
        gvec[0, :] = prof / a
        gvec[1, :] = power(prof, 2.0)*temp*c/(a*b)
        gvec[2, :] = (-1.0*power(prof, 2.0)*(temp/a)*log(_np.abs(XX/b)))
        return gvec

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)

        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)

        dfda = 1.0/(1+power(x/b, c))
        dfdb = a*c*(x/b)^c/(b*(1+(x/b)^c)^2)
             = prof^2*(c/ab)*(x/b)^c
        dfdc = -1.0*a*(x/b)^c*ln|x/b|/ (1+(x/b)^c)^2
             = -1*(prof^2/a)*(x/b)^c*ln|x/b|

        d2fdx2 = a*c*(x/b)^c*(c*(x/b)^c+(x/b)^c-c+1.0)/(x^2*(1+(x/b)^c)^3)
        """
        a, b, c = tuple(aa)
        temp = power(XX/b, c)
        x2 = power(XX,2.0)
        temp3 = power(1.0+temp,3.0)
        return a*c*temp*(c*temp+temp-c+1.0)/(x2*temp3)

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)

        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)

        d2fdx2 = a*c*(x/b)^c*(c*(x/b)^c+(x/b)^c-c+1.0)/(x^2*(1+(x/b)^c)^3)

        dfda = 1.0/(1+power(x/b, c))
        dfdb = a*c*(x/b)^c/(b*(1+(x/b)^c)^2)
             = prof^2*(c/ab)*(x/b)^c
        dfdc = -1.0*a*(x/b)^c*ln|x/b|/ (1+(x/b)^c)^2
             = -1*(prof^2/a)*(x/b)^c*ln|x/b|

        d2fdxda = -1 * c*(x/b)^c / ( x* ( 1+(x/b)^c )^2 )
                = dfdx / a
        d2fdxdb = -1 * a * c^2 * (x/b)^(c-1) * ((x/b)^c-1.0) / (b^2*( 1+(x/b)^c )^3 )
                = prof * -1 * c^2 * (x/b)^(c-1) * ((x/b)^c-1.0) / (b^2*( 1+(x/b)^c )^2 )
                = prof * x*dprofdx * c/a * (x/b)^(-1) * ((x/b)^c-1.0) / (b^2)
                = prof * dprofdx * c/a * ((x/b)^c-1.0) / b
        d2fdxdc = a * (x/b)^c * ( -(x/b)^c + c*(x/b)^c*ln|x/b| - c*ln|x/b| - 1.0  )
                 / ( x*( (x/b)^c + 1 )^3 )
                = -1.0*prof*dfdx/(a*c)*( -(x/b)^c + c*(x/b)^c*ln|x/b| - c*ln|x/b| - 1.0  )
        """
        nx = _np.size(XX)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, b, c = tuple(aa)
        temp = power(XX/b, c)
        prof = flattop(XX, aa)
        dprofdx = deriv_flattop(XX, aa)

        dgdx = _np.zeros((3, nx), dtype=_np.float64)
        dgdx[0, :] = dprofdx / a
        dgdx[1, :] = prof * dprofdx * (c/a)*(temp-1.0)/b
        dgdx[2, :] = -1.0*prof*dprofdx/(a*c) *( -1.0*temp + c*temp*log(_np.abs(XX/b)) - c*log(_np.abs(XX/b))  - 1.0 )
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)

        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)

        d2fdx2 = a*c*(x/b)^c*(c*(x/b)^c+(x/b)^c-c+1.0)/(x^2*(1+(x/b)^c)^3)

        dfda = 1.0/(1+power(x/b, c))
        dfdb = a*c*(x/b)^c/(b*(1+(x/b)^c)^2)
             = prof^2*(c/ab)*(x/b)^c
        dfdc = -1.0*a*(x/b)^c*ln|x/b|/ (1+(x/b)^c)^2
             = -1*(prof^2/a)*(x/b)^c*ln|x/b|

        d2fdxda = -1 * c*(x/b)^c / ( x* ( 1+(x/b)^c )^2 )
                = dfdx / a
        d2fdxdb = -1 * a * c^2 * (x/b)^(c-1) * ((x/b)^c-1.0) / (b^2*( 1+(x/b)^c )^3 )
                = prof * -1 * c^2 * (x/b)^(c-1) * ((x/b)^c-1.0) / (b^2*( 1+(x/b)^c )^2 )
                = prof * x*dprofdx * c/a * (x/b)^(-1) * ((x/b)^c-1.0) / (b^2)
                = prof * dprofdx * c/a * ((x/b)^c-1.0) / b
        d2fdxdc = a * (x/b)^c * ( -(x/b)^c + c*(x/b)^c*ln|x/b| - c*ln|x/b| - 1.0  )
                 / ( x*( (x/b)^c + 1 )^3 )
                = -1.0*prof*dfdx/(a*c)*( -(x/b)^c + c*(x/b)^c*ln|x/b| - c*ln|x/b| - 1.0  )

        d3fdx2da = d2fdx2/a
        d3fdx2db = dprofdx^2*(c/a * ((x/b)^c-1.0) / b)
                + prof * d2profdx2 * c/a * ((x/b)^c-1.0) / b
                + prof * dprofdx * c/a * c*(x/b)^(c-1)
        d3fdx2dc = -1.0*dfdx^2/(a*c)*( -(x/b)^c + c*(x/b)^c*ln|x/b| - c*ln|x/b| - 1.0  )
               -1.0*prof*d2fdx2/(a*c)*( -(x/b)^c + c*(x/b)^c*ln|x/b| - c*ln|x/b| - 1.0  )
               -1.0*prof*dfdx/(a*c)*( -b*c*(x/b)^(c-1) + c*c*b*(x/b)^(c-1)*ln|x/b| + c*(x/b)^c*(1/x) - c*(1/x)  )

              = -1.0*(dfdx^2+prof*d2fdx2)/(a*c)*( (x/b)^c*(c*ln|x/b|-1) - c*ln|x/b| - 1.0  )
               -1.0*prof*dfdx/(a*c)*( b*c*(x/b)^(c-1)*(c*ln|x/b|-1.0) + (c/x)*((x/b)^c - 1.0) )

        """
        nx = _np.size(XX)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, b, c = tuple(aa)

        temp = power(XX/b, c)
        bmc = power(1.0/b, c)
        xc1 = power(XX, c-1.0)
        dtdx = c*bmc*xc1  # c*x^(c-1)/b^c = c*tmp/x

        logaXb = log(_np.abs(XX/b))

        prof = ModelFlattop._model(XX, aa, **kwargs)
        dprofdx = ModelFlattop._deriv(XX, aa, **kwargs)
        d2profdx2 = ModelFlattop._deriv2(XX, aa, **kwargs)

        d2gdx2 = _np.zeros((3, nx), dtype=_np.float64)
        d2gdx2[0, :] = d2profdx2 / a
        d2gdx2[1, :] = (dprofdx * dprofdx * (c/a)*(temp-1.0)/b
                       + prof * d2profdx2 * (c/a)*(temp-1.0)/b
                       + prof * dprofdx * (c/a)*(dtdx)/b )

        d2gdx2[2, :] = -1.0*(dprofdx*dprofdx+prof*d2profdx2)/(a*c) *(
            -1.0*temp + c*temp*logaXb - c*logaXb - 1.0 )
        d2gdx2[2, :] -= prof*dprofdx/(a*c) *(
            -1.0*dtdx + c*dtdx*logaXb+c*temp/XX - c/XX)

#        if _np.isnan(d2gdx2).any():
#            print('pause')
#        d2gdx2[2, :] = -1.0*power(dprofdx, 2.0)/(a*c) *(
#            -1.0*temp + c*temp*_np.log(_np.abs(XX/b)) - c*_np.log(_np.abs(XX/b)) - 1.0 )
#        d2gdx2[2, :] += -1.0*prof*d2profdx2/(a*c) *(
#            -1.0*temp + c*temp*_np.log(_np.abs(XX/b)) - c*_np.log(_np.abs(XX/b)) - 1.0 )
#        d2gdx2[2, :] += -1.0*prof*dprofdx/(a*c) *(
#            -1.0*dtdx + c*dtdx*_np.log(_np.abs(XX/b)) + c*temp/XX - c/XX )

        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        y = = a / (1 + (x/b)^c)
        y-scaling:  y'=y/ys
           y = ys*a' / (1 + (x/b')^c')
           a = ys*a'
           b = b'
           c = c'
        y-shifting:  y'=(y-yo)/ys
           y = ys*a' / (1 + (x/b')^c')  + yo*(1 + (x/b')^c') / (1 + (x/b')^c')
               can't make constant coefficients
        x-scaling:  x'=x/xs
           y = ys*a' / (1 + (x/(xs*b'))^c')
               a = ys*a'
               b = xs*b'
               c = c'
        x-shifting:  x'=(x-xo)/xs
           y = ys*a' / (1 + ((x-xo)/(xs*b'))^c')
               can't make constant coefficients

        a = ys*a'
        b = xs*b'
        c = c'
        iff xo, yo = 0.0
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[0] *= ys
        aout[1] *= xs
        return aout

    def scaleaf(self, ain, **kwargs):
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[0] /= ys
        aout[1] /= xs
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xoffset = 0.0
        self.offset = 0.0
        return super(ModelFlattop, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelFlattop, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelFlattop


# ========================================================================== #
# ========================================================================== #


def slopetop(XX, aa):
    return ModelSlopetop._model(XX, aa)

def deriv_slopetop(XX, aa):
    return ModelSlopetop._deriv(XX, aa)

def partial_slopetop(XX, aa):
    return ModelSlopetop._partial(XX, aa)

def partial_deriv_slopetop(XX, aa):
    return ModelSlopetop._partial_deriv(XX, aa)

def model_slopetop(XX=None, af=None, **kwargs):
    return _model(ModelSlopetop, XX, af, **kwargs)

class ModelSlopetop(ModelClass):
    """
    Commonly referred to the Slopetop profile and used in a lot of W7-AS
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
    _af = _np.asarray([1.0, 0.4, 5.0, 0.5], dtype=_np.float64)
    _LB = _np.asarray([1e-18, 0.0, 1.0, -1], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, 1.0, _np.inf, 1], dtype=_np.float64)
    _fixed = _np.zeros( (4,), dtype=int)
    _analytic_xscaling = True
    _analytic_yscaling = True
    def __init__(self, XX, af=None, **kwargs):
        super(ModelSlopetop, self).__init__(XX, af, **kwargs)
    # end def __init__

#    @staticmethod
#    def _model(XX, aa, **kwargs):
#        a, b, c, h = tuple(aa)
#        return a*(1.0-h*(XX/b)) / (1+power(XX/b, c))

    def __str__(self):
        return "Model: f(x)= %3.1f(1-%3.1fx/%3.1f)/(1+(x/%3.1f)^%3.1f)"%(self.af[0], self.af[3], self.af[1], self.af[2])

    def __repr__(self):
        return "Flattop Model(a=%3.1f, b=%3.1f, c=%3.1f, h=%3.1f)"%(self.af[0], self.af[1], self.af[2], self.af[3])

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        f = a*(1-h*x/b)/(1+(x/b)^c)
          = flattop*(1-h*x/b)
        """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, b, c, h = tuple(aa)
        prft = flattop(XX, [a, b, c])
        temp = XX/b
        return prft * (1-h*temp)

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        f = a*(1-h*x/b)/(1+(x/b)^c)
          = flattop*(1-h*x/b)

        dfdx = dflattopdx*(1-h*x/b) - flattop*h/b
        """
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, b, c, h = tuple(aa)
        prft = flattop(XX,[a, b, c])
        drft = deriv_flattop(XX, [a, b, c])

        temp = XX/b
        return drft*(1-h*temp) - prft*h/b

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        f = a*(1-h*x/b)/(1+(x/b)^c)
          = flattop*(1-h*x/b)

        dfdx = dflattopdx*(1-h*x/b) - flattop*h/b
        """
        a, b, c, h = tuple(aa)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
#        prft = ModelFlattop._model(XX,[a, b, c])
        drft = ModelFlattop._deriv(XX, [a, b, c])
        temp = XX/b

        dpdx = ModelFlattop._deriv(XX,[a, b, c])
        drdx = ModelFlattop._deriv2(XX, [a, b, c])
        dtdx = 1.0/b
        return (drdx*(1-h*temp) + drft*(-h*dtdx) - dpdx*h/b)

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        f = a*(1-h*x/b)/(1+(x/b)^c)
          = flattop*(1-h*x/b)

        dfdx = dflattopdx*(1-h*x/b) - flattop*h/b

        dfda = (1-h*x/b)/(1+(x/b)^c) = flattop*(1-h*x/b) / a
        dfdb = dflattop/db *(1-h*x/b) + flattop * h*x/(b*b)
        dfdc = dflattop/dc *(1-h*x/b)
        dfdh = dflattop/dh *(1-h*x/b) - flattop*x/b
             = -flattop*x/b
        """
        nx = _np.size(XX)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, b, c, h = tuple(aa)
        prft = flattop(XX, [a, b, c])
        gft = partial_flattop(XX, [a, b, c])

        temp = XX/b
        prof = slopetop(XX, aa)

        gvec = _np.zeros((4, nx), dtype=_np.float64)
        gvec[0, :] = prof / a
        gvec[1, :] = ( gft[1,:]*(1.0-h*temp) + prft*h*XX/power(b, 2.0) )
        gvec[2, :] = gft[2,:]*(1.0-h*temp)
        gvec[3, :] = (-1.0*XX / b)*prft
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        f = a*(1-h*x/b)/(1+(x/b)^c)
          = flattop*(1-h*x/b)

        dfdx = dflattopdx*(1-h*x/b) - flattop*h/b

        dfda = (1-h*x/b)/(1+(x/b)^c) = flattop*(1-h*x/b) / a
        dfdb = dflattop/db *(1-h*x/b) + flattop * h*x/(b*b)
        dfdc = dflattop/dc *(1-h*x/b)
        dfdh = dflattop/dh *(1-h*x/b) - flattop*x/b
             = -flattop*x/b

        d2fdxda = d2flattopdxda * (1-h*x/b) - dflattopda*h/b
        d2fdxdb = d2flattopdxdb*(1-h*x/b) - dflattopdb*h/b
                 + dflattopdx*(h*x/b^2) + flattop*h/b^2
        d2fdxdc = d2flattopdxdc*(1-h*x/b) - dflattopdc*h/b
        d2fdxdh = dflattopdx*(-x/b) - flattop/b
        """
        nx = _np.size(XX)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, b, c, h = tuple(aa)
    #    prof = slopetop(XX, af)
#        dprofdx = deriv_slopetop(XX, aa)

        temp = XX/b

        prft = flattop(XX, [a, b, c])
        dflatdx = deriv_flattop(XX, [a, b, c])
        gft = partial_flattop(XX, [a, b, c])
        dgft = partial_deriv_flattop(XX, [a, b, c])

        dgdx = _np.zeros((4, nx), dtype= float)
        dgdx[0, :] = dgft[0,:]*(1.0-h*temp) - gft[0,:]*h/b
        dgdx[1, :] = ( dgft[1,:]*(1.0-h*temp) - gft[1,:]*h/b
                     + dflatdx*h*XX/power(b,2.0) + prft*h/power(b,2.0) )
        dgdx[2, :] = dgft[2,:]*(1.0-h*temp) - gft[2,:]*h/b
        dgdx[3, :] = dflatdx*(-1.0*XX/b) - prft/b

#        dgdx[1,:] = dgft[1,:]*(1.0-h*temp) + dflatdx * h*XX/power(b, 2.0)
#        dgdx[1,:] += prft*h/power(b, 2.0) - (h/b)*gft[1,:]
#        dgdx[2,:] = dgft[2,:]*(1.0-h*temp) - gft[2, :]*h/b
#        dgdx[3,:] = -1.0*(XX/b)*prft
        return dgdx


    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        f = a*(1-h*x/b)/(1+(x/b)^c)
          = flattop*(1-h*x/b)

        dfdx = dflattopdx*(1-h*x/b) - flattop*h/b

        dfda = (1-h*x/b)/(1+(x/b)^c) = flattop*(1-h*x/b) / a
        dfdb = dflattop/db *(1-h*x/b) + flattop * h*x/(b*b)
        dfdc = dflattop/dc *(1-h*x/b)
        dfdh = dflattop/dh *(1-h*x/b) - flattop*x/b
             = -flattop*x/b

        d2fdxda = d2flattopdxda * (1-h*x/b) - dflattopda*h/b
        d2fdxdb = d2flattopdxdb*(1-h*x/b) - dflattopdb*h/b
                 + dflattopdx*(h*x/b^2) + flattop*h/b^2
        d2fdxdc = d2flattopdxdc*(1-h*x/b) - dflattopdc*h/b
        d2fdxdh = dflattopdx*(-x/b) - flattop/b
        """
        nx = _np.size(XX)
        XX = _np.copy(XX)
#        XX = _np.abs(XX)
        a, b, c, h = tuple(aa)
    #    prof = slopetop(XX, af)
#        dprofdx = deriv_slopetop(XX, aa)

        temp = XX/b
        dtdx = 1.0/b

#        prft = ModelFlattop._model(XX, [a, b, c])
        dflatdx = ModelFlattop._deriv(XX, [a, b, c])
#        gft = ModelFlattop._partial(XX, [a, b, c])
        dgft = ModelFlattop._partial_deriv(XX, [a, b, c])

        d2flatdx2 = ModelFlattop._deriv2(XX, [a, b, c])
        d2gfdx2 = ModelFlattop._partial_deriv2(XX, [a, b, c])

        d2gdx2 = _np.zeros((4, nx), dtype=_np.float64)
        d2gdx2[0, :] = d2gfdx2[0,:]*(1.0-h*temp)+dgft[0,:]*(-h*dtdx) - dgft[0,:]*h/b
        d2gdx2[1, :] = ( d2gfdx2[1,:]*(1.0-h*temp) + dgft[1,:]*(-h*dtdx) - dgft[1,:]*h/b
                     + d2flatdx2*h*XX/power(b, 2.0) + 2.0*dflatdx*h/power(b,2.0))

        d2gdx2[2, :] = d2gfdx2[2,:]*(1.0-h*temp)+dgft[2,:]*(-h*dtdx) - dgft[2,:]*h/b
        d2gdx2[3, :] = d2flatdx2*(-1.0*XX/b) + dflatdx*(-1.0/b) - dflatdx/b
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain, **kwargs):
        """
        f = a*(1-h*x/b)/(1+(x/b)^c)
          = flattop*(1-h*x/b)

        y-scaling:  y'=y/ys
           y = ys*a'*(1-h'*x/b') / (1 + (x/b')^c')
           a = ys*a'
           b = b'
           c = c'
        y-shifting:  y'=(y-yo)/ys
           y = ys*a'*(1-h'*x/b') / (1 + (x/b')^c')  + yo*(1 + (x/b')^c') / (1 + (x/b')^c')
             num:    handle, then check if it works
                 a*(1-h*x/b) = yo + yo*(x/b')^c' + ys*a' - ys*a'*h'*x/b'
                 a:   a = yo + ys*a'
                 -a*h*x/b = yo*(x/b')^c' - ys*a'*h'*x/b'
                 -a*h/b = yo*(x/b')^c'/x - ys*a'*h'/b'
               can't make constant coefficients
        x-scaling:  x'=x/xs
           y = ys*a' / (1 + (x/(xs*b'))^c')
               a = ys*a'
               b = xs*b'
               c = c'
        x-shifting:  x'=(x-xo)/xs
           y = ys*a' / (1 + ((x-xo)/(xs*b'))^c')
               can't make constant coefficients

        a = ys*a'
        b = xs*b'
        c = c'
        iff xo, yo = 0.0
        """
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore
        aout[0] *= ys
        aout[1] *= xs
        return aout

    def scaleaf(self, ain, **kwargs):
        ain, aout = _np.copy(ain), _np.copy(ain)
        ys = kwargs.setdefault('ys', self.slope)
        yo = kwargs.setdefault('yo', self.offset)  # analysis:ignore
        xs = kwargs.setdefault('xs', self.xslope)
        xo = kwargs.setdefault('xo', self.xoffset)  # analysis:ignore

        aout[0] /= ys
        aout[1] /= xs
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xoffset = 0.0
        self.offset = 0.0
        return super(ModelSlopetop, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelSlopetop, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelSlopetop

# ========================================================================== #
# ========================================================================== #
# These two haven't been checked yet!!! also need to add analytic jacobian
# for the derivatives

def model_Heaviside(XX=None, af=None, **kwargs):
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
    npoly = kwargs.setdefault('npoly', 4)
    rinits = kwargs.setdefault('rinits', [0.30, 0.35])
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
    info.fixed = _np.zeros( _np.shape(info.Lbounds), dtype=int)
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

    # d(tanh(x))/dx = 1-tanh(x)^2 = _ut.sech(x)^2
    # dfdx  = (a1*2*x^1+a2+0) + 0.5*k*a4*(_ut.sech(k*(x-a5))^2 - _ut.sech(k*(x-a6))^2)
    info.dprofdx = _np.zeros((nx,), dtype=_np.float64)
    for ii in range(npoly):  # ii = 1:(num_fit-4)
        kk = npoly - (ii+1)
        info.dprofdx = info.dprofdx+af[ii]*kk*(XX**(kk-1))
    # endfor
    info.dprofdx = info.dprofdx + 0.5*af[num_fit-3]*zz*(
                      (_ut.sech(zz*(XX-af[num_fit-2]))**2)
                      - (_ut.sech(zz*(XX-af[num_fit-1]))**2))

    # dfda1 = x^2
    # dfda2 = x
    # dfda3 = 1
    # dfda4 = a4<XX<a5 = 0.5*(tanh(kk*(x-a5)) - tanh(kk*(x-a5)))
    # dfda5 = -0.5*a4*kk*_ut.sech(kk*(x-a5))^2
    # dfda6 = -0.5*a4*kk*_ut.sech(kk*(x-a6))^2
    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
    for ii in range(npoly+1):  # ii=1:(num_fit-3)
        kk = npoly+1 - (ii+1)
        gvec[ii, :] = (XX**kk)
    # endfor

    gvec[num_fit-3, :] = (0.5*(_np.tanh(zz*(XX-af[num_fit-2]))
                          - _np.tanh(zz*(XX-af[num_fit-1]))))
    gvec[num_fit-2, :] = (-1.0*0.5*af[num_fit-3]*zz
                          * _ut.sech(zz*(XX-af[num_fit-2]))**2)
    gvec[num_fit-1, :] = (-1.0*0.5*af[num_fit-3]*zz*(-1
                          * _ut.sech(zz*(XX-af[num_fit-1]))**2))

    return prof, gvec, info
# end def model_Heaviside()


#class ModelHeaviside(ModelClass):
#    """
#
#    """
#    _af = _np.asarray([1.0], dtype=_np.float64)
#    _LB = _np.asarray([0.0], dtype=_np.float64)
#    _UB = _np.asarray([_np.inf], dtype=_np.float64)
#    _fixed = _np.zeros( (1,), dtype=int)
#    def __init__(self, XX, af=None, **kwargs):
#        super(ModelHeaviside, self).__init__(XX, af, **kwargs)
#    # end def __init__
#
#    @staticmethod
#    def _model(XX, aa, **kwargs):
#
#    @staticmethod
#    def _deriv(XX, aa, **kwargs):
#
#    @staticmethod
#    def _partial(XX, aa, **kwargs):
#
#    @staticmethod
#    def _partial_deriv(XX, aa, **kwargs):
#
##    @staticmethod
##    def _hessian(XX, aa):
#
#    # ====================================== #
#
##    @staticmethod
##    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
##        """
##        """
##        ain = _np.copy(ain)
##        aout = _np.copy(ain)
##        return aout
#
#    # ====================================== #
#
##    def checkbounds(self, dat):
##        return super(ModelHeaviside, self).checkbounds(dat, self.aa, mag=None)
#
#    # ====================================== #
## end def ModelHeaviside

# ========================================================================== #
# ========================================================================== #


def model_StepSeries(XX=None, af=None, **kwargs):
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
    npoly = kwargs.setdefault('npoly',4)

    if af is None:
        af = _np.hstack((5.0, 1.0*_np.random.randn(npoly,)))
#        af *= 0.1*_np.random.normal(0.0, 1.0, len(af))
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
    info.fixed = _np.zeros( _np.shape(info.Lbounds), dtype=int)
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
        # dfdx = 0.5*zz*sum_k(ak*_ut.sech(zz*(x-b^ak))^2)
        # dfdak = 0.5*(1 + tanh(zz*(x-b^ak))
        #                - 0.5*zz*ln(b)*b^ak*_ut.sech(zz*(x-b^ak))^2)

        # f    = a1*tanh(zz(x-x1))+a2*tanh(zz(x-x2))+...an*tanh(zz(x-xn))
        temp = _np.tanh(zz*(XX-bb**af[ii]))
        prof = prof + 0.5*af[ii]*(1 + temp)

        info.dprofdx = info.dprofdx+0.5*af[ii]*zz*(1 - temp**2)
        # info.dprofdx = info.dprofdx+0.5*af[ii]*zz*_ut.sech(zz*(XX-bb**af[ii]))**2

        gvec[ii, :] = (0.5*(1 + temp)
                       - 0.5*zz*log(bb)*(bb**af[ii])*(1 - temp**2))

#        #indice of transitions
#        bx = _np.floor(1+bb/(XX[2]-XX[1]))
#        gvec[num_fit-1,ba-1:bx-1] = (zz*log(bb)*(-bb)**af[num_fit-1]
#                            * _ut.sech(zz*(XX[ba-1:bx-1]-bb**af[num_fit-1]))**2
#        ba = _np.floor(1+bb/(XX(2)-XX(1)))
    # endfor

    return prof, gvec, info
# end def model_StepSeries()


#class ModelStepSeries(ModelClass):
#    """
#
#    """
#    _af = _np.asarray([1.0], dtype=_np.float64)
#    _LB = _np.asarray([0.0], dtype=_np.float64)
#    _UB = _np.asarray([_np.inf], dtype=_np.float64)
#    _fixed = _np.zeros( (1,), dtype=int)
#    def __init__(self, XX, af=None, **kwargs):
#        super(ModelStepSeries, self).__init__(XX, af, **kwargs)
#    # end def __init__
#
#    @staticmethod
#    def _model(XX, aa, **kwargs):
#
#    @staticmethod
#    def _deriv(XX, aa, **kwargs):
#
#    @staticmethod
#    def _partial(XX, aa, **kwargs):
#
#    @staticmethod
#    def _partial_deriv(XX, aa, **kwargs):
#
##    @staticmethod
##    def _hessian(XX, aa):
#
#    # ====================================== #
#
##    @staticmethod
##    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
##        """
##        """
##        ain = _np.copy(ain)
##        aout = _np.copy(ain)
##        return aout
#
#    # ====================================== #
#
##    def checkbounds(self, dat):
##        return super(ModelStepSeries, self).checkbounds(dat, self.aa, mag=None)
#
#    # ====================================== #
## end def ModelStepSeries

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
       4 - Exponential on Background - f(x) ~ a1*(exp(a2*XX^a3) + XX^a4)
       5 - Polynomial with a Heaviside (Logistics)
                                        - f(x) ~ a1x^2+a2x+a3+a4*(XX>a5)*(XX<a6)
       6 - Series of Step functions  - H(x1)-H(x2) ~ 1/2*(tanh(ka5)-tanh(ka6))
       7 - Quasi-parabolic fit       - f(x) / af[0] ~ af[1]-af[4]+(1-af[1]+af[4])*(1-XX^af[2])^af[3]
                                                + af[4]*(1-exp(-XX^2/af[5]^2))
       8 - Even order polynomial     - f(x) ~ sum( af(ii)*XX^2*(polyorder-ii))
       9 - 2-power profile           - f(x) ~ (Core-Edge)*(1-x^pow1)^pow2 + Edge
       10 - Parabolic fit            - f(x) ~ a*(1.0-x^2)
       11 - Flat top profile         - f(x) ~ a / (1 + (x/b)^c)
       12 - Slopetop profile         - f(x) ~ a * (1-h*(x/b)) / (1+(x/b)^c) = flattop*(1-h*(x/b))
    """
#    if af is not None:
#        if (af==0).any():
#            af[_np.where(af==0)[0]] = 1e-14
#        # end if
#    # end if
    if XX is None:
        XX = _np.linspace(1e-4, 1, 200)
    # endif
    XX = clean_XX(XX)

    # ====================================================================== #

    if model_number == 1:
        if verbose: print('Modeling with an order %i product of Exponentials'%(npoly,))  # endif
        [prof, gvec, info] = model_ProdExp(XX, af, npoly=npoly)
        info.func = model_ProdExp

    elif model_number == 2:
        if verbose: print('Modeling with an order %i polynomial'%(npoly,))  # endif
        [prof, gvec, info] = model_poly(XX, af, npoly=npoly)
        info.func = model_poly

    elif model_number == 3:
        if verbose: print('Modeling with an order %i power law'%(npoly,))  # endif
        [prof, gvec, info] = model_PowerLaw(XX, af, npoly=npoly)
        info.func = model_PowerLaw

    elif model_number == 4:
        if verbose: print('Modeling with an exponential on order %i polynomial background'%(npoly,))  # endif
        [prof, gvec, info] = model_Exponential(XX, af, npoly=npoly)
        info.func = model_Exponential

    elif model_number == 5:
        if verbose: print('Modeling with an order %i polynomial+Heaviside fn'%(npoly,))  # endif
        [prof, gvec, info] = model_Heaviside(XX, af, npoly=npoly)
        info.func = model_Heaviside

    elif model_number == 6:
        if verbose: print('Modeling with a %i step profile'%(npoly,))  # endif
        [prof, gvec, info] = model_StepSeries(XX, af, npoly=npoly)
        info.func = model_StepSeries

    elif model_number == 7:
        if verbose: print('Modeling with a quasiparabolic profile')  # endif
        [prof, gvec, info] = model_qparab(XX, af)
        info.func = model_qparab

    elif model_number == 8:
        if verbose: print('Modeling with an order %i even polynomial'%(npoly,))  # endif
        [prof, gvec, info] = model_evenpoly(XX, af, npoly=npoly)
        info.func = model_evenpoly

    elif model_number == 9:  # Two power fit
        if verbose: print('Modeling with a 2-power profile')  # endif
        [prof, gvec, info] = _model_twopower(XX, af)
        info.func = _model_twopower

    elif model_number == 10:
        if verbose: print('Modeling with a parabolic profile')  # endif
        [prof, gvec, info] = model_parabolic(XX, af)
        info.func = model_parabolic

    elif model_number == 11:
        if verbose: print('Modeling with a flat-top profile')  # endif
        [prof, gvec, info] = model_flattop(XX, af)
        info.func = model_flattop

    elif model_number == 12:
        if verbose: print('Modeling with a Slopetop-style profile')  # endif
        [prof, gvec, info] = model_slopetop(XX, af)
        info.func = model_slopetop

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
       4 - Exponential on Background - chi ~ a1*(exp(a2*XX^a3) + XX^a4)
       5 - Polynomial with a Heaviside (Logistics)
                                        - chi ~ a1x^2+a2x+a3+a4*(XX>a5)*(XX<a6)
       6 - Series of Step functions  - H(x1)-H(x2) ~ 1/2*(tanh(ka5)-tanh(ka6))
       7 - Quasi-parabolic fit       - 10.0 - qparab
       8 - Even order polynomial     - chi ~ sum( af(ii)*XX^2*(polyorder-ii))
       9 - 2-power profile           - chi ~ (Core-Edge)*(1-x^pow1)^pow2 + Edge
    """
#    if af is not None:
#        if (af==0).any():
#            af[_np.where(af==0)[0]] = 1e-14
#        # end if
#    # end if

    if XX is None:
        XX = _np.linspace(1e-4, 1, 200)
    # endif
    XX = clean_XX(XX)

    # ====================================================================== #

    if model_number == 1:
        if verbose: print('Modeling with an order %i product of Exponentials'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_ProdExp(XX, af, npoly=npoly)
        info.func = model_ProdExp

    elif model_number == 2:
        if verbose: print('Modeling with an order %i polynomial'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_poly(XX, af, npoly=npoly)
        info.func = model_poly

    elif model_number == 3:
        if verbose: print('Modeling with an order %i power law'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_PowerLaw(XX, af, npoly=npoly)
        info.func = model_PowerLaw

    elif model_number == 4:
        if verbose: print('Modeling with an exponential on order %i polynomial background'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_Exponential(XX, af, npoly=npoly)
        info.func = model_Exponential

    elif model_number == 5:
        if verbose: print('Modeling with an order %i polynomial+Heaviside fn'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_Heaviside(XX, af, npoly=npoly)
        info.func = model_Heaviside

    elif model_number == 6:
        if verbose: print('Modeling with a %i step profile'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_StepSeries(XX, af, npoly=npoly)
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
                _, _, info = tfunc(XX, af, npoly=npoly)
                af = _np.asarray(info.af.tolist()+[3.0], dtype=_np.float64)
#                info.fixed = _np.asarray(info.fixed.tolist()+[int(0)], dtype=int)
            else:
                _, _, info = tfunc(XX, af[:-1], npoly=npoly)
            # end if
            info.af = _np.copy(af)
            info.prof += af[-1].copy()
            info.gvec = _np.insert(info.gvec, [-1], _np.ones(_np.shape(info.gvec[0,:]), dtype=_np.float64), axis=0)
            info.dgdx = _np.insert(info.dgdx, [-1], _np.zeros(_np.shape(info.dgdx[0,:]), dtype=_np.float64), axis=0)
            info.Lbounds = _np.asarray(info.Lbounds.tolist()+[-20.0], dtype=_np.float64)
            info.Ubounds = _np.asarray(info.Ubounds.tolist()+[ 20.0], dtype=_np.float64)
            info.fixed = _np.asarray(info.fixed.tolist()+[int(0)], dtype=int)
            return info.prof, info.gvec, info
        [chi_eff, gvec, info] = tfunc(XX, af)
        info.func = tfunc

    elif model_number == 8:
        if verbose: print('Modeling with an order %i even polynomial'%(npoly,))  # endif
        [chi_eff, gvec, info] = model_evenpoly(XX, af, npoly=npoly)
        info.func = model_evenpoly

    elif model_number == 9:
        if verbose: print('Modeling with a 2-power profile')  # endif
        def tfunc(XX, af, npoly=None):
            return _model_twopower(XX, af)
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
    dPdx = exp(xvar**exppow)-1.0       # Radiation profile shape [MW/rho]

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

#    dPdx = (exp(-0.5*((xvar-rloc)/rhalfwidth)**2)
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

def _checkbounds(ain, LB, UB):
    LB = _np.copy(LB)
    UB = _np.copy(UB)
    ain = _np.copy(ain)
    for ii in range(len(LB)):
        if ain[ii]<LB[ii]:                ain[ii] += 1e-12     # end if
        if ain[ii]>UB[ii]:                ain[ii] -= 1e-12     # end if
    # end for
    return ain

def randomize_initial_conditions(LB, UB, af0=None, varaf0=None, inf=10, modelinfo=None):
    """
    randomize initial conditions, while assuming that the problem is
    normalized to unit magnitude
    """
    LB = _np.atleast_1d(LB)
    UB = _np.atleast_1d(UB)
    LB, UB = LB.copy(), UB.copy()

    af = _np.zeros_like(LB)
    for ii in range(len(LB)):
        if _np.isinf(LB[ii]):            LB[ii] = -inf   # end if
        if _np.isinf(UB[ii]):            UB[ii] = inf    # end if

        if af0 is None or varaf0 is None:
            af[ii] = _np.random.uniform(low=LB[ii]+1e-10, high=UB[ii]-1e-10, size=None)
        else:
            af0 = _np.atleast_1d(af0)
            varaf0 = _np.atleast_1d(varaf0)
            if len(varaf0)<2:
                varaf0 = varaf0*_np.ones_like(af0)
            # end if
            af[ii] = _np.copy(af0[ii]) + _np.sqrt(varaf0[ii])*_np.random.normal(low=-1.0, high=1.0, size=None)
            if af[ii]>UB[ii]:
                af[ii] = _np.copy(UB[ii]) - 0.01*_np.sqrt(varaf0[ii])
            elif af[ii]<LB[ii]:
                af[ii] = _np.copy(LB[ii]) + 0.01*_np.sqrt(varaf0[ii])
            # end if
    # end for
    if modelinfo is None:
        af = _checkbounds(af, LB, UB)
    else:
        af = modelinfo.checkbounds(af=af, LB=LB, UB=UB)
    # end if
    return af

#def rescale_xlims(XX, forward=True, ascl=None):
#    if ascl is None:
#        ascl = max(_np.max(XX), 1.0)
#    # end if
#    if forward:
#        return XX/ascl
#    else:
#        return XX*ascl

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
#        xslope = _np.nanmax(xdat)-_np.nanmin(xdat)
#        xoffset = _np.nanmin(xdat)
#        return pdat, vdat, slope, offset, xslope, xoffset
    elif info is not None:
        slope = info.slope
        offset = info.offset

        if hasattr(info, 'xslope'):
            xslope = info.xslope
        else:
            xslope = 1.0
            info.xslope = xslope
        # end if

        if hasattr(info, 'xoffset'):
            xoffset = info.xoffset
        else:
            xoffset = 0.0
            info.xoffset = xoffset
        # end if

        if hasattr(info,'pdat'):
            info.pdat = info.pdat*slope+offset
            info.vdat = info.vdat * (slope**2.0)
        # end if
        if hasattr(info,'dprofdx'):
            info.dprofdx = slope*info.dprofdx
            info.vardprofdx = (slope**2.0)*info.vardprofdx

            info.dprofdx /= xslope
            info.vardprofdx /= xslope*xslope
        # end if

        if hasattr(info,'prof'):
            info.prof = slope*info.prof+offset
            info.varprof = (slope**2.0)*info.varprof
        # end if

#        info.af = info.unscaleaf(info.af, info.slope, info.offset, info.xslope)
#        _, _, info = info.model(info.XX, info.af, **info.kwargs)
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
    return Y0*exp(-(tt-t0)/tau)

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

# ========================================================================== #
# ========================================================================== #

# ========================================================================== #

def _model(mod, XX, af=None, **kwargs):
    """
    This function is for backwards compatability and will be deprecated!
    """
    mod = mod(XX, af, **kwargs)
    if XX is None:
        return mod
#    offset = 0.0 if not hasattr(mod, '_secretoffset') else mod._secretoffset    # end if
    mod.update(XX, af, **kwargs)

    mod.modfunc = lambda _x, _a: mod.model(_x, _a, **kwargs) #+ offset
    mod.moddfunc = lambda _x, _a: mod.derivative(_x, _a, **kwargs)
    mod.modgfunc = lambda _x, _a: mod.jacobian(_x, _a, **kwargs)
    mod.moddgfunc = lambda _x, _a: mod.derivative_jacobian(_x, _a, **kwargs)
    return mod.prof, mod.gvec, mod


# ========================================================================== #
# ========================================================================== #

#def sech(x):
#    """
#    sech(x)
#    Uses numpy's cosh(x).
#    """
#    return 1.0/_np.cosh(x)

def clean_XX(XX):
    """
    It is debateable whether this is beneficial.
    ex //
        ln(x) is only defined for x>0    (ln(0) = -inf)
        ln(-x) is only defined for x<0
        exp(+x) != exp(-x)

    For profiles you might think, hey no problem, we want even functions!
    but then the derivatives of even functions are odd
    and this matters for error propagation!

    Pitfall:     y = a*ln(x) == ln(abs(x))
                 dy/dx = a/x != a/abs(x)
                 dy/da = a*ln(x) == a*ln(abs(x))
                 d^2y/dxda = 1/x != 1/abs(x)
    absolute value signs can mess up error propagation (especially on derivatives)
             implemented solultion is to code in abs value signs: y = a*ln( abs(x) )
                                   ... not sure if correctly done everywhere

    In practice, the user needs to define limits through
        Upper / Lower bounds options while fitting
        abs(x) when appropriate

    """
    XX = _np.copy(XX)
#    XX = _np.abs(XX)
#    if len([XX])>1 and (XX==0).any():
#        XX[_np.where(XX==0)] = 0.111e-2
#    # end if
    return XX

# ========================================================================== #

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
#    XX = _np.linspace(1e-3, 0.99, num=61)
#    XX = _np.linspace(1e-3, 0.99, num=100)

    # Numerical testing for errors in models
    # Analytic testing for errors in forward/reverse scalings
    mod = ModelLine().test_numerics(num=10)   # checked
    mod = ModelLine().test_scaling(num=10)   #
#
#    mod = ModelSines().test_numerics(num=int((6.0/33.0-0.0)*5.0e2), start=-3.5/33.0, stop=6.0/33.0, fmod=33.0)   # checked
#    mod = ModelSines().test_scaling(num=int((6.0/33.0-0.0)*5.0e2), start=-3.0/33.0, stop=6.0/33.0, fmod=33.0)   #
#
#    mod = ModelSines().test_numerics(nfreqs=2, num=int((6.0/33.0-0.0)*5.0e2), start=-1.0/33.0, stop=6.0/33.0, fmod=33.0)   # checked
#    mod = ModelSines().test_scaling(nfreqs=2, num=int((6.0/33.0-0.0)*5.0e2), start=-1.0/33.0, stop=6.0/33.0, fmod=33.0)   # checked

#    mod = ModelSines().test_numerics(nfreqs=5, num=int((6.0/33.0-0.0)*5.0e3), start=0.0, stop=6.0/33.0, fmod=33.0) # checked
#    mod = ModelSines().test_scaling(nfreqs=5, num=int((6.0/33.0-0.0)*5.0e3), start=0.0, stop=6.0/33.0, fmod=33.0) # checked
#
#    mod = ModelSines().test_numerics(nfreqs=5, num=int((6.0/33.0-0.0)*5.0e3), start=0.0, stop=6.0/33.0,
#                                     fmod=5.0, shape='square', duty=0.40, fmod=33.0) # checked
#    mod = ModelSines().test_scaling(nfreqs=5, num=int((6.0/33.0-0.0)*5.0e3), start=0.0, stop=6.0/33.0,
#                                     fmod=5.0, shape='square', duty=0.40, fmod=33.0) # checked
#    mod = ModelFourier().test_numerics(num=int((6.0/33.0-0.0)*5.0e2), start=0.0, stop=6.0/33.0) # checked
#    mod = ModelFourier.test_numerics(nfreqs=5, num=20*int((6.0/33.0-0.0)*5.0e2), start=0.0, stop=6.0/33.0,
#                                     shape='square', duty=0.40, fmod=33.0)  # checked

#    mod = ModelPoly.test_numerics(npoly=1) # checked
#    mod = ModelPoly.test_numerics(npoly=2) # checked
#    mod = ModelPoly.test_numerics(npoly=5)  # checked
#    mod = ModelPoly.test_numerics(npoly=12)  # checked
#    mod = ModelPoly.test_scaling(npoly=1)  # checked
#    mod = ModelPoly.test_scaling(npoly=2)  # checked
#    mod = ModelPoly.test_scaling(npoly=5)  # checked
#    mod = ModelPoly.test_scaling(npoly=12)  # checked

#    mod = ModelProdExp.test_numerics(npoly=1) # checked
#    mod = ModelProdExp.test_numerics(npoly=2) # checked
#    mod = ModelProdExp.test_numerics(npoly=5) # checked
#    mod = ModelProdExp.test_numerics(npoly=12) # checked
#    mod = ModelProdExp.test_scaling(npoly=1) # checked
#    mod = ModelProdExp.test_scaling(npoly=2) # checked
#    mod = ModelProdExp.test_scaling(npoly=5) # checked
#    mod = ModelProdExp.test_scaling(npoly=12) # checked

#    mod = ModelEvenPoly.test_numerics(npoly=1) # checked
#    mod = ModelEvenPoly.test_numerics(npoly=2) # checked
#    mod = ModelEvenPoly.test_numerics(npoly=3) # checked
#    mod = ModelEvenPoly.test_numerics(npoly=4) # checked
#    mod = ModelEvenPoly.test_numerics(npoly=8) # checked
#    mod = ModelEvenPoly.test_scaling(npoly=1) # checked
#    mod = ModelEvenPoly.test_scaling(npoly=2) # checked
#    mod = ModelEvenPoly.test_scaling(npoly=3) # checked
#    mod = ModelEvenPoly.test_scaling(npoly=4) # checked
#    mod = ModelEvenPoly.test_scaling(npoly=8) # checked

#    mod = ModelParabolic.test_numerics() # checked
#    mod = ModelParabolic.test_scaling() # checked
#    mod = ModelExpEdge.test_numerics()  # checked
#    mod = ModelExpEdge.test_scaling()  # checked
#    mod = ModelTwoPower.test_numerics() # checked
#    mod = ModelTwoPower.test_scaling() # checked
#    mod = ModelTwoPower.test_numerics(num=100) # checked

#    mod = ModelGaussian.test_numerics() # checked
#    mod = ModelOffsetGaussian.test_numerics()  # checked
#    mod = ModelNormal.test_numerics()  # checked
#    mod = ModelOffsetNormal.test_numerics() # checked
#    mod = ModelLogGaussian.test_numerics()
#    mod = ModelLorentzian.test_numerics(num=301)  # checked
#    mod = ModelPseudoVoigt.test_numerics(num=301)  # checked
#    mod = ModelLogLorentzian.test_numerics(num=301)  # checked
#    mod = ModelDoppler.test_numerics(num=301)      # checked
#    mod = ModelLogGaussian.test_numerics(num=301)  # checked
#    mod = ModelLogDoppler.test_numerics(num=301)

#    mod = ModelGaussian.test_scaling(start=-1.0, stop=1.0, num=301) # checked
#    mod = ModelOffsetGaussian.test_scaling(start=-1.0, stop=1.0, num=301)  # checked
#    mod = ModelNormal.test_scaling(start=-1.0, stop=1.0, num=301)  # checked
#    mod = ModelOffsetNormal.test_scaling(start=-1.0, stop=1.0, num=301) # checked
#    mod = ModelLorentzian.test_scaling(start=-1.0, stop=1.0, num=301)  # checked
#    mod = ModelPseudoVoigt.test_scaling(start=-1.0, stop=1.0, num=301)  # checked
#    mod = ModelDoppler.test_scaling(start=-1.0, stop=1.0, num=301)      # checked
#    # mod = ModelLogGaussian.test_scaling(start=-1.0, stop=1.0, num=301)
#    # mod = ModelLogLorentzian.test_scaling(start=-1.0, stop=1.0, num=301)  # checked
#    # mod = ModelLogDoppler.test_scaling(start=-1.0, stop=1.0, num=301)

#    # ----- Checked within bounds:
#    mod = _ModelTwoPower.test_numerics(start=0.1, stop=0.9)   # checked
#    mod = _ModelTwoPower.test_numerics(start=0.1, stop=1.3)   # checked
#    mod = _ModelTwoPower.test_scaling() # checked
#    mod = ModelQuasiParabolic.test_numerics(start=0.1, stop=0.9)  # checked
#    mod = ModelQuasiParabolic.test_scaling(start=0.1, stop=0.9)  # checked
#    mod = ModelQuasiParabolic.test_scaling()  # checked
#    mod = ModelPowerLaw.test_numerics(npoly=2, start=0.1, stop=0.9, num=100)
#    mod = ModelPowerLaw.test_numerics(npoly=3, start=0.1, stop=0.9, num=100)
#    mod = ModelPowerLaw.test_numerics(npoly=4, start=0.1, stop=0.9) # checked
#    mod = ModelPowerLaw.test_numerics(npoly=8, start=0.1, stop=0.9) # checked
#    mod = ModelExponential.test_numerics(start=0.1, stop=0.9) # checked
#    mod = ModelExponential.test_numerics(start=0.1, stop=1.0) # checked
#    mod = ModelFlattop.test_numerics(start=0.1, stop=1.0) # checked
#    mod = ModelSlopetop.test_numerics(start=0.1, stop=1.0) # checked
#    mod = ModelFlattop.test_scaling() # checked
#    mod = ModelSlopetop.test_scaling() # checked

#    mod = ModelExp.test_numerics(start=0.1, stop=0.9) # checked
#    mod = ModelExp.test_numerics(start=0.1, stop=1.0) # checked
#    mod = ModelExp.test_scaling(start=0.0, stop=1.0) # checked
#    mod = ModelExp.test_scaling(start=0.0, stop=0.9) # checked
#    mod = ModelExp.test_scaling(start=0.1, stop=1.0) # checked

#    Fs = 1.0
#    XX = _np.linspace(-0.5, 0.5, num=int(1.5*10e-3*10.0e6/8))
#    model_order = 2
#    chi_eff, gvec, info = model_doppler(XX=XX, af=None, Fs=Fs, noshift=True, model_order=model_order)

#    XX = _np.linspace(0, 1, 200)
#    npoly = 10
#    model_number = 7
##    af = [0.1,0.1,0.1]
#    af = None
##    af = [2.0,0.1,0.1,0.5]
#
#    [chi_eff, gvec, info] = \
#        model_chieff(af=af, XX=XX, model_number=model_number, npoly=npoly, nargout=3, verbose=True)

# #    [chi_eff, gvec, info] = \
# #        model_profile(af=af, XX=XX, model_number=model_number, npoly=npoly, nargout=3, verbose=True)
#    info.dchidx = info.dprofdx
#
#    varaf = (0.1*info.af)**2
#    varchi = _np.zeros_like(chi_eff)
#    for ii in range(_np.size(gvec, axis=0)):
#        varchi = varchi + varaf[ii]*(gvec[ii, :]**2)
#    # endfor
#
#    _plt.figure()
#    _plt.plot(XX, chi_eff, 'k-')
#    _plt.plot(XX, chi_eff+_np.sqrt(varchi), 'k--')
#    _plt.plot(XX, chi_eff-_np.sqrt(varchi), 'k--')
#
#    _plt.figure()
#    _plt.plot(XX, info.dchidx, '-')
#
#    _plt.figure()
#    for ii in range(_np.size(gvec, axis=0)):
#        _plt.plot(XX, gvec[ii, :], '-')
#    # endfor
##    _plt.plot(XX, gvec[1, :], '-')
##    _plt.plot(XX, gvec[2, :], '-')

# endif


# =========================================================================== #
# =========================================================================== #

#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        a0, a1, a2 = parse_noshift(ain)
#        ain = _np.copy(ain)
#        aout = _np.copy(ain)
# #        aout = i0.unscaleaf(a0, slope, offset, xslope, xoffset)
# #        aout = _np.append(aout, i1.unscaleaf(a1, slope, offset, xslope, xoffset), axis=0)
# #        aout = _np.append(aout, i2.unscaleaf(a2, slope, offset, xslope, xoffset), axis=0)
#        return aout
#    info.unscaleaf = unscaleaf
#

# ========================================================================== #
# ========================================================================== #

