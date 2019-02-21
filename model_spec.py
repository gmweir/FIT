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
from pybaseutils.utils import sech, interp_irregularities, interp, argsort # analysis:ignore


# ========================================================================== #
# ========================================================================== #

class ModelClass(Struct):
    def __init__(self, XX, af=None, **kwargs):
        LB, UB, fixed = self.defaults(**kwargs)
        if af is None:   af = _np.copy(self._af)    # end if
        self.af = _np.copy(af)
        self.Lbounds = _np.copy(LB)
        self.Ubounds = _np.copy(UB)
        self.fixed = _np.copy(fixed)
        self.__dict__.update(kwargs)
        self.kwargs = kwargs
    # end def __init__

    def parse_in(self, XX, aa):
        if aa is None: aa = _np.copy(self.af)  # end if
        if XX is None: XX = _np.copy(self.XX)  # end if
        XX = _np.copy(XX)
        aa = _np.copy(aa)
        return XX, aa

    def model(self, XX, aa=None):
        """
        Returns the derivative of the model.
            f(x) = f(x, a, b, c, ...)
            out: f(x)
        """
        XX, aa = self.parse_in(XX, aa)
        return self._model(XX, aa)

    def derivative(self, XX, aa=None):
        """
        Returns the derivative of the model.
            f(x) = f(x, a, b, c, ...)
            out: dfdx
        """
        XX, aa = self.parse_in(XX, aa)
        return self._deriv(XX, aa)

    def jacobian(self, XX, aa=None):
        """
        Returns the jacobian of the model.
            f(x) = f(x, a, b, c)
            out: [dfda(x), dfdb(x), dfdc(x), ...]
        """
        XX, aa = self.parse_in(XX, aa)
        return self._partial(XX, aa)

    def derivative_jacobian(self, XX, aa=None):
        """
        Returns the jacobian of the derivative of the model.
            f(x) = f(x, a, b, c)
            out: [d2fdxda(x), d2fdxdb(x), d2fdxdc(x), ...]
        """
        XX, aa = self.parse_in(XX, aa)
        return self._partial_deriv(XX, aa)

    def hessian(self, XX, aa=None):
        if hasattr(self, '_hessian'):
            return self._hessian(XX, aa)
        else:
            return 0
    # end def

    # ====================================================== #
    # ====================================================== #

    def update(self, XX, af):
        self.prof = self.model(XX, aa=af)
        self.gvec = self.jacobian(XX, aa=af)
        self.dprofdx = self.derivative(XX, aa=af)
        self.dgdx = self.derivative_jacobian(XX, aa=af)
        return self.prof, self.gvec, self.dprofdx, self.dgdx
    # end def

    def defaults(self, **kwargs):
        if 'Lbounds' in kwargs:
            LB = kwargs.setdefault('Lbounds', self._LB)
            UB = kwargs.setdefault('Ubounds', self._UB)
        else:
            LB = kwargs.setdefault('LB', self._LB)
            UB = kwargs.setdefault('UB', self._UB)
        fixed = kwargs.setdefault('fixed', self._fixed)
        return LB, UB, fixed

    # ====================================================== #
    # ====================================================== #

    def checkbounds(self, dat, ain, mag=None):
        LB = _np.copy(self.Lbounds)
        UB = _np.copy(self.Ubounds)
        ain = _checkbounds(ain, LB=LB, UB=UB)
        return dat, ain

    # ============================= #

    def scale(self, pdat, vdat, **kwargs):
        """
        When fitting problems it is convenient to scale it to order 1:
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
        covariance / gvec / dgdx before rescaling because that is arbitrarily complex

        ... but we can no longer reproduce the original data with our fitted model!
            prof !=line(x, a) if prof was scaled pre-fitting

        Here we check if we can rescale the analytic function to reproduce the data
        """
        xdat = kwargs.setdefault('xdat', None)
        if hasattr(self, 'scaled') and (self.scaled is True):
            # Unscale the data if it has already been scaled
            pdat = pdat*self._slope+self._offset
            vdat = vdat * (self._slope**2.0)

            if hasattr(self,'dprofdx'):
                self.dprofdx = self._slope*self.dprofdx
                self.vardprofdx = (self._slope**2.0)*self.vardprofdx
                if hasattr(self, '_xslope'):
                    self.dprofdx /= self._xslope
                    self.vardprofdx /= self._xslope*self._xslope
                # end if
            # end if

            if hasattr(info,'prof'):
                self.prof = self._slope*self.prof+self._offset
                self.varprof = (self._slope**2.0)*self.varprof
            # end if

            if xdat is not None:
                xdat = xdat*self._xslope+self._xoffset
            # end if

            # =============================== #

            # Scaling model parameters to reproduce original data
            self.af = self.unscaleaf(self.af, self._slope, self._offset, self._xslope, self._xoffset)
            self.scaled = False
        else:
            self._slope = _np.nanmax(pdat)-_np.nanmin(pdat)
            self._offset = _np.nanmin(pdat)
            self.scaled = True

            pdat = (pdat.copy()-self._offset)/self._slope
            vdat = vdat.copy()/_np.abs(self._slope)**2.0

            if xdat is not None:
                self._xslope = _np.nanmax(xdat)-_np.nanmin(xdat)
                self._xoffset = _np.nanmin(xdat)
                xdat = (xdat.copy()-self._xoffset)/self._xslope
            # end if
        # end if
        if xdat is not None:
            return pdat, vdat, xdat
        else:
            return pdat, vdat
    # end def scale

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
        aout = _np.copy(ain)
        print('Unscaling model parameters not supported: \n'+
              '   Either not implemented in model or \n'+
              '   there is a nonlinearity in the analytic model that precludes analytic scaling!')
        return aout
# end def class


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
#    def _model(XX, aa):
#
#    @staticmethod
#    def _deriv(XX, aa):
#
#    @staticmethod
#    def _partial(XX, aa):
#
#    @staticmethod
#    def _partial_deriv(XX, aa):
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
##        aout = _np.copy(ain)
##        return aout
#
#    # ====================================== #
#
##    def checkbounds(self, dat):
##        return super(ModelExample, self).checkbounds(dat, self.aa, mag=None)
#
#    # ====================================== #
## end def ModelExample


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

def model_line(XX, af=None, **kwargs):
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
    def __init__(self, XX, af=None, **kwargs):
        super(ModelLine, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
        model of a line
            y = a*x + b
        """
        y = aa[0]*XX+aa[1]
        return y

    @staticmethod
    def _deriv(XX, aa):
        """
        derivative of a line
            y = a*x + b
            dydx = a
        """
        dydx = aa[0]*_np.ones_like(XX)
        return dydx

    @staticmethod
    def _partial(XX, aa):
        """
        Jacobian of a line
            y = a*x + b
            dyda = x
            dydb = 1
        """
        gvec = _np.zeros( (2,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = XX.copy() # aa[0]
        gvec[1,:] = 1.0       # aa[1]
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        """
        Jacobian of the derivative of a line
            y = a*x + b
            dydx = a
            d2ydxda = 1.0
            d2ydxdb = 0.0
        """
        dgdx = _np.zeros( (2,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = _np.ones_like(XX)  # aa[0]
        dgdx[1,:] = _np.zeros_like(XX) # aa[1]
        return dgdx

    @staticmethod
    def _hessian(XX, aa):
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

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
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
        """
        aout = _np.copy(ain)
        aout[1] = slope*(aout[1]-xoffset*aout[0]/xslope) + offset
        aout[0] = slope*aout[0]/xslope
        return aout

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelLine, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelLine

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

def model_poly(XX, af=None, **kwargs):
    return _model(ModelPoly, XX, af, **kwargs)

# =========================================== #


class ModelPoly(ModelClass):
    """
    --- Straight Polynomial ---
    Model - chi ~ sum( af(ii)*XX^(polyorder-ii))
    af    - estimate of fitting parameters
    XX    - independent variable
    """
    def __init__(self, XX, af=None, **kwargs):
        if af is not None:
            num_fit = _np.size(af)  # Number of fitting parameters
            npoly = _np.int(num_fit-1)  # Polynomial order from input af
        else:
            npoly = kwargs.setdefault('npoly', 4)
        self._af = 0.1*_np.ones((npoly+1,), dtype=_np.float64)
        self._LB = -_np.inf*_np.ones((npoly+1,), dtype=_np.float64)
        self._UB = _np.inf*_np.ones((npoly+1,), dtype=_np.float64)
        self._fixed = _np.zeros( _np.shape(self._LB), dtype=int)
        super(ModelPoly, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        pp = _np.poly1d(aa)
        return pp(XX)

    @staticmethod
    def _deriv(XX, aa):
        pp = _np.poly1d(aa).deriv()
        return pp(XX)

    @staticmethod
    def _partial(XX, aa):
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
            kk = num_fit - (ii + 1)
            gvec[ii, :] = XX**kk
        # endfor
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        """
         The jacobian for the derivative
         f = a1*x^2+a2*x+a3
         dfdx = 2*a1*x+a2
         dfda1 = 2*x;
         dfda2 = 1;
         dfda3 = 0;
         dgdx(1,1:nx) = 2*XX;
         dgdx(2,1:nx) = 1.0;
         dgdx(3,1:nx) = 0.0;
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)

        dgdx = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(num_fit-1):
            kk = (num_fit-1) - (ii + 1)
            dgdx[ii,:] = (kk+1)*(XX**kk)
        # end for
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
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
                .... I won't show the math, look up Shaw and Traub method of "Taylor Shift"
            if q(x) = p(x+xo)
                g(x) = p(xo*x)
                f(x) = g(x+1)
                q(x) = f(x/xo)
        """
        aout = _np.copy(ain)
        # First the y-scaling:
        aout = slope*aout
        # next the x-stretching:
        for ii in range(len(aout)):
            aout[ii] = aout[ii]/_np.power(xslope, ii)
        # end for
#        # 3rd the complicated x-shifting
#        aoutf = _np.copy(aout)
#        for jj in range(len(ain),0,-1):
#            for kk in range(jj):
#            aoutf[ii]
#        # end for
#        aout = aoutf
        # finally the y-shifting
        aout[-1] += offset
        print('X-offsets do not work with the polynomial model yet')
        return aout

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

def model_ProdExp(XX, af=None, **kwargs):
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
    def __init__(self, XX, af=None, **kwargs):
        if af is not None:
            num_fit = _np.size(af)  # Number of fitting parameters
            npoly = _np.int(2*(num_fit-1))  # Polynomial order from input af
        else:
            npoly = kwargs.setdefault('npoly', 4)
        self._af = 0.9*_np.ones((npoly+1,), dtype=_np.float64)
        self._LB = -_np.inf*_np.ones_like(self._af)
        self._UB = _np.inf*_np.ones_like(self._af)
        self._fixed = _np.zeros(_np.shape(self._LB), dtype=int)
        super(ModelProdExp, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
        Product of exponentials
         y = exp( a0 + a1*x + a2*x^2 + ...)
        """
        # Polynomial of order len(af)
        return _np.exp( poly(XX, aa) )

    @staticmethod
    def _deriv(XX, aa):
        """
        derivative of product of exponentials w.r.t. x is analytic as well:
         y = exp( a0 + a1*x + a2*x^2 + ...)

         f = exp(a1x^n+a2x^(n-1)+...a(n+1))
         f = exp(a1x^n)exp(a2x^(n-1))...exp(a(n+1)));
         dfdx = (n*a1*x^(n-1)+(n-1)*a2*x^(n-2)+...a(n))*f
        """
        return deriv_poly(XX, aa)*prodexp(XX, aa)

    @staticmethod
    def _partial(XX, aa):
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
        num_fit = _np.size(aa)  # Number of fitting parameters
        prof = prodexp(XX, aa)

        gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(num_fit):  # 1:num_fit
            # Formulated this way, there is an analytic jacobian:
            kk = num_fit - (ii + 1)
            gvec[ii, :] = (XX**kk)*prof
        # endif
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        """
         The g-vector (jacobian) for the derivative
         dfdx = (...+2*a1*x + a2)*exp(...+a1*x^2+a2*x+a3)
        """
        num_fit = _np.size(aa)  # Number of fitting parameters

        prof = prodexp(XX, aa)
        dprofdx = deriv_prodexp(XX, aa)
        gvec = partial_prodexp(XX, aa)

        # Product rule:  partial derivatives of the exponential term times the leading derivative polynomial
        dgdx = gvec.copy()*(_np.ones((num_fit,1), dtype=float)*_np.atleast_2d(dprofdx))

        # Product rule:  exponential term times the partial derivatives of the derivative polynomial
        for ii in range(num_fit-1):  # 1:num_fit
            # Formulated this way, there is an analytic jacobian:
            kk = num_fit-1 - (ii + 1)
            dgdx[ii, :] += (kk+1)*(XX**kk)*prof
        # endif
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
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
                    not trivial with constant coefficients

        x-scaling: x' = x/xs
            y'=  y/ys = exp( sum( a_i'*(x/xs)^i ) ) = prod( exp( a_i'*(x/xs)^i ) )
                a_o = ln(ys) + a_o'
                a_i = a_i' / xs^i

        x-shifting: x' = (x-xo)/xs
            possible but complicated ... requires binomial / multinomial theorem
        """
        aout = _np.copy(ain)
        aout[0] = _np.log(slope) + aout[0]
        for ii in range(len(aout), 1, -1):
            aout[ii] /= _np.power(xslope, ii)
        return aout

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

def model_evenpoly(XX, af=None, **kwargs):
    return _model(ModelEvenPoly, XX, af, **kwargs)

# =========================================== #


class ModelEvenPoly(ModelClass):
    """
    --- Polynomial with only even powers ---
    Model - y ~ sum( af(ii)*XX^2*(numfit-ii))
    af    - estimate of fitting parameters (npoly=4, numfit=3, poly= a0*x^4+a1*x^2+a3)
    XX    - independent variable
    """
    def __init__(self, XX, af=None, **kwargs):
        if af is not None:
            num_fit = _np.size(af)  # Number of fitting parameters
            npoly = _np.int(2*(num_fit-1))  # Polynomial order from input af
        else:
            npoly = kwargs.setdefault('npoly', 4)
        self._af = 0.1*_np.ones((npoly//2+1,), dtype=_np.float64)
        self._LB = -_np.inf*_np.ones((npoly//2+1,), dtype=_np.float64)
        self._UB = _np.inf*_np.ones((npoly//2+1,), dtype=_np.float64)
        self._fixed = _np.zeros( _np.shape(self._LB), dtype=int)
        super(ModelEvenPoly, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
        Even Polynomial of order num_fit, Insert zeros for the odd powers
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        a0 = _np.insert(aa, _np.linspace(1, num_fit-1, 2), 0.0)
        return poly(XX, a0)

    @staticmethod
    def _deriv(XX, aa):
        """
        Derivative of an even polynomial
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        a0 = _np.insert(aa, _np.linspace(1, num_fit-1, 2), 0.0)
        return deriv_poly(XX, a0)

    @staticmethod
    def _partial(XX, aa):
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
        for ii in range(num_fit):  # ii=1:num_fit
            #2*(num_fit-1)
            kk = num_fit - (ii + 1)
            kk *= 2
            gvec[ii, :] = XX**kk
        # endfor
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
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
        for ii in range(num_fit-1):
            kk = num_fit - (ii + 1)    # ii=0, kk = num_fit-1;   ii=num_fit-2, kk=+1
            kk *= 2                    #       kk = 2*num_fit-2;               kk=+2
            dgdx[ii,:] = kk * XX**(kk-1)
        # end for
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
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
                .... I won't show the math, look up Shaw and Traub method of "Taylor Shift"
            if q(x) = p(x+xo)
                g(x) = p(xo*x)
                f(x) = g(x+1)
                q(x) = f(x/xo)
        """
        aout = _np.copy(ain)
        # First the y-scaling and shifting:
        aout = slope*aout
        aout[-1] += offset
        print('X-shifting/offsets do not work with the even polynomial model yet')
        return aout

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

def model_PowerLaw(XX, af=None, **kwargs):
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
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
         Curved power-law:
         fc = x^(a1*x^(n+1)+a2*x^n+...a(n+1))

         With exponential cut-off:
         f  = a(n+2)*exp(a(n+1)*XX)*fc(x);
        """
    #    num_fit = _np.size(af)  # Number of fitting parameters
    #    npoly = (num_fit-1)-2

        polys = poly(XX, aa[:-2])
        exp_factor = _np.exp(aa[-2]*XX)
        return aa[-1]*exp_factor*(XX**polys)

    @staticmethod
    def _deriv(XX, aa):
        """
         dfdx = dfcdx*(a(n+2)*e^a(n+1)x) +a(n+1)*f(x);
              = (dfcdx/fc)*f(x) + a(n+1)*f(x)
              =  (dln(fc)dx + a(n+1)) * f(x)

            fc = x^(a1*x^(n+1)+a2*x^n+...a(n+1))
                 dfcdx/fc = d ln(fc)/dx = d/dx poly*ln(x)
                 d ln(fc)/dx = dpolydx*ln(x)+poly/x
                 (dfcdx = fc*( dpolydx*x+poly ))

         dfdx = = (  dpolydx*ln(x)+poly/x + a(n+1)) * f(x)

         dfcdx = XX^(-1)*prof*(polys+ddx(poly)*XX*log(XX),
         log is natural logarithm
        """
    #    num_fit = _np.size(af)  # Number of fitting parameters
    #    npoly = (num_fit-1)-2

        prof = powerlaw(XX, aa)
        polys = poly(XX, aa[:-2])
        dpolys = deriv_poly(XX, aa[:-2])
        return prof*( aa[-1] + polys/XX + _np.log(_np.abs(XX))*dpolys )

    @staticmethod
    def _partial(XX, aa):
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
        nx = _np.size(XX)
        num_fit = _np.size(aa)  # Number of fitting parameters
        npoly = len(aa[:-2])    #    npoly = num_fit-3?

        prof = powerlaw(XX, aa)
        gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
        ii = 0
        for kk in range(npoly,0,-1):
            gvec[ii, :] = _np.power(XX, kk) * _np.log(_np.abs(XX)) * prof
            ii += 1
        # endfor
        gvec[-2, :] = prof*XX
        gvec[-1, :] = prof/aa[-1]
        return gvec
    #    prof = powerlaw(XX, af)
    #    gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
    #    for ii in range(npoly+1):  # ii=1:num_fit
    #        kk = npoly+1 - (ii + 1)
    #        gvec[ii, :] = prof*_np.log(_np.abs(XX))*XX**kk
    #    # endfor
    #    gvec[num_fit-1, :] = prof/af[num_fit-1]
    #    gvec[num_fit, :] = prof*XX
    #    return gvec


    @staticmethod
    def _partial_deriv(XX, aa):
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
        nx = _np.size(XX)
        num_fit = _np.size(aa)  # Number of fitting parameters
        npoly = len(aa[:-2])    #    npoly = num_fit-3?

        prof = powerlaw(XX, aa)
        dprofdx = deriv_powerlaw(XX, aa)

        dgdx = _np.zeros((num_fit, nx), dtype=_np.float64)
        ii = 0
        for kk in range(npoly,0,-1):
    #        dgdx[ii, :] = (XX**(npoly-ii)*_np.log(_np.abs(XX))*dprofdx
    #                          + (1.0+(npoly-ii)*_np.log(_np.abs(XX)))*XX**(npoly-ii-1)*prof )
            dgdx[ii, :] = (_np.power(XX, kk)*_np.log(_np.abs(XX))*dprofdx
                              + (1.0+kk*_np.log(_np.abs(XX)))*_np.power(XX, kk-1.0)*prof )
            ii += 1
        # endfor
        dgdx[-2, :] = XX*dprofdx
        dgdx[-1, :] = dprofdx/aa[-1]
        return dgdx
    #    for ii in range(npoly+1):  # ii=1:(npoly-1)
    #        kk = npoly+1 - (ii + 1)
    #        dgdx[ii, :] = info.dprofdx*_np.log(_np.abs(XX))*(XX**kk)
    ##        dgdx[ii, :] += prof*af[num_fit]*_np.log(_np.abs(XX))*(XX**kk)
    #        dgdx[ii, :] += prof*af[num_fit-1]*_np.log(_np.abs(XX))*(XX**kk)
    #
    #        if ii<npoly:
    #            dgdx[ii, :] += prof*(XX**(kk-1))*(1.0 + kk*_np.log(_np.abs(XX)))     # 3 = dcoeffs / af[:npoly+1]
    #        else:
    #            dgdx[ii, :] += prof*(XX**(kk-1))
    #        # endif
    #    # endfor
    #    dgdx[num_fit-2, :] = (info.dprofdx/(af[num_fit-2]) + af[num_fit-1])*prof/af[num_fit-1]
    #    dgdx[num_fit-1, :] = prof*( af[num_fit-1]*XX + 1.0 + XX*info.dprofdx )
    #    dgdx[num_fit-1, :] = (info.dprofdx/(af[num_fit-1]) + af[num_fit])*prof/af[num_fit-1]
    #    dgdx[num_fit  , :] = prof*( af[num_fit]*XX + 1.0 + XX*info.dprofdx )
    #    return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #
#
#    @staticmethod
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        """
#        """
#        aout = _np.copy(ain)
#        return aout

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

def model_parabolic(XX, af, **kwargs):
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
    _LB = _np.asarray([0.0], dtype=_np.float64)
    _UB = _np.asarray([_np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (1,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelParabolic, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        return aa*(1.0 - _np.power(XX, 2.0))

    @staticmethod
    def _deriv(XX, aa):
        return -2.0*aa*XX

    @staticmethod
    def _partial(XX, aa):
        return _np.atleast_2d(parabolic(XX, aa) / aa)

    @staticmethod
    def _partial_deriv(XX, aa):
        return -2.0*XX

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

#    @staticmethod
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        """
#        y-scaling: y'=y/ys
#            a = ys*a'
#        y-shifting: y'=(y-yo)/ys
#            y = yo + ys*a'*(1.0-x^2) = yo + ys*a' - ys*a'x^2
#                hard to do with constant coefficients
#        x-scaling: x'=x/xs
#            y = ys*a'*(1.0-(x/xs)^2)
#              = ys*a'/xs^2 * (xs^2 - x^2) + ys*a'/xs^2 - ys*a'/xs^2
#              = ys*a'/xs^2 * (1.0 - x^2) + ys*a'*(1.0 - 1.0/xs^2)
#                hard to do with constant coefficients
#        x-shifting: x'=(x-xo)/xs
#            y = ys*a'*(1.0-(x-xo)^2) = ys*a'*(1.0-x^2-2xo*x+xo^2)
#              = ys*a'*(1.0-x^2)-ys*a'*xo*(2*x+xo)
#                hard to do with constant coefficients
#        """
#        aout = _np.copy(ain)
#        return aout

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelParabolic, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelParabolic


# ========================================================================== #
# ========================================================================== #


def expcutoff(XX, aa):
    return ModelExponential._model(XX, aa)

def deriv_expcutoff(XX, aa):
    return ModelExponential._deriv(XX, aa)

def partial_expcutoff(XX, aa):
    return ModelExponential._partial(XX, aa)

def partial_deriv_expcutoff(XX, aa):
    return ModelExponential._partial_deriv(XX, aa)

def model_Exponential(XX, af=None, **kwargs):
    return _model(ModelExponential, XX, af, **kwargs)

# =========================================== #


class ModelExponential(ModelClass):
    """
    --- Exponential on Background ---
    Model - y = a1*(exp(a2*XX^a3) + XX^a4)
    af    - estimate of fitting parameters
    XX    - independent variables
    """
    _af = 0.1*_np.ones((4,), dtype=_np.float64)
    _LB = _np.array([0.0, -_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (4,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelExponential, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, separate=False):
        """
         f     = a1*(exp(a2*XX^a3) + XX^a4) = f1+f2;
        """
        prof1, prof2 = ModelExponential._separate_model(XX, aa)
        return prof1 + prof2

    @staticmethod
    def _separate_model(XX, aa):
        prof1 = aa[0]*_np.exp(aa[1]* _np.power(XX, aa[2]))
        prof2 = aa[0]*_np.power(XX, aa[3])
        return prof1, prof2

    @staticmethod
    def _deriv(XX, aa):
        """
         dfdx  = a1*(a2*a3*XX^(a3-1)*exp(a2*XX^a3) + a4*XX^(a4-1));
        """
        dprof1dx, dprof2dx = ModelExponential._separate_deriv(XX, aa)
        return dprof1dx+dprof2dx

    @staticmethod
    def _separate_deriv(XX, aa):
        prof1, prof2 = ModelExponential._separate_model(XX, aa)
        dprof1dx = aa[1]*aa[2]*XX**(aa[2]-1.0)*prof1
        dprof2dx = aa[0]*aa[3]*(XX**(aa[3]-1))
        return dprof1dx, dprof2dx

    @staticmethod
    def _partial(XX, aa):
        """
         dfda1 = f/a1;
         dfda2 = XX^a3*f1;
         dfda3 = f1*XX^a3*log10(XX)
         dfda4 = a1*XX^a4*log10(XX) = log10(XX)*f2;
        """
        nx = _np.size(XX)
        num_fit = _np.size(aa)
        gvec = _np.zeros( (num_fit, nx), dtype=float)

        prof1, prof2 = ModelExponential._separate_model(XX, aa)
        prof = prof1 + prof2
        gvec[0, :] = prof/aa[0]
        gvec[1, :] = prof1*_np.power(XX, aa[2])
        gvec[2, :] = prof1*aa[1]*_np.log(_np.abs(XX))*_np.power(XX, aa[2])
        gvec[3, :] = aa[0]*_np.log(_np.abs(XX))*_np.power(XX, aa[3])
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        nx = _np.size(XX)
        num_fit = _np.size(aa)
        dgdx = _np.zeros( (num_fit,nx), dtype=float)

        dprof1dx, dprof2dx = ModelExponential._separate_deriv(XX, aa)
        dprofdx = dprof1dx + dprof2dx

        dgdx[0, :] = dprofdx / aa[0]
        dgdx[1, :] = dprof1dx*_np.power(XX, aa[2]) + dprof1dx/aa[1]
        dgdx[2, :] = dprof1dx*(aa[1]*_np.log(_np.abs(XX))*_np.power(XX, aa[2]) + _np.log(_np.abs(XX)) + 1.0/aa[2] )
        dgdx[3, :] = dprof2dx*_np.log(_np.abs(XX)) + aa[0]*_np.power(XX, aa[3]-1.0)
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

#    @staticmethod
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        """
#        """
#        aout = _np.copy(ain)
#        return aout

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

def _model_gaussian(XX, af=None, **kwargs):
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

        note that if normalized, then the PDF form is used so that the integral
        is equal to af[0]:
            af[0]*_np.exp(-(XX-af[1])**2/(2.0*af[2]**2))/_np.sqrt(2.0*pi*af[2]**2.0)
    """
    _af = _np.asarray([1.0e-1/0.05, -0.3, 0.1], dtype=_np.float64)
    _LB = _np.array([-_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.array([0, 0, 0], dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelGaussian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
        model of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
            A = af[0]
            xo = af[1]
            ss = af[2]
        """
        AA, x0, ss = tuple(aa)
        return AA*_np.exp(-(XX-x0)**2/(2.0*ss**2))

    @staticmethod
    def _deriv(XX, aa):
        """
        derivative of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2 *f
        """
        AA, x0, ss = tuple(aa)
        prof = gaussian(XX, aa)
        return -1.0*(XX-x0)/_np.power(ss, 2.0) * prof

    @staticmethod
    def _partial(XX, aa):
        """
        Jacobian of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdA = f/A
        dfdxo = A*(x-xo)*exp(-(x-xo)**2.0/(2.0*ss**2.0))/ss^2 = (x-xo)/ss^2 * f
        dfdss =  (x-xo)^2/ss^3 * f
        """
        AA, x0, ss = tuple(aa)
        prof = gaussian(XX, aa)
        gvec = _np.zeros( (3,len(XX)), dtype=_np.float64)
        gvec[0,:] = prof/AA
        gvec[1,:] = ((XX-x0)/(ss**2.0))*prof
        gvec[2,:] = (((XX-x0)**2.0)/(ss**3.0)) * prof
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        """
        Jacobian of the derivative of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2 *f
        term1 = (-(x-xo)/ss^2)

        d^2f/dxdA = 0*f + df/dA * term1
                  = partial_gaussian[0]*term1
        d^2f/dxdxo = 1.0/(ss^2)*f + df/dxo * term1
                   =  prof/(ss^2) + partial_gaussian[1]*term1
        d^2f/dxdss = 2.0*(x-xo)/(ss^3)*f + df/dxo * term1
                   = -2.0*term1*prof/ss + partial_gaussian[2]*term1
        """
        AA, x0, ss = tuple(aa)

        prof = gaussian(XX, aa)
        term1 = (-1.0*(XX-x0)/_np.power(ss,2.0))
        gvec = partial_gaussian(XX, aa)

        dgdx = _np.zeros( (3,len(XX)), dtype=_np.float64)
        dgdx[0,:] = gvec[0,:]*term1
        dgdx[1,:] = prof/_np.power(ss,2.0) + gvec[1,:]*term1
        dgdx[2,:] = -2.0*term1*prof/ss + gvec[2,:]*term1
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):
#       """
#        Hessian of a gaussian
#        [d2fda2, d2fdadb, d2fdadc; d2fdadb, d2fdb2, d2fdbdc; d2fdadc, d2fdbdc, d2fdc2]
#       d2fda2 = 0.0
#       d2fdb2 = d/dxo (x-xo)/ss^2 * f
#       """
#       hmat = _np.zeros((3, 3, _np.size(XX)), dtype=_np.float64)
#       return hmat
    # ====================================== #

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
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
        aout = _np.copy(ain)
        aout[0] = slope*aout[0]
        aout[1] = xslope*aout[1]+xoffset
        aout[2] = xslope*aout[2]
#        self._secretoffset = offset
        print('Data offsets do not work with the gaussian model:\n'
             +'Remember the offset is now included in the models!')
        return aout

    # ====================================== #

    def checkbounds(self, dat):
        return super(ModelGaussian, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelGaussian


# ========================================================================== #
# ========================================================================== #


def offsetgaussian(XX, aa):
    return ModelGaussian._model(XX, aa[1:]) + aa[0]

def partial_offsetgaussian(XX, aa):
    return _np.concatenate( (_np.ones((1,_np.size(XX)), dtype=_np.float64),
                             ModelGaussian._partial(XX, aa[1:])), axis=0)

def deriv_offsetgaussian(XX, aa):
    return ModelGaussian._deriv(XX, aa[1:])

def partial_deriv_offsetgaussian(XX, aa):
    return _np.concatenate( (_np.zeros((1,_np.size(XX)), dtype=_np.float64),
                             ModelGaussian._partial_deriv(XX, aa[1:])), axis=0)

def _model_offsetgaussian(XX, af=None, **kwargs):
    """
    """
    return _model(ModelOffsetGaussian, XX, af, **kwargs)

# ==================================== #

class ModelOffsetGaussian(ModelClass):
    """
    A gaussian with four free parameters:
        XX - x - independent variable
        af - y-offset, magnitude, shift, width

        af[0] + af[1]*_np.exp(-(XX-af[2])**2/(2.0*af[3]**2))

        note that if normalized, then the PDF form is used so that the integral
        is equal to af[1]:
            af[0] + af[1]*_np.exp(-(XX-af[2])**2/(2.0*af[3]**2))/_np.sqrt(2.0*pi*af[3]**2.0)
    """
    _af = _np.asarray([0, 1.0e-1/0.05, -0.3, 0.1], dtype=_np.float64)
    _LB = _np.array([-_np.inf, -_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.array([0, 0, 0, 0], dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelOffsetGaussian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        return offsetgaussian(XX, aa)

    @staticmethod
    def _deriv(XX, aa):
        return deriv_offsetgaussian(XX, aa)

    @staticmethod
    def _partial(XX, aa):
        return partial_offsetgaussian(XX, aa)

    @staticmethod
    def _partial_deriv(XX, aa):
        return partial_deriv_offsetgaussian(XX, aa)

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

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
        """
        If fitting with scaling, then the algebra necessary to unscale the problem
        to original units is:

            y-scaling:  y' = (y-yo)/ys
             (y-miny)/(maxy-miny) = (y-yo)/ys
             (y-yo)/ys = a0 + a' exp(-1*(x-b')^2/(2*c'^2))

             y-yo = ys*a0 + ys * a' * exp(-1*(x-b')^2/(2*c'^2))
                 assume a = ys*a'

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
        aout = _np.copy(ain)
        aout[0] = slope*aout[0] + offset
        aout[1] = slope*aout[1]
        aout[2] = xslope*aout[2]+xoffset
        aout[3] = xslope*aout[3]
        return aout

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelOffsetGaussian, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelOffsetGaussian

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

def model_normal(XX, af, **kwargs):
    norm = kwargs.pop('norm', True)  # analysis:ignore
    return model_gaussian(XX, af, norm=True, **kwargs)

# ==================================== #

class ModelNormal(ModelClass):
    """
    A gaussian with three free parameters:
        XX - x - independent variable
        af - magnitude, shift, width

        af[0]*_np.exp(-(XX-af[1])**2/(2.0*af[2]**2))

        note that if normalized, then the PDF form is used so that the integral
        is equal to af[0]:
            af[0]*_np.exp(-(XX-af[1])**2/(2.0*af[2]**2))/_np.sqrt(2.0*pi*af[2]**2.0)
    """
    _af = _np.asarray([1.0, -0.3, 0.1], dtype=_np.float64)
    _LB = _np.array([-_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.array([0, 0, 0], dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelNormal, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
        Model of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
            A = af[0]
            xo = af[1]
            ss = af[2]
        """
        AA, x0, ss = tuple(aa)
        nn = _np.sqrt(2.0*_np.pi*_np.power(ss, 2.0))
        return gaussian(XX,aa)/nn

    @staticmethod
    def _deriv(XX, aa):
        """
        derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2 *f
        """
        AA, x0, ss = tuple(aa)
        nn = _np.sqrt(2.0*_np.pi*_np.power(ss, 2.0))
        return deriv_gaussian(XX, aa)/nn

    @staticmethod
    def _partial(XX, aa):
        """
        Jacobian of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)

        dfdA = f/A
        dfdxo = (x-xo)/ss^2 * f
        dfdss =  (x-xo)^2/ss^3 * f + A*exp(-(x-xo)**2.0/(2.0*ss**2.0))*d/ds( sqrt(2*pi*ss^2.0)**-1 )
            d/ds( sqrt(2*pi*ss^2.0)**-1 ) = sqrt(2*pi)^-1 * d/ds( (ss^2.0)**-0.5 ) = sqrt(2*pi)^-1 * d/ds( abs(ss)^-1.0 )
                = sqrt(2*pi)^-1 * (2*ss)*-0.5 * (ss^2.0)^-1.5 = sqrt(2*pi)^-1 *-ss/(ss^3.0) = sqrt(2*pi)^-1 *1.0/ss^2
                = sqrt(2*pi*ss**2.0)^-1 *1.0/abs(ss)
        dfdss =  (x-xo)^2/ss^3 * f + 1.0/abs(ss) * f
        """
        AA, x0, ss = tuple(aa)
        gvec = _np.zeros( (3,len(XX)), dtype=_np.float64)
        prof = normal(XX, aa)
        gvec = _np.zeros( (3,len(XX)), dtype=_np.float64)
        gvec[0,:] = prof/AA
        gvec[1,:] = ((XX-x0)/(ss**2.0))*prof
        gvec[1,:] = (((XX-x0)**2.0)/(ss**3.0)) * prof -prof/_np.abs(ss)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        """
        Jacobian of the derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)

        dfdx = -(x-xo)/ss^2 *f

        d2fdxdA = dfdx/A = -(x-xo)/ss^2 *dfdA
        d2fdxdxo = f/ss^2 -(x-xo)/ss^2 *dfdxo
        d2fdxds = 2*(x-xo)/ss^3 *f -(x-xo)/ss^2 *dfds

        """
        AA, x0, ss = tuple(aa)
        gvec = _np.zeros( (3,len(XX)), dtype=_np.float64)
        prof = normal(XX, aa)
        gvec = partial_normal(XX, aa)
        dgdx = _np.zeros((3,len(XX)), dtype=_np.float64)
        dgdx[0,:] = -(XX-x0)/_np.power(ss, 2.0)*gvec[0,:]
        dgdx[1,:] = prof/_np.power(XX, 2.0) - ((XX-x0)/_np.power(ss, 2.0))*gvec[1,:]
        dgdx[1,:] = 2.0*(XX-x0)*prof/_np.power(XX, 3.0) - ((XX-x0)/_np.power(ss, 2.0))*gvec[2,:]
        return dgdx

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

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
        """
        If fitting with scaling, then the algebra necessary to unscale the problem
        to original units is:

            y-scaling:  y' = (y-yo)/ys
             (y-miny)/(maxy-miny) = (y-yo)/ys
             (y-yo)/ys = a' exp(-1*(x-b')^2/(2*c'^2))/sqrt(2*pi*c'^2.0)

             y-yo = ys * a' * exp(-1*(x-b')^2/(2*c'^2))/sqrt(2*pi*c'^2.0)
                 assume a = ys*a'+yo

             yo + ys*a'*exp(-1*(x-b')^2/(2*c'^2))/sqrt(2*pi*c'^2.0) = a*exp(-1*(x-b)^2/(2*c^2))/sqrt(2*pi*c^2.0)
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
        aout = _np.copy(ain)
        aout[0] = slope*xslope*aout[0]
        aout[1] = xslope*aout[1]+xoffset
        aout[2] = xslope*aout[2]
#        self._secretoffset = offset
        print('Data offsets do not work with the normalized gaussian model:\n'
             +'Remember the offset is now included in the models!')
        return aout

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelNormal, self).checkbounds(dat, self.aa, mag=None)

    # ========================#============= #
# end def ModelNormal

# =========================================================================== #
# =========================================================================== #


def model_gaussian(XX, af=None, **kwargs):
    """
    """
    normalized = kwargs.setdefault('norm', False)
    offset = kwargs.pop('offset', False)
    if normalized:
        return _model(ModelNormal, XX, af, **kwargs)
    else:
        if offset:
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

def model_loggaussian(XX, af=None, **kwargs):
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
    _af = _np.asarray([1.0, -0.3, 0.1], dtype=_np.float64)
    _LB = _np.asarray([0.0, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelLogGaussian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
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
        return (10.0/_np.log(10))*_np.abs(_np.log(AA)-_np.power(XX-x0, 2.0)/(2.0*_np.power(ss, 2.0)))

    @staticmethod
    def _deriv(XX, aa):
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
        return -10.0*(XX-xo)/(_np.power(ss, 2.0) * _np.log(10))

    @staticmethod
    def _partial(XX, aa):
        """
        f = 10*_np.log10( A*exp(-(x-xo)**2.0/(2.0*ss**2.0)) )
          = 10/ln(10)*( ln(A) -(x-xo)**2.0/(2.0*ss**2.0)  )

        dfdA = 10/(A*ln(10))
        dfdxo = 10*(x-xo)/(ss^2 * ln(10))
        dfdss =  10*(x-xo)^2/(ss^3 * ln(10))
        """
        AA, x0, ss = tuple(aa)
        gvec = _np.zeros( (3,len(XX)), dtype=_np.float64)
        gvec[0,:] = 10.0/(AA*_np.log(10.0))
        gvec[1,:] = 10.0*((XX-x0)/(_np.power(ss, 2.0)*_np.log(10.0)))
        gvec[2,:] = 10.0*_np.power(XX-x0, 2.0)/(_np.power(ss, 3.0)*_np.log(10.0))
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        """
        f = 10*_np.log10( A*exp(-(x-xo)**2.0/(2.0*ss**2.0)) )
          = 10/ln(10)*( ln(A) -(x-xo)**2.0/(2.0*ss**2.0)  )

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

        dgdx = _np.zeros( (3,len(XX)), dtype=_np.float64)
        dgdx[1,:] = 10.0/(_np.log(10.0)*_np.power(ss,2.0))
        dgdx[2,:] = 20.0*(XX-x0)/(_np.log(10)*_np.power(ss,3.0))
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
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
        aout = _np.copy(ain)
        aout[0] = _np.exp( _np.log(10.0)*offset/10.0 + slope*_np.log(_np.abs(aout[0])))
        aout[1] = xslope*aout[1]+xoffset
        aout[2] = xslope*aout[2]*_np.sqrt( _np.log(10.0)/(10.0*slope))
        return aout

    # ====================================== #

#    def checkbounds(self, dat):
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

def model_lorentzian(XX, af=None, **kwargs):
    return _model(ModelLorentzian, XX, af, **kwargs)

# =========================================== #


class ModelLorentzian(ModelClass):
    """
    A lorentzian with three free parameters:
        XX - x - independent variable
        af - magnitude, shift, width

        AA*0.5*ss/((XX-x0)^2+0.25*ss^2)/_np.pi
    """
    _af = _np.asarray([1.0, 0.4, 0.05], dtype=_np.float64)
    _LB = _np.asarray([-_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelLorentzian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
        Lorentzian normalization such that integration equals AA (af[0])
            f = 0.5*A*s / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi
        """
        AA, x0, ss = tuple(aa)
        return AA*0.5*ss/((XX-x0)*(XX-x0)+0.25*ss*ss)/_np.pi

    @staticmethod
    def _deriv(XX, aa):
        AA, x0, ss = tuple(aa)
        return -1.0*AA*16.0*(XX-x0)*ss/(_np.pi*(4.0*(XX-x0)*(XX-x0)+ss*ss)**2.0)

    @staticmethod
    def _partial(XX, aa):
        AA, x0, ss = tuple(aa)

        gvec = _np.zeros( (3,len(XX)), dtype=_np.float64)
        gvec[0,:] = 0.5*ss/((XX-x0)*(XX-x0)+0.25*ss*ss)/_np.pi
        gvec[1,:] = AA*ss*(XX-x0)/( ((XX-x0)*(XX-x0)+0.25*ss*ss)**2.0 )/_np.pi
        gvec[2,:] = AA*(-0.125*ss*ss+0.5*XX*XX-XX*x0+0.5*x0*x0)/((0.25*ss*ss+(XX-x0)*(XX-x0))**2.0)/_np.pi
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        AA, x0, ss = tuple(aa)

        gvec = _np.zeros( (3,len(XX)), dtype=_np.float64)
        gvec[0,:] = -1.0*16.0*(XX-x0)*ss/(_np.pi*(4.0*(XX-x0)*(XX-x0)+ss*ss)**2.0)
        gvec[1,:] = 16.0*AA*ss*(-12.0*x0*x0+24.0*x0*XX+ss*ss-12.0*XX*XX)/(_np.pi*(4.0*x0*x0-8.0*x0*XX+ss*ss+4.0*XX*XX)**3.0)
        gvec[2,:] = 16.0*AA*(XX-x0)*(3.0*ss*ss-4.0*(XX*XX-2.0*XX*x0+x0*x0))/(_np.pi*(ss*ss+4.0*(XX*XX-2.0*XX*x0+x0*x0))**3.0)
        return gvec

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
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
        aout = _np.copy(ain)
        aout[0] = slope*xslope*aout[0]
        aout[1] = xslope*aout[1]+xoffset
        aout[2] = xslope*aout[2]
#        self._secretoffset = offset
        print('Data offsets do not work with the lorentzian model')
        return aout

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelLorentzian, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelLorentzian

# =========================================================================== #
# =========================================================================== #


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
    def __init__(self, XX, af=None, **kwargs):
        super(ModelPseudoVoigt, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        return (aa[0]*ModelLorentzian._model(XX, aa[1:4])
                (1.0-aa[0])*ModelNormal._model(XX, aa[4:]))

    @staticmethod
    def _deriv(XX, aa):
        return (aa[0]*ModelLorentzian._deriv(XX, aa[1:4])
                (1.0-aa[0])*ModelNormal._deriv(XX, aa[4:]))

    @staticmethod
    def _partial(XX, aa):
        gvec = _np.concatenate( (ModelPseudoVoigt.model/aa[0],
                                 aa[0]*ModelLorentzian._partial(XX, aa[1:4]),
                           (1.0-aa[0])*ModelNormal._partial(XX, aa[4:])), axis=0)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        dgdx = _np.concatenate( (ModelPseudoVoigt.derivative/aa[0],
                                 aa[0]*ModelLorentzian._partial_deriv(XX, aa[1:4]),
                           (1.0-aa[0])*ModelNormal._partial_deriv(XX, aa[4:])), axis=0)
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

#    @staticmethod
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        """
#        """
#        aout = _np.copy(ain)
#        return aout

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

def model_loglorentzian(XX, af=None, **kwargs):
    return _model(ModelLogLorentzian, XX, af, **kwargs)

# =========================================== #


class ModelLogLorentzian(ModelClass):
    """
    A log-lorentzian with three free parameters:
        XX - x - independent variable
        af - magnitude, shift, width

      f =  10.0*_np.log10( AA*0.5*ss/((XX-x0)^2+0.25*ss^2)/_np.pi )

      f = 10*log10(A) - 10*log10(s) + 10*log10(0.5) - 10*log10(pi)
        - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )

    A log-lorentzian is a shifted log-parabola in x-s space that is centered at (xo,0)
    """
    _af = _np.asarray([1.0, 0.4, 0.05], dtype=_np.float64)
    _LB = _np.asarray([0.0, -_np.inf, 0.0], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelLogLorentzian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
        log of a Lorentzian
            f = 10*log10(  0.5*A*s / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi  )
              = 10*log10( 0.5*A*s )
                - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )
                - 10*log10(pi)
            f = 10*log10( 0.5*A*s ) - 10*log10(pi)
              - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )

              = line plus log of a shifted parabola
        """
        return 10.0*_np.log(lorentzian(XX, aa))/_np.log(10)
    #    AA, x0, ss = tuple(af)
    #    return ( 10.0*_np.log10( 0.5*AA*ss )
    #           - 10.0*_np.log10(_np.pi)
    #           - 10.0*_np.log10((XX-x0)**2.0 + 0.25*ss**2.0 ) )

    @staticmethod
    def _deriv(XX, aa):
        """
        derivative of the log of a Lorentzian
            f = 10*log10( 0.5*A*s ) - 10*log10(pi)
              - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )

            dfdx = d/dx ( - 10*_np.log10((x-xo)**2.0 + 0.25*ss**2.0 ) )
                 = -2.0*(x-xo)*10.0/((x-xo)**2.0 + 0.25*ss**2.0 )/_np.log(10)
        """
        AA, x0, ss = tuple(aa)
        return -20.0*(XX-x0)/( _np.log(10) *( _np.power(XX-x0, 2.0)+0.25*_np.power(ss, 2.0)) )

    @staticmethod
    def _partial(XX, aa):
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

        gvec = _np.zeros( (3,len(XX)), dtype=_np.float64)
        gvec[0,:] = 10.0/(_np.log(10.0)*AA)
        gvec[1,:] = 20.0*(XX-x0)/( _np.log(10.0)*(_np.power(XX-x0, 2.0) + 0.25*_np.power(ss, 2.0 )))
        gvec[2,:] = 10.0/(_np.log(10.0)*ss) - 5.0*ss/( _np.log(10.0)*(_np.power(XX-x0, 2.0) + 0.25*_np.power(ss, 2.0 )))
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        """
        derivative of the log of a Lorentzian
        f = 10*log10( 0.5*A*s ) - 10*log10(pi)
          - 10*log10((x-xo)**2.0 + 0.25*ss**2.0 )

        dfdx = d/dx ( - 10*_np.log10((x-xo)**2.0 + 0.25*ss**2.0 ) )
             = -20.0*(x-xo)/((x-xo)**2.0 + 0.25*ss**2.0 )/_np.log(10)

        d2f/dxdAA = 0.0
        d2f/dxdx0 = 80.0*(s-2.0*(x-xo))*(s+2.0*(x-xo))/(log10*(4.0*(x-xo)**2.0 + ss**2.0)**2.0 )
        d2f/dxds =
        """
        AA, x0, ss = tuple(aa)

        gvec = _np.zeros( (3,len(XX)), dtype=_np.float64)
        gvec[1,:] = 80.0*(ss-2.0*(XX-x0))*(ss+2.0*(XX-x0))/( _np.log(10)*_np.power( 4.0*_np.power(XX-x0, 2.0) + _np.power(ss, 2.0) , 2.0) )
        gvec[2,:] = 10.0*ss*(XX-x0)/(_np.log(10.0)*_np.power(_np.power(XX-x0, 2.0)+0.25*_np.power(ss, 2.0), 2.0) )
        return gvec

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
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
        aout = _np.copy(ain)
        aout[0] = _np.exp( _np.log(10.0)*offset/10.0 - 2.0*slope*_np.log(_np.abs(xslope))+slope*_np.log(_np.abs(aout[0])))
        aout[1] = xslope*aout[1]+xoffset
        aout[2] = xslope*aout[2]
        return aout

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
#
#def _parse_noshift(ain, model_order=2):
#    if model_order>1:
#        return ain[0:3], ain[3:6], ain[6:]
#    elif model_order>0:
#        return ain[0:3], ain[3:6], []
#    else:
#        return ain[0:3], [], []
#    # end if
## end def
#
#def doppler(xdat, ain, model_order=2):
#    a0, a1, a2 = _parse_noshift(ain, model_order=model_order)
#    prof = normal(xdat, af=a0)
#    if model_order>0:
#        prof += lorentzian(xdat, af=a1)
#    if model_order>1:
#        prof += normal(xdat, af=a2)
#    return prof
#
#def deriv_doppler(xdat, ain, model_order=2):
#    a0, a1, a2 = _parse_noshift(ain, model_order=model_order)
#    prof = deriv_normal(xdat,af=a0)
#    if model_order>0:
#        prof += deriv_lorentzian(xdat, af=a1)
#    if model_order>1:
#        prof += deriv_normal(xdat,af=a2)
#    return prof
#
#def partial_doppler(xdat, ain, model_order=2):
#    a0, a1, a2 = _parse_noshift(ain, model_order=model_order)
#    gvec = partial_normal(xdat, af=a0)
#    if model_order>0:
#        gvec = _np.append(gvec, partial_lorentzian(xdat, af=a1), axis=0)
#    if model_order>1:
#        gvec = _np.append(gvec, partial_normal(xdat, af=a2), axis=0)
#    return gvec
#
#def partial_deriv_doppler(xdat, ain, model_order=2):
#    a0, a1, a2 = _parse_noshift(ain, model_order=model_order)
#    gvec = partial_deriv_normal(xdat, af=a0)
#    if model_order>0:
#        gvec = _np.append(gvec, partial_deriv_lorentzian(xdat, af=a1), axis=0)
#    if model_order>1:
#        gvec = _np.append(gvec, partial_deriv_normal(xdat, af=a2), axis=0)
#    return gvec
#
#def _doppler_logdata(xdat, ain, model_order=2):
#    return 10.0*_np.log10(doppler(xdat, ain, model_order))
#
#def _deriv_doppler_logdata(xdat, ain, model_order=2):
#    """
#        f = 10*log10(y)
#        dfdx = 10.0/ln(10) d ln(y)/dx
#            d ln(y)/dx = 1/y * dydx
#    """
#    prof = doppler(xdat, ain, model_order)
#    deriv = deriv_doppler(xdat, ain, model_order)
#    return 10.0*deriv/(prof*_np.log(10.0))
#
#def _partial_doppler_logdata(xdat, ain, model_order=2):
#    """
#        f = 10*log10(y)
#        dfdx = 10.0/ln(10) d ln(y)/dx
#            d ln(y)/dx = 1/y * dydx
#        dfda = 10.0 / ln(10) * d ln(y)/da
#             = 10.0/ ln(10) *dyda/y
#    """
#    prof = doppler(xdat, ain, model_order)
#    gvec = partial_doppler(xdat, ain, model_order)
#    for ii in range(len(ain)):
#        gvec[ii,:] /= prof
#    # end for
#    return 10.0*gvec/_np.log(10.0)
#
#def _partial_deriv_doppler_logdata(xdat, ain, model_order=2):
#    """
#        f = 10*log10(y)
#        dfdx = 10.0/ln(10) d ln(y)/dx
#            d ln(y)/dx = 1/y * dydx
#        d2fdxda = d(df/dx)/da
#                = 10.0/ln(10) d(dy/dx/y)/da
#                = 10.0/ln(10) * ( 1/y*d2y/dxda - dyda/y^2 )
#    """
#    prof = doppler(xdat, ain, model_order)
#    gvec = partial_doppler(xdat, ain, model_order)
#    dgdx = partial_deriv_doppler(xdat, ain, model_order)
#    dlngdx = _np.zeros_like(gvec)
#    for ii in range(len(ain)):
#        dlngdx[ii,:] = dgdx[ii,:]/prof -gvec[ii,:]/(prof*prof)
#    # end for
#    return 10.0*dlngdx/_np.log(10)
#
#
#
## fit model to the data
#def model_doppler(XX, af=None, **kwargs):
#    logdata = kwargs.setdefault('logdata', False)
#    if logdata:
#        return _model_doppler_logdata(XX, af=af, **kwargs)
#    else:
#        return _model_doppler_lindata(XX, af=af, **kwargs)
#    # end if
## end def
#
#
#
#
#def _model_doppler_lindata(XX, af=None, **kwargs):
#    """
#         simple shifted gaussian - one doppler peak
#         3 parameter fit
#            modelspec.gaussian AA*_np.exp(-(XX-x0)**2/(2.0*ss**2))
#         shifted gaussian and centered lorentzian - one doppler peak and reflection
#         6 parameter fit
#       +    modelspec.lorentzian AA*0.5*ss/((XX-x0)*(XX-x0)+0.25*ss*ss)/_np.pi
#         two shifted gaussians and centered lorentzian - one doppler peak and reflection
#         9 parameter fit
#       +    modelspec.gaussian ... initialized with opposite shifts
#    """
#    model_order = kwargs.setdefault('model_order', 0)
#    Fs = kwargs.setdefault('Fs', None)  # default to normalized frequencies
#    noshift = kwargs.setdefault('noshift', True)
#    if af is None:
#        af = _np.ones((9,), dtype=_np.float64)
#        # frequencies are normalized to sampling frequency
#        af[0] = 1.0e-4      # integral of gaussian, -30 dBm ~ 0.001 mW = 1e-6 uW
#        af[1] = -0.04       # (-0.4 MHz) doppler shift of gaussian,
#        af[2] = 0.005       # doppler width of peak
#        af[3] = 1.0e-3      # integrated value of lorentzian
#        af[4] = 0.0         # shift of lorentzian
#        af[5] = 0.01        # width of lorentzian peak
#        af[6] = 0.5*_np.copy(af[0])   # integral of second gaussian
#        af[7] =-0.75*_np.copy(af[1])  # doppler shift of second gaussian
#        af[8] = 2.0*_np.copy(af[2])   # width of doppler peak of second gaussian
#
#        if Fs is not None:
#            af[1] *= Fs;            af[2] *= Fs
#            af[4] *= Fs;            af[5] *= Fs
#            af[7] *= Fs;            af[8] *= Fs
#        # end if
#
#        if model_order<2:
#            af = af[:6]
#        elif model_order<1:
#            af = af[:3]
#        # end if
#    # end if
#    if Fs is None:                  Fs = 1.0            # end if
#    a0, a1, a2 = _parse_noshift(af, model_order=model_order)
##    af = _np.asarray(a0.tolist()+a1.tolist()+a2.tolist(), dtype=_np.float64)
#
#    i0 = model_gaussian(None, a0, norm=True)
#    info = Struct()
#    info.af = _np.copy(i0.af)
#    info.Lbounds = _np.array([0.0,-Fs/2, -Fs/2], dtype=_np.float64)
#    info.Ubounds = _np.array([3.0, Fs/2, Fs/2], dtype=_np.float64)  # update this based on experience
#    info.fixed = _np.zeros( _np.shape(info.Lbounds), dtype=int)
#
#    if model_order>0:
#        i1 = model_lorentzian(None, a1)
#        if noshift:
#            i1.fixed[1] = int(1)
#            i1.af[1] = 0.0
#        # end if
#        info.af = _np.append(info.af, i1.af, axis=0)
#        i1.Lbounds = _np.array([0.0,-Fs/2.0, -Fs/2.0], dtype=_np.float64)
#        i1.Ubounds = _np.array([3.0, Fs/2.0, Fs/2.0], dtype=_np.float64)
#        info.Lbounds = _np.append(info.Lbounds, i1.Lbounds, axis=0)
#        info.Ubounds = _np.append(info.Ubounds, i1.Ubounds, axis=0)
#        info.fixed = _np.append(info.fixed, i1.fixed, axis=0)
#    # end if
#
#    if model_order>1:
#        i2 = model_gaussian(None, a2, norm=True)
#        info.af = _np.append(info.af, i2.af, axis=0)
#        i2.Lbounds = _np.array([0.0,-Fs/2.0, -Fs/2.0], dtype=_np.float64)
#        i2.Ubounds = _np.array([3.0, Fs/2.0, Fs/2.0], dtype=_np.float64)
#        info.Lbounds = _np.append(info.Lbounds, i2.Lbounds, axis=0)
#        info.Ubounds = _np.append(info.Ubounds, i2.Ubounds, axis=0)
#        info.fixed = _np.append(info.fixed, i2.fixed, axis=0)
#
#    # ====== #
#
#    info.Lbounds = kwargs.setdefault('LB', info.Lbounds)
#    info.Ubounds = kwargs.setdefault('UB', info.Ubounds)
#    info.fixed = kwargs.setdefault('fixed', info.fixed)
#    def checkbounds(dat, ain):
#        LB = _np.copy(info.Lbounds)
#        UB = _np.copy(info.Ubounds)
#        ain = _checkbounds(ain, LB=LB, UB=UB)
#        return dat, ain
#    info.checkbounds = checkbounds
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        a0, a1, a2 = _parse_noshift(ain, model_order=model_order)
#        aout = _np.copy(ain)
#        aout = i0.unscaleaf(a0, slope, offset=0.0, xslope=1.0, xoffset=0.0)
#        if model_order>0:
#            aout = _np.append(aout, i1.unscaleaf(a1, slope, offset=0.0, xslope=xslope, xoffset=xoffset), axis=0)
#        if model_order>1:
#            aout = _np.append(aout, i2.unscaleaf(a2, slope, offset=0.0, xslope=xslope, xoffset=xoffset), axis=0)
#        info._secretoffset = offset
#        print('Data offsets do not work with the lorentzian/normal models:\n'
#             +'Remember the offset is now included in the model calls!')
#        return aout
#    info.unscaleaf = unscaleaf
#    offset = 0.0 if not hasattr(info, '_secretoffset') else info._secretoffset    # end if
#    if XX is None:
#        return info
#    # end if
#
#    # ===== #
#
#    model = doppler(XX, af, model_order=model_order) + offset
#    gvec = partial_doppler(XX, af, model_order=model_order)
#
#    info.prof = model
#    info.gvec = gvec
#    info.dprofdx = deriv_doppler(XX, af, model_order=model_order)
#    info.dgdx = partial_deriv_doppler(XX, af, model_order=model_order)
#
#    info.func = lambda _x, _a: doppler(_x, _a, model_order=model_order)+offset
#    info.dfunc = lambda _x, _a: deriv_doppler(_x, _a, model_order=model_order)
#    info.gfunc = lambda _x, _a: partial_doppler(_x, _a, model_order=model_order)
#    info.dgfunc = lambda _x, _a: partial_deriv_doppler(_x, _a, model_order=model_order)
#    return model, gvec, info
## end def
#
#
#def _model_doppler_logdata(XX, af=None, **kwargs):
#    model_order = kwargs.setdefault('model_order', 0)
#
#
#    if XX is None:
#        info = _model_doppler_lindata(None, af=af, **kwargs)
#        return info
#    # end if
#
#    model, gvec, info = _model_doppler_lindata(XX, af=af, **kwargs)
#    af = info.af.copy()
#
#    # ===== #
#    def checkbounds(dat, ain):
#        LB = _np.copy(info.Lbounds)
#        UB = _np.copy(info.Ubounds)
#        ain = _checkbounds(ain, LB=LB, UB=UB)
#
#        dat = _np.copy(dat)
#        if (dat<0.0).any(): dat[dat<0] = min((1e-10, 1e-3*_np.min(dat[dat>0]))) # end if
#        return dat, ain
#    info.checkbounds = checkbounds
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):  # check this ... leaving noshift to user
#        aout = info.unscaleaf(ain, slope, 0.0, xslope, xoffset)
#        info._secretoffset = offset
#        print('Data offsets do not work with the lorentzian model:\n'
#             +'Remember the offset is now included in the models!')
#        return aout
#    info.unscaleaf = unscaleaf
#    offset = 0.0 if not hasattr('_secretoffset',info) else info._secretoffset    # end if
#
#    # ===== #
#
#    model = _doppler_logdata(XX, af, model_order=model_order) + offset
#    gvec = _partial_doppler_logdata(XX, af, model_order=model_order)
#
#    info.prof = model
#    info.gvec = gvec
#    info.dprofdx = _deriv_doppler_logdata(XX, af, model_order=model_order)
#    info.dgdx = _partial_deriv_doppler_logdata(XX, af, model_order=model_order)
#
#    info.func = lambda _x, _a: _doppler_logdata(_x, _a, model_order=model_order) + offset
#    info.dfunc = lambda _x, _a: _deriv_doppler_logdata(_x, _a, model_order=model_order)
#    info.gfunc = lambda _x, _a: _partial_doppler_logdata(_x, _a, model_order=model_order)
#    info.dgfunc = lambda _x, _a: _partial_deriv_doppler_logdata(_x, _a, model_order=model_order)
#    return model, gvec, info
#
#
#class ModelDoppler(ModelClass):
#    """
#
#    """
#    _af = _np.asarray([1.0], dtype=_np.float64)
#    _LB = _np.asarray([0.0], dtype=_np.float64)
#    _UB = _np.asarray([_np.inf], dtype=_np.float64)
#    _fixed = _np.zeros( (1,), dtype=int)
#    def __init__(self, XX, af=None, **kwargs):
#        super(ModelDoppler, self).__init__(XX, af, **kwargs)
#    # end def __init__
#
#    @staticmethod
#    def _model(XX, aa):
#
#    @staticmethod
#    def _deriv(XX, aa):
#
#    @staticmethod
#    def _partial(XX, aa):
#
#    @staticmethod
#    def _partial_deriv(XX, aa):
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
##        aout = _np.copy(ain)
##        return aout
#
#    # ====================================== #
#
##    def checkbounds(self, dat):
##        return super(ModelDoppler, self).checkbounds(dat, self.aa, mag=None)
#
#    # ====================================== #
## end def ModelDoppler


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


def _model_twopower(XX, af=None, **kwargs):
    return _model(_ModelTwoPower, XX, af, **kwargs)

class _ModelTwoPower(ModelClass):
    """
    A two power profile fit with three free parameters:
    prof ~ f(x) = a*(1-x^pow1)^pow2
    """
    _af = _np.asarray([1.0, 2.0, 1.0], dtype=_np.float64)
    _LB = _np.asarray([0.0, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(_ModelTwoPower, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model_base(XX, aa):
        """
            f = (1.0 - x**c)**d
        non-trival domain:  c>0 and 0<=x<1  or   c<0 and x>1
        """
        XX = _np.copy(XX)
        c, d = tuple(aa)
        return _np.power(1.0-_np.power(_np.abs(XX), c), d)

    @staticmethod
    def _model(XX, aa):
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
    def _deriv(XX, aa):
        """
            f = a*(1.0 - x^c)^d
         dfdx = -a*c*d*x^(c-1)*(1.0 - x**c)**(d-1)
              = -c*d*x^(c-1)*_twopower(x, [c, d-1])
       """
        XX = _np.copy(_np.abs(XX))
        a, c, d = tuple(aa)
        return -1.0*c*d*_np.power(XX, c-1.0)*_ModelTwoPower._model(XX, [a, c, d-1.0])
    #    return -1.0*a*c*d*_np.power(XX, c-1.0)*_np.power(1.0-_np.power(XX,c), d-1.0)

    @staticmethod
    def _partial(XX, aa):
        """
            f = a*(1.0 - x^c)^d
         dfdx = -a*c*d*x^(c-1)*(1.0 - x**c)**(d-1)

         dfda = f/a = _twopower/a
         dfdc = -a*d*(1.0 - x^c)^(d-1)*x^c*ln|x|
              = -d*x^c*ln(x)*_twopower(x, [c, d-1])
         dfdd = a*(1.0 - x^c)^d * ln|1.0 - x^c|
              = _twopower * ln|_twopower(x, [1.0, c, 1.0])|
       """
        XX = _np.copy(XX)
        a, c, d = tuple(aa)

        gvec = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = _ModelTwoPower._model_base(XX, [c, d])
        gvec[1,:] = -1.0*d*_np.power(XX, c)*_np.log(_np.abs(XX))*_ModelTwoPower._model(XX, [a, c, d-1.0])
        gvec[2,:] = _ModelTwoPower._model(XX, aa)*_np.log(_np.abs(_ModelTwoPower._model(XX, [1.0, c, 1.0])))
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        """
            f = a*(1.0 - x^c)^d
         dfdx = -a*c*d*x^(c-1)*(1.0 - x**c)**(d-1)

         dfda = f/a = _twopower/a
         dfdc = -a*d*(1.0 - x^c)^(d-1)*x^c*ln|x|
              = -d*x^c*ln(x)*_twopower(x, [c, d-1])
         dfdd = a*(1.0 - x^c)^d * ln|1.0 - x^c|
              = _twopower * ln|_twopower(x, [1.0, c, 1.0])|

        d2fdxda = dfdx/a
        d2fdxdc = dfdx/c + ln|x|*dfdx - a*c*d*x^(c-1)*(d-1)*(1.0 - x^c)^(d-2)*-1*x^c*ln|x|
                = dfdx/c + ln|x|*dfdx - a*c*d*x^(c-1)*(d-1)*(1.0 - x^c)^(d-2)*-1*x^c*ln|x|
                = dfdx/c + ln|x|*dfdx - dfdx*(d-1)*x^c*ln|x|*(1.0 - x^c)^-1
                = dfdx*(1.0/c + ln|x| - (d-1)*x^c*ln|x|*_twopower(x, [1.0, c, -1.0]))
        d2fdxdd = dfdx/d + dfdx*ln|(1.0 - x^c)^(d-1)|
       """
        XX = _np.copy(XX)
        a, c, d = tuple(aa)
        dfdx = _ModelTwoPower._deriv(XX, aa)

        dgdx = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = dfdx/a
        dgdx[1,:] = dfdx*(_np.power(c, -1.0) + _np.log(_np.abs(XX))
            - (d-1.0)*_np.power(XX, c)*_np.log(_np.abs(XX))*_ModelTwoPower._model(XX, [1.0, c, -1.0]))
        dgdx[2,:] = dfdx*(_np.power(d, -1.0) + _np.log(_np.abs(_ModelTwoPower._model_base(XX, [c, d-1.0]))))
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

#    @staticmethod
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        """
#         y-scaling is possible, but everything else is non-linear
#        """
#        aout = _np.copy(ain)
#        return aout

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

def model_twopower(XX, af=None, **kwargs):
    return _model(ModelTwoPower, XX, af, **kwargs)

class ModelTwoPower(ModelClass):
    """
    model a two-power fit
        first-half of a quasi-parabolic (no hole depth width or decaying edge)
                .... dumb: reproduced everywhere: y-offset = slope*(1-x^b)^c
        y = a*( (1-b)*(1.0 - x^c)^d + b )

        a = amplitude of core
        b = edge / core
        c = power scaling factor 1
        d = power scaling factor 2
    """
    _af = _np.asarray([1.0, 0.0, 12.0, 3.0], dtype=_np.float64)
    _LB = _np.asarray([0.0, 0.0, -20.0, -20.0], dtype=_np.float64)
    _UB = _np.asarray([20, 1.0, 20.0, 20.0], dtype=_np.float64)
    _fixed = _np.zeros( (4,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelTwoPower, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
         one option: f/a = (1-b)*(1-x^c)^d + b
         2nd option: f = (a-b)*(1- x^c)^d + b
             we'll take the first to build with it easier

        non-trival domain:  c>0 and 0<=x<1  or   c<0 and x>1
        """
        a, b, c, d = tuple(aa)
        return (1.0-b)*_twopower(XX, [a, c, d]) + a*b

    @staticmethod
    def _deriv(XX, aa):
        """
        f/a = (1-b)*(1-x^c)^d + b
        dfdx = a*(1-b)*c*d*x^(c-1)*(1-x^c)^(d-1)
             = (1-b)*_deriv_twopower

        """
        a, b, c, d = tuple(aa)
        return (1.0-b)*_deriv_twopower(XX, [a, c, d])

    @staticmethod
    def _partial(XX, aa):
        """
        f/a = (1-b)*(1-x^c)^d + b
        dfdx = a*b*c*d*x^(c-1)*(1-x^c)^(d-1)

         dfda = b+(1-b)*(1-x^c)^d
         dfdb = a-a*(1-x^c)^d = a-_twopower(XX, [a, c, d])
         dfdc = -a*(1-b)*d*x^c*ln|x|*(1-x^c)^(d-1)
              = -d*ln|x|*x^c*twopower(x, [a,b,c,d-1])
         dfdd = a*(1.0-b)(1.0 - x^c)^d * ln|1.0 - x^c|
              = (1.0-b)*_twopower() * ln|_twopower(x, [1.0, c, 1.0])|
        """
        XX = _np.copy(XX)
        a, b, c, d = tuple(aa)

        nx = len(XX)
        gvec = _np.zeros((4, nx), dtype=_np.float64)
        gvec[0, :] = twopower(XX, [1.0, b, c, d])
        gvec[1, :] = a-_twopower(XX, [a, c, d])
        gvec[2, :] = -d*_np.log(_np.abs(XX))*_np.power(_np.abs(XX), c)*twopower(XX, [a, b, c, d-1.0])
        gvec[3, :] = (1.0-b)*_twopower(XX, [a, c, d])*_np.log(_np.abs(_twopower(XX, [1.0, c, 1.0])))
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
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
        dfdx = deriv_twopower(XX, aa)

        nx = len(XX)
        dgdx = _np.zeros((4, nx), dtype=_np.float64)
        dgdx[0, :] = dfdx/a
        dgdx[1, :] = -dfdx/(1.0-b)
        dgdx[2, :] = dfdx*( 1.0/c + _np.log(_np.abs(XX))
                - (d-1.0)*_np.power(XX, c)*_np.log(_np.abs(XX))/_ModelTwoPower._model_base(XX,[c, 1.0]))
        dgdx[3, :] = dfdx*( 1.0/d + _np.log(_np.abs(_ModelTwoPower._model_base(XX, [c, 1.0]))) )
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

#    @staticmethod
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#    """
#        To unscale the problem
#        y-scaling: y'= (y-yo)/ys
#            y = yo+ys*a'*(1-b')(1-x^c')^d' + ys*a'b'
#             (1)  ab = yo+ys*a'b'
#             (2)  a*(1-b) = ys*a'*(1-b')
#             a = yo/b + ys*a'*b'/b   from (1)
#             b = 1-ys*(1-b')*a'/a    from (2)
#             c= c'
#             d= d'
#
#        x-scaling:  x' = (x-xo)/xs
#            y = yo+ys*a'*(1-b')(1-(x-xo)^c/xs^c')^d' + ys*a'b'
#                doesn't work due to the nonlinearity, xo=0, xs=1
#
#             a = yo/b + ys*a'*b'/b   from (1)
#             b = 1-ys*(1-b')*a'/a    from (2)
#             c= c'
#             d= d'
#    """
#        aout = _np.copy(ain)
#        return aout

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

def model_expedge(XX, af, **kwargs):
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
    _af = _np.asarray([2.0, 1.0], dtype=_np.float64)
    _LB = _np.asarray([-_np.inf,-_np.inf], dtype=_np.float64)
    _UB = _np.asarray([ _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (2,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelExpEdge, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
        model an exponential edge
            y = e*(1-exp(-x^2/h^2))

              = e *( 1 - gaussian(XX, [1.0, 0.0, _np.sqrt(0.5)*h]) )
            second-half of a quasi-parabolic (no edge, or power factors)
            e = hole width
            h = hole depth
        """
        e, h = tuple(aa)
        prof = e*(1.0-_np.exp(-_np.square(XX)/_np.square(h)))
        return prof

    @staticmethod
    def _deriv(XX, aa):
        e, h = tuple(aa)
        return 2.0*XX*e*_np.exp(-_np.square(XX)/_np.square(h))/_np.square(h)

    @staticmethod
    def _partial(XX, aa):
        e, h = tuple(aa)
        gvec = _np.zeros((2,len(XX)), dtype=_np.float64)
        gvec[0,:] =  1.0 - _np.exp(-_np.square(XX)/_np.square(h))
        gvec[1,:] = -2.0*_np.square(XX)*e*_np.exp(-_np.square(XX)/_np.square(h))/_np.power(h,3)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        e, h = tuple(aa)
        gvec = _np.zeros((2,len(XX)), dtype=_np.float64)
        gvec[0,:] = 2.0*XX*_np.exp(-_np.square(XX)/_np.square(h))/_np.square(h)
        gvec[1,:] = 4.0*XX*e*_np.exp(-_np.square(XX)/_np.square(h)) * (_np.square(XX)-_np.square(h))/_np.power(h,5)
        return gvec
#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    @staticmethod
    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
        """
            y-scaling: y' = y/ys
            y' = y/ys = e'*(1-exp(-x^2/h'^2))
                e = e'*ys
                h = h'
                possible with constant coefficients

          y-shifting: y' = (y-yo)/ys
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
          x-shifting not possible with constant coefficients
        """
        aout = _np.copy(ain)
        aout[0] = slope*aout[0]
        aout[1] = xslope*aout[1]
        print('No shifting is possible while unscaling')
        return aout

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelExpEdge, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelExpEdge


# ========================================================================== #
# ========================================================================== #

def qparab(XX, aa):
    return ModelQuasiParabolic._model(XX, aa)

def deriv_qparab(XX, aa):
    return ModelQuasiParabolic._deriv(XX, aa)

def partial_qparab(XX, aa):
    return ModelQuasiParabolic._partial(XX, aa)

def partial_deriv_qparab(XX, aa):
    return ModelQuasiParabolic._partial_deriv(XX, aa)

def deriv2_qparab(XX, aa):
    """
    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
    This subfunction calculates the second derivative of a quasi-parabolic fit
    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
        XX - r/a
    aa[0] - Y0 - function value on-axis
    aa[1] - gg - Y1/Y0 - function value at edge over core
    aa[2],aa[3]-  pp, qq - power scaling parameters
    aa[4],aa[5]-  hh, ww - hole depth and width
    """
    aa = _np.asarray(aa,dtype=_np.float64)
    a, b, c, d, e, f = tuple(aa)

    d2pdx2 = aa[3]*(aa[2]**2.0)*(aa[3]-1.0)*(1.0+aa[4]-aa[1])*_np.power(_np.abs(XX), 2.*aa[2]-2.0)*(1-_np.power(_np.abs(XX),aa[2]))**(aa[3]-2.0)
    d2pdx2 -= (aa[2]-1.0)*aa[2]*aa[3]*(1.0+aa[4]-aa[1])*_np.power(_np.abs(XX), aa[2]-2.0)*_np.power(1-_np.power(_np.abs(XX), aa[2]), aa[3]-1.0)
    d2pdx2 += (2.0*aa[4]*_np.exp(-_np.power(XX,2.0)/_np.power(aa[5], 2.0)))/_np.power(aa[5], 2.0)
    d2pdx2 -= (4*aa[4]*_np.power(XX, 2.0)*_np.exp(-_np.power(XX, 2.0)/_np.power(aa[5], 2.0)))/_np.power(aa[5], 4.0)
    d2pdx2 *= a
    return d2pdx2
# end def derive_qparab

def partial_deriv2_qparab(XX, aa):
    """
    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
    This subfunction calculates the jacobian of the second derivative of a
    quasi-parabolic fit (partial derivatives of the second derivative of a quasi-parabolic fit)

    quasi-parabolic fit:
    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
        XX - r/a

    aa[0] - Y0 - function value on-axis
    aa[1] - gg - Y1/Y0 - function value at edge over core
    aa[2],aa[3]-  pp, qq - power scaling parameters
    aa[4],aa[5]-  hh, ww - hole depth and width
    """
    aa = _np.asarray(aa,dtype=_np.float64)
    Y0 = aa[0]
    g = aa[1]
    p = aa[2]
    q = aa[3]
    h = aa[4]
    w = aa[5]

    gvec = _np.zeros( (6,_np.size(XX)), dtype=_np.float64)
    gvec[0,:] = deriv2_qparab(XX, aa) / Y0
    gvec[1,:] = -p*q*Y0*_np.power(_np.abs(XX), p-2.0)*_np.power(1.0-_np.power(_np.abs(XX), p), q-2.0)*(p*(q*_np.power(_np.abs(XX), p)-1.0)-_np.power(XX, p)+1.0)
    gvec[2,:] = p*_np.log(_np.abs(XX))*(p*(_np.power(q, 2.0)*_np.power(_np.abs(XX), 2.0*p)-3.0*q*_np.power(_np.abs(XX), p)+_np.power(_np.abs(XX), p)+1.0)-(_np.power(XX, p)-1.0)*(q*_np.power(_np.abs(XX), p)-1.0))
    gvec[2,:] += (_np.power(_np.abs(XX), p)-1.0)*(2.0*p*(q*_np.power(_np.abs(XX), p)-1.0)-_np.power(_np.abs(XX), p)+1.0)
    gvec[2,:] *= q*Y0*(g-h-1.0)*_np.power(_np.abs(XX), p-2.0)*(_np.power(1.0-_np.power(_np.abs(XX), p), q-3.0))
    gvec[3,:] = p*Y0*(-(g-h-1.0))*_np.power(_np.abs(XX), p-2.0)*_np.power(1.0-_np.power(_np.abs(XX), p), q-2.0)*(p*(2.0*q*_np.power(_np.abs(XX), p)-1.0)+q*(p*(q*_np.power(_np.abs(XX), p)-1.0)-_np.power(_np.abs(XX), p)+1.0)*_np.log(_np.abs(1.0-_np.power(_np.abs(XX)**p)))-_np.power(_np.abs(XX), p)+1.0)
    gvec[4,:] = Y0*(p*q*_np.power(_np.abs(XX), p-2.0)*_np.power(1.0-_np.power(_np.abs(XX), p), q-2.0)*(p*(q*_np.power(_np.abs(XX), p)-1.0)-_np.power(_np.abs(XX), p)+1.0)+(2.0*_np.exp(-_np.power(XX, 2.0)/_np.power(w, 2.0))*(_np.power(w, 2.0)-2.0*_np.power(_np.abs(XX), 2.0)))/_np.power(w, 4.0))
    gvec[5,:] = -(4.0*h*Y0*_np.exp(-_np.power(XX, 2.0)/_np.power(w, 2.0))*(_np.power(w, 4.0)-5*_np.power(w, 2.0)*_np.power(_np.abs(XX), 2.0)+2.0*_np.power(_np.abs(XX), 4.0)))/_np.power(w, 7.0)

    return gvec
# end def partial_deriv2_qparab


def model_qparab(XX, af=None, **kwargs):
    """

    """
    return _model(ModelQuasiParabolic, XX, af, **kwargs)

# ======================================= #


class ModelQuasiParabolic(ModelClass):
    """
    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
        XX - r/a

    aa[0] - Y0 - function value on-axis
    aa[1] - gg - Y1/Y0 - function value at edge over core
    aa[2],aa[3]-  pp, qq - power scaling parameters
    aa[4],aa[5]-  hh, ww - hole depth and width

        b+(1-b)*(1-XX^c)^d
    {x element R: (d>0 and c=0 and x>0) or (d>0 and x=1)
            or (c>0 and 0<=x<1) or (c<0 and x>1) }
        nohollow = True
        af = _np.hstack((af,0.0))
        af = _np.hstack((af,1.0))

        Y0 = core
        YO*aa[1] = edge

    """
    _af = _np.asarray([1.0, 0.002, 2.0, 0.7, -0.24, 0.30], dtype=_np.float64)
    _LB = _np.asarray([  0.0, 0.0,  0,-10,-1, 0], dtype=_np.float64)
    _UB = _np.asarray([ 20.0, 1.0, 10, 10, 1, 1], dtype=_np.float64)
    _fixed = _np.zeros( (6,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        nohollow = kwargs.setdefault('nohollow', False)
        if nohollow:
            self._af[4] = 0.0
            self._af[5] = 1.0
            self._fixed[4:] = int(1)
        super(ModelQuasiParabolic, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        a, b, c, d, e, f = tuple(aa)
        prof = ( ModelTwoPower._model(XX, [a, b-e, c, d])
                 + a*ModelExpEdge._model(XX, [e, f]) )
        return prof

    @staticmethod
    def _deriv(XX, aa):
        a, b, c, d, e, f = tuple(aa)
        return ( ModelTwoPower._deriv(XX, [a, b-e, c, d])
                 + a*ModelExpEdge._deriv(XX, [e, f]) )

    @staticmethod
    def _partial(XX, aa):
        a, b, c, d, e, f = tuple(aa)
        gvec = _np.zeros((6,len(XX)), dtype=_np.float64)
        gvec[0, :] = ModelQuasiParabolic._model(XX, aa)/a
        gvec[1:4,:] = ModelTwoPower._partial(XX, [a, b-e, c, d])[1:,:]
        gvec[4:, :] = a*ModelExpEdge._partial(XX, [e, f])
    #    gvec = _np.append(gvec, ModelTwoPower._partial(XX, [a, b-e, c, d]), axis=0)
    #    gvec = _np.append(gvec, a*ModelExpEdge._partial(XX, [e, f]), axis=0)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        a, b, c, d, e, f = tuple(aa)
        gvec = _np.zeros((6,len(XX)), dtype=_np.float64)
        gvec[0, :] = ModelQuasiParabolic._deriv(XX, aa)/a
        gvec[1:4,:] = ModelTwoPower._partial_deriv(XX, [a, b-e, c, d])[1:,:]
        gvec[4:,:] = a*ModelExpEdge._partial_deriv(XX, [e, f])
    #    gvec = _np.append(gvec, a*partial_deriv_edgepower(XX, [b-e, c, d]), axis=0)
    #    gvec = _np.append(gvec, a*partial_deriv_expedge(XX, [e, f]), axis=0)
        return gvec


#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

#    @staticmethod
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        """
#        rescaling: y'=(y-yo)/ys
#            y' = a0'*( a1'-a4'+(1-a1'+a4')*(1-x^a2')^a3' + a4'*(1-exp(x^2/a5'^2)))
#               = a0'*a1'-a0'*a4' + a0'*(1-a1'+a4')*(...) + a0'*a4'*(...)
#               = -a4' + (1+a4')*(...) + a4'*(...)
#
#            y = yo+a0'*(...)
#                by inspection
#                (1)
#            constants :    a0*a1-a0*a4 = yo+a0'*a1'-a0'*a4'
#
#            and (2)
#            (1-x^a2)^a3 :   a0*(1-a1+a4) = a0'*(1-a1'+a4')
#                     a0-a1*a0+a0*a4 = a0'-a0'*a1'+a0'*a4'
#            and (3)
#            (1-exp(x^2/a5^2)) : a0*a4 = a0'*a4'
#                                    --------
#                    use (3) in (1):  a0*a1 = yo+a0'*a1'
#                        (3) in (2):  a0-a1*a0 = a0'-a0'*a1' ->
#                                    a0*(1-a1) = a0'*(1-a1')
#                                    --------
#                  a0 = a0'*(1-a1)/(1-a1') = (yo+a0'*a1')/a1
#                   a0'*(1-a1)/(1-a1') = (yo+a0'*a1')/a1
#                        a0'*(1-a1)*a1 = (1-a1')*(yo+a0'*a1')
#                        a1-a1^2 = (1-a1')*(yo/a0'+a1')
#                        a1-a1^2 = yo/a0'+a1'-yo*a1'/a0'-a1'^2
#
#                            a1 = yo/a0' + a1' - yo*a1'/a0'
#                            a0 = a0'*(1-a1)/(1-a1')
#                            a4 = a0'*a4'/a0
#
#        rescaling: x'=(x-xo)/xs
#            is not possible because of the non-linearities in the x-functions
#                (1-x^a2)^a3 = (1-(x-xo)^a2'/xs^a2')^a3'
#
#          and   exp(x^2/a5^2) = exp((x-xo)^2/(xs*a5')^2)
#                exp(x/a5) = exp((x-xo)/(xs*a5'))
#                    x/a5 = (x-xo)/(xs*a5')
#                    x*xs*a5' = x*a5-xo*a5
#                    independence of x means that x*(xs*a5'-a5) = 0, a5 = xs*a5'
#                    but x*xs*a5' = x*xs*a5'-xo*a5 requires that xo == 0
#
#                (1-x^a2)^a3 = (1-x^a2'/xs^a2')^a3'
#                a3*ln(1-x^a2) = a3'*ln(1-x^a2'/xs^a2')
#                    assume a3 == a3'
#                    x^a2 = x^a2'/xs^a2'
#                    a2*ln(x) = a2'*ln(x)-a2'*ln(xs)
#                        a2 = a2'*(1-ln(xs)/ln(x))  ... only works if xs = 1.0
#        a1 = yo/a0' + a1' - yo*a1'/a0'
#        a0 = a0'*(1-a1)/(1-a1')
#        a4 = a0'*a4'/a0
#        """
#        aout = _np.copy(ain)
#        aout[1] = offset/ain[0] + ain[1] - offset*ain[1]/ain[0]
#        aout[0] = ain[0]*(1-aout[1])/(1-ain[1])
#        aout[4] = ain[0]*ain[4]/aout[0]
#        return aout

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

def model_flattop(XX, af, **kwargs):
    """
    """
    return _model(ModelFlattop, XX, af, **kwargs)

class ModelFlattop(ModelClass):
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
    _af = _np.asarray([1.0, 0.4, 5.0], dtype=_np.float64)
    _LB = _np.asarray([0.0, 0.0, 1.0], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, 1.0, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelFlattop, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa):
        """
        A flat-top plasma parameter profile with three free parameters:
            a, b, c
        prof ~ f(x) = a / (1 + (x/b)^c)

        """
        a, b, c = tuple(aa)
        temp = _np.power(XX/b, c)
        return a / (1.0 + temp)

    @staticmethod
    def _deriv(XX, aa):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)
        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)
        """
        a, b, c = tuple(aa)
        temp = _np.power(XX/b, c)
        return -1.0*a*c*temp/(XX*_np.power(1.0+temp,2.0))

    @staticmethod
    def _partial(XX, aa):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)
        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)

        dfda = 1.0/(1+_np.power(x/b, c))
        dfdb = a*c*(x/b)^c/(b*(1+(x/b)^c)^2)            # check math
        dfdc = a*_np.log(x/b)*(x/b)^c / (1+(x/b)^c)^2   # check math
        """
        nx = len(XX)
        a, b, c = tuple(aa)
        temp = _np.power(XX/b, c)
        prof = flattop(XX, aa)

        gvec = _np.zeros((3, nx), dtype=_np.float64)
        gvec[0, :] = prof / a
        gvec[1, :] = a*c*temp / (b*_np.power(1.0+temp,2.0) )
        gvec[2, :] = a*temp*_np.log(_np.abs(XX/b)) / _np.power(1.0+temp,2.0)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)
        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)

        dfda = 1.0/(1+_np.power(x/b, c))
        dfdb = a*c*(x/b)^c/(b*(1+(x/b)^c)^2)            # check math
        dfdc = a*_np.log(x/b)*(x/b)^c / (1+(x/b)^c)^2   # check math

        d2fdxda
        d2fdxdb
        d2fdxdc
        """
        nx = len(XX)
        a, b, c = tuple(aa)
        temp = _np.power(XX/b, c)
        prof = flattop(XX, aa)
        dprofdx = deriv_flattop(XX, aa)

        dgdx = _np.zeros((3, nx), dtype=_np.float64)
        dgdx[0, :] = dprofdx / a
        dgdx[1, :] = prof * dprofdx * (XX/temp) * (c/a) * (temp-1.0) / (b*b)
        dgdx[2, :] = dprofdx/c
        dgdx[2, :] += dprofdx*_np.log(_np.abs(XX/b))
        dgdx[2, :] -= 2.0*(dprofdx**2.0)*(_np.log(_np.abs(XX/b))/prof)
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

#    @staticmethod
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        """
#        y = = a / (1 + (x/b)^c)
#        y-scaling:  y'=y/ys
#           y = ys*a' / (1 + (x/b')^c')
#           a = ys*a'
#           b = b'
#           c = c'
#        y-shifting:  y'=(y-yo)/ys
#           y = ys*a' / (1 + (x/b')^c')  + yo*(1 + (x/b')^c') / (1 + (x/b')^c')
#               can't make constant coefficients
#        x-scaling:  x'=x/xs
#           y = ys*a' / (1 + (x/(xs*b'))^c')
#               a = ys*a'
#               b = xs*b'
#               c = c'
#        x-shifting:  x'=(x-xo)/xs
#           y = ys*a' / (1 + ((x-xo)/(xs*b'))^c')
#               can't make constant coefficients
#        """
#        aout = _np.copy(ain)
#        aout[0] *= slope
#        aout[1] *= xslope
#        return aout

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelFlattop, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelFlattop


# ========================================================================== #
# ========================================================================== #


def massberg(XX, aa):
    return ModelMassberg._model(XX, aa)

def deriv_massberg(XX, aa):
    return ModelMassberg._deriv(XX, aa)

def partial_massberg(XX, aa):
    return ModelMassberg._partial(XX, aa)

def partial_deriv_massberg(XX, aa):
    return ModelMassberg._partial_deriv(XX, aa)

def model_massberg(XX, af, **kwargs):
    return _model(ModelMassberg, XX, af, **kwargs)

class ModelMassberg(ModelClass):
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
    _af = _np.asarray([1.0, 0.4, 5.0, 0.5], dtype=_np.float64)
    _LB = _np.asarray([0.0, 0.0, 1.0, -1], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, 1.0, _np.inf, 1], dtype=_np.float64)
    _fixed = _np.zeros( (4,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelMassberg, self).__init__(XX, af, **kwargs)
    # end def __init__

#    @staticmethod
#    def _model(XX, aa):
#        a, b, c, h = tuple(aa)
#        return a*(1.0-h*(XX/b)) / (1+_np.power(XX/b, c))

    @staticmethod
    def _model(XX, aa):
        a, b, c, h = tuple(aa)
        prft = flattop(XX, [a, b, c])
        temp = XX/b
        return prft * (1-h*temp)

    @staticmethod
    def _deriv(XX, aa):
        a, b, c, h = tuple(aa)
        prft = flattop(XX,[a, b, c])
        drft = deriv_flattop(XX, [a, b, c])

        temp = XX/b
        return drft*(1-h*temp) - prft*h/b

    @staticmethod
    def _partial(XX, aa):
        nx = len(XX)
        a, b, c, h = tuple(aa)
        prft = flattop(XX, [a, b, c])
        gft = partial_flattop(XX, [a, b, c])

        temp = XX/b
        prof = massberg(XX, aa)

        gvec = _np.zeros((4, nx), dtype=_np.float64)
        gvec[0, :] = prof / a
        gvec[1, :] = gft[1,:]*(1.0-h*temp) + prft*h*XX/_np.power(b, 2.0)
        gvec[2, :] = gft[2,:]*(1.0-h*temp)
        gvec[3, :] = (-1.0*XX / b)*prft
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa):
        nx = len(XX)
        a, b, c, h = tuple(aa)
    #    prof = massberg(XX, af)
        dprofdx = deriv_massberg(XX, aa)

        temp = XX/b

        prft = flattop(XX, [a, b, c])
        dflatdx = deriv_flattop(XX, [a, b, c])
        gft = partial_flattop(XX, [a, b, c])
        dgft = partial_deriv_flattop(XX, [a, b, c])

        dgdx = _np.zeros((4, nx), dtype= float)
        dgdx[0,:] = dprofdx / a
        dgdx[1,:] = dgft[1,:]*(1.0-h*temp) + dflatdx * h*XX/_np.power(b, 2.0)
        dgdx[1,:] += prft*h/_np.power(b, 2.0) - (h/b)*gft[1,:]
        dgdx[2,:] = dgft[2,:]*(1.0-h*temp) - gft[2, :]*h/b
        dgdx[3,:] = -1.0*(XX/b)*prft
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

#    @staticmethod
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        """
#        """
#        aout = _np.copy(ain)
#        return aout

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelMassberg, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelMassberg

# ========================================================================== #
# ========================================================================== #
# These two haven't been checked yet!!! also need to add analytic jacobian
# for the derivatives

def model_Heaviside(XX, af=None, **kwargs):
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
#    def _model(XX, aa):
#
#    @staticmethod
#    def _deriv(XX, aa):
#
#    @staticmethod
#    def _partial(XX, aa):
#
#    @staticmethod
#    def _partial_deriv(XX, aa):
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


def model_StepSeries(XX, af=None, **kwargs):
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
        # dfdx = 0.5*zz*sum_k(ak*sech(zz*(x-b^ak))^2)
        # dfdak = 0.5*(1 + tanh(zz*(x-b^ak))
        #                - 0.5*zz*ln(b)*b^ak*sech(zz*(x-b^ak))^2)

        # f    = a1*tanh(zz(x-x1))+a2*tanh(zz(x-x2))+...an*tanh(zz(x-xn))
        temp = _np.tanh(zz*(XX-bb**af[ii]))
        prof = prof + 0.5*af[ii]*(1 + temp)

        info.dprofdx = info.dprofdx+0.5*af[ii]*zz*(1 - temp**2)
        # info.dprofdx = info.dprofdx+0.5*af[ii]*zz*sech(zz*(XX-bb**af[ii]))**2

        gvec[ii, :] = (0.5*(1 + temp)
                       - 0.5*zz*_np.log(_np.abs(bb))*(bb**af[ii])*(1 - temp**2))

#        #indice of transitions
#        bx = _np.floor(1+bb/(XX[2]-XX[1]))
#        gvec[num_fit-1,ba-1:bx-1] = (zz*_np.log(_np.abs(bb))*(-bb)**af[num_fit-1]
#                            * sech(zz*(XX[ba-1:bx-1]-bb**af[num_fit-1]))**2
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
#    def _model(XX, aa):
#
#    @staticmethod
#    def _deriv(XX, aa):
#
#    @staticmethod
#    def _partial(XX, aa):
#
#    @staticmethod
#    def _partial_deriv(XX, aa):
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
       12 - Massberg profile         - f(x) ~ a * (1-h*(x/b)) / (1+(x/b)^c) = flattop*(1-h*(x/b))
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
    return Y0*_np.exp(-(tt-t0)/tau)

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
    offset = 0.0 if not hasattr(mod, '_secretoffset') else mod._secretoffset    # end if
    mod.update(XX, af)

    mod.modfunc = lambda _x, _a: mod.model(_x, _a) + offset
    mod.moddfunc = lambda _x, _a: mod.derivative(_x, _a)
    mod.modgfunc = lambda _x, _a: mod.jacobian(_x, _a)
    mod.moddgfunc = lambda _x, _a: mod.derivative_jacobian(_x, _a)
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

    Fs = 1.0
    XX = _np.linspace(-0.5, 0.5, num=int(1.5*10e-3*10.0e6/8))
    model_order = 2
    chi_eff, gvec, info = model_doppler(XX=XX, af=None, Fs=Fs, noshift=True, model_order=model_order)

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

# ========================================================================== #
# ========================================================================== #





# ================= #

#def _gaussian(XX, af, **kwargs):
#    af = _parse_gaussian_inputs(af, **kwargs)
#    return gaussian(XX, af)
#
#def _partial_gaussian(XX, af, **kwargs):
#    noshift = kwargs.setdefault('noshift',False)
#    af = _parse_gaussian_inputs(af, **kwargs)
#    gvec = partial_gaussian(XX, af)
#    if noshift:
#        gvec = _np.delete(gvec, (1), axis=0)
#    # end if
#    return gvec
#
#def _deriv_gaussian(XX, af, **kwargs):
#    af = _parse_gaussian_inputs(af, **kwargs)
#    return deriv_gaussian(XX, af)
#
#def _partial_deriv_gaussian(XX, af, **kwargs):
#    noshift = kwargs.setdefault('noshift',False)
#    af = _parse_gaussian_inputs(af, **kwargs)
#    gvec = partial_deriv_gaussian(XX, af)
#    if noshift:
#        gvec = _np.delete(gvec, (1), axis=0)
#    # end if
#    return gvec
#
#def model_gaussian(XX, af=None, **kwargs):
#    """
#    A gaussian with three free parameters:
#        XX - x - independent variable
#        af - magnitude, shift, width
#
#        af[0]*_np.exp(-(XX-af[1])**2/(2.0*af[2]**2))
#
#    If fitting with scaling, then the algebra necessary to unscale the problem
#    to original units is:
#        #    wrong af[0] = slope*af[0]
#        #    wrong offset = 0.0  (in practice, this is not included in this filt)
#
#        y-scaling:  y' = (y-yo)/ys
#         (y-miny)/(maxy-miny) = (y-yo)/ys
#         (y-yo)/ys = a' exp(-1*(x-b')^2/(2*c'^2))
#
#         y-yo = ys * a' * exp(-1*(x-b')^2/(2*c'^2))
#         a = ys*a'
#             not possible to shift and maintain constant coefficients unless yo=0.0
#
#        x-scaling: x' = (x-xo)/xs
#         y(x') - yo = a'*exp(-1*(x'-b')^2.0/(2*c'^2))
#                    = a'*exp(-1*(x/xs-xo/xs-b')^2.0/(2*c'^2))
#                    = a'*exp(-1*(x-xo-xs*b')^2.0/(2*c'^2*xs^2))
#         here:
#             a = a'*ys
#             b = b'*xs + xo
#             c = c'*xs
#        only works if yo = 0.0
#     found in the info Structure
#    """
#    noshift = kwargs.setdefault('noshift', False)
#    if af is None:
#        af = 0.7*_np.ones((3,), dtype=_np.float64)
#        if noshift:
#            af = _np.delete(af, (1), axis=0)
#        # end if
#    # endif
#
#    if len(af) == 2:
#        noshift = True
##        af = _np.atleast_1d(af)
##        af = _np.insert(af,1,0.0, axis=0)
#    # endif
#
#    info = Struct()
#    info.Lbounds = _np.array([-_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
#    info.Ubounds = _np.array([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
#    info.af = af
#    info.kwargs = kwargs
#
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        aout = _np.copy(ain)
# #        aout[0] = slope*aout[0]
# #        aout[1] = xslope*aout[1]+xoffset
# #        aout[2] = xslope*aout[2]
#        return aout
#    info.unscaleaf = unscaleaf
#    if noshift:
##        info.af = _np.delete(info.af, (1), axis=0)
#        info.Lbounds = _np.delete(info.Lbounds, (1), axis=0)
#        info.Ubounds = _np.delete(info.Ubounds, (1), axis=0)
#    # end if
#    if XX is None:
#        return info
#
#    prof = gaussian(XX, af, noshift=noshift)
#    gvec = partial_gaussian(XX, af, noshift=noshift)
#
#    info.prof = prof
#    info.gvec = gvec
#    info.dprofdx = deriv_gaussian(XX, af, noshift=noshift)
#    info.dgdx = partial_deriv_gaussian(XX, af, noshift=noshift)
#
#    info.func = lambda _x, _a: gaussian(_x, _a, noshift=noshift)
#    info.dfunc = lambda _x, _a: deriv_gaussian(_x, _a, noshift=noshift)
#    info.gfunc = lambda _x, _a: partial_gaussian(_x, _a, noshift=noshift)
#    info.dgfunc = lambda _x, _a: partial_deriv_gaussian(_x, _a, noshift=noshift)
#    return prof, gvec, info
#
#def _lorentzian(XX, af, **kwargs):
#    """
#    Lorentzian normalization such that integration equals AA (af[0])
#    """
#    af = _parse_gaussian_inputs(af, **kwargs)
#    AA, x0, ss = tuple(af)
#    return lorentzian(XX, [AA, x0, ss])
#
#def _deriv_lorentzian(XX, af, **kwargs):
#    af = _parse_gaussian_inputs(af, **kwargs)
#    AA, x0, ss = tuple(af)
#    return deriv_lorentzian(XX, [AA, x0, ss])
#
#def _partial_lorentzian(XX, af, **kwargs):
#    noshift = kwargs.setdefault('noshift', False)
#    af = _parse_gaussian_inputs(af, **kwargs)
#    AA, x0, ss = tuple(af)
#
#    gvec = partial_lorentzian(XX, [AA, x0, ss])
#    if noshift:
#        gvec = _np.delete(gvec, (1), axis=0)
#    # end if
#    return gvec
#
#def _partial_deriv_lorentzian(XX, af, **kwargs):
#    noshift = kwargs.setdefault('noshift', False)
#    af = _parse_gaussian_inputs(af, **kwargs)
#    AA, x0, ss = tuple(af)
#
#    gvec = partial_deriv_lorentzian(XX, [AA, x0, ss])
#    if noshift:
#        gvec = _np.delete(gvec, (1), axis=0)
#    # end if
#    return gvec
#
#def model_lorentzian(XX, af=None, **kwargs):
#    """
#    A lorentzian with three free parameters:
#        XX - x - independent variable
#        af - magnitude, shift, width
#
#        AA*0.5*ss/((XX-x0)^2+0.25*ss^2)/_np.pi
#
#    If fitting with scaling, then the algebra necessary to unscale the problem
#    to original units is:
#        af[0] = slope*af[0]
#        offset = 0.0  (in practice, this is necessary for this fit)
#
#       y-scaling: y' = (y-yo)/ys
#        (y-yo) / ys = 0.5*a'*c' / ( (x-b')^2 + 0.25*c'^2 ) / pi
#
#        y-yo = 0.5*ys*a'*c' / ( (x-b')^2 + 0.25*c'^2 ) / pi
#        a = ys*a'
#
#       x-scaling: x' = (x-xo)/xs
#        y-yo = 0.5*ys*a'*c' / ( (x-xo-xs*b')^2/xs^2 + 0.25*c'^2 ) / pi
#             = 0.5 * ys*xs*a'*xs*c' /( (x-xo-xs*b')^2 + 0.25*xs^2*c'^2 )/pi
#        a = ys*xs*a'
#        b = xs*b'+xo
#        c = xs*c'
#
#    found in the info Structure
#    """
#    noshift = kwargs.setdefault('noshift', False)
#    if af is None:
#        af = 0.7*_np.ones((3,), dtype=_np.float64)
#    # end if
#
#    if len(af) == 2:
#        noshift = True
#        af = _np.atleast_1d(_np.copy(af))
#        af = _np.insert(af,1,0.0, axis=0)
#    # end if
#
#    if len(af) == 3 and noshift:
#        af = _np.delete(_np.copy(af), (1), axis=0)
#    # endif
#
#    info = Struct()
#    info.Lbounds = _np.array([-_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
#    info.Ubounds = _np.array([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
#    info.af = _np.copy(af)
#
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):  # check this ... leaving noshift to user
#        aout = _np.copy(ain)
# #        aout[0] = slope*xslope*aout[0]
# #        aout[1] = xslope*aout[1]+xoffset
# #        aout[2] = xslope*aout[2]
#        return aout
#    info.unscaleaf = unscaleaf
#    if noshift:
##        info.af = _np.delete(info.af, (1), axis=0)
#        info.Lbounds = _np.delete(info.Lbounds, (1), axis=0)
#        info.Ubounds = _np.delete(info.Ubounds, (1), axis=0)
#    # end if
#
#    if XX is None:
#        return info
#
#    prof = lorentzian(XX, af, noshift=noshift)
#    gvec = partial_lorentzian(XX, af, noshift=noshift)
#    info.prof = prof
#    info.gvec = gvec
#    info.dprofdx = deriv_lorentzian(XX, af, noshift=noshift)
#    info.dgdx = partial_deriv_lorentzian(XX, af, noshift=noshift)
#
#    info.func = lambda _x, _a: lorentzian(_x, _a, noshift=noshift)
#    info.dfunc = lambda _x, _a: deriv_lorentzian(_x, _a, noshift=noshift)
#    info.gfunc = lambda _x, _a: partial_lorentzian(_x, _a, noshift=noshift)
#    info.dgfunc = lambda _x, _a: partial_deriv_lorentzian(_x, _a, noshift=noshift)
#    return prof, gvec, info
#
#
## =========================================================================== #
## =========================================================================== #
#
## fit model to the data
#def model_doppler(XX, af=None, noshift=True):
#    """
#         simple shifted gaussian - one doppler peak
#         3 parameter fit
#            modelspec.gaussian AA*_np.exp(-(XX-x0)**2/(2.0*ss**2))
#         shifted gaussian and centered lorentzian - one doppler peak and reflection
#         6 parameter fit
#       +    modelspec.lorentzian AA*0.5*ss/((XX-x0)*(XX-x0)+0.25*ss*ss)/_np.pi
#         two shifted gaussians and centered lorentzian - one doppler peak and reflection
#         9 parameter fit
#       +    modelspec.gaussian ... initialized with opposite shifts
#    """
#    if af is None:
#        af = _np.ones((3,), dtype=_np.float64)
##        af = _np.ones((9,), dtype=_np.float64)
#        af[0] = 1.0         # amplitude of gaussian
#        af[1] = -500.0e3    # doppler shift of gaussian
#        af[2] = 100.0e3     # doppler width of peak
##        af[3] = 10.0        # integrated value of lorentzian
##        #  af[4] is added here for simplicity.  In this model we are using an unshifted lorentzian at zero-frequency.  It will be deleted shortly.
##        af[4] = 0.0         # shift of lorentzian
##        af[5] = 10.0e3      # width of lorentzian peak
##        af[6] = 0.2*af[0]   # amplitude of second gaussian
##        af[7] =-1.0*_np.copy(af[1])    # doppler shift of second gaussian
##        af[8] = 50.0e3      # width of doppler peak of second gaussian
#
##        af = _np.delete(af, (4), axis=0)
##    elif len(af) == 8:
##        noshift = True
##        af = _np.insert(_np.copy(af), (4), 0.0, axis=0)
##    # end if
##    if XX is None: XX = _np.linspace(-2e6, 2e6, num=200)   # end if
#
#    def parse_noshift(ain):
#        if noshift:
#            a0 = ain[0:3]
#            a1 = ain[3:6]
#            a2 = ain[6:]
#        else:
#            a0 = ain[0:3]
#            a1 = ain[3:5]
#            a2 = ain[5:]
#        # end if
#        return a0, a1, a2
#    a0, a1, a2 = parse_noshift(af)
#
#    i0 = model_gaussian(None, a0)
#    info = Struct()
#    info.af = _np.copy(i0.af)
#    info.Lbounds = _np.copy(i0.Lbounds)
#    info.Ubounds = _np.copy(i0.Ubounds)
#
#    i1 = model_lorentzian(None, a1, noshift=noshift)
#    info.af = _np.append(info.af, i1.af, axis=0)
#    info.Lbounds = _np.append(info.Lbounds, i1.Lbounds, axis=0)
#    info.Ubounds = _np.append(info.Ubounds, i1.Ubounds, axis=0)
#
#    i2 = model_gaussian(None, a2)
#    info.af = _np.append(info.af, i2.af, axis=0)
#    info.Lbounds = _np.append(info.Lbounds, i2.Lbounds, axis=0)
#    info.Ubounds = _np.append(info.Ubounds, i2.Ubounds, axis=0)
#
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        a0, a1, a2 = parse_noshift(ain)
#        aout = _np.copy(ain)
# #        aout = i0.unscaleaf(a0, slope, offset, xslope, xoffset)
# #        aout = _np.append(aout, i1.unscaleaf(a1, slope, offset, xslope, xoffset), axis=0)
# #        aout = _np.append(aout, i2.unscaleaf(a2, slope, offset, xslope, xoffset), axis=0)
#        return aout
#    info.unscaleaf = unscaleaf
#
#    if XX is None:
#        return info
#    # end if
#
##    # shifted gaussian
###    model, gvec, info = model_gaussian(XX, a0)
##    model = gaussian(XX, a0)
##    info.dprofdx = deriv_gaussian(XX, a0)
##    gvec = partial_gaussian(XX, a0)
##    info.dgdx = partial_deriv_gaussian(XX, a0)
##
##    # shifted gaussian with lorentzian
###    m1, g1, i1 = model_lorentzian(XX, a1, noshift=noshift)
##    model += lorentzian(XX, a1, noshift=noshift)
##    info.dprofdx += deriv_lorentzian(XX, a1, noshift=noshift)
##    gvec = _np.append(gvec, partial_lorentzian(XX, a1, noshift=noshift), axis=0)
##    info.dgdx = _np.append(info.dgdx, partial_deriv_lorentzian(XX, a1, noshift=noshift), axis=0)
##
##    # two shifted gaussians with lorentzian
###    m2, g2, i2 = model_gaussian(XX, a2)
##    model += gaussian(XX, a2)
##    info.dprofdx += deriv_gaussian(XX, a2)
##    gvec = _np.append(gvec, partial_gaussian(XX, a2), axis=0)
##    info.dgdx = _np.append(info.dgdx, partial_deriv_gaussian(XX, a2), axis=0)
#
#    # ===== #
#
#    def dopplermodel(xdat, ain):
#        a0, a1, a2 = parse_noshift(ain)
#        return gaussian(xdat,af=a0) \
#             + lorentzian(xdat, af=a1, noshift=noshift) \
#             + gaussian(xdat,af=a2)
#
#    def deriv_model(xdat, ain):
#        a0, a1, a2 = parse_noshift(ain)
#        return deriv_gaussian(xdat,af=a0) \
#             + deriv_lorentzian(xdat, af=a1, noshift=noshift) \
#             + deriv_gaussian(xdat,af=a2)
#
#    def partial_model(xdat, ain):
#        a0, a1, a2 = parse_noshift(ain)
#        gvec = partial_gaussian(xdat,af=a0)
#        gvec = _np.append(gvec, partial_lorentzian(xdat, af=a1, noshift=noshift), axis=0)
#        if noshift:
#            gvec = _np.insert(gvec, (4), _np.zeros((len(xdat),), dtype=_np.float64), axis=0)
#        gvec = _np.append(gvec, partial_gaussian(xdat,af=a2), axis=0)
#        return gvec
#
#    def partial_deriv_model(xdat, ain):
#        a0, a1, a2 = parse_noshift(ain)
#        gvec = partial_deriv_gaussian(xdat,af=a0)
#        gvec = _np.append(gvec, partial_deriv_lorentzian(xdat, af=a1, noshift=noshift), axis=0)
#        if noshift:
#            gvec = _np.insert(gvec, (4), _np.zeros((len(xdat),), dtype=_np.float64), axis=0)
#        gvec = _np.append(gvec, partial_deriv_gaussian(xdat,af=a2), axis=0)
#        return gvec
#
#    model = dopplermodel(XX, af)
#    gvec = partial_model(XX, af)
#    info.prof = model
#    info.gvec = gvec
#    info.dprofdx = deriv_model(XX, af)
#    info.dgdx = partial_deriv_model(XX, af)
#
#    info.func = lambda _x, _a: dopplermodel(_x, _a)
#    info.dfunc = lambda _x, _a: deriv_model(_x, _a)
#    info.gfunc = lambda _x, _a: partial_model(_x, _a)
#    info.dgfunc = lambda _x, _a: partial_deriv_model(_x, _a)
#    return model, gvec, info
## end def
#
## =========================================================================== #
#
## Set the plasma density, temperature, and Zeff profiles (TRAVIS INPUTS)
#def qparab(XX, *aa, **kwargs):
#    """
#    ex// ne_parms = [0.30, 0.002, 2.0, 0.7, -0.24, 0.30]
#    This subfunction calculates the quasi-parabolic fit
#    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
#        XX - r/a
#    aa[0] - Y0 - function value on-axis
#    aa[1] - gg - Y1/Y0 - function value at edge over core
#    aa[2],aa[3]-  pp, qq - power scaling parameters
#    aa[4],aa[5]-  hh, ww - hole depth and width
#    """
#    options = {}
#    options.update(kwargs)
#    nohollow = options.get('nohollow', False)
#    aedge = options.get('aedge', 1.0)
#    if len(aa)>6:
#        nohollow = aa.pop(6)
#    XX = _np.abs(XX)/aedge
#    if (type(aa) is tuple) and (len(aa) == 2):
#        nohollow = aa[1]
#        aa = aa[0]
#    elif (type(aa) is tuple) and (len(aa) == 1):
#        aa = aa[0]
#    # endif
#    aa = _np.asarray(aa, dtype=_np.float64)
#    if nohollow and (_np.size(aa)==4):
#        aa = _np.vstack((aa,_np.atleast_1d(0.0)))
#        aa = _np.vstack((aa,_np.atleast_1d(1.0)))
#    elif nohollow:
#        aa[4] = 0.0
#        aa[5] = 1.0
#    # endif
#    prof = aa[0]*( aa[1]-aa[4]
#                   + (1.0-aa[1]+aa[4])*_np.abs(1.0-XX**aa[2])**aa[3]
#                   + aa[4]*(1.0-_np.exp(-XX**2.0/aa[5]**2.0)) )
#    return prof
## end def qparab
#
#def deriv_qparab(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
#    """
#    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
#    This subfunction calculates the derivative of a quasi-parabolic fit
#    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
#        XX - r/a
#    aa[0] - Y0 - function value on-axis
#    aa[1] - gg - Y1/Y0 - function value at edge over core
#    aa[2],aa[3]-  pp, qq - power scaling parameters
#    aa[4],aa[5]-  hh, ww - hole depth and width
#    """
#    XX = _np.abs(XX)
#    aa = _np.asarray(aa,dtype=_np.float64)
#    if nohollow and (_np.size(aa)==4):
#        aa = _np.vstack((aa,_np.atleast_1d(0.0)))
#        aa = _np.vstack((aa,_np.atleast_1d(1.0)))
#    elif nohollow:
#        aa[4] = 0.0
#        aa[5] = 1.0
#    # endif
#
#    dpdx = aa[0]*( (1.0-aa[1]+aa[4])*(-1.0*aa[2]*XX**(aa[2]-1.0))*aa[3]*(1.0-XX**aa[2])**(aa[3]-1.0)
#                   - aa[4]*(-2.0*XX/aa[5]**2.0)*_np.exp(-XX**2.0/aa[5]**2.0) )
#
#    return dpdx
## end def derive_qparab
#
#def partial_qparab(XX,aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
#    """
#    ex// ne_parms = [0.30, 0.002 2.0, 0.7 -0.24 0.30]
#    This subfunction calculates the jacobian of a quasi-parabolic fit
#
#    quasi-parabolic fit:
#    Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
#        XX - r/a
#    aa[0] - Y0 - function value on-axis
#    aa[1] - gg - Y1/Y0 - function value at edge over core
#    aa[2],aa[3]-  pp, qq - power scaling parameters
#    aa[4],aa[5]-  hh, ww - hole depth and width
#    """
#    XX = _np.abs(XX)
#    aa = _np.asarray(aa,dtype=_np.float64)
#    if nohollow and (_np.size(aa)==4):
#        aa = _np.vstack((aa,_np.atleast_1d(0.0)))
#        aa = _np.vstack((aa,_np.atleast_1d(1.0)))
#    elif nohollow:
#        aa[4] = 0.0
#        aa[5] = 1.0
#    # endif
#    Y0 = aa[0]
#    g = aa[1]
#    p = aa[2]
#    q = aa[3]
#    h = aa[4]
#    w = aa[5]
#
#    gvec = _np.zeros( (6,_np.size(XX)), dtype=_np.float64)
#    gvec[0,:] = g-h+(1.0-g+h)*_np.abs(1.0-XX**p)**q + h*(1.0-_np.exp(-XX**2.0/w**2.0))
#    gvec[1,:] = Y0*( 1.0-_np.abs(1.0-XX**p)**q )    # aa[0]
#    gvec[2,:] = -1.0*Y0*q*(g-h-1.0)*(XX**p)*_np.log(_np.abs(XX))*(_np.abs(1-XX**p)**q)/(XX**p-1.0)
#    gvec[3,:] = Y0*(-g+h+1.0)*(_np.abs(1-XX**p)**q)*_np.log(_np.abs(1.0-XX**p))
#    gvec[4,:] = Y0*(_np.abs(1.0-XX**p)**q) - Y0*_np.exp(-(XX/w)**2.0)
#    gvec[5,:] = -2.0*h*(XX**2.0)*Y0*_np.exp(-(XX/w)**2.0) / w**3.0
#
#    return gvec
## end def partial_qparab
#
#def partial_deriv_qparab(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
#    """
#    ex// ne_parms = [0.30, 0.002 2.0 0.7 -0.24 0.30]
#    This subfunction calculates the jacobian of the derivative of a
#    quasi-parabolic fit (partial derivatives of the derivative of a quasi-parabolic fit)
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
#    XX = _np.abs(XX)
#    aa = _np.asarray(aa,dtype=_np.float64)
#    if nohollow and (_np.size(aa)==4):
#        aa = _np.vstack((aa,_np.atleast_1d(0.0)))
#        aa = _np.vstack((aa,_np.atleast_1d(1.0)))
#    elif nohollow:
#        aa[4] = 0.0
#        aa[5] = 1.0
#    # endif
#    Y0 = aa[0]
#    g = aa[1]
#    p = aa[2]
#    q = aa[3]
#    h = aa[4]
#    w = aa[5]
#
#    gvec = _np.zeros( (6,_np.size(XX)), dtype=_np.float64)
#    gvec[0,:] = 2.0*h*XX*_np.exp(-(XX/w)**2.0)/(w**2.0) - p*q*(-g+h+1.0)*(XX**(p-1.0))*((1.0-XX**p)**(q-1.0))
#    gvec[1,:] = p*q*Y0*(XX**(p-1.0))*((1-XX**p)**(q-1.0))
#
#    gvec[2,:] = q*Y0*(-1.0*(g-h-1.0))*((1.0-XX**p)**(q-2.0))
#    gvec[2,:] *= p*_np.log(_np.abs(XX))*((q-1.0)*(XX**(2.0*p-1.0))
#                    - (XX**(p-1.0))*(1.0-XX**p))+(XX**p-1.0)*XX**(p-1.0)
#
#    gvec[3,:] = p*Y0*(g-h-1.0)*(XX**(p-1.0))*((1.0-XX**p)**(q-1.0))*(q*_np.log(_np.abs(1.0-XX**p))+1.0)
#    gvec[4,:] = (2.0*XX*Y0*_np.exp(-1.0*(XX/w)**2.0))/(w**2.0) - p*q*Y0*(XX**(p-1.0))*((1.0-XX**p)**(q-1.0))
#
#    gvec[5,:] = h*Y0*_np.exp(-1.0*(XX/w)**2.0)*((4.0*(XX**3.0))/(w**5.0)-(4.0*XX)/(w**3.0))
#
#    return gvec
## end def partial_deriv_qparab
#
#def deriv2_qparab(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
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
#    XX = _np.abs(XX)
#    aa = _np.asarray(aa,dtype=_np.float64)
#    if nohollow and (_np.size(aa)==4):
#        aa = _np.vstack((aa,_np.atleast_1d(0.0)))
#        aa = _np.vstack((aa,_np.atleast_1d(1.0)))
#    elif nohollow:
#        aa[4] = 0.0
#        aa[5] = 1.0
#    # endif
#    d2pdx2 = aa[3]*(aa[2]**2.0)*(aa[3]-1.0)*(1.0+aa[4]-aa[1])*(XX**(2.*aa[2]-2.0))*(1-XX**aa[2])**(aa[3]-2.0)
#    d2pdx2 -= (aa[2]-1.0)*aa[2]*aa[3]*(1.0+aa[4]-aa[1])*(XX**(aa[2]-2.0))*(1-XX**aa[2])**(aa[3]-1.0)
#    d2pdx2 += (2.0*aa[4]*_np.exp(-XX**2.0/(aa[5]**2.0)))/(aa[5]**2.0)
#    d2pdx2 -= (4*aa[4]*(XX**2.0)*_np.exp(-XX**2.0/(aa[5]**2.0)))/(aa[5]**4.0)
#    d2pdx2 *= aa[0]
#    return d2pdx2
## end def derive_qparab
#
#def partial_deriv2_qparab(XX, aa=[0.30, 0.002, 2.0, 0.7, -0.24, 0.30], nohollow=False):
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
#    XX = _np.abs(XX)
#    aa = _np.asarray(aa,dtype=_np.float64)
#    if nohollow and (_np.size(aa)==4):
#        aa = _np.vstack((aa,_np.atleast_1d(0.0)))
#        aa = _np.vstack((aa,_np.atleast_1d(1.0)))
#    elif nohollow:
#        aa[4] = 0.0
#        aa[5] = 1.0
#    # endif
#    Y0 = aa[0]
#    g = aa[1]
#    p = aa[2]
#    q = aa[3]
#    h = aa[4]
#    w = aa[5]
#
#    gvec = _np.zeros( (6,_np.size(XX)), dtype=_np.float64)
#    gvec[0,:] = deriv2_qparab(XX, aa, nohollow) / Y0
#    gvec[1,:] = -p*q*Y0*(XX**(p-2.0))*(1.0-XX**p)**(q-2.0)*(p*(q*(XX**p)-1.0)-XX**p+1.0)
#    gvec[2,:] = p*_np.log(_np.abs(XX))*(p*((q**2.0)*(XX**(2.0*p))-3.0*q*(XX**p)+XX**p+1.0)-(XX**p-1.0)*(q*XX**p-1.0))
#    gvec[2,:] += (XX**p-1.0)*(2.0*p*(q*(XX**p)-1.0)-XX**p+1.0)
#    gvec[2,:] *= q*Y0*(g-h-1.0)*(XX**(p-2.0))*((1.0-XX**p)**(q-3.0))
#    gvec[3,:] = p*Y0*(-(g-h-1.0))*(XX**(p-2.0))*((1.0-XX**p)**(q-2.0))*(p*(2.0*q*XX**p-1.0)+q*(p*(q*XX**p-1.0)-XX**p+1.0)*_np.log(_np.abs(1.0-XX**p))-XX**p+1.0)
#    gvec[4,:] = Y0*(p*q*(XX**(p-2.0))*((1.0-XX**p)**(q-2.0))*(p*(q*XX**p-1.0)-XX**p+1.0)+(2.0*_np.exp(-XX**2.0/w**2.0)*(w**2.0-2.0*XX**2.0))/w**4.0)
#    gvec[5,:] = -(4.0*h*Y0*_np.exp(-XX**2.0/w**2.0)*(w**4.0-5*w**2.0*XX**2.0+2.0*XX**4.0))/w**7.0
#
#    return gvec
## end def partial_deriv2_qparab
#
#
#def model_qparab(XX, af=None, nohollow=False, prune=False, rescale=False, info=None):
#    """
#    ex// ne_parms = [0.30, 0.002, 2.0, 0.7, -0.24, 0.30]
#    This function calculates the quasi-parabolic fit
#        y/a = b-e+(1-b+e)*(1-x^c)^d + e*(1-exp(-x^2/f^2))
#        y/a = edgepower(x,[b-e,c,d]) + expedge(x,[e,f])
#            where
#
#    Y/Y0 = af[1]-af[4]+(1-af[1]+af[4])*(1-XX^af[2])^af[3]
#                + af[4]*(1-exp(-XX^2/af[5]^2))
#    XX - r/a
#a    af[0] - Y0 - function value on-axis
#b    af[1] - gg - Y1/Y0 - function value at edge over core
#c,d    af[2],af[3]-  pp, qq - power scaling parameters
#e,f    af[4],af[5]-  hh, ww - hole depth and width
#
#    If fitting with scaling, then the algebra necessary to unscale the problem
#    to original units is:
#            af[0] is Y0, af[1] if Y1/Y0; Y1 = af[1]*af[0]
#
#        af[1] = (slope*af[1]*af[0]+offset)/(slope*af[0]+offset)
#        af[0] = slope*af[0]+offset
#    found in the info Structure
#
#    """
#    if info is None:
#        info = Struct()  # Custom class that makes working with dictionaries easier
##    info.Lbounds = _np.array([    0.0, 0.0,-_np.inf,-_np.inf,-_np.inf,-_np.inf], dtype=_np.float64)
##    info.Ubounds = _np.array([_np.inf, _np.inf, _np.inf, _np.inf, _np.inf, _np.inf], dtype=_np.float64)
##    info.Lbounds = _np.array([  0.0, 0.0,-20,-20,-1,-1], dtype=_np.float64)
##    info.Ubounds = _np.array([ 20.0, 1.0, 20, 20, 1, 1], dtype=_np.float64)
#    info.Lbounds = _np.array([  0.0, 0.0,-10,-10,-1,-1], dtype=_np.float64)
#    info.Ubounds = _np.array([ 20.0, 1.0, 10, 10, 1, 1], dtype=_np.float64)
#
#    if af is None:
#        af = _np.array([1.0, 0.002, 2.0, 0.7, -0.24, 0.30], dtype=_np.float64)
##        af = randomize_initial_conditions(info.Lbounds, info.Ubounds)
#        if nohollow:
#            af[4] = 0.0
#            af[5] = 1.0
#        # endif
#    if len(af) == 4:
#        nohollow = True
#        af = _np.hstack((af,0.0))
#        af = _np.hstack((af,1.0))
#    # endif
#    info.af = _np.copy(af)
#
#    def unscaleaf(ain, slope, offset):
#        aout = _np.copy(ain)
# #        aout[1] = (slope*ain[1]*ain[0]+offset)/(slope*ain[0]+offset)
# #        aout[0] = slope*ain[0]+offset
#        return aout
#    info.unscaleaf = unscaleaf
#    if XX is None:
#        if prune:
#            info.af = info.af[:4]
#            info.Lbounds = info.Lbounds[:4]
#            info.Ubounds = info.Ubounds[:4]
#        # end if
#        return info
#    # endif
#
#    # ========= #
#    XX = _np.abs(XX)
#    if rescale:
#        XX = rescale_xlims(XX, forward=True, ascl=rescale)
#    else:
#        rescale = 1.0
#    # end if
#
#    # ========= #
#
#    af = af.reshape((len(af),))
#    if _np.isfinite(af).any() == 0:
#        print("checkit! No finite values in fit coefficients! (from model_spec: model_qparab)")
##    print(_np.shape(af))
#
#    try:
#        prof = qparab(XX, af, nohollow)
##        prof = interp_irregularities(prof, corezero=False)
#        info.prof = prof
#
#        gvec = partial_qparab(XX*rescale, af, nohollow)
##        gvec = interp_irregularities(gvec, corezero=False)  # invalid slice
#        info.gvec = gvec
#
#        info.dprofdx = deriv_qparab(XX, af, nohollow)
##        info.dprofdx = interp_irregularities(info.dprofdx, corezero=True)
#
#        info.dgdx = partial_deriv_qparab(XX*rescale, af, nohollow)
##        info.dgdx = interp_irregularities(info.dgdx, corezero=False)
#    except:
#        pass
#        raise
#
#    if prune:
#        af = af[:4]
#        info.Lbounds = info.Lbounds[:4]
#        info.Ubounds = info.Ubounds[:4]
#
#        gvec = gvec[:4, :]
#        info.dgdx = info.dgdx[:4, :]
#    # endif
#    info.af = af
#
#    return prof, gvec, info
## end def model_qparab

# ========================================================================== #
# ========================================================================== #