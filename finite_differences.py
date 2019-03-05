# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:09:10 2019

@author: gawe
"""

# ========================================================================== #
# ========================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

# ========================================================================== #
# ========================================================================== #

import numpy as _np
from pybaseutils.Struct import Struct
from pybaseutils import utils as _ut

# ========================================================================== #
# ========================================================================== #


class FD(Struct):
    """
    Designed for analytic functions

    model is a function that takes the form:

    self._model(XX, aa=None, **kwargs)
    """
    machine_epsilon = (_np.finfo(_np.float64).eps)

    def __init__(self, model, **kwargs):
        self._model = model
        self.__dict__.update(kwargs)
        self.kwargs = kwargs
    # end def

    def parse_in(self, XX, aa, **kwargs):
        if aa is None: aa = _np.copy(self.af)  # end if
        if XX is None: XX = _np.copy(self.XX)  # end if
        XX = _np.copy(XX)
        aa = _np.copy(aa)
        if len(kwargs.keys()) == 0:
            kwargs = self.kwargs
        return XX, aa, kwargs

    def __step_size(self, model, XX, aa, deriv_order=1, **kwargs):
        if 'hh' not in kwargs:
            tst = model(XX, aa, **kwargs)

#            xsh = _np.shape(XX)
#            tsh = _np.shape(tst)
#            if xsh[0] == tsh[0]:
#                tst = _np.diff(tst, axis=0)/_np.diff(XX)
#            else:  #xsh[0] == tsh[1]:
            if 1:
                tst = _np.diff(tst, axis=1)/_np.diff(XX)
#            tst = model(XX, aa, **kwargs)
            dmax = _np.max((tst.max()/0.2, 5.0))

            dmax = _np.max((_np.max(_np.abs(aa))/0.2, dmax))   # maximum derivative, arbitraily say 10/0.2
            hh = _ut.cbrt(6.0*self.machine_epsilon/dmax)

            if deriv_order == 2:
                hh = _np.power(hh, 0.5)
            elif deriv_order == 3:
                hh = _np.power(hh, 1.0/3.0)
            # end if
            kwargs.setdefault('hh', hh)
        # end if
        return hh, kwargs

    def derivative(self, XX, aa=None, **kwargs):
        """
        Returns the derivative of the model.
            f(x) = f(x, a, b, c, ...)
            out: dfdx

        Leave it to the user to define the method '_deriv' with an analytic derivative.
        otherwise, calculate the derivative numerically by finite-differencing the model.
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)
        hh, kwargs = self.__step_size(self.model, XX, aa, deriv_order=1, **kwargs)

        # check the model for boundary conditions (non-negative, etc.)
        kwargs.setdefault('order', 6)

#        if hasattr(self, 'badleftboundary') and self.badleftboundary:
#            # Use

        return self._1stderiv_fd(self.model, XX=XX, aa=aa, **kwargs)
        # end if
    # end def derivative

    # ============================================================= #

    def second_derivative(self, XX, aa=None, **kwargs):
        """
        Returns the 2nd derivative of the model.
            f(x) = f(x, a, b, c, ...)
            out: d2fdx2

        Leave it to the user to define the method '_deriv2' with an analytic 2nd derivative.
        otherwise, calculate the derivative numerically by finite-differencing the model.
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)
        hh, kwargs = self.__step_size(self.model, XX, aa, deriv_order=2, **kwargs)

        # 2nd order accurate will not pass standard numpy assertion test: atol=1e-5, rtol=1e-8
        # 4th order accurate passes standard numpy assertion test: atol=1e-5, rtol=1e-8
        # 6th order and 8th order also passes
        kwargs.setdefault('order', 8)
        return self._2ndderiv_fd(self.model, XX=XX, aa=aa, **kwargs)

    # ============================================================= #

    def jacobian(self, XX, aa=None, **kwargs):
        """
        Returns the jacobian of the model.
            f(x) = f(x, a, b, c)
            out: [dfda(x), dfdb(x), dfdc(x), ...]

        Leave it to the user to define the method '_partial' with an analytic Jacobian.
        Otherwise, calculate the jacobian numerically by finite-differencing the model.
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)

        _, kwargs = FD.__step_size(self.model, XX, aa, deriv_order=1, **kwargs)
        kwargs['hh'] = kwargs['hh']*_np.ones_like(aa)
        kwargs.setdefault('order', 6)

        def func(ai, xi, **kwargs):
            return self.model(xi, ai, **kwargs)

        numfit = _np.size(aa)
        gvec = _np.zeros((numfit, _np.size(XX)), dtype=_np.float64)
        for ii in range(numfit):
            msk = _np.zeros(aa.shape, dtype=bool)
            msk[ii] = True
            gvec[ii, :] = self._1stderiv_fd(func, XX=aa, aa=XX, msk=msk, **kwargs)
#        # end for
        return gvec

    # ============================================================= #

    def derivative_jacobian(self, XX, aa=None, **kwargs):
        """
        Returns the jacobian of the derivative of the model.
            f(x) = f(x, a, b, c)
            out: [d2fdxda(x), d2fdxdb(x), d2fdxdc(x), ...]

        Leave it to the user to define the method '_partial_deriv' with an analytic Jacobian of the derivative.
        Otherwise, calculate the jacobian of the derivative numerically by finite-differencing the model derivative.
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)

        hx, _ = self.__step_size(self.model, XX, aa, deriv_order=1, **kwargs)
        kwargs.setdefault('hx', hx)
        ha, _ = self.__step_size(self.model, XX, aa, deriv_order=1, **kwargs)
        kwargs.setdefault('ha', ha*_np.ones_like(aa))

        order = kwargs.setdefault('order', 4)
        if order<5:
            numfit = _np.size(aa)
            dgdx = _np.zeros((numfit, _np.size(XX)), dtype=_np.float64)
            for ii in range(numfit):
                msk = _np.zeros(aa.shape, dtype=bool)
                msk[ii] = True
                dgdx[ii, :] = self._2ndderiv_mixedfd(self.model, XX=XX, aa=aa, msk=msk, **kwargs)
            # end for
        else:
            # Numerically unstable to take derivative of a numerical derivative
            # this is only for testing!
            # use reversal of differentiation orders to do it on one-step here
            #    based on the jacobian (if numerical, this degrades accuracy)
            # for simple models (line/sine):
            #    passes at 4th order for rtol, atol = (1e-5, 1e-4)
            #    passes at 6th order for rtol, atol = (1e-4, 1e-4)
            #    passes at 8th order for rtol, atol = (1e-5, 1e-4),
            dgdx = self._1stderiv_fd(self.jacobian, XX=XX, hh=hx, aa=aa, **kwargs)
        return dgdx

    # ============================================================= #

    def second_derivative_jacobian(self, XX, aa=None, **kwargs):
        """
        Returns the jacobian of the second derivative of the model.
            f(x) = f(x, a, b, c)
            out: [d3fdx2da(x), d3fdx2db(x), d3fdx2dc(x), ...]

        Leave it to the user to define the method '_partial_deriv2' with an analytic Jacobian of the 2nd derivative.
        Otherwise, calculate the jacobian of the 2nd derivative numerically by finite-differencing the model derivative.
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)
        if hasattr(self, '_partial_deriv2') and self.analytic:
            return self._partial_deriv2(XX, aa, **kwargs)

        hx, _ = self.__step_size(self.model, XX, aa, deriv_order=2, **kwargs)
        kwargs.setdefault('hx', hx)
        ha, _ = self.__step_size(self.model, XX, aa, deriv_order=1, **kwargs)
        kwargs.setdefault('ha', ha*_np.ones_like(aa))

        order = kwargs.setdefault('order', 2)

        if order<2:
            numfit = _np.size(aa)
            d2gdx2 = _np.zeros((numfit, _np.size(XX)), dtype=_np.float64)
            for ii in range(numfit):
                msk = _np.zeros(aa.shape, dtype=bool)
                msk[ii] = True
                d2gdx2[ii, :] = self._3rdderiv_mixedfd(self.model, XX=XX, aa=aa, msk=msk, **kwargs)
    #        # end for
        else:
            # Numerically unstable to take 2nd derivative of a numerical derivative
            # this is only for testing!
            if 'hh' not in kwargs:
                hh, _ = self.__step_size(self.derivative_jacobian, XX, aa, deriv_order=2, **kwargs)
                kwargs.setdefault('hx', hx)
            # end if
            d2gdx2 = self._2ndderiv_fd(self.jacobian, XX=XX, aa=aa, **kwargs)
        return d2gdx2

    # ============================================================= #

    def hessian(self, XX, aa=None, **kwargs):
        """
        Returns the hessian of the model
            f(x) = f(x, a, b, c)
            out: [[d2fda2(x), d2fdbda(x), d2fdcda(x), ...]
                  [d2fdadb(x), d2fdb2(x), d2fdcdb(x), ...]
                  [               ...                    ] ]

        Leave it to the user to define the method '_hessian' with an analytic Hessian.
        Otherwise, calculate the hessian numerically by finite-differencing the jacobian.
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)

        hh, _ = self.__step_size(self.jacobian, XX, aa, deriv_order=1, **kwargs)
        kwargs.setdefault('hh', hh*_np.ones_like(hh))

#        hx, _ = self.__step_size(self.jacobian, XX, aa, deriv_order=1, **kwargs)
#        ha, _ = self.__step_size(self.jacobian, XX, aa, deriv_order=2, **kwargs)
#        kwargs.setdefault('hx', hx)
#        kwargs.setdefault('ha', ha)
#
#        def func(ai, xi, **kwargs):
#            return self.model(xi, ai, **kwargs)

        numfit = _np.size(aa)
        hess = _np.zeros((numfit, numfit, _np.size(XX)), dtype=_np.float64)
        for ii in range(numfit):
            tmp = _np.zeros_like(aa)
            tmp[ii] = 1.0
            hess[ii, :, :] = (self.jacobian(XX, aa + hh*tmp, **kwargs) - self.jacobian(XX, aa-hh*tmp, **kwargs))/(2*hh[ii]) # TODO!: use second derivative of model wrt a
        # end for
        return hess
    # end def

    # ============================================================= #
    # ============================================================= #

    def centered_coeffs(self, deriv_order, order):

        if deriv_order == 1:
            if order<3:
                num_coeffs = [-1.0, 0.0, 1.0]
                den_coeffs = 2.0
            elif order<5:
                num_coeffs = [1.0, -8.0, 0.0, 8.0, -1.0]
                den_coeffs = 12.0
            elif order<7:
                num_coeffs = [-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0 ]
                den_coeffs = 60.0
            elif order<9:
                num_coeffs = [3.0, -32.0, 168.0, -672.0, 0.0, 672.0, -168.0, 32.0, -3.0]
                den_coeffs = 840.0
            # end if
        elif deriv_order == 2:
            if order<3:
                num_coeffs = [ 1.0, -2.0, 1.0]
                den_coeffs = 1.0
            elif order<5:
                num_coeffs = [-1.0, 16.0, -30.0, 16.0, -1.0]
                den_coeffs = 12.0
            elif order<7:
                num_coeffs = [2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]
                den_coeffs = 180.0
            elif order<9:
                num_coeffs = [-1.0/560.0, 8.0/315.0, -1.0/5.0, 8.0/5.0, -205.0/72.0, 8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0]
                den_coeffs = 1.0
#                num_coeffs = [-9.0, 128.0, -1008.0, 8064.0, -14350.0, 8064.0, -1008.0, 128.0, -9.0]
#                den_coeffs = 5040.0
            # end if
        elif deriv_order == 3:
            if order<3:
                num_coeffs = [-1.0, 2.0, 0.0, -2.0, 1.0]
                den_coeffs = 2.0
            elif order<5:
                num_coeffs = [1.0, -8.0, 13.0, 0.0, -13.0, 8.0, -1.0]
                den_coeffs = 8.0
            # end if
        # end if
        return num_coeffs, den_coeffs

    @staticmethod
    def parse_unmixed_inputs(XX, hh, aa, order, msk):
        if XX is None or hh is None or aa is None:
            raise Exception('improper inputs to finite difference function')
        if msk is None:
            msk = _np.ones_like(XX)
        msk = _np.asarray(msk, dtype=bool)
        tmp = _np.zeros_like(XX)
        tmp[msk] = 1.0

        hh = _np.atleast_1d(hh)
        if len(hh)>1:
            hs = _np.copy(hh[msk])
        else:
            hs = _np.copy(hh)
            hh = hh*_np.ones_like(XX)
        # end if
        return tmp, msk, hh, hs

    @staticmethod
    def centered_fd(func, XX=None, hh=None, aa=None, deriv_order, order=2, msk=None, **kwargs):
        tmp, msk, hh, hs = FD.parse_unmixed_inputs(XX, hh, aa, order, msk)

        num_coeffs, den_coeffs = FD.centered_coeffs(deriv_order=1, order=order)

        ncoeffs = len(num_coeffs)
        nstencil = ncoeffs // 2
        for ii in range(ncoeffs):
            jj = -1.0*nstencil + ii

            ysum = ncoeffs[ii]*func(XX-jj*hh*tmp, aa, **kwargs)
        # end for
        return ysum / (den_coeffs*_np.power(hs, deriv_order))

    @staticmethod
    def centered_1stderiv(func, XX=None, hh=None, aa=None, order=2, msk=None, **kwargs):
        return centered_fd(func, XX, hh, aa, deriv_order=1, order=order, msk=msk, **kwargs)

#        if order<3:
#            # 2nd order accurate
#            return ( 0.0  # -1/2, 0, +1/2
#            - 1.0*func(XX-1.0*hh*tmp, aa, **kwargs)
#            + 1.0*func(XX+1.0*hh*tmp, aa, **kwargs))/(2.0*hs)
#        elif order<5:
#            # 4th order accurate
#            return ( 0.0 # +1/12, -2/3, 0, +2/3, -1/12
#            + 1.0*func(XX-2.0*hh*tmp, aa, **kwargs)
#            - 8.0*func(XX-1.0*hh*tmp, aa, **kwargs)
#            + 8.0*func(XX+1.0*hh*tmp, aa, **kwargs)
#            - 1.0*func(XX+2.0*hh*tmp, aa, **kwargs))/(12.0*hs)
#        elif order<7:
#            # 6th order accurate
#            return ( 0.0 # -1/60, +3/20, -3/4, 0, +3/4, -3/20, +1/60
#            -  1.0*func(XX-3.0*hh*tmp, aa, **kwargs)
#            +  9.0*func(XX-2.0*hh*tmp, aa, **kwargs)
#            - 45.0*func(XX-1.0*hh*tmp, aa, **kwargs)
#            + 45.0*func(XX+1.0*hh*tmp, aa, **kwargs)
#            -  9.0*func(XX+2.0*hh*tmp, aa, **kwargs)
#            +  1.0*func(XX+3.0*hh*tmp, aa, **kwargs))/(60.0*hs)
#        elif order<9:
#            # 8th order accurate
#            return ( 0.0 # +1/280, -4/105, +1/5 -4/5 0, +4/5, -1/5, +4/105, -1/280
#            +   3.0*func(XX-4.0*hh*tmp, aa, **kwargs)
#            -  32.0*func(XX-3.0*hh*tmp, aa, **kwargs)
#            + 168.0*func(XX-2.0*hh*tmp, aa, **kwargs)
#            - 672.0*func(XX-1.0*hh*tmp, aa, **kwargs)
#            + 672.0*func(XX+1.0*hh*tmp, aa, **kwargs)
#            - 168.0*func(XX+2.0*hh*tmp, aa, **kwargs)
#            +  32.0*func(XX+3.0*hh*tmp, aa, **kwargs)
#            -   3.0*func(XX+4.0*hh*tmp, aa, **kwargs))/(840.0*hs)

    @staticmethod
    def centered_2ndderiv(func, XX=None, hh=None, aa=None, order=2, msk=None, **kwargs):
        return centered_fd(func, XX, hh, aa, deriv_order=2, order=order, msk=msk, **kwargs)

#        if order<3:
#            # 2nd order accurate
#            return ( 0.0 # 1 -2, 1
#           + 1.0*func(XX-1.0*hh*tmp, aa, **kwargs)
#           - 2.0*func(XX, aa, **kwargs)
#           + 1.0*func(XX+1.0*hh*tmp, aa, **kwargs))/(_np.power(hs,2.0))
#        elif order<5:
#            # 4th order accurate
#            return ( 0.0 # -1/12, 4/3, -5/2, 4/3, -1/12
#           -  1.0*func(XX-2.0*hh*tmp, aa, **kwargs)
#           + 16.0*func(XX-1.0*hh*tmp, aa, **kwargs)
#           - 30.0*func(XX, aa, **kwargs)
#           + 16.0*func(XX+1.0*hh*tmp, aa, **kwargs)
#           -  1.0*func(XX+2.0*hh*tmp, aa, **kwargs))/(12.0*_np.power(hs,2.0))
#        elif order<7:
#            # 6th order accurate
#            return ( 0.0 # 1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90
#          +   2.0*func(XX-3.0*hh*tmp, aa, **kwargs)
#          -  27.0*func(XX-2.0*hh*tmp, aa, **kwargs)
#          + 270.0*func(XX-1.0*hh*tmp, aa, **kwargs)
#          - 490.0*func(XX, aa, **kwargs)
#          + 270.0*func(XX+1.0*hh*tmp, aa, **kwargs)
#          -  27.0*func(XX+2.0*hh*tmp, aa, **kwargs)
#          +   2.0*func(XX+3.0*hh*tmp, aa, **kwargs) )/(180.0*_np.power(hs,2.0))
#        elif order<9:
#            # 8th order accurate
#            coeff = [-1.0/560.0, 8.0/315.0, -1.0/5.0, 8.0/5.0, -205.0/72.0, 8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0]
#            return ( 0.0  # -1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5,  8/315, -1/560
#         + coeff[0]*func(XX-4.0*hh*tmp, aa, **kwargs)
#         + coeff[1]*func(XX-3.0*hh*tmp, aa, **kwargs)
#         + coeff[2]*func(XX-2.0*hh*tmp, aa, **kwargs)
#         + coeff[3]*func(XX-1.0*hh*tmp, aa, **kwargs)
#         + coeff[4]*func(XX, aa, **kwargs)
#         + coeff[5]*func(XX+1.0*hh*tmp, aa, **kwargs)
#         + coeff[6]*func(XX+2.0*hh*tmp, aa, **kwargs)
#         + coeff[7]*func(XX+3.0*hh*tmp, aa, **kwargs)
#         + coeff[8]*func(XX+4.0*hh*tmp, aa, **kwargs) )/_np.power(1.0*hs, 2.0)
#            # not 8th order accurate -returns 0
##            return ( 0.0  # -1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5,  8/315, -1/560
##         -    9.0*func(XX-4.0*hh*tmp, aa, **kwargs)
##         +  128.0*func(XX-3.0*hh*tmp, aa, **kwargs)
##         - 1008.0*func(XX-2.0*hh*tmp, aa, **kwargs)
##         + 8064.0*func(XX-1.0*hh*tmp, aa, **kwargs)
##         -14350.0*func(XX, aa, **kwargs)
##         + 8064.0*func(XX+1.0*hh*tmp, aa, **kwargs)
##         - 1008.0*func(XX+2.0*hh*tmp, aa, **kwargs)
##         +  128.0*func(XX+3.0*hh*tmp, aa, **kwargs)
##         -    9.0*func(XX+4.0*hh*tmp, aa, **kwargs) )/_np.power(5040.0*hs, 2.0)

    @staticmethod
    def centered_3rdderiv(func, XX=None, hh=None, aa=None, order=2, msk=None, **kwargs):
        return centered_fd(func, XX, hh, aa, deriv_order=3, order=order, msk=msk, **kwargs)

#        if order<3:
#            # 2nd order accurate
#            return ( 0.0 # -1/2, 1, 0, -1, 1/2
#            - 1.0*func(XX-2.0*hh*tmp, aa, **kwargs)
#            + 2.0*func(XX-1.0*hh*tmp, aa, **kwargs)
#            - 2.0*func(XX+1.0*hh*tmp, aa, **kwargs)
#            + 1.0*func(XX-2.0*hh*tmp, aa, **kwargs))/(2.0*_np.power(hs,3.0))
#        elif order<5:
#            # 4th order accurate
#            return ( 0.0 # +1/8, -1, +13/8, -13/8, +1, -1/8
#            +  1.0*func(XX-3.0*hh*tmp, aa, **kwargs)
#            -  8.0*func(XX-2.0*hh*tmp, aa, **kwargs)
#            + 13.0*func(XX-1.0*hh*tmp, aa, **kwargs)
#            - 13.0*func(XX+1.0*hh*tmp, aa, **kwargs)
#            +  8.0*func(XX+2.0*hh*tmp, aa, **kwargs)
#            -  1.0*func(XX+3.0*hh*tmp, aa, **kwargs))/(8.0*_np.power(hs,3.0))
#        elif order<7:
#            # 6th order accurate
#            return
#        elif order<9:
#            # 8th order accurate
#            return

    # ========================= mixed derivatives ============================ #

    @staticmethod
    def parse_mixed_inputs(XX, hx, aa, ha, order, msk):
        if XX is None or hx is None or aa is None or ha is None:
            raise Exception('improper inputs to finite difference function')

        if msk is None:
            msk = _np.ones_like(aa)
        msk = _np.asarray(msk, dtype=bool)
        tmp = _np.zeros_like(aa)
        tmp[msk] = 1.0

        # a-derivative
        ha = _np.atleast_1d(ha)
        if len(ha)>1:
            da = _np.copy(ha[msk])
        else:
            da = _np.copy(ha)
            ha = ha*_np.ones_like(aa)
        # end if

        # x-derivative
        dx = _np.copy(hx)
        hx = hx*_np.ones_like(XX)
        return tmp, msk, hx, ha, dx, da

    @staticmethod
    def centered_2ndderiv_mixed(func, XX=None, hx=None, aa=None, ha=None, order=2, msk=None, **kwargs):

        tmp, msk, hx, ha, dx, da = FD.parse_mixed_inputs(XX, hx, aa, ha, order, msk)

        if order<3:
            # 2nd order accurate
            return ( 0.0 # 1, -1, -1, 1
             + 1.0*func(XX+1.0*hx, aa+1.0*ha*tmp, **kwargs)
             - 1.0*func(XX+1.0*hx, aa-1.0*ha*tmp, **kwargs)
             - 1.0*func(XX-1.0*hx, aa+1.0*ha*tmp, **kwargs)
             + 1.0*func(XX-1.0*hx, aa-1.0*ha*tmp, **kwargs)
            )/(4.0*dx*da)
        elif order<5:
            # 4th order accurate
            return ( 0.0 #
             +  1.0*func(XX+2.0*hx, aa+2.0*ha*tmp, **kwargs)
             -  8.0*func(XX+2.0*hx, aa+1.0*ha*tmp, **kwargs)
             +  8.0*func(XX+2.0*hx, aa-1.0*ha*tmp, **kwargs)
             -  1.0*func(XX+2.0*hx, aa-2.0*ha*tmp, **kwargs)

             -  8.0*func(XX+1.0*hx, aa+2.0*ha*tmp, **kwargs)
             + 64.0*func(XX+1.0*hx, aa+1.0*ha*tmp, **kwargs)
             - 64.0*func(XX+1.0*hx, aa-1.0*ha*tmp, **kwargs)
             +  8.0*func(XX+1.0*hx, aa-2.0*ha*tmp, **kwargs)

             +  8.0*func(XX-1.0*hx, aa+2.0*ha*tmp, **kwargs)
             - 64.0*func(XX-1.0*hx, aa+1.0*ha*tmp, **kwargs)
             + 64.0*func(XX-1.0*hx, aa-1.0*ha*tmp, **kwargs)
             -  8.0*func(XX-1.0*hx, aa-2.0*ha*tmp, **kwargs)

             -  1.0*func(XX-2.0*hx, aa+2.0*ha*tmp, **kwargs)
             +  8.0*func(XX-2.0*hx, aa+1.0*ha*tmp, **kwargs)
             -  8.0*func(XX-2.0*hx, aa-1.0*ha*tmp, **kwargs)
             +  1.0*func(XX-2.0*hx, aa-2.0*ha*tmp, **kwargs)
            )/(144.0*dx*da)
#             - 44.0*func(XX+2.0*hx, aa+2.0*ha*tmp, **kwargs)
#             + 63.0*func(XX+2.0*hx, aa+1.0*ha*tmp, **kwargs)
#             - 63.0*func(XX+2.0*hx, aa-1.0*ha*tmp, **kwargs)
#             + 63.0*func(XX+2.0*hx, aa-2.0*ha*tmp, **kwargs)
#
#             + 63.0*func(XX+1.0*hx, aa+2.0*ha*tmp, **kwargs)
#             + 74.0*func(XX+1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             - 74.0*func(XX+1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 63.0*func(XX+1.0*hx, aa-2.0*ha*tmp, **kwargs)
#
#             - 63.0*func(XX-1.0*hx, aa+2.0*ha*tmp, **kwargs)
#             - 74.0*func(XX-1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 74.0*func(XX-1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             + 63.0*func(XX-1.0*hx, aa-2.0*ha*tmp, **kwargs)
#
#             + 44.0*func(XX-2.0*hx, aa+2.0*ha*tmp, **kwargs)
#             - 63.0*func(XX-2.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 63.0*func(XX-2.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 44.0*func(XX-2.0*hx, aa-2.0*ha*tmp, **kwargs)
#            )/(600.0*dx*da)
#        elif order<7:
#            # 6th order accurate
#            return ( 0.0 # 1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90
##                 )/(600.0*_np.power(hs,2.0))
#        elif order<9:
#            # 8th order accurate
#            coeff = []
#            return ( 0.0  # -1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5,  8/315, -1/560
#                #)/_np.power(1.0*hs, 2.0)
#            # not 8th order accurate -returns 0
##            return ( 0.0  # -1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5,  8/315, -1/560
#                #)/_np.power(5040.0*hs, 2.0)

    @staticmethod
    def centered_3rdderiv_mixed(func, XX=None, hx=None, aa=None, ha=None, order=2, msk=None, **kwargs):
        """
        For d3fdxdy2 and d3fdx2dy.
        Implementing d3fdx2dy for 2nd derivative of the jacobian
        """
        tmp, msk, hx, ha, dx, da = FD.parse_mixed_inputs(XX, hx, aa, ha, order, msk)

        if order<3:
            # 2nd order accurate
            return ( 0.0 # 1, -1, -1, 1
             + 1.0*func(XX+1.0*hx, aa+1.0*ha*tmp, **kwargs)
             + 1.0*func(XX-1.0*hx, aa+1.0*ha*tmp, **kwargs)
             - 2.0*func(XX+0.0*hx, aa+1.0*ha*tmp, **kwargs)
             + 2.0*func(XX+0.0*hx, aa-1.0*ha*tmp, **kwargs)
             - 1.0*func(XX+1.0*hx, aa-1.0*ha*tmp, **kwargs)
             - 1.0*func(XX-1.0*hx, aa-1.0*ha*tmp, **kwargs)
            )/(2.0*da*_np.power(dx,2.0))

    # ============================================================= #
    # ============================================================= #

# end def class