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
import matplotlib.pyplot as _plt
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

#    def boundaryconditions(self, **kwargs):
#
#        if 'leftboundary' in kwargs:
#            # Trigger forward difference derivative at left boundary
#            # also set Neumann or Dirichlet boundary conditions if requested
#            self.leftboundary = kwargs['leftboundary']
#        if 'rightboundary' in kwargs:
#            # Trigger backward difference derivative at right boundary
#            # also set Neumann or Dirichlet boundary conditions if requested
#            self.rightboundary = kwargs['rightboundary']
#        # end if
#    # end def boundary conditions

    def __step_size(self, model, XX, aa, deriv_order=1, order=6, **kwargs):
        deriv_in = kwargs.pop('deriv_in', 'x')

        if 'hh' not in kwargs:
            aa = _np.asarray(aa)

            if deriv_in.lower().find('x')>-1:
                tst = self.centered_fd(model, XX=XX, hh=0.01, aa=aa,
                        deriv_order=deriv_order, order=order, msk=None, **kwargs)
                dmax = _np.abs(tst).max()
            else:
                numfit = _np.size(aa)
                tst = _np.zeros((numfit, _np.size(XX)), dtype=_np.float64)
                for ii in range(numfit):
                    msk = _np.zeros(aa.shape, dtype=bool)
                    msk[ii] = True
                    tst[ii, :] = self.centered_fd(model, XX=aa, hh=0.01, aa=XX,
                        deriv_order=deriv_order, order=order, msk=msk, **kwargs)
                # end for
                dmax = _np.max(_np.abs(tst), axis=1)
            # end if

#            dmax = _np.max((tst.max()/0.2, 5.0))
#            dmax = _np.max((_np.max(_np.abs(aa))/0.2, dmax))   # maximum derivative, arbitraily say 10/0.2
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
        order = kwargs.setdefault('order', 6)
        kwargs.setdefault('deriv_order', 1)

        dydx = self.centered_fd(self.model, XX=XX, aa=aa, **kwargs)

#        kwargs.setdefault('order', 8)
        if hasattr(self, 'leftboundary') and (XX - 2*hh*order<self.leftboundary).any():
            ileft = _np.where(XX - 2*hh*order<self.leftboundary)[0]
            dydx[ileft] = self.forward_fd(self.model, XX=XX[ileft], aa=aa, **kwargs)
        # end if
        if hasattr(self, 'rightboundary') and (XX + 2*hh*order>self.rightboundary).any():
            iright = _np.where((XX + 2*hh*order>self.rightboundary))[0]
            dydx[iright] = self.backward_fd(self.model, XX=XX[iright], aa=aa, **kwargs)
        # end if
        return dydx
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
        order = kwargs.setdefault('order', 6)
        kwargs.setdefault('deriv_order', 2)

        d2ydx2 = self.centered_fd(self.model, XX=XX, aa=aa, **kwargs)

#        kwargs.pop('order')
#        order = kwargs.setdefault('order', 5) # order>5 is broken in forward/backward difference currently on 2nd derivative
        if hasattr(self, 'leftboundary') and (XX - 2*hh*order<self.leftboundary).any():
            ileft = _np.where(XX - 2*hh*order<self.leftboundary)[0]
            d2ydx2[ileft] = self.forward_fd(self.model, XX=XX[ileft], aa=aa, **kwargs)
        # end if
        if hasattr(self, 'rightboundary') and (XX + 2*hh*order>self.rightboundary).any():
            iright = _np.where(XX + 2*hh*order>self.rightboundary)[0]
            d2ydx2[iright] = self.backward_fd(self.model, XX=XX[iright], aa=aa, **kwargs)
        # end if
        return d2ydx2

    # ============================================================= #

    def jacobian(self, XX, aa=None, **kwargs):
        """
        Returns the jacobian of the model.
            f(x) = f(x, a, b, c)
            out: [dfda(x), dfdb(x), dfdc(x), ...]

        Leave it to the user to define the method '_partial' with an analytic Jacobian.
        Otherwise, calculate the jacobian numerically by finite-differencing the model.
        """
        def func(ai, xi, **kwargs):
            return self.model(xi, ai, **kwargs)

        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)

        ha, kwargs = self.__step_size(func, XX=XX, aa=aa, deriv_order=1, deriv_in='a', **kwargs)
#        kwargs['hh'] = kwargs['hh']*_np.ones_like(aa)
        kwargs.setdefault('order', 6)
        kwargs.setdefault('deriv_order', 1)

        numfit = _np.size(aa)
        gvec = _np.zeros((numfit, _np.size(XX)), dtype=_np.float64)
        for ii in range(numfit):
            msk = _np.zeros(aa.shape, dtype=bool)
            msk[ii] = True
            gvec[ii, :] = self.centered_fd(func, XX=aa, aa=XX, msk=msk, **kwargs)
#        # end for

#        order = kwargs['order']
#        if hasattr(self, 'leftboundary') and (aa - 2*hh*order<self.leftboundary).any():
#            ileft = _np.where(XX - 2*hh*order<self.leftboundary)[0]
#            for ii in range(numfit):
#                msk = _np.zeros(aa.shape, dtype=bool)
#                msk[ii] = True
#                gvec[ii, ileft] = self.forward_fd(func, XX=aa, aa=XX[ileft], msk=msk, **kwargs)
#            # end for
#        # end if
#        if hasattr(self, 'rightboundary') and (XX + 2*hh*order>self.rightboundary).any():
#            iright = _np.where(XX + 2*hh*order>self.rightboundary)[0]
#            for ii in range(numfit):
#                msk = _np.zeros(aa.shape, dtype=bool)
#                msk[ii] = True
#                gvec[ii, iright] = self.backward_fd(func, XX=aa, aa=XX[iright], msk=msk, **kwargs)
#            # end for
#        # end if
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
        def func(ai, xi, **kwargs):
            return self.derivative(xi, ai, **kwargs)

        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)

        order = kwargs.setdefault('order_x', 6)
        kwargs['deriv_order_x'] = 1
        hx, _ = self.__step_size(self.model, XX, aa, deriv_order=kwargs['deriv_order_x'], deriv_in='x',  **kwargs)

        kwargs.setdefault('order_a', 6)
        kwargs['deriv_order_a'] = 1
        ha, _ = self.__step_size(func, XX, aa, deriv_order=kwargs['deriv_order_a'], deriv_in='a', **kwargs)

        # Set optimized step sizes
        kwargs.setdefault('hx', hx)
        kwargs.setdefault('ha', ha)
#        kwargs.setdefault('ha', ha*_np.ones_like(aa))

        if 1:
            numfit = _np.size(aa)
            dgdx = _np.zeros((numfit, _np.size(XX)), dtype=_np.float64)
            for ii in range(numfit):
                msk = _np.zeros(aa.shape, dtype=bool)
                msk[ii] = True
                dgdx[ii, :] = self.centered_mixed_fd(self.model, XX=XX, aa=aa, msk=msk, **kwargs)
            # end for

#            order = 6
#            kwargs['order_x'] = order
            if hasattr(self, 'leftboundary') and (XX - 2*hx*order<self.leftboundary).any():
                ileft = _np.where(XX - 2*hx*order<self.leftboundary)[0]
                for ii in range(numfit):
                    msk = _np.zeros(aa.shape, dtype=bool)
                    msk[ii] = True
                    dgdx[ii, ileft] = self.forward_mixed_fd(self.model, XX=XX[ileft], aa=aa, msk=msk, **kwargs)
                # end for
            # end if
            if hasattr(self, 'rightboundary') and (XX + 2*hx*order>self.rightboundary).any():
                iright = _np.where(XX + 2*hx*order>self.rightboundary)[0]
                for ii in range(numfit):
                    msk = _np.zeros(aa.shape, dtype=bool)
                    msk[ii] = True
                    dgdx[ii, iright] = self.backward_mixed_fd(self.model, XX=XX[iright], aa=aa, msk=msk, **kwargs)
                # end for
            # end if
        else:
            # Numerically unstable to take derivative of a numerical derivative
            # this is only for testing!
            # use reversal of differentiation orders to do it on one-step here
            #    based on the jacobian (if numerical, this degrades accuracy)
            # for simple models (line/sine):
            #    passes at 4th order for rtol, atol = (1e-5, 1e-4)
            #    passes at 6th order for rtol, atol = (1e-4, 1e-4)
            #    passes at 8th order for rtol, atol = (1e-5, 1e-4),
#            dgdx = self._1stderiv_fd(self.jacobian, XX=XX, hh=hx, aa=aa, **kwargs)
            dgdx = self.centered_1stderiv(self.jacobian, XX=XX, hh=hx, aa=aa, **kwargs)

            if hasattr(self, 'leftboundary') and (XX - 2*hx*order<self.leftboundary).any():
                ileft = _np.where(XX - 2*hx*order<self.leftboundary)[0]
                dgdx[ileft] = self.forward_fd(self.jacobian, XX=XX[ileft], aa=aa, **kwargs)
            # end if
            if hasattr(self, 'rightboundary') and (XX + 2*hx*order>self.rightboundary).any():
                iright = _np.where(XX + 2*hx*order>self.rightboundary)[0]
                dgdx[iright] = self.backward_fd(self.jacobian, XX=XX[iright], aa=aa, **kwargs)
            # end if
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
        def func(ai, xi, **kwargs):
            return self.second_derivative(xi, ai, **kwargs)

        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)

        order = kwargs.setdefault('order_x', 6)
        kwargs['deriv_order_x'] = 2
        hx, _ = self.__step_size(self.model, XX, aa, deriv_in='x', deriv_order=kwargs['deriv_order_x'], **kwargs)

        kwargs.setdefault('order_a', 6)
        kwargs['deriv_order_a'] = 1
        ha, _ = self.__step_size(func, XX, aa, deriv_in='a', deriv_order=kwargs['deriv_order_a'], **kwargs)
#        ha = 0.1*ha

        # set optimized step sizes
        kwargs.setdefault('hx', hx)
        kwargs.setdefault('ha', ha)
#        kwargs.setdefault('ha', ha*_np.ones_like(aa))

        if 1:
            numfit = _np.size(aa)
            d2gdx2 = _np.zeros((numfit, _np.size(XX)), dtype=_np.float64)
            for ii in range(numfit):
                msk = _np.zeros(aa.shape, dtype=bool)
                msk[ii] = True
                d2gdx2[ii, :] = self.centered_mixed_fd(self.model, XX=XX, aa=aa, msk=msk, **kwargs)
            # end for

#            order = 10
#            kwargs['order_x'] = order
            kwargs['backward_in'] = 'x'
            kwargs['forward_in'] = 'x'
            kwargs['order_x'] = 2
            kwargs['order_a'] = 6
            if hasattr(self, 'leftboundary') and (XX - 2*hx*order<self.leftboundary).any():
                ileft = _np.where(XX - 2*hx*order<self.leftboundary)[0]
                for ii in range(numfit):
                    msk = _np.zeros(aa.shape, dtype=bool)
                    msk[ii] = True
                    d2gdx2[ii, ileft] = self.forward_mixed_fd(self.model, XX=XX[ileft], aa=aa, msk=msk, **kwargs)
                # end for
            # end if
            if hasattr(self, 'rightboundary') and (XX + 2*hx*order>self.rightboundary).any():
                iright = _np.where(XX + 2*hx*order>self.rightboundary)[0]
                for ii in range(numfit):
                    msk = _np.zeros(aa.shape, dtype=bool)
                    msk[ii] = True
                    d2gdx2[ii, iright] = self.backward_mixed_fd(self.model, XX=XX[iright], aa=aa, msk=msk, **kwargs)
                # end for
            # end if
        else:
            # Numerically unstable to take 2nd derivative of a numerical derivative
            # this is only for testing!
            if 'hh' not in kwargs:
                hh, _ = self.__step_size(self.derivative_jacobian, XX, aa, deriv_order=2, **kwargs)
                kwargs.setdefault('hx', hx)
            # end if
#            d2gdx2 = self._2ndderiv_fd(self.jacobian, XX=XX, aa=aa, **kwargs)
            d2gdx2 = self.centered_1stderiv(self.jacobian, XX=XX, aa=aa, **kwargs)
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

    @staticmethod
    def centered_coeffs(deriv_order, order):

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
            else:
                num_coeffs = []
                den_coeffs = 1.0
            # end if
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
#                num_coeffs = [-1.0/560.0, 8.0/315.0, -1.0/5.0, 8.0/5.0, -205.0/72.0, 8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0]
#                den_coeffs = 1.0
                num_coeffs = [-9.0, 128.0, -1008.0, 8064.0, -14350.0, 8064.0, -1008.0, 128.0, -9.0]
                den_coeffs = 5040.0
            else:
                num_coeffs = []
                den_coeffs = 1.0
            # end if
            # end if
        elif deriv_order == 3:
            if order<3:
                num_coeffs = [-1.0, 2.0, 0.0, -2.0, 1.0]
                den_coeffs = 2.0
            elif order<5:
                num_coeffs = [1.0, -8.0, 13.0, 0.0, -13.0, 8.0, -1.0]
                den_coeffs = 8.0
            else:
                num_coeffs = []
                den_coeffs = 1.0
            # end if
        # end if
        if len(num_coeffs) == 0:
            print('Calculating coefficients by matrix inversion')
#            if deriv_order % 2:
#               stencil=_np.linspace(start=-(order+2//2), stop=(order+2)//2, num=order+2, endpoint=True, dtype=int)
            num_coeffs, order_accuracy = FD.__coeff_generator(deriv_order, order,
                      stencil=None, centered=True)
        # end if
        return num_coeffs, den_coeffs

    @staticmethod
    def forward_coeffs(deriv_order, order):
        if deriv_order == 1:
            if order<2:                # 1st order accuracy
                num_coeffs = [-1.0, 1.0]
                den_coeffs = 1.0
            elif order<3:                # 2nd order accuracy
                num_coeffs = [-3.0, 4.0, -1.0]
                den_coeffs = 2.0
            elif order<4:                # 3rd order accuracy
                num_coeffs = [-11.0, 18.0, -9.0, 2.0]
                den_coeffs = 6.0
            elif order<5:                # 4th order accuracy
                num_coeffs = [-25.0, 48.0, -36.0, 16.0, -3.0]
                den_coeffs = 12.0
            elif order<6:                # 5th order accuracy
                num_coeffs = [-137.0, 300.0, -300.0, 200.0, -75.0, 12.0]
                den_coeffs = 60.0
            elif order<7:                # 6th order accuracy
                num_coeffs = [-147.0, 360.0, -450.0, 400.0, -225.0, 72.0, -10.0]
                den_coeffs = 60.0
            else:
                num_coeffs = []
                den_coeffs = 1.0
        elif deriv_order == 2:
            if order<2:
                num_coeffs = [1.0, -2.0, 1.0]
                den_coeffs = 1.0
            elif order<3:                # 2nd order accuracy
                num_coeffs = [2.0, -5.0, 4.0, -1.0]
                den_coeffs = 1.0
            elif order<4:                # 3rd order accuracy
                num_coeffs = [35.0, -104.0, 114.0, -56.0, 11.0]
                den_coeffs = 12.0
            elif order<5:                # 4th order accuracy
                num_coeffs = [45.0, -154.0, 214.0, -156.0, 61.0, -10.0]
                den_coeffs = 12.0
            elif order<6:                # 5th order accuracy
#                num_coeffs = [203.0/45.0, -87.0/5.0, 117.0/4.0, -254.0/9.0, 33.0/2.0, -27.0/5.0, 137.0/180.0]
#                den_coeffs = 1.0
                num_coeffs = [812.0, -3132.0, 5265.0, -5080.0, 2970.0, -972.0, 137.0]
                den_coeffs = 180.0
            elif order<7:                # 6th order accuracy  -  TODO!: check
                num_coeffs = [29531.0, -138528.0, 312984.0, -448672.0, 435330.0, -284256.0, 120008.0, -29664.0, 3267.0]
                den_coeffs = 5040.0
            elif order<8:
                num_coeffs = [177133.0, -972200.0, 2754450.0, -5232800.0, 7088550.0, -6932016.0, 4872700.0, -2407200.0, 794925.0, -157800.0, 14258.0]
                den_coeffs = 25200.0
            else:
                num_coeffs = []
                den_coeffs = 1.0
        elif deriv_order == 3:
            if order<2:                # 1st order accuracy
                num_coeffs = [-1.0, 3.0, -3.0, 1.0]
                den_coeffs = 1.0
            elif order<3:                # 2nd order accuracy
#                num_coeffs = [-5.0/2.0, 9.0, -12.0, 7.0, -3.0/2.0]
#                den_coeffs = 1.0
                num_coeffs = [-5.0, 18.0, -24.0, 14.0, -6.0]
                den_coeffs = 2.0
            elif order<4:                # 3rd order accuracy
#                num_coeffs = [-17.0/4.0, 71.0/4.0, -59.0/2.0, 49/2.0, -41.0/4.0, 7/4.0]
#                den_coeffs = 1.0
                num_coeffs = [-17.0, 71.0, -118.0, 98.0, -41.0, 7.0]
                den_coeffs = 4.0
            elif order<5:                # 4th order accuracy
#                num_coeffs = [-49.0/8.0, 29.0, -461.0/8.0, 62.0, -307.0/8.0, 13.0, -15.0/8.0]
#                den_coeffs = 1.0
                num_coeffs = [-49.0, 232.0, -461.0, 496.0, -307.0, 104.0, -15.0]
                den_coeffs = 8.0
            elif order<6:                # 5th order accuracy
#                num_coeffs = [-967.0/120.0, 638.0/15.0, -3929.0/40.0, 389.0/3.0, -2545.0/24.0, 268.0/5.0, -1849.0/120.0, 29.0/15.0]
#                den_coeffs = 1.0
                num_coeffs = [-967.0, 5104.0, -11787.0, 15560.0, -12725.0, 6432.0, -1849.0, 232.0]
                den_coeffs = 120.0
#            else:
            elif order<7:                # 6th order accuracy
#                num_coeffs = [-801.0/80.0, 349.0/6.0, -18353.0/120.0, 2391.0/10.0, -1457.0/6.0, 4891.0/30.0, -561.0/8.0, 527.0/30.0, -469.0/240.0]
#                den_coeffs = 1.0
                num_coeffs = [-2403, 13960, -36706, 57384, -58280, 39128, -16830, 4216, -469]
                den_coeffs = 240.0
            else:
                num_coeffs = []
                den_coeffs = 1.0
        elif deriv_order == 4:
            if order<2:                # 1st order accuracy
                num_coeffs = [1.0, -4.0, 6.0, -4.0, 1.0]
                den_coeffs = 1.0
            elif order<3:                # 2nd order accuracy
                num_coeffs = [3.0, -14.0, 26.0, -24.0, 11.0, -2.0]
                den_coeffs = 1.0
            elif order<4:                # 3rd order accuracy
#                num_coeffs = [35.0/6.0, -31.0, 137.0/2.0, -242.0/3.0, 107.0/2.0, -19.0, 17.0/6.0]
#                den_coeffs = 1.0
                num_coeffs = [35.0, -186.0, 411.0, -484.0, 321.0, -114.0, 17.0]
                den_coeffs = 6.0
            elif order<5:                # 4th order accuracy
#                num_coeffs = [28.0/3.0, -111.0/2.0, 142.0, -1219.0/6.0, 176.0, -185.0/2.0, 82.0/3.0, -7.0/2.0]
#                den_coeffs = 1.0
                num_coeffs = [56.0, -333.0, 852.0, -1219.0, 1056.0, -555.0, 164.0, -21.0]
                den_coeffs = 6.0
            elif order<6:                # 5th order accuracy
#                num_coeffs = [1069.0/80.0, -1316.0/15.0, 15289.0/60.0, -2144.0/5.0, 10993.0/24.0, -4772.0/15.0, 2803.0/20.0, -536.0/15.0, 967.0/240.0]
#                den_coeffs = 1.0
                num_coeffs = [3207.0, -21056.0, 61156.0, -102912.0, 109930.0, -76352.0, 33636.0, -8576.0, 967.0]
                den_coeffs = 240.0
            else:
                num_coeffs = []
                den_coeffs = 1.0
#            elif order<7:                # 6th order accuracy
##                num_coeffs = []
##                den_coeffs = 1.0
#                num_coeffs = []
#                den_coeffs = 1.0
            # end if
        # end if
        if len(num_coeffs) == 0:
            print('Calculating coefficients by matrix inversion')
            num_coeffs, order_accuracy = FD.__coeff_generator(deriv_order, order,
                      stencil=None, centered=False)
        # end if
        return num_coeffs, den_coeffs

    def __coeff_generator(deriv_order, order, stencil=None, centered=False, fraction_out=False):
        deriv_order = int(deriv_order)
        order = int(order)
        if stencil is None and centered:
#            if deriv_order % 2:   # even deriv orders have no coefficient at 0
#                ncoef = order
            ncoef = int(2*_np.floor((deriv_order+1)/2)  - 1 + order)
            pp = (ncoef-1)/2.0
            stencil = _np.linspace(start=-pp, stop=pp, num=ncoef, endpoint=True)
        elif stencil is None:
            # Create a forward difference
            ncoef = order+1
            stencil = _np.asarray(range(ncoef), dtype=int)
        # end for
        ncoef = len(stencil)
        stencil = _np.asarray(stencil, dtype=_np.float64)
        lhs = _np.zeros((ncoef, ncoef), dtype=_np.float64)

        for ii in range(ncoef):
            lhs[ii,:] = _np.power(_np.copy(stencil), ii)
        # end for
        rhs = _np.zeros((ncoef,1), dtype=_np.float64)
        rhs[deriv_order] = _ut.factorial(deriv_order)
        coef = _np.linalg.solve(lhs, rhs)  # exact solution to Ax=b with harsh requirements
#        coef, residuals, rank, singular_values = _np.linalg.lstsq(lhs, rhs, rcond=None) # least-squares solution, returns exact solution if possible

        order_accuracy = len(stencil)-deriv_order

        coef = coef.flatten().tolist()
        if fraction_out:
            # Convert decimal result to fractions
            import fractions
            num, den = [], []
            for ii in range(ncoef):
                tst = fractions.Fraction(coef[ii])
                tst.limit_denominator(20000)
                num.append(tst.numerator)
                den.append(tst.numerator)

#                print((coef[ii], num[ii]/den[ii], tst))
                coef[ii] = num[ii]/den[ii]
            # end for

            # Get least common denominator
        return coef, order_accuracy

    @staticmethod
    def backward_coeffs(deriv_order, order):
        coeffs, den_coeffs = FD.forward_coeffs(deriv_order, order)
        if deriv_order % 2:    # even
            coeffs = [-1.0*coeffs[ii] for ii in range(len(coeffs))]
        # end if
        return coeffs[::-1], den_coeffs

    @staticmethod
    def centered_mixed_coeffs(deriv_order_x, order_x, deriv_order_a, order_a):
        coefx, denx =FD.centered_coeffs(deriv_order_x, order_x)
        coefa, dena = FD.centered_coeffs(deriv_order_a, order_a)

        coeff = []
        for ii in range(len(coefx)):
            for jj in range(len(coefa)):
                coeff.append(coefx[ii]*coefa[jj])
        return coeff, denx*dena, coefx, denx, coefa, dena

    @staticmethod
    def forward_mixed_coeffs(deriv_order_x, order_x, deriv_order_a, order_a, forward_in="x"):
        if forward_in.lower().find("a")>-1:
            coefx, denx = FD.centered_coeffs(deriv_order_x, order_x)
            coefa, dena = FD.forward_coeffs(deriv_order_a, order_a)
        else:
            coefx, denx = FD.forward_coeffs(deriv_order_x, order_x)
            coefa, dena = FD.centered_coeffs(deriv_order_a, order_a)

        coeff = []
        for ii in range(len(coefx)):
            for jj in range(len(coefa)):
                coeff.append(coefx[ii]*coefa[jj])
        return coeff, denx*dena, coefx, denx, coefa, dena

    @staticmethod
    def backward_mixed_coeffs(deriv_order_x, order_x, deriv_order_a, order_a, backward_in="x"):
        if backward_in.lower().find("a")>-1:
            coefx, denx = FD.centered_coeffs(deriv_order_x, order_x)
            coefa, dena = FD.backward_coeffs(deriv_order_a, order_a)
        else:
            coefx, denx = FD.backward_coeffs(deriv_order_x, order_x)
            coefa, dena = FD.centered_coeffs(deriv_order_a, order_a)

        coeff = []
        for ii in range(len(coefx)):
            for jj in range(len(coefa)):
                coeff.append(coefx[ii]*coefa[jj])
        return coeff, denx*dena, coefx, denx, coefa, dena

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
    def centered_fd(func, XX=None, hh=None, aa=None, deriv_order=1, order=2, msk=None, **kwargs):
        """
        Note: the looping requires a stencil point be defined at 0
        """
        tmp, msk, hh, hs = FD.parse_unmixed_inputs(XX, hh, aa, order, msk)

        num_coeffs, den_coeffs = FD.centered_coeffs(deriv_order=deriv_order, order=order)

        ncoeffs = len(num_coeffs)
        nstencil = ncoeffs // 2
        ysum = 0.0
        for ii in range(ncoeffs):
            jj = -1.0*nstencil + ii

            ysum += num_coeffs[ii]*func(XX+jj*hh*tmp, aa, **kwargs)
        # end for
        return ysum / (den_coeffs*_np.power(hs, deriv_order))

    @staticmethod
    def forward_fd(func, XX=None, hh=None, aa=None, deriv_order=1, order=2, msk=None, **kwargs):
        """
        Note: the looping requires a stencil point be defined at 0
        """
        tmp, msk, hh, hs = FD.parse_unmixed_inputs(XX, hh, aa, order, msk)

        num_coeffs, den_coeffs = FD.forward_coeffs(deriv_order=deriv_order, order=order)

        ncoeffs = len(num_coeffs)
        ysum = 0.0
        for ii in range(ncoeffs):
            ysum += num_coeffs[ii]*func(XX+ii*hh*tmp, aa, **kwargs)
        # end for
        return ysum / (den_coeffs*_np.power(hs, deriv_order))

    @staticmethod
    def backward_fd(func, XX=None, hh=None, aa=None, deriv_order=1, order=2, msk=None, **kwargs):
        """
        Note: the looping requires a stencil point be defined at 0
        """
        tmp, msk, hh, hs = FD.parse_unmixed_inputs(XX, hh, aa, order, msk)

        num_coeffs, den_coeffs = FD.backward_coeffs(deriv_order=deriv_order, order=order)

        ncoeffs = len(num_coeffs)
        nstencil = ncoeffs
        ysum = 0.0
        for ii in range(ncoeffs):
            jj = -1.0*nstencil + ii

            ysum += num_coeffs[ii]*func(XX+jj*hh*tmp, aa, **kwargs)
        # end for
        return ysum / (den_coeffs*_np.power(hs, deriv_order))

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
    def centered_mixed_fd(func, XX=None, hx=None, aa=None, ha=None, deriv_order=1, order=2, msk=None, **kwargs):
        """
        Note: the looping requires a stencil point be defined at 0
        """
        tmp, msk, hx, ha, dx, da = FD.parse_mixed_inputs(XX, hx, aa, ha, order, msk)

        deriv_order_a = kwargs.pop("deriv_order_a", 1)
        order_a = kwargs.pop("order_a", order)
        deriv_order_x = kwargs.pop("deriv_order_x", deriv_order)
        order_x = kwargs.pop("order_x", order)

        coeff, denx, coefx, denx, coefa, dena = \
            FD.centered_mixed_coeffs(deriv_order_x, order_x, deriv_order_a, order_a)

        ncoefx = len(coefx)
        nxstencil = ncoefx // 2
        ncoefa = len(coefa)
        nastencil = ncoefa // 2

        ysum = 0.0
        for ii in range(ncoefx):
            istencil = -1.0*nxstencil + ii
            for jj in range(len(coefa)):
                jstencil = -1.0*nastencil + jj
                ysum += coefx[ii]*coefa[jj]*func(XX+istencil*hx, aa+jstencil*ha*tmp, **kwargs)
            # end for
        # end for
        return ysum / (denx*dena*_np.power(dx, deriv_order_x)*_np.power(da, deriv_order_a))

    @staticmethod
    def forward_mixed_fd(func, XX=None, hx=None, aa=None, ha=None, deriv_order=1, order=2, msk=None, **kwargs):
        """
        Note: the looping requires a stencil point be defined at 0
        """
        tmp, msk, hx, ha, dx, da = FD.parse_mixed_inputs(XX, hx, aa, ha, order, msk)

        deriv_order_a = kwargs.pop("deriv_order_a", 1)
        order_a = kwargs.pop("order_a", order)
        deriv_order_x = kwargs.pop("deriv_order_x", deriv_order)
        order_x = kwargs.pop("order_x", order)
        forward_in = kwargs.pop("forward_in", "x")

        coeff, den, coefx, denx, coefa, dena = \
            FD.forward_mixed_coeffs(deriv_order_x, order_x, deriv_order_a, order_a, forward_in)

        ncoefx = len(coefx)
        ncoefa = len(coefa)
        if forward_in.lower().find("a")>-1:
            nxstencil = ncoefx // 2
            nastencil = 0
        else:
            nastencil = ncoefa // 2
            nxstencil = 0
        istencil = lambda ii: -1.0*nxstencil+ii
        jstencil = lambda jj: -1.0*nastencil+jj

        ysum = 0.0
        for ii in range(ncoefx):
            istep = istencil(ii)
            for jj in range(len(coefa)):
                jstep = jstencil(jj)
                ysum += coefx[ii]*coefa[jj]*func(XX+istep*hx, aa+jstep*ha*tmp, **kwargs)
            # end for
        # end for
        return ysum / (denx*dena*_np.power(dx, deriv_order_x)*_np.power(da, deriv_order_a))

    @staticmethod
    def backward_mixed_fd(func, XX=None, hx=None, aa=None, ha=None, deriv_order=1, order=2, msk=None, **kwargs):
        """
        Note: the looping requires a stencil point be defined at 0
        """
        tmp, msk, hx, ha, dx, da = FD.parse_mixed_inputs(XX, hx, aa, ha, order, msk)

        deriv_order_a = kwargs.pop("deriv_order_a", 1)
        order_a = kwargs.pop("order_a", order)
        deriv_order_x = kwargs.pop("deriv_order_x", deriv_order)
        order_x = kwargs.pop("order_x", order)
        backward_in = kwargs.pop("backward_in", "x")

        coeff, den, coefx, denx, coefa, dena = \
            FD.backward_mixed_coeffs(deriv_order_x, order_x, deriv_order_a, order_a, backward_in)

        ncoefx = len(coefx)
        ncoefa = len(coefa)
        if backward_in.lower().find("a")>-1:
            nxstencil = ncoefx // 2
            nastencil = ncoefa
        else:
            nastencil = ncoefa // 2
            nxstencil = ncoefx
        istencil = lambda ii: -1.0*nxstencil+ii
        jstencil = lambda jj: -1.0*nastencil+jj

        ysum = 0.0
        for ii in range(ncoefx):
            istep = istencil(ii)
            for jj in range(len(coefa)):
                jstep = jstencil(jj)
                ysum += coefx[ii]*coefa[jj]*func(XX+istep*hx, aa+jstep*ha*tmp, **kwargs)
            # end for
        # end for
        return ysum / (denx*dena*_np.power(dx, deriv_order_x)*_np.power(da, deriv_order_a))
#
#
#    @staticmethod
#    def centered_1stderiv(func, XX=None, hh=None, aa=None, order=2, msk=None, **kwargs):
#        return FD.centered_fd(func, XX, hh, aa, deriv_order=1, order=order, msk=msk, **kwargs)
#
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
#
#    @staticmethod
#    def centered_2ndderiv(func, XX=None, hh=None, aa=None, order=2, msk=None, **kwargs):
#        return FD.centered_fd(func, XX, hh, aa, deriv_order=2, order=order, msk=msk, **kwargs)
#
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
#
#    @staticmethod
#    def centered_3rdderiv(func, XX=None, hh=None, aa=None, order=2, msk=None, **kwargs):
#        return FD.centered_fd(func, XX, hh, aa, deriv_order=3, order=order, msk=msk, **kwargs)
#
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

#    @staticmethod
#    def forward_2ndderiv_mixed(func, XX=None, hx=None, aa=None, ha=None, order=2, msk=None, **kwargs):
#        """ forward difference in x, centered in a: d2fdxda"""
#        return FD.forward_mixed_fd(func, XX, hx, aa, ha, deriv_order=1, order=order, msk=msk, **kwargs)
#
#        tmp, msk, hx, ha, dx, da = FD.parse_mixed_inputs(XX, hx, aa, ha, order, msk)
#
#        if order<2:
#            # 1st order accurate in x, 2nd in a
#            return ( 0.0 # -1, 1
#             + 1.0*func(XX+1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX+1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX-0.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 1.0*func(XX-0.0*hx, aa-1.0*ha*tmp, **kwargs)
#            )/(2.0*dx*da)
#        else: # order<3:
#            # 2nd order accurate in x, 2nd in a
#            return ( 0.0 # -3, 4, -1
#             - 3.0*func(XX-0.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 3.0*func(XX-0.0*hx, aa-1.0*ha*tmp, **kwargs)
#             + 4.0*func(XX+1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             - 4.0*func(XX+1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX+2.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 1.0*func(XX+2.0*hx, aa-1.0*ha*tmp, **kwargs)
#            )/(4.0*dx*da)
#
#    @staticmethod
#    def backward_2ndderiv_mixed(func, XX=None, hx=None, aa=None, ha=None, order=2, msk=None, **kwargs):
#        """ backward difference in x, centered in a: d2fdxda"""
#        return FD.backward_mixed_fd(func, XX, hx, aa, ha, deriv_order=1, order=order, msk=msk, **kwargs)
#        tmp, msk, hx, ha, dx, da = FD.parse_mixed_inputs(XX, hx, aa, ha, order, msk)
#
#        if order<2:
#            # 1st order accurate in x, 2nd in a
#            return ( 0.0 # -1, 1
#             - 1.0*func(XX-1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 1.0*func(XX-1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             + 1.0*func(XX-0.0*hx, aa+1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX-0.0*hx, aa-1.0*ha*tmp, **kwargs)
#            )/(2.0*dx*da)
#        else: # order<3:
#            # 2nd order accurate in x, 2nd in a
#            return ( 0.0 # -3, 4, -1
#             + 3.0*func(XX-0.0*hx, aa+1.0*ha*tmp, **kwargs)
#             - 3.0*func(XX-0.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 4.0*func(XX-1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 4.0*func(XX-1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX-2.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 1.0*func(XX-2.0*hx, aa-1.0*ha*tmp, **kwargs)
#            )/(4.0*dx*da)
#
#    @staticmethod
#    def centered_2ndderiv_mixed(func, XX=None, hx=None, aa=None, ha=None, order=2, msk=None, **kwargs):
#        """ d2fdxda """
#        return FD.centered_mixed_fd(func, XX, hx, aa, ha, deriv_order=1, order=order, msk=msk, **kwargs)

#        tmp, msk, hx, ha, dx, da = FD.parse_mixed_inputs(XX, hx, aa, ha, order, msk)
#
#        if order<3:
#            # 2nd order accurate
#            return ( 0.0 # 1, -1, -1, 1
#             + 1.0*func(XX+1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX+1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX-1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 1.0*func(XX-1.0*hx, aa-1.0*ha*tmp, **kwargs)
#            )/(4.0*dx*da)
#        elif order<5:
#            # 4th order accurate
#            return ( 0.0 #
#             +  1.0*func(XX+2.0*hx, aa+2.0*ha*tmp, **kwargs)
#             -  8.0*func(XX+2.0*hx, aa+1.0*ha*tmp, **kwargs)
#             +  8.0*func(XX+2.0*hx, aa-1.0*ha*tmp, **kwargs)
#             -  1.0*func(XX+2.0*hx, aa-2.0*ha*tmp, **kwargs)
#
#             -  8.0*func(XX+1.0*hx, aa+2.0*ha*tmp, **kwargs)
#             + 64.0*func(XX+1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             - 64.0*func(XX+1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             +  8.0*func(XX+1.0*hx, aa-2.0*ha*tmp, **kwargs)
#
#             +  8.0*func(XX-1.0*hx, aa+2.0*ha*tmp, **kwargs)
#             - 64.0*func(XX-1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 64.0*func(XX-1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             -  8.0*func(XX-1.0*hx, aa-2.0*ha*tmp, **kwargs)
#
#             -  1.0*func(XX-2.0*hx, aa+2.0*ha*tmp, **kwargs)
#             +  8.0*func(XX-2.0*hx, aa+1.0*ha*tmp, **kwargs)
#             -  8.0*func(XX-2.0*hx, aa-1.0*ha*tmp, **kwargs)
#             +  1.0*func(XX-2.0*hx, aa-2.0*ha*tmp, **kwargs)
#            )/(144.0*dx*da)
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

#    @staticmethod
#    def forward_3rdderiv_mixed(func, XX=None, hx=None, aa=None, ha=None, order=2, msk=None, **kwargs):
#        """
#        Forward difference in x, centered in a
#        For d3fdxdy2 and d3fdx2dy.
#        Implementing d3fdx2dy for 2nd derivative of the jacobian
#        """
#        return FD.forward_mixed_fd(func, XX, hx, aa, ha, deriv_order=1, order=order, msk=msk,
#                                   deriv_order_a=2, deriv_order_x=1, **kwargs)
#        tmp, msk, hx, ha, dx, da = FD.parse_mixed_inputs(XX, hx, aa, ha, order, msk)
#
##        if order<3:
#        if 1:
#            # 2nd order accurate
#            return ( 0.0 # 1, -1, -1, 1
#             + 1.0*func(XX+1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 1.0*func(XX-1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             - 2.0*func(XX+0.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 2.0*func(XX+0.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX+1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX-1.0*hx, aa-1.0*ha*tmp, **kwargs)
#            )/(2.0*da*_np.power(dx,2.0))
#
#    @staticmethod
#    def centered_3rdderiv_mixed(func, XX=None, hx=None, aa=None, ha=None, order=2, msk=None, **kwargs):
#        """
#        For d3fdxdy2 and d3fdx2dy.
#        Implementing d3fdx2dy for 2nd derivative of the jacobian
#        """
#        return FD.centered_mixed_fd(func, XX, hx, aa, ha, deriv_order=1, order=order, msk=msk,
#                                   deriv_order_a=1, deriv_order_x=2, **kwargs)
#        tmp, msk, hx, ha, dx, da = FD.parse_mixed_inputs(XX, hx, aa, ha, order, msk)
#
##        if order<3:
#        if 1:
#            # 2nd order accurate
#            return ( 0.0 # 1, -1, -1, 1
#             + 1.0*func(XX+1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 1.0*func(XX-1.0*hx, aa+1.0*ha*tmp, **kwargs)
#             - 2.0*func(XX+0.0*hx, aa+1.0*ha*tmp, **kwargs)
#             + 2.0*func(XX+0.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX+1.0*hx, aa-1.0*ha*tmp, **kwargs)
#             - 1.0*func(XX-1.0*hx, aa-1.0*ha*tmp, **kwargs)
#            )/(2.0*da*_np.power(dx,2.0))

    # ============================================================= #
    # ============================================================= #
# end def class


# ========================================================================== #

class ModelClass(FD):
    def __init__(self, XX=None, af=None, **kwargs):
        LB, UB, fixed = self.defaults(**kwargs)
        if af is None:   af = _np.copy(self._af)    # end if
        self.af = _np.copy(af)
        self.Lbounds = _np.copy(LB)
        self.Ubounds = _np.copy(UB)
        self.fixed = _np.copy(fixed)
        kwargs.setdefault('analytic', True) # default to analytic derivatives/jacob. etc.
#        kwargs.setdefault('analytic', False) # default to analytic derivatives/jacob. etc.
        self.__dict__.update(kwargs)
        self.kwargs = kwargs
        if XX is not None:
            self.update(XX=XX)
    # end def __init__


    def model(self, XX, aa=None, **kwargs):
        """
        Returns the model.
            f(x) = f(x, a, b, c, ...)
            out: f(x)
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)
        return self._model(XX, aa, **kwargs)

    def derivative(self, XX, aa=None, **kwargs):
        """
        Returns the derivative of the model.
            f(x) = f(x, a, b, c, ...)
            out: dfdx

        Leave it to the user to define the method '_deriv' with an analytic derivative.
        otherwise, calculate the derivative numerically by finite-differencing the model.
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)
        if hasattr(self, '_deriv') and self.analytic:
            return self._deriv(XX, aa, **kwargs)
        return super(ModelClass, self).derivative(XX, aa, **kwargs)
        # end if
    # end def derivative

    def jacobian(self, XX, aa=None, **kwargs):
        """
        Returns the jacobian of the model.
            f(x) = f(x, a, b, c)
            out: [dfda(x), dfdb(x), dfdc(x), ...]

        Leave it to the user to define the method '_partial' with an analytic Jacobian.
        Otherwise, calculate the jacobian numerically by finite-differencing the model.
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)
        if hasattr(self, '_partial') and self.analytic:
            return self._partial(XX, aa, **kwargs)
        return super(ModelClass, self).jacobian(XX, aa, **kwargs)

    def derivative_jacobian(self, XX, aa=None, **kwargs):
        """
        Returns the jacobian of the derivative of the model.
            f(x) = f(x, a, b, c)
            out: [d2fdxda(x), d2fdxdb(x), d2fdxdc(x), ...]

        Leave it to the user to define the method '_partial_deriv' with an analytic Jacobian of the derivative.
        Otherwise, calculate the jacobian of the derivative numerically by finite-differencing the model derivative.
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)
        if hasattr(self, '_partial_deriv') and self.analytic:
            return self._partial_deriv(XX, aa, **kwargs)
        return super(ModelClass, self).derivative_jacobian(XX, aa, **kwargs)

    def second_derivative(self, XX, aa=None, **kwargs):
        """
        Returns the 2nd derivative of the model.
            f(x) = f(x, a, b, c, ...)
            out: d2fdx2

        Leave it to the user to define the method '_deriv2' with an analytic 2nd derivative.
        otherwise, calculate the derivative numerically by finite-differencing the model.
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)
        if hasattr(self, '_deriv2') and self.analytic:
            return self._deriv2(XX, aa, **kwargs)
        return super(ModelClass, self).second_derivative(XX, aa, **kwargs)

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
        return super(ModelClass, self).second_derivative_jacobian(XX, aa, **kwargs)

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
        if hasattr(self, '_hessian') and self.analytic:
            return self._hessian(XX, aa, **kwargs)
        return super(ModelClass, self).hessian(XX, aa, **kwargs)
    # end def


    # ============================================================= #
    # ============================================================= #


    def update_minimal(self, XX, af, **kwargs):
        XX, af, kwargs = self.parse_in(XX, af, **kwargs)
        self.prof = self.model(XX, aa=af, **kwargs)
        self.gvec = self.jacobian(XX, aa=af, **kwargs)
        return self.prof, self.gvec

    def update(self, XX=None, af=None, **kwargs):
        XX, af, kwargs = self.parse_in(XX, af, **kwargs)
        self.XX = _np.copy(XX)
        self.af = _np.copy(af)
        self.kwargs = kwargs
        self.prof = self.model(XX, aa=af, **kwargs)
        self.gvec = self.jacobian(XX, aa=af, **kwargs)
        self.dprofdx = self.derivative(XX, aa=af, **kwargs)
        self.dgdx = self.derivative_jacobian(XX, aa=af, **kwargs)
        self.updatevar(**kwargs)

        return self.prof, self.gvec, self.dprofdx, self.dgdx
    # end def

    def updatevar(self, **kwargs):
        if 'covmat' in kwargs:
            self.covmat = kwargs['covmat']
        if hasattr(self, 'covmat'):
            self.varprof = self.properror(self.XX, self.covmat, self.gvec)
            self.vardprofdx = self.properror(self.XX, self.covmat, self.dgdx)
        # end if
    # end def

    @staticmethod
    def properror(XX, covmat, gvec):
        return _ut.properror(XX, covmat, gvec)


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

#    def checkbounds(self, dat, ain, mag=None):
#        LB = _np.copy(self.Lbounds)
#        UB = _np.copy(self.Ubounds)
#        ain = _checkbounds(ain, LB=LB, UB=UB)
#        return dat, ain

    # ============================= #

    def scalings(self, xdat, ydat, **kwargs):
        self.xdat, self.ydat = _np.copy(xdat), _np.copy(ydat)
        if not hasattr(self, 'xoffset'):
            self.xoffset = _np.nanmin(xdat)
        if not hasattr(self, 'xslope'):
            self.xslope = _np.nanmax(xdat) - _np.nanmin(xdat)
        if not hasattr(self, 'offset'):
            self.offset = _np.nanmin(ydat)
        if not hasattr(self, 'slope'):
            self.slope = _np.nanmax(ydat)-_np.nanmin(ydat)
    # end def

    def scaledat(self, xdat, ydat, vdat, vxdat=None, **kwargs):
        """
        When fitting problems it is convenient to scale it to order 1:
            slope = _np.nanmax(pdat)-_np.nanmin(pdat)
            offset = _np.nanmin(pdat)

            pdat = (pdat-offset)/slope
            vdat = vdat/slope**2.0

        In general we cannot scale both x- and y-data due to non-linearities.
        Return scalings on all data by default, but overwrite in each model with NL
        (x-xmin)/(xmax-xmin) = (x-xoffset)/xslope
        """
        self.scalings(xdat, ydat, **kwargs)
        self.xdat = (xdat-self.xoffset)/self.xslope
        self.ydat = (ydat-self.offset)/self.slope
        self.vdat = vdat/(self.slope*self.slope)
        self.scaled = True
        if hasattr(self, 'XX'):
            self.XX = (self.XX-self.xoffset)/self.xslope
        if vxdat is None:
            self.vxdat = None
            return self.xdat, self.ydat, self.vdat
        else:
            self.vxdat = _np.copy(vxdat)/(self.xslope*self.xslope)
            return self.xdat, self.ydat, self.vdat, self.vxdat
        # end if
    # end def

    def unscaledat(self, **kwargs):
        """
        After fitting, then the algebra necessary to unscale the problem to original
        units is:
            prof = slope*prof+offset
            varp = varp*slope**2.0

            dprofdx = (slope*dprofdx)
            vardprofdx = slope**2.0 * vardprofdx

        Note that when scaling the problem, it is best to propagate errors from
        covariance / gvec / dgdx before rescaling because that is arbitrarily complex

        In general we cannot scale both x- and y-data due to non-linearities.
        Return scalings on all data by default, but overwrite in each model with NL
        (x-xmin)/(xmax-xmin) = (x-xoffset)/xslope
        """
#        if not hasattr(self, 'xslope'):  self.xslope = 1.0   # end if
#        if not hasattr(self, 'xoffset'): self.xoffset = 0.0  # end if
#        if not hasattr(self, 'slope'):   self.slope = 1.0    # end if
#        if not hasattr(self, 'offset'):  self.offset = 0.0   # end if
        self.update(**kwargs)
        if self.scaled:
            # Fitted data
            self.xdat = self.xdat*self.xslope+self.xoffset
            self.ydat = self.ydat*self.slope+self.offset
            self.vdat = self.vdat*self.slope*self.slope
            if hasattr(self, 'vxdat') and self.vxdat is not None:
                self.vxdat = self.vxdat*self.xslope*self.xslope
            # end if
            if hasattr(self, 'XX') and self.XX is not None:
                self.XX = self.XX*self.xslope + self.xoffset
            # Scaling model parameters to reproduce original data
            self.af = self.unscaleaf(self.af)

            # unscale the input covariance if there is one from a fitter
            if hasattr(self, 'covmat'):
                self.covmat = self.unscalecov(self.covmat)
            # end if

            # Update with the unscaled parameters
            self.update()

            # Calculated quantities  (if covmat already exists, this was done
            # in update through propper error propagation)
            if not hasattr(self, 'covmat'):
                self.varprof *= self.slope*self.slope
                self.vardprofdx *= (self.slope*self.slope)/(self.xslope*self.xslope)
            # end if
            self.scaled = False
       # end if
        if hasattr(self, 'vxdat') and self.vxdat is not None:
            return self.xdat, self.ydat, self.vdat, self.vxdat
        else:
            return self.xdat, self.ydat, self.vdat
        # end if
    # end def

#    @staticmethod
#    def unscaleaf(ain, slope, offset=0.0, xslope=1.0, xoffset=0.0):
#        ain = _np.copy(ain)
#        aout = _np.copy(ain)
#        print('Unscaling model parameters not supported: \n'+
#              '   Either not implemented in model or \n'+
#              '   there is a nonlinearity in the analytic model that precludes analytic scaling!')
    def unscaleaf(self, ain):
        return NotImplementedError
    # end def

#    def unscalecov(self, covin):
#        return NotImplementedError

    # ========================================================= #

    @classmethod
    def test_numerics(cls, **kwargs):
        num = kwargs.pop('num', 21)
        start = kwargs.pop('start', 1e-3)
        stop = kwargs.pop('stop', 0.99)
        endpoint = kwargs.pop('endpoint', True)
        dtyp = kwargs.pop('dtype', None)
        XX = kwargs.pop('XX', _np.linspace(start=start, stop=stop, num=int(num), endpoint=endpoint, dtype=dtyp))

        # call the analytic version and the numerical derivative version
        modanal = cls(XX, **kwargs)
        modnum = cls(XX, modanal.af, analytic=False)
#        na = len(modanal.af)

        # Manually call the second derivative because it is not calculated by default
        modnum.d2profdx2 = modnum.second_derivative(XX)
        modanal.d2profdx2 = modanal.second_derivative(XX)
        modnum.d2gdx2 = modnum.second_derivative_jacobian(XX)
        modanal.d2gdx2 = modanal.second_derivative_jacobian(XX)

        # ======= #

        # plot the model and its derivatives from both forms.  It should match.

        def getylims(func1, func2, _ax):
            ylims = _ax.get_ylim()
            mn = _np.min((_np.nanmin(func1.flatten()), _np.nanmin(func2.flatten()), ylims[0]))
            mx = _np.max((_np.nanmax(func1.flatten()), _np.nanmax(func2.flatten()), ylims[1]))
            if mn == 0:     mn = -0.1*(ylims[1]-ylims[0])   # end if
            if mx == 0:     mx = 0.1*(ylims[1]-ylims[0])    # end if
            ylims = (_np.nanmin((0.8*mn, 1.2*mn)), _np.nanmax((0.8*mx, 1.2*mx)))
            return ylims

        fignum_base = cls.__name__
        if hasattr(modanal, 'npoly'):
            fignum_base += '_npoly%i'%(modanal.npoly,)
        fignum, ii = fignum_base, 0
        while _plt.fignum_exists(fignum) and ii<10:
            fignum = fignum_base+'_%i'%(ii,)
            ii += 1

        _plt.figure(fignum)
        _ax1 = _plt.subplot(3,2,1)
        if not _np.isnan(modnum.prof).all():
            _ax1.plot(XX, modnum.prof, '-')
        _ax1.plot(XX, modanal.prof, '.')
        ylims = getylims(modnum.prof, modanal.prof, _ax1)
        _ax1.set_ylim(ylims)
        _ax1.set_title('model')

        _ax2 = _plt.subplot(3,2,2, sharex=_ax1)
        if not _np.isnan(modnum.gvec).all():
            _ax2.plot(XX, modnum.gvec.T, '-')
        _ax2.plot(XX, modanal.gvec.T, '.')
        ylims = getylims(modnum.gvec, modanal.gvec, _ax2)
        _ax2.set_ylim(ylims)
        _ax2.set_title('jacobian')

        _ax3 = _plt.subplot(3,2,3, sharex=_ax1)
        if not _np.isnan(modnum.dprofdx).all():
            _ax3.plot(XX, modnum.dprofdx, '-')
        _ax3.plot(XX, modanal.dprofdx, '.')
        ylims = getylims(modnum.dprofdx, modanal.dprofdx, _ax3)
        _ax3.set_ylim(ylims)
#        _ax3.set_title('derivative')
        _ax3.set_ylabel('deriv')

        _ax4 = _plt.subplot(3,2,4, sharex=_ax1)
        if not _np.isnan(modnum.dgdx).all():
            _ax4.plot(XX, modnum.dgdx.T, '-')
        _ax4.plot(XX, modanal.dgdx.T, '.')
        ylims = getylims(modnum.dgdx, modanal.dgdx, _ax4)
        _ax4.set_ylim(ylims)
#        _ax4.set_title('deriv jacobian')

        _ax5 = _plt.subplot(3,2,5, sharex=_ax1)
        if not _np.isnan(modnum.d2profdx2).all():
            _ax5.plot(XX, modnum.d2profdx2, '-')
        _ax5.plot(XX, modanal.d2profdx2, '.')
        ylims = getylims(modnum.d2profdx2, modanal.d2profdx2, _ax5)
        _ax5.set_ylim(ylims)
#        _ax5.set_title('2nd deriv')
        _ax5.set_ylabel('2nd deriv')

        _ax6 = _plt.subplot(3,2,6, sharex=_ax1)
        if not _np.isnan(modnum.d2gdx2).all():
            _ax6.plot(XX, modnum.d2gdx2.T, '-')
        _ax6.plot(XX, modanal.d2gdx2.T, '.')
        ylims = getylims(modnum.d2gdx2, modanal.d2gdx2, _ax6)
        _ax6.set_ylim(ylims)
#        _ax6.set_title('2nd deriv jacobian')

        _plt.tight_layout()

        # ======= #

        try:
            ii = -1
            # Check the maximum error between quantities
    #        rtol, atol = (1e-5, 1e-8)
            rtol, atol = (1e-5, _np.max((1e-8, _np.nanmax(1e-8*modanal.prof))))
            aerr = _np.nanmax(modnum.prof - modanal.prof)
            rerr = 1.0-_np.nanmax(modnum.prof/modanal.prof)
            assert _ut.allclose(modnum.prof, modanal.prof, rtol=rtol, atol=atol, equal_nan=True)  # same functions, this has to be true
    ##        assert _ut.allclose(modnum.gvec, modanal.gvec, rtol=rtol, atol=atol, equal_nan=True)  # jacobian of model

    ##        rtol, atol = (1e-4, 1e-5)
            rtol, atol = (1e-5, _np.max((1e-8, _np.nanmax(1e-8*modanal.dprofdx))))
            aerr = _np.nanmax(modnum.dprofdx - modanal.dprofdx)
            rerr = 1.0-_np.nanmax(modnum.dprofdx/modanal.dprofdx)
            assert _ut.allclose(modnum.dprofdx, modanal.dprofdx, rtol=rtol, atol=atol, equal_nan=True)  # derivative of model
    ##        assert _ut.allclose(modnum.dgdx, modanal.dgdx, rtol=rtol, atol=atol, equal_nan=True)  # jacobian of the model derivative

#            rtol, atol = (1e-3, 1e-4)
            rtol, atol = (1e-5, _np.max((1e-8, _np.nanmax(1e-8*modanal.d2profdx2))))
            aerr = _np.nanmax(modnum.d2profdx2 - modanal.d2profdx2)
            rerr = 1.0-_np.nanmax(modnum.d2profdx2/modanal.d2profdx2)
            assert _ut.allclose(modnum.d2profdx2, modanal.d2profdx2, rtol=rtol, atol=atol, equal_nan=True)  # model second derivative
    ##        assert _ut.allclose(modnum.d2gdx2, modanal.d2gdx2, rtol=rtol, atol=atol, equal_nan=True)  # jacobian of the model second derivative

            for ii in range(modanal.gvec.shape[0]):
#                print(ii)
#                rtol, atol = (1e-5, 1e-8)
                rtol, atol = (1e-5, _np.max((1e-8, _np.nanmax(1e-8*modanal.gvec[ii,:]))))  # mixed 2nd deriv implemented up to 4th order
                aerr = _np.nanmax(modnum.gvec[ii,:] - modanal.gvec[ii,:])
                rerr = 1.0-_np.nanmax(modnum.gvec[ii,:]/modanal.gvec[ii,:])
                assert _ut.allclose(modnum.gvec[ii,:], modanal.gvec[ii,:], rtol=rtol, atol=atol, equal_nan=True)  # jacobian of model

#                rtol, atol = (1e-4, 1e-3)
                rtol, atol = (1e-4, _np.max((1e-3, _np.nanmax(1e-3*modanal.dgdx[ii,:]))))  # mixed 2nd deriv implemented up to 4th order
                rerr = 1.0-_np.nanmax(modnum.dgdx[ii,:]/modanal.dgdx[ii,:])
                aerr = 1.0-_np.nanmax(modnum.dgdx[ii,:] - modanal.dgdx[ii,:])
#                _np.testing.assert_allclose(modnum.dgdx[ii,:], modanal.dgdx[ii,:], rtol=rtol)
                assert _ut.allclose(modnum.dgdx[ii,:], modanal.dgdx[ii,:], rtol=rtol, atol=atol, equal_nan=True)  # jacobian of the model derivative

                rtol, atol = (1e-3, _np.max((1e-3, _np.nanmax(1e-3*modanal.d2gdx2[ii,:])))) # mixed 3rd deriv. implemented only up to 2nd order
                aerr = _np.nanmax(modnum.d2gdx2[ii,:] - modanal.d2gdx2[ii,:])
                rerr = 1.0-_np.nanmax(modnum.d2gdx2[ii,:]/modanal.d2gdx2[ii,:])
                assert _ut.allclose(modnum.d2gdx2[ii,:], modanal.d2gdx2[ii,:], rtol=rtol, atol=atol, equal_nan=True)  # jacobian of the model second derivative
        except:
            print('Numerical testing failed at parameter %i \n abs. err. level %e\n rel. err. level %e'%(ii, aerr, rerr))
            raise
        # end try
    # end def test numerics
# end def class


# ========================================================================== #
# ========================================================================== #


