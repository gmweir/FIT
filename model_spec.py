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
from pybaseutils import utils as _ut
#from pybaseutils.utils import properror, sech

# ========================================================================== #
# ========================================================================== #

class ModelClass(Struct):
    def __init__(self, XX=None, af=None, **kwargs):
        LB, UB, fixed = self.defaults(**kwargs)
        if af is None:   af = _np.copy(self._af)    # end if
        self.af = _np.copy(af)
        self.Lbounds = _np.copy(LB)
        self.Ubounds = _np.copy(UB)
        self.fixed = _np.copy(fixed)
        kwargs.setdefault('analytic', True) # default to analytic derivatives/jacob. etc.
#        kwargs.setdefault('analytic', False) # default to analytic derivatives/jacob. etc.
        self.machine_epsilon = (_np.finfo(_np.float64).eps)
        self.__dict__.update(kwargs)
        self.kwargs = kwargs
        if XX is not None:
            self.update(XX=XX)
    # end def __init__

    def parse_in(self, XX, aa, **kwargs):
        if aa is None: aa = _np.copy(self.af)  # end if
        if XX is None: XX = _np.copy(self.XX)  # end if
        XX = _np.copy(XX)
        aa = _np.copy(aa)
        if len(kwargs.keys()) == 0:
            kwargs = self.kwargs
        return XX, aa, kwargs

    def model(self, XX, aa=None, **kwargs):
        """
        Returns the derivative of the model.
            f(x) = f(x, a, b, c, ...)
            out: f(x)
        """
        XX, aa, kwargs = self.parse_in(XX, aa, **kwargs)
#        return self._model(XX, aa, **kwargs)
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
        hstep = kwargs.setdefault('hstep', None)
        if hstep is None:
            tst = self.model(XX, aa, **kwargs)
            dmax = _np.max((tst.max()/0.2, 5.0))

            dmax = _np.max((_np.max(_np.abs(aa))/0.2, dmax))   # maximum derivative, arbitraily say 10/0.2
            hh = _ut.cbrt(6.0*self.machine_epsilon/dmax)
#            hh = _np.sqrt(self.machine_epsilon)*XX
#            if (hh<100*self.machine_epsilon).any():
##                hh[hh<_np.sqrt(self.machine_epsilon)] = _np.sqrt(self.machine_epsilon)
        else:
            hh = hstep
        # end if
        return self._1stderiv_fd(self.model, XX, hh, aa, order=4, **kwargs)
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
        hstep = kwargs.setdefault('hstep', None)
        if hstep is None:
            tst = self.model(XX, aa, **kwargs)
            dmax = _np.max((tst.max()/0.2, 5.0))

            dmax = _np.max((_np.max(_np.abs(aa))/0.2, dmax))   # maximum derivative, arbitraily say 10/0.2
            hh = _ut.cbrt(6.0*self.machine_epsilon/dmax)
            hh = hh*_np.ones_like(aa)

#            hh = _np.sqrt(self.machine_epsilon)*aa
#            if (hh<100*self.machine_epsilon).any():
#                hh[hh<_np.sqrt(self.machine_epsilon)] = _np.sqrt(self.machine_epsilon)
        else:
            hh = hstep
        # end if

        numfit = _np.size(aa)
        gvec = _np.zeros((numfit, _np.size(XX)), dtype=_np.float64)
        for ii in range(numfit):
            tmp = _np.zeros_like(aa)
            tmp[ii] += 1.0
#            gvec[ii, :] = (self.model(XX, aa + hstep*tmp, **kwargs) - self.model(XX, aa-hstep*tmp, **kwargs))/(2*hstep)
            # 1st derivative, 2nd order accurate
#            gvec[ii, :] = (self.model(XX, aa + hh*tmp, **kwargs) - self.model(XX, aa-hh*tmp, **kwargs))/(2*hh[ii])
            # 1st derivative, 4th order accurate
            gvec[ii, :] = (self.model(XX, aa-2.0*hh*tmp, **kwargs) - 8.0*self.model(XX, aa-hh*tmp, **kwargs)
                        + 8.0*self.model(XX, aa+hh*tmp, **kwargs) - self.model(XX, aa+2.0*hh*tmp, **kwargs))/(12.0*hh[ii])

            # 2nd order accuratefor unequally spaced data: Lagrange interpolation
#            gvec[ii, :] = (self.model(XX, aa -hh*tmp, **kwargs)*(2.0*aa[ii])/(()*())
        # end for
        return gvec

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
        hstep = kwargs.setdefault('hstep', None)
        if hstep is None:
#            tst = self.derivative(XX, aa, **kwargs)
#            dmax = _np.max((tst.max()/0.1, 5.0))
#            dmax = _np.max((_np.max(_np.abs(aa))/0.2, dmax))   # maximum derivative, arbitraily say 10/0.2

            tst = self.jacobian(XX, aa, **kwargs)
            dmax = _np.max((tst.max()/0.1, 5.0))

            hh = _ut.cbrt(6.0*self.machine_epsilon/dmax)      # estimation error: R = f'''(x)/6 * h^2
#            hh = hh*_np.ones_like(aa)
#            hh = _np.sqrt(self.machine_epsilon)*aa
#            if (hh<100*self.machine_epsilon).any():
#                hh[hh<_np.sqrt(self.machine_epsilon)] = _np.sqrt(self.machine_epsilon)
        else:
            hh = hstep
        # end if
        dgdx = self._1stderiv_fd(self.jacobian, XX, hh, aa, order=6, **kwargs)
        return dgdx

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
        hstep = kwargs.setdefault('hstep', None)
        if hstep is None:
            tst = self.derivative(XX, aa, **kwargs)
            dmax = _np.max((tst.max()/0.2, 5.0))
#            dmax = 1
#            dmax = _np.max((_np.max(_np.abs(aa))/0.2, dmax))   # maximum derivative, arbitraily say 10/0.2
            hh = _ut.cbrt(6.0*self.machine_epsilon/dmax)
            hh = _np.power(hh, 0.5)
#            hh = 10.0*hh
        else:
            hh = hstep
        # end if
        return self._2ndderiv_fd(self.model, XX, hh, aa, order=6, **kwargs)

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
        hstep = kwargs.setdefault('hstep', None)
        if hstep is None:
#            tst = self.second_derivative(XX, aa, **kwargs)
#            dmax = _np.max((tst.max()/0.1, 5.0))
#            dmax = _np.max(_np.vstack(_np.abs(aa)/0.1, dmax))   # maximum derivative, arbitraily say 10/0.2
#            hh = _ut.cbrt(6.0*self.machine_epsilon/dmax)
#            hh = hh*_np.ones_like(aa)

            tst = self.derivative_jacobian(XX, aa, **kwargs)
            tst = _np.diff(tst, axis=1)/_np.diff(XX)
            dmax = _np.max((tst.max(), 5.0))
#            dmax = _np.min((dmax,10))
            hh = _ut.cbrt(6.0*self.machine_epsilon/dmax)
            hh = _np.power(hh, 0.5)
        else:
            hh = hstep
        # end if
        d2gdx2 = self._2ndderiv_fd(self.jacobian, XX, hh, aa, order=6, **kwargs)
        return d2gdx2

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
        hstep = kwargs.setdefault('hstep', None)
        if hstep is None:
            tst = self.jacobian(XX, aa, **kwargs)
            dmax = _np.max((tst.flatten().max()/0.2, 5.0))
            dmax = _np.max(_np.max(aa)/0.2, dmax)   # maximum derivative, arbitraily say 10/0.2
            hh = _ut.cbrt(6.0*self.machine_epsilon/dmax)
#            hh = _np.power(hh, 0.5)
            hh = hh*_np.ones_like(hh)

#            hh = _np.sqrt(self.machine_epsilon)*aa
#            if (hh<100*self.machine_epsilon).any():
#                hh[hh<_np.sqrt(self.machine_epsilon)] = _np.sqrt(self.machine_epsilon)
        else:
            hh = hstep
        # end if
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

    def _1stderiv_fd(self, func, XX, hh, aa, order=2, **kwargs):
        if 1:
#        if order<3:
            # 2nd order accurate
            return (-func(XX-hh, aa, **kwargs) + func(XX+hh, aa, **kwargs))/(2.0*hh)
        elif order<5:
            # 4th order accurate
            return (func(XX-2*hh, aa, **kwargs) - 8.0*func(XX-hh, aa, **kwargs)
                 + 8.0*func(XX+hh, aa, **kwargs) - func(XX+2.0*hh, aa, **kwargs))/(12.0*hh)
        elif order<7:
            # 6th order accurate
            return (-func(XX-3.0*hh, aa, **kwargs) + 9.0*func(XX-2.0*hh, aa, **kwargs)
                  -45.0*func(XX-hh, aa, **kwargs) + 45.0*func(XX+hh, aa, **kwargs)
                  -9.0*func(XX+2.0*hh, aa, **kwargs) + func(XX+3.0*hh, aa, **kwargs))/(60.0*hh)
        elif order<9:
            # 8th order accurate
            return (1.0/280.0*func(XX-4.0*hh, aa, **kwargs) -4.0/105.0*func(XX-3.0*hh, aa, **kwargs)
                  + 1.0/5.0*func(XX-2.0*hh, aa, **kwargs) -4.0/5.0*func(XX-1.0*hh, aa, **kwargs)
                  + 4.0/5.0*func(XX+1.0*hh, aa, **kwargs) -1.0/5.0*func(XX+2.0*hh, aa, **kwargs)
                  + 4.0/105.0*func(XX+3.0*hh, aa, **kwargs) - 1.0/280.0*func(XX+4.0*hh, aa, **kwargs))/(hh)

    def _2ndderiv_fd(self, func, XX, hh, aa, order=2, **kwargs):
        if 1:
#        if order<3:
            # 2nd order accurate
            return (func(XX-1.0*hh, aa, **kwargs) -2.0*func(XX, aa, **kwargs)+func(XX+1.0*hh, aa, **kwargs))/(_np.power(hh,2.0))
        elif order<5:
            # 4th order accurate
            return (-1.0*func(XX-2.0*hh, aa, **kwargs)+16.0*func(XX-hh, aa, **kwargs)
                -30.0*func(XX, aa, **kwargs)+16.0*func(XX+1.0*hh, aa, **kwargs)
                    -1.0*func(XX+2.0*hh, aa, **kwargs))/(12.0*_np.power(hh,2.0))
        elif order<7:
            # 6th order accurate
            return (1.0/90.0*func(XX-3.0*hh, aa, **kwargs)-3.0/20.0*func(XX-2*hh, aa, **kwargs)
                  + 3.0/2.0*func(XX-hh, aa, **kwargs) - 49.0/18.0*func(XX, aa, **kwargs)
                  + 3.0/2.0*func(XX+1.0*hh, aa, **kwargs) - 3.0/20.0*func(XX+2.0*hh, aa, **kwargs)
                  + 1.0/90.0*func(XX+3.0*hh, aa, **kwargs) )/(_np.power(hh,2.0))
        elif order<9:
            # 8th order accurate
            return (-1.0/560.0*func(XX-4.0*hh, aa, **kwargs) + 8.0/315.0*func(XX-3.0*hh, aa, **kwargs)
                    -1.0/5.0*func(XX-2.0*hh, aa, **kwargs) +8.0/5.0*func(XX-1.0*hh, aa, **kwargs)
                    -205.0/72.0*func(XX, aa, **kwargs)+8.0/5.0*func(XX+1.0*hh, aa, **kwargs)
                    -1.0/5.0*func(XX+2.0*hh, aa, **kwargs)+8.0/315.0*func(XX+3.0*hh, aa, **kwargs)
                    -1.0/560.0*func(XX+4.0*hh, aa, **kwargs) )/_np.power(hh, 2.0)

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

    def checkbounds(self, dat, ain, mag=None):
        LB = _np.copy(self.Lbounds)
        UB = _np.copy(self.Ubounds)
        ain = _checkbounds(ain, LB=LB, UB=UB)
        return dat, ain

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

    def unscalecov(self, covin):
        return NotImplementedError

    # ========================================================= #

    @classmethod
    def test_numerics(cls, **kwargs):
        XX = kwargs.pop('XX', _np.linspace(1e-3, 0.99, num=100))

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

        _plt.figure()
        _ax1 = _plt.subplot(3,2,1)
        _ax1.plot(XX, modnum.prof, '-')
        _ax1.plot(XX, modanal.prof, '.')
        _ax1.relim()
        _ax1.autoscale_view()
        _ax1.set_title('model')

        _ax2 = _plt.subplot(3,2,2, sharex=_ax1)
        _ax2.plot(XX, modnum.gvec.T, '-')
        _ax2.plot(XX, modanal.gvec.T, '.')
        _ax2.relim()
        _ax2.autoscale_view()
        _ax2.set_title('jacobian')

        _ax3 = _plt.subplot(3,2,3, sharex=_ax1)
        _ax3.plot(XX, modnum.dprofdx, '-')
        _ax3.plot(XX, modanal.dprofdx, '.')
        _ax3.relim()
        _ax3.autoscale_view()
#        _ax3.set_title('derivative')
        _ax3.set_ylabel('deriv')

        _ax4 = _plt.subplot(3,2,4, sharex=_ax1)
        _ax4.plot(XX, modnum.dgdx.T, '-')
        _ax4.plot(XX, modanal.dgdx.T, '.')
        _ax4.relim()
        _ax4.autoscale_view()
#        _ax4.set_title('deriv jacobian')

        _ax5 = _plt.subplot(3,2,5, sharex=_ax1)
        _ax5.plot(XX, modnum.d2profdx2, '-')
        _ax5.plot(XX, modanal.d2profdx2, '.')
        _ax5.relim()
        _ax5.autoscale_view()
#        _ax5.set_title('2nd deriv')
        _ax5.set_ylabel('2nd deriv')

        _ax6 = _plt.subplot(3,2,6, sharex=_ax1)
        _ax6.plot(XX, modnum.d2gdx2.T, '-')
        _ax6.plot(XX, modanal.d2gdx2.T, '.')
        _ax6.relim()
        _ax6.autoscale_view()
#        _ax6.set_title('2nd deriv jacobian')

        _plt.tight_layout()

        # ======= #

#        # Check the maximum error between quantities
#        rtol, atol = (1e-5, 1e-8)
#        assert _ut.allclose(modnum.prof, modanal.prof, rtol=rtol, atol=atol, equal_nan=True)  # same functions, this has to be true
##        assert _ut.allclose(modnum.gvec, modanal.gvec, rtol=rtol, atol=atol, equal_nan=True)  # jacobian of model
##        rtol, atol = (1e-4, 1e-5)
#        assert _ut.allclose(modnum.dprofdx, modanal.dprofdx, rtol=rtol, atol=atol, equal_nan=True)  # derivative of model
##        assert _ut.allclose(modnum.dgdx, modanal.dgdx, rtol=rtol, atol=atol, equal_nan=True)  # jacobian of the model derivative
##        rtol, atol = (1e-3, 1e-4)
#        assert _ut.allclose(modnum.d2profdx2, modanal.d2profdx2, rtol=rtol, atol=atol, equal_nan=True)  # model second derivative
##        assert _ut.allclose(modnum.d2gdx2, modanal.d2gdx2, rtol=rtol, atol=atol, equal_nan=True)  # jacobian of the model second derivative
#        try:
#            for ii in range(na):
#                print(ii)
#                rtol, atol = (1e-5, 1e-8)
#                assert _ut.allclose(modnum.gvec[ii,:], modanal.gvec[ii,:], rtol=rtol, atol=atol, equal_nan=True)  # jacobian of model
#                rtol, atol = (1e-4, 1e-5)
#                assert _ut.allclose(modnum.dgdx[ii,:], modanal.dgdx[ii,:], rtol=rtol, atol=atol, equal_nan=True)  # jacobian of the model derivative
#                rtol, atol = (1e-3, 1e-3)
#                assert _ut.allclose(modnum.d2gdx2[ii,:], modanal.d2gdx2[ii,:], rtol=rtol, atol=atol, equal_nan=True)  # jacobian of the model second derivative
#        except:
#            print('Testing failed at parameter %i'%(ii,))
#            raise
#        # end try
    # end def test numerics
# end def class



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


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        xs = self.xslope
        xo = self.xoffset
        ys = self.slope
        yo = self.offset
        aout[0] = ys*ain[0] / xs
        aout[1] = ys*(ain[1]-xo*ain[0]/xs) + yo
        return aout

    def unscalecov(self, covin):
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

        xo = self.xoffset
        xs = self.xslope
#        yo = self.offset
        ys = self.slope
        covout[0,0] = _np.power(ys/xs, 2.0)*covin[0,0]
        covout[0,1] = _np.power(ys, 2.0)*( covin[0,1]/xs - xo*covin[0,0]/_np.power(xs, 2.0) )
        covout[1,0] = _np.copy(covout[0,1])
        covout[1,1] = _np.power(ys, 2.0)*( covin[1,1] + _np.power(xo/xs, 2.0)*covin[0,0] - 2.0*xo/xs*covin[1,0])
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
    _af = _np.asarray([0.0]+[1.0, 5.0, _np.pi/3], dtype=_np.float64)
    _LB = _np.asarray([-_np.inf]+[-_np.inf,     0.0, -2*_np.pi], dtype=_np.float64)
    _UB = _np.asarray([ _np.inf]+[ _np.inf, _np.inf,  2*_np.pi], dtype=_np.float64)
    _fixed = _np.zeros( (4,), dtype=int)
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
        else:
            for ii in range(self.nfreqs):
                self._af[3*ii + 1] = self._af[1]/(ii+1)  # amplitudes
                self._af[3*ii + 2] = self._af[2]*(ii+1)  # i'th harm. of default
                self._af[3*ii + 3] = _np.random.uniform(0.0, _np.pi, size=1)  # phase of i'th harm. of default
            # end for
        # end if

    def _default_plot(self, XX=None):
#        if XX is None:  XX = _np.copy(self.XX) # end if
        if XX is None:
            XX = _np.linspace(-2.0/self.fmod, 2.0/self.fmod, num=500)
        _plt.figure()
        _plt.plot(XX, self.model(XX, self._af))

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
        w = _np.atleast_2d(2*_np.pi*aa[2::3]).T  # cyclic freq. (nfreq,1)
        p = _np.atleast_2d(aa[3::3]).T*_np.ones_like(XX) # phase (nfreq, nx)
        return 0.5*aa[0] + _np.sum( a*_np.sin(w*XX+p), axis=0)

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        f = 0.5*ao + sum_ii a_ii *sin((2pi*f_ii)*XX+p_ii)
        dfdx = sum_ii 2*pi*f_ii*a_ii *cos((2pi*f_ii)*XX+p_ii)
        """
        XX = _np.copy(XX)
#        XX = _np.atleast_2d(XX)                  # (1,len(XX))
#        a = _np.atleast_2d(aa[1::3]).T*_np.ones_like(XX) # amp (nfreq, nx)
#        w = _np.atleast_2d(2*_np.pi*aa[2::3]).T  # cyclic freq. (nfreq,1)
#        p = _np.atleast_2d(aa[3::3]).T*_np.ones_like(XX) # phase (nfreq, nx)
#        return _np.sum( (w*_np.ones_like(XX))*a*_np.cos(w*XX+p), axis=0)
        nfreqs = ModelSines.getnfreqs(aa)
        ysum = 0.0 #_np.zeros_like(XX)
        for ii in range(nfreqs):
            a = aa[1+3*ii]
            f = aa[2+3*ii]
            p = aa[3+3*ii]
            ysum += 2.0*_np.pi*f*a*_np.cos(2.0*_np.pi*f*XX+p)
        # end for
        return ysum

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
        a = _np.atleast_2d(aa[1::3]).T*_np.ones_like(XX) # amp (nfreq, nx)
        w = _np.atleast_2d(2*_np.pi*aa[2::3]).T  # cyclic freq. (nfreq,1)
        p = _np.atleast_2d(aa[3::3]).T*_np.ones_like(XX) # phase (nfreq, nx)

        gvec = _np.zeros( (len(aa), _np.size(XX)), dtype=_np.float64)
        gvec[0, :] = 0.5
        gvec[1::3, :] = _np.sin(w*XX+p)
        gvec[2::3, :] = (2.0*_np.pi*_np.ones_like(w)*XX)*a*_np.cos(w*XX+p)
        gvec[3::3, :] = a*_np.cos(w*XX+p)

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
        w = 2.0*_np.pi*_np.copy(aa[2::3])
        p = _np.copy(aa[3::3])

        XX = _np.atleast_2d(XX)  # (1,nx)
        tmp = _np.ones_like(XX)  # (1,nx)
        a = _np.atleast_2d(a).T  # (nfreq,1)
        w = _np.atleast_2d(w).T  # (nfreq,1)
        p = _np.atleast_2d(p).T  # (nfreq,1)

        return -1*_np.sum( (_np.power(w, 2.0)*tmp) * _np.sin(w*XX + p*tmp)  , axis=0)

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
        XX = _np.atleast_2d(XX)                  # (1,len(XX))
        a = _np.atleast_2d(aa[1::3]).T*_np.ones_like(XX) # amp (nfreq, nx)
        w = _np.atleast_2d(2*_np.pi*aa[2::3]).T  # cyclic freq. (nfreq,1)
        p = _np.atleast_2d(aa[3::3]).T*_np.ones_like(XX) # phase (nfreq, nx)

        dgdx = _np.zeros( (len(aa), _np.size(XX)), dtype=_np.float64)
        dgdx[0, :] = 0.0
        dgdx[1::3, :] = (w*_np.ones_like(XX))*_np.cos(w*XX+p)
        dgdx[2::3, :] = (2.0*_np.pi)*a*(_np.cos(w*XX+p) - (w*XX)*_np.sin(w*XX+p))
        dgdx[3::3, :] = (-1*w*_np.ones_like(XX))*a*_np.sin(w*XX+p)
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
        d2gdx2[1::3, :] = -1*(_np.power(w, 2.0)*tmp)*_np.sin(w*XX+p*tmp)
        d2gdx2[2::3, :] = ( -2.0*_np.power(2.0*_np.pi, 2.0)*((f*a)*tmp)*_np.sin(w*XX+p*tmp)
                            - ((_np.power(w, 2.0)*a)*(2.0*_np.pi*XX))*_np.cos(w*XX+p*tmp))
        d2gdx2[3::3, :] = -1*((_np.power(w, 2.0)*a)*tmp)*_np.cos(w*XX+p*tmp)
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
        a = _np.atleast_2d(aa[1::3]).T*_np.ones_like(XX) # amp (nfreq, nx)
        p = _np.atleast_2d(aa[3::3]).T*_np.ones_like(XX) # phase (nfreq, nx)

        hess = _np.zeros((numfit, numfit, _np.size(XX)), dtype=_np.float64)
        # diagonal:
        # hess[0::3, 0::3, :] = 0.0                                                    # d2fao2
        # hess[1::3, 1::3, :] = 0.0                                                    # d2fdai2
        hess[2::3, 2::3, :] = -1.0*(f*_np.power(2.0*_np.pi*XX, 2.0))*a*_np.sin(w*XX+p) # d2fdfi2
        hess[3::3, 3::3, :] = -1.0*a*_np.sin(w*XX+p)                                   # d2fdpi2

        # Upper triangle
        # hess[0, :, :] = 0.0                                                                   #d2fdaod_
        hess[1::3, 2::3, :] = 2.0*_np.pi*(_np.ones(w.shape, dtype=_np.float64)*XX)*_np.cos(w*XX+p)    # d2fdaidfi
        hess[1::3, 3::3, :] =_np.cos(w*XX+p)                                                          # d2fdaidpi
        hess[2::3, 3::3, :] = -2.0*_np.pi*(_np.ones(w.shape, dtype=_np.float64)*XX)*a*_np.sin(w*XX+p) # d2fdfidpi

        # Lower triangle by symmetry
        # hess[:, 0::3, :] =
        for ii in range(numfit):    # TODO!:  CHECK THIS!
            for jj in range(numfit):
                hess[jj, ii, :] = hess[ii, jj, :]
            # end for
        # end for
#        hess[2::3, 1::3, :] = hess[1::3, 2::3, :].T
#        hess[3::3, 1::3, :] = hess[1::3, 3::3, :].T
#        hess[3::3, 2::3, :] = hess[2::3, 3::3, :].T
        return hess

    # ====================================== #

#    def scaledat(self, xdat, ydat, vdat, vxdat=None, **kwargs):
#        super(ModelSines, self).scaledat(xdat, ydat, vdat, vxdat=vxdat, **kwargs)

    def unscaleaf(self, ain):
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
                p_i = p_i'-2pif_i'xo/xs
            [a_i, f_i, p_i] = [ys*a_i', f_i'/xs, p_i' - 2pif_i'*xo/xs]
        """
        ain = _np.copy(ain)
        aout = _np.zeros_like(ain)
        ys = self.slope
        yo = self.offset
        xs = self.xslope
        xo = self.xoffset

        aout[0] = 2*yo + ys*ain[0]
        aout[1::3] = ys*ain[1::3]
        aout[2::3] = ain[2::3]/xs
        aout[3::3] = ain[3::3] - 2.0*_np.pi*ain[2::3]*xo/xs
        return aout
#        for ii in range(nfreq):
#            aout[3*ii+1] = ain[3*ii+1]*ys
#            aout[3*ii+2] = ain[3*ii+2]/xs
#            aout[3*ii+3] = ain[3*ii+3] - 2*_np.pi*ain[3*ii+2]*xo/xs
#        # end for
#        return aout

    def unscalecov(self, covin):
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
        covin = _np.copy(covin)
        covout = _np.zeros_like(covin)

        nfreq = self.getnfreqs(self.af)
#        numfit = _np.shape(covout)[0]
        xo = self.xoffset
        xs = self.xslope
#        yo = self.offset
        ys = self.slope

        # loop over covariance entries
        covout[0,0] = _np.power(ys, 2.0)*covin[0,0]
        for ii in range(nfreq):
            # diagonal terms (block-tridiagonal matrix)
            covout[1+3*ii, 1+3*ii] = _np.power(ys, 2.0)*covin[1+3*ii, 1+3*ii]
            covout[2+3*ii, 2+3*ii] = _np.power(1.0/xs, 2.0)*covin[2+3*ii, 2+3*ii]
            covout[3+3*ii, 3+3*ii] = (covin[3+3*ii, 3+3*ii]
                   + _np.power(2.0*_np.pi*xo/xs, 2.0)*covin[2+3*ii,2+3*ii]
                   - 2.0*_np.pi*xo/xs*covin[3+3*ii, 2+3*ii])

            # top row
            covout[0,(1+3*ii)::3] = _np.power(ys, 2.0)*covin[0,(1+3*ii)::3]  # ao-a_i
            covout[0,(2+3*ii)::3] = (ys/xs)*covin[0,(2+3*ii)::3]             # ao-f_i
            covout[0,(3+3*ii)::3] = ys*(covin[0,(3+3*ii)::3] - 2.0*_np.pi*xo/xs*covin[0,(2+3*ii)::3]) # ao-p_i

            # first column
            covout[(1+3*ii)::3,0] = _np.copy(covout[0,(1+3*ii)::3])  # a_i-ao
            covout[(2+3*ii)::3,0] = _np.copy(covout[0,(2+3*ii)::3])  # f_i-ao
            covout[(3+3*ii)::3,0] = _np.copy(covout[0,(3+3*ii)::3])  # p_i-ao

            for jj in range(ii+1):
                # mixed terms from sines in upper triangle:
                covout[1+3*ii,2+3*jj] = (ys/xs)*covin[1+3*ii,2+3*jj] # a_i-f_i
                covout[1+3*ii,3+3*jj] = ys*(covin[1+3*ii,3+3*jj]
                               - 2.0*_np.pi*xo/xs*covin[1+3*ii,2+3*jj])    # a_i-p_i
                covout[2+3*ii,3+3*jj] = (1.0/xs)*( covin[2+3*ii, 3+3*jj]
                               - 2.0*_np.pi*xo/xs*covin[2+3*ii,2+3*jj] )   # f_i-p_i

                # lower triangle is hermitian symmetric to upper triangle
                covout[2+3*ii,1+3*jj] = _np.copy(covout[1+3*ii,2+3*jj])  # f_i-a_i
                covout[3+3*ii,1+3*jj] = _np.copy(covout[1+3*ii,3+3*jj])  # p_i-a_i
                covout[3+3*ii,2+3*jj] = _np.copy(covout[2+3*ii,3+3*jj])  # p_i-f_i

#        aout[0] = 2*yo + ys*ain[0]
#        aout[1::3] = ys*aout[1::3]
#        aout[2::3] = aout[2::3]/xs
#        aout[3::3] = aout[3::3] - 2*_np.pi*aout[3::3]*xo

#        # diagonal terms
#        covout[0,0] = _np.power(ys, 2.0)*covin[0,0]
#        covout[1::3, 1::3] = _np.power(ys, 2.0)*covin[1::3, 1::3]
#        covout[2::3, 2::3] = _np.power(1.0/xs, 2.0)*covin[2::3, 2::3]
#        covout[3::3, 3::3] = (covin[3::3, 3::3] + _np.power(2.0*_np.pi*xo/xs, 2.0)*covin[2::3,2::3]
#               - 2.0*_np.pi*xo/xs*covin[3::3, 2::3])
#
#        for ii in range(numfit-3):
#            # first row
#            covout[0,(1+3*ii)::3] = _np.power(ys, 2.0)*covin[0,(1+3*ii)::3]  # ao-a_i
#            covout[0,(2+3*ii)::3] = (ys/xs)*covin[0,(2+3*ii)::3]             # ao-f_i
#            covout[0,(3+3*ii)::3] = ys*(covin[0,(3+3*ii)::3] - 2.0*_np.pi*xo/xs*covin[0,(2+3*ii)::3]) # ao-p_i
#
#            # first column
#            covout[(1+3*ii)::3,0] = _np.copy(covout[0,(1+3*ii)::3])  # a_i-ao
#            covout[(2+3*ii)::3,0] = _np.copy(covout[0,(2+3*ii)::3])  # f_i-ao
#            covout[(3+3*ii)::3,0] = _np.copy(covout[0,(3+3*ii)::3])  # p_i-ao
#
#            # mixed terms from sines:
#            covout[(1+3*ii)::3,(2+3*ii)::3] = (ys/xs)*covin[(1+3*ii)::3,(2+3*ii)::3] # a_i-f_i
#            covout[(1+3*ii)::3,(3+3*ii)::3] = ys*(covin[(1+3*ii)::3,(3+3*ii)::3]
#                           - 2.0*_np.pi*xo/xs*covin[(1+3*ii)::3,(2+3*ii)::3])    # a_i-p_i
#            covout[(2+3*ii)::3,(3+3*ii)::3] = (1.0/xs)*( covin[(2+3*ii)::3, (3+3*ii)::3]
#                           - 2.0*_np.pi*xo/xs*covin[(2+3*ii)::3,(2+3*ii)::3] )   # f_i-p_i
#
#            covout[(2+3*ii)::3,(1+3*ii)::3] = _np.copy(covout[(1+3*ii)::3,(2+3*ii)::3])  # f_i-a_i
#            covout[(3+3*ii)::3,(1+3*ii)::3] = _np.copy(covout[(1+3*ii)::3,(3+3*ii)::3])  # p_i-a_i
#            covout[(3+3*ii)::3,(2+3*ii)::3] = _np.copy(covout[(2+3*ii)::3,(3+3*ii)::3])  # p_i-f_i
#        # end for
        return covout

    def scalings(self, xdat, ydat, **kwargs):
        """
        Although we can scale the input fitting data. The frequency scaling isn't working yet
        """
        self.xslope = 1.0
        self.xoffset = 0.0
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
    _af = _np.asarray([   5.0,     1.0,     0.5,     0.5,    0.25,    0.25], dtype=_np.float64)
    _LB = _np.asarray([  1e-18,-_np.inf,-_np.inf,-_np.inf,-_np.inf,-_np.inf], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf, _np.inf, _np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (6,), dtype=int)
    def __init__(self, XX=None, af=None, **kwargs):
        # Tile defaults to the number of frequencies requested
        if af is not None:
            self.nfreqs = self.getnfreqs(af)
        else:
            self.nfreqs = kwargs.setdefault('nfreqs', 2)
            self.fmod = kwargs.setdefault('fmod', self._af[0])
            self._af[0] = _np.copy(self.fmod)
        # end if

        for ii in range(self.nfreqs):
            self._af = _np.asarray(self._af.tolist()+[     0.0,     0.0], dtype=_np.float64)
            self._LB = _np.asarray(self._LB.tolist()+[-_np.inf,-_np.inf], dtype=_np.float64)
            self._UB = _np.asarray(self._UB.tolist()+[ _np.inf, _np.inf], dtype=_np.float64)
            self._fixed = _np.asarray(self._fixed.tolist()+[ 0, 0], dtype=_np.float64)
        # end for

        self._shape(**kwargs)
        super(ModelFourier, self).__init__(XX, af, **kwargs)
    # end def __init__

    def _shape(self, **kwargs):
        sq = kwargs.setdefault('shape', 'sine')
        duty = kwargs.setdefault('duty', 0.5)
        if sq.lower().find('square')>-1:# and duty!=0.5:
            # duty cycled square wave
            # an = 2A/npi * sin(n*pi*tp/T)
            # an = 2A/npi * sin(n*pi*dutycycle)
            ff = _np.copy(self._af[0])
            AA = 1.0
            self._af = _np.zeros((2+2*self.nfreqs,), dtype=_np.float64)
            self._af[0] = ff
            self._af[1] = self._af[1] + AA*duty
            for ii in range(self.nfreqs):
                if (ii+1) % 2 == 0:  # if the frequency is even
                    continue
#                nn = 2*(ii+1)-1
                nn = ii+1
                self._af[2*ii + 2 + 0] = 2.0*AA*_np.sin(nn*_np.pi*duty)/(_np.pi*nn)  # amplitudes
                self._af[2*ii + 2 + 1] = 0.0  # amplitudes of sine
            # end for
        else:
            for ii in range(self.nfreqs):
                self._af[2*ii + 2 + 0 ] = 0.5*self._af[1]/(ii+1)  # amplitudes
                self._af[2*ii + 2 + 1 ] = 0.0  # i'th harm. of default
#                self._af[2*ii + 2 + 1 ] = 0.5*self._af[1]/(ii+1)  # i'th harm. of default
            # end for
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
        aout[1::3] = _np.sqrt( _np.power(aa[2::2], 2.0) + _np.power(aa[3::2], 2.0)) # amplitude
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
        w = 2.0*_np.pi*_np.copy(aa[0])
        ai = _np.atleast_2d(_np.copy(aa[2::2])).T
        bi = _np.atleast_2d(_np.copy(aa[3::2])).T

        tmp = _np.ones_like(_np.atleast_2d(XX))
        ii = _np.atleast_2d(_np.asarray(range(nfreqs), dtype=_np.float64) + 1.0).T
        return _np.power(w, 3.0)*_np.sum( (_np.power(ii, 3.0)*tmp)*(
                (ai*tmp)*_np.sin(w*(ii*_np.atleast_2d(XX)))
              - (bi*tmp)*_np.cos(w*(ii*_np.atleast_2d(XX)))
                ), axis=0)

    @staticmethod
    def _partial(XX, aa, **kwargs):
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
        """
        nfreqs = ModelFourier.getnfreqs(aa)
        f  = aa[0]
        ao = aa[1]
        gvec = ao*_np.zeros( (len(aa), _np.size(XX)), dtype=_np.float64)
        gvec[0, :] = (XX/f)*ModelFourier._deriv(XX, aa, **kwargs)
        gvec[1, :] = 1.0   # dfdao
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
        d2gdx2[2::2, :] = -_np.power(w, 2.0)*(_np.power(ii, 2.0)*tmp)*_np.cos(w*ii*_np.atleast_2d(XX))
        d2gdx2[3::2, :] = -_np.power(w, 2.0)*(_np.power(ii, 2.0)*tmp)*_np.sin(w*ii*_np.atleast_2d(XX))
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa, *kwargs):
#        return NotImplementedError

    # ====================================== #

    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        nfreqs = ModelFourier.getnfreqs(ain)
        ys = self.slope
        yo = self.offset
        xs = self.xslope
        xo = self.xoffset
        tmp = _np.asarray(range(nfreqs))+1

        aout[0] = ain[0]/xs
        aout[1] = yo + ys*ain[1]
        w  = 2*_np.pi*ain[0]/xs
        aout[2::2] = ys*(ain[2::2]*_np.cos(tmp*w*xo) + ain[3::2]*_np.sin(w*tmp*xo))
        aout[3::2] = ys*(ain[3::2]*_np.sin(tmp*w*xo) - ain[2::2]*_np.sin(w*tmp*xo))
        return aout

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
            gvec[ii, :] = _np.power(XX, ii)
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
            dgdx[ii,:] = ii*_np.power(XX, ii-1.0)
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
            d2gdx2[ii,:] = (ii)*(ii-1.0)*_np.power(XX, ii-2.0)
        # end for
        return d2gdx2[::-1, :]

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain):
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
             = sum( a_i'/xs^i*sum( (i,k)*x^k*xo^(i-k)) )
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        nn = _np.size(aout)
        # First the y-scaling:
        yo = self.offset
        ys = self.slope
        xo = self.xoffset
        xs = self.xslope

        coeffs = _np.zeros_like(aout)
        for ii in range(nn):
            coeffs[:ii] += _ut.binomial_expansion(-xo, ii)
        # end for
        coeffs = coeffs / _np.power(xs, _np.asarray(range(nn))+1)
        coeffs = coeffs[::-1]

        aout = ys*coeffs*aout
        aout[-1] += yo
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
        return _np.exp( ModelPoly._model(XX, aa, **kwargs) )

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
            gvec[ii, :] = _np.power(XX, ii)*prof
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
        return _np.power(ModelProdExp._deriv(XX,aa), 2.0)/ModelProdExp._model(XX,aa) \
                      + ModelPoly._deriv2(XX, aa)*ModelProdExp._model(XX, aa)

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
#        d2fdx2 = _np.power(ModelProdExp._deriv(XX,aa), 2.0)/ModelProdExp._model(XX,aa) \
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

    def unscaleaf(self, ain):
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
                    may not be possible with constant coefficients

            translate the coefficients into a polynomial model, then unshift there

        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)

        PM = ModelPoly(None)
        PM.offset = _np.log(_np.abs(self.slope))
        PM.slope = 1.0
        PM.xoffset = self.xoffset
        PM.xslope = self.xslope
        aout = PM.unscaleaf(aout)
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
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
#    def __init__(self, XX, af=None, **kwargs):
#        if af is not None:
#            num_fit = _np.size(af)  # Number of fitting parameters
#            npoly = _np.int(2*(num_fit-1))  # Polynomial order from input af
#        else:
#            npoly = kwargs.setdefault('npoly', 4)
#        self._af = 0.1*_np.ones((npoly//2+1,), dtype=_np.float64)
#        self._LB = -_np.inf*_np.ones((npoly//2+1,), dtype=_np.float64)
#        self._UB = _np.inf*_np.ones((npoly//2+1,), dtype=_np.float64)
#        self._fixed = _np.zeros( _np.shape(self._LB), dtype=int)
#        super(ModelEvenPoly, self).__init__(XX, af, **kwargs)
#    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        Even Polynomial of order num_fit, Insert zeros for the odd powers
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)
        prof = _np.zeros((nx,), dtype=_np.float64)
        for ii in range(num_fit):
            prof += aa[-(ii+1)]*_np.power(XX, 2.0*ii)
        # end for
        return prof
#        return ModelPoly._model(_np.power(XX, 2.0), aa, **kwargs)
#        num_fit = _np.size(aa)  # Number of fitting parameters
#        a0 = _np.insert(aa, _np.linspace(1, num_fit-1, 2), 0.0)
#        return poly(XX, a0)

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
            gvec[ii, :] = _np.power(XX, 2.0*ii)
        # end for
        return gvec[::-1, :]
#        return ModelPoly._partial(_np.power(XX, 2.0), aa, **kwargs)

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        Derivative of an even polynomial
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)
        dfdx = _np.zeros((nx,), dtype=_np.float64)
        for ii in range(1, num_fit):
            dfdx += (2.0*ii)*aa[-(ii+1)]*_np.power(XX, 2.0*ii-1.0)
        # end for
        return dfdx
#        return ModelPoly._deriv(_np.power(XX, 2.0), aa, **kwargs)
#        num_fit = _np.size(aa)  # Number of fitting parameters
#        a0 = _np.insert(aa, _np.linspace(1, num_fit-1, 2), 0.0)
#        return deriv_poly(XX, a0)

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
            dgdx[ii, :] = (2.0*ii)*_np.power(XX, 2.0*ii-1.0)
        # end for
        return dgdx[::-1, :]
#        return ModelPoly._partial_deriv(_np.power(XX, 2.0), aa, **kwargs)

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        Derivative of an even polynomial
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)
        d2fdx2 = _np.zeros((nx,), dtype=_np.float64)
        for ii in range(1,num_fit):
            d2fdx2 += (2.0*ii)*(2.0*ii-1.0)*aa[-(ii+1)]*_np.power(XX, 2.0*ii-2.0)
        # end for
#        d2fdx2 += 2.0*aa[-2]
        return d2fdx2
#        return ModelPoly._deriv2(_np.power(XX, 2.0), aa, **kwargs)


    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        Jacobian of the 2nd derivative
        """
        num_fit = _np.size(aa)  # Number of fitting parameters
        nx = _np.size(XX)
        d2gdx2 = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(1,num_fit):
            d2gdx2[ii, :] = (2.0*ii)*(2.0*ii-1.0)*_np.power(XX, 2.0*ii-2.0)
        # end for
#        d2gdx2[1,:] = 2.0
        return d2gdx2[::-1, :]
#        return ModelPoly._partial_deriv2(_np.power(XX, 2.0), aa, **kwargs)

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain):
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
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)

#        # Note that this might not work, because in the binomial expansion there will be odd terms
#        atmp = _np.zeros( (2*_np.size(aout),), dtype=_np.float64)
#        atmp[::2] = _np.copy(aout)
#        PM = ModelPoly(None)
#        PM.offset = self.offset
#        PM.slope = self.slope
#        PM.xoffset = self.xoffset
#        PM.xslope = self.xslope
#        atmp = PM.unscaleaf(atmp)

        # x-scaling:
        aout = aout/_np.power(self.xslope, _np.asarray(range(_np.size(aout))))
        # y-scaling and shifting:
        aout = self.slope*aout
        aout[-1] += self.offset
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xoffset = 0.0
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
    def _model(XX, aa, **kwargs):
        """
         Curved power-law:
         fc = x^(a1*x^(n+1)+a2*x^n+...a(n+1))

         With exponential cut-off:
         f  = a(n+2)*exp(a(n+1)*XX)*fc(x);
        """
        XX = _np.abs(XX)
        polys = ModelPoly._model(XX, aa[:-2])
        exp_factor = _np.exp(aa[-2]*XX)
        return aa[-1]*exp_factor*_np.power(XX, polys)

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
        XX = _np.abs(XX)
        prof = ModelPowerLaw._model(XX, aa)
        polys = ModelPoly._model(XX, aa[:-2])
        dpolys = ModelPoly._deriv(XX, aa[:-2])
        return prof*( aa[-1] + polys/XX + _np.log(_np.abs(XX))*dpolys )

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
        """
        XX = _np.abs(XX)
        prof = ModelPowerLaw._model(XX, aa)
        dprofdx = ModelPowerLaw._deriv(XX, aa)
        polys = ModelPoly._model(XX, aa[:-2])
        dpolys = ModelPoly._deriv(XX, aa[:-2])
        d2poly = ModelPoly._deriv2(XX, aa[:-2])
        return _np.power(dprofdx, 2.0)/prof + prof*( d2poly*_np.log(_np.abs(XX))
                + 2.0*dpolys/XX - polys/_np.power(XX, 2.0) )

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
        XX = _np.abs(XX)
        nx = _np.size(XX)
        num_fit = _np.size(aa)  # Number of fitting parameters

        gvec = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(2, num_fit):
            gvec[ii, :] = _np.power(XX, ii-2.0)
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
        XX = _np.abs(XX)
        nx = _np.size(XX)
        num_fit = _np.size(aa)  # Number of fitting parameters
        npoly = len(aa[:-2])    #    npoly = num_fit-3?

        prof = ModelPowerLaw._model(XX, aa)
        dprofdx = ModelPowerLaw._deriv(XX, aa)

        dgdx = _np.zeros((num_fit, nx), dtype=_np.float64)
        for ii in range(2, npoly+2):
            dgdx[ii, :] = (_np.power(XX, ii)*_np.log(_np.abs(XX))*dprofdx
                              + (1.0+ii*_np.log(_np.abs(XX)))*_np.power(XX, ii-1.0)*prof )
        # end for
        dgdx = dgdx[::-1, :]
        dgdx[-2, :] = XX*dprofdx
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
        XX = _np.abs(XX)
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
               poly_deriv2*_np.log(_np.abs(XX)) + 2.0*poly_deriv/XX - polys/_np.power(XX, 2.0) - _np.power(dlnfdx, 2.0)))
               + (tmp*_np.atleast_2d(f/_np.power(XX, 2.0)))*(
                  poly_partial_deriv2*(tmp*_np.atleast_2d(_np.power(XX, 2.0)*_np.log(_np.abs(XX))))
                        + (2.0*tmp*XX)*poly_partial_deriv - poly_partial) )
        return d2gdx2
#        d2gdx2 = (tmp*dlnfdx)*(2.0*dgdx - (tmp*dlnfdx)*gvec)
#        d2gdx2 += gvec*(tmp*_np.atleast_2d(
#                       poly_deriv2*_np.log(_np.abs(XX)) + 2.0*poly_deriv/XX - polys/_np.power(XX,2.0)
#                       ))
#        d2gdx2 += (tmp*_np.atleast_2d(f))*(
#                poly_partial_deriv2*(tmp*_np.atleast_2d(_np.log(_np.abs(XX))))
#              + 2.0*poly_partial_deriv*(tmp*_np.atleast_2d(1.0/XX))
#              - poly_partial*(tmp*_np.atleast_2d(1.0/_np.power(XX, 2.0)))
#                )
#        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain):
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

         a(n+2) = ys* an2'
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)  # TODO!:   Finish scalings
        aout[-1] *= self.slope
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0      # TODO!:   Finish scalings
        self.xoffset = 0.0     # TODO!:   Finish scalings
        self.xscaling = 1.0    # TODO!:   Finish scalingss
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
    def __init__(self, XX, af=None, **kwargs):
        super(ModelParabolic, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        return aa*(1.0 - _np.power(XX, 2.0))

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        return -2.0*aa*XX

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        return -2.0*aa*_np.ones_like(XX)

    @staticmethod
    def _partial(XX, aa, **kwargs):
        return _np.atleast_2d(parabolic(XX, aa) / aa)

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        return _np.atleast_2d(-2.0*XX)

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        return _np.atleast_2d(-2.0*_np.ones_like(XX))

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain):
        """
        y = a*(1.0-x^2)

        y-scaling: y'=y/ys
            a = ys*a'
        y-shifting: y'=(y-yo)/ys
            y = yo + ys*a'*(1.0-x^2) = yo + ys*a' - ys*a'x^2
                Not possible with constant coefficients
        x-scaling: x'=x/xs
            y = ys*a'*(1.0-(x/xs)^2)
              = ys*a'/xs^2 * (xs^2 - x^2) + ys*a'/xs^2 - ys*a'/xs^2
              = ys*a'/xs^2 * (1.0 - x^2) + ys*a'*(1.0 - 1.0/xs^2)
                Not possible with constant coefficients

        x-shifting: x'=(x-xo)/xs
            y = ys*a'*(1.0-(x-xo)^2) = ys*a'*(1.0-x^2-2xo*x+xo^2)
              = ys*a'*(1.0-x^2)-ys*a'*xo*(2*x+xo)
                Not possible with constant coefficients
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        return aout*self.slope

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
    Model - y = a1*(exp(a2*XX^a3) + XX^a4)

    af    - estimate of fitting parameters
    XX    - independent variables
    """
    _af = 0.1*_np.ones((4,), dtype=_np.float64)
    _LB = _np.array([1e-18, -_np.inf, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.array([_np.inf, _np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (4,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelExponential, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
         f     = a*(exp(b*XX^c) + XX^d) = f1+f2;
        """
        prof1, prof2 = ModelExponential._separate_model(XX, aa)
        return prof1 + prof2

    @staticmethod
    def _separate_model(XX, aa, **kwargs):
        a, b, c, d = tuple(aa)
        XX = _np.abs(XX)
        prof1 = a*_np.exp(b* _np.power(XX, c))
        prof2 = a*_np.power(XX, d)
        return prof1, prof2

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
         dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
               = b*c*XX^(c-1) * (a*exp(b*XX^c))  + a*d*XX^(d-1)
               = b*c*XX^(c-1) * prof1 + dprof2dx
        """
        dprof1dx, dprof2dx = ModelExponential._separate_deriv(XX, aa)
        return dprof1dx+dprof2dx

    @staticmethod
    def _separate_deriv2(XX, aa, **kwargs):
        a, b, c, d = tuple(aa)
        XX = _np.abs(XX)
        prof1, prof2 = ModelExponential._separate_model(XX, aa, **kwargs)
        dprof1dx, dprof2dx = ModelExponential._separate_deriv(XX, aa, **kwargs)
        d2prof1dx2 = b*c*_np.power(XX, c-1.0)*( (c-1.0)*prof1/XX + dprof1dx )
        d2prof2dx2 = (d-1.0)*prof2/XX
        return d2prof1dx2, d2prof2dx2

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
         dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))
               = b*c*XX^(c-1) * (a*exp(b*XX^c))  + a*d*XX^(d-1)
               = b*c*XX^(c-1) * prof1 + dprof2dx

         d2fdx2 = b*c*(c-1)*XX^(c-2)*prof1
                 + b*c*XX^(c-1)*dprof1dx
                 + a*d*(d-1)*XX^(d-2)
                = b*c*XX^(c-1)*( (c-1)*prof1/XX + dprof1dx )
                  + (d-1)*prof2/XX
        """
        d2prof1dx2, d2prof2dx2 = ModelExponential._separate_deriv2(XX, aa, **kwargs)
        return d2prof1dx2 + d2prof2dx2

    @staticmethod
    def _separate_deriv(XX, aa, **kwargs):
        prof1, prof2 = ModelExponential._separate_model(XX, aa)
        a, b, c, d = tuple(aa)
        XX = _np.abs(XX)
        dprof1dx = b*c*_np.power(XX, c-1.0)*prof1
        dprof2dx = a*d*_np.power(XX, d-1.0)
        return dprof1dx, dprof2dx

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
         f     = a*(exp(b*XX^c) + XX^d) = f1+f2;
         dfdx  = a*(b*c*XX^(c-1)*exp(b*XX^c) + d*XX^(d-1))

         dfda = f/a
         dfdb = XX^c*f1;
         dfdc = f1*XX^c*log10(XX)
         dfdd = a*XX^d*log10(XX) = log10(XX)*f2;
        """
        nx = _np.size(XX)
        num_fit = _np.size(aa)
        gvec = _np.zeros( (num_fit, nx), dtype=float)

        prof1, prof2 = ModelExponential._separate_model(XX, aa)
        prof = prof1 + prof2

        a, b, c, d = tuple(aa)
        XX = _np.abs(XX)

        gvec[0, :] = prof/a
        gvec[1, :] = prof1*_np.power(XX, c)
        gvec[2, :] = _np.log10(_np.abs(XX))*_np.power(XX, c)*prof1
        gvec[3, :] = _np.log10(_np.abs(XX))*prof2
#        gvec[2, :] = b*prof1*_np.log(_np.abs(XX))*_np.power(XX, c)
#        gvec[3, :] = a*_np.log(_np.abs(XX))*_np.power(XX, d)
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
        XX = _np.abs(XX)

        dgdx[0, :] = dprofdx / a
        dgdx[1, :] = dprof1dx*( _np.power(XX, c) + 1.0/b )
        dgdx[2, :] = dprof1dx*( b*_np.log(_np.abs(XX))*_np.power(XX, c) + _np.log(_np.abs(XX)) + 1.0/c )
        dgdx[3, :] = a*_np.power(XX, d-1.0)*( 1.0 + d*_np.log(_np.abs(XX)) )
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
        XX = _np.abs(XX)

        d2gdx2 = _np.zeros( (num_fit,nx), dtype=float)
        d2gdx2[0, :] = d2profdx2 / a
        d2gdx2[1, :] = d2prof1dx2*( _np.power(XX, c) + 1.0/b ) + dprof1dx*c*_np.power(XX, c-1.0)
        d2gdx2[2, :] = d2prof1dx2*( b*_np.log(_np.abs(XX))*_np.power(XX, c) + _np.log(_np.abs(XX)) + 1.0/c ) \
                    + dprof1dx*( b*c*_np.log(_np.abs(XX))*_np.power(XX, c-1.0) + b*_np.power(XX, c-1.0) + 1.0/XX )
        d2gdx2[3, :] = d2prof2dx2/d + d2prof2dx2*_np.log(_np.abs(XX)) + dprof2dx/XX
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] *= self.slope
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
    def __init__(self, XX, af=None, **kwargs):
        super(ModelGaussian, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        model of a Gaussian
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))
            A = af[0]
            xo = af[1]
            ss = af[2]
        """
        AA, x0, ss = tuple(aa)
        return AA*_np.exp(-_np.power(XX-x0, 2.0)/(2.0*_np.power(ss, 2.0)))

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
        return -1.0*(XX-x0)/_np.power(ss, 2.0) * prof

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
        return -1.0*prof/_np.power(ss, 2.0) - dprofdx*(XX-x0)/_np.power(ss, 2.0)

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
        gvec[1,:] = ((XX-x0)/_np.power(ss, 2.0))*prof
        gvec[2,:] = (_np.power(XX-x0, 2.0)/_np.power(ss, 3.0)) * prof
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
        dgdx[2,:] = -2.0*dfdx/ss + dfdx*_np.power(XX-x0,2.0)/_np.power(ss, 3.0)
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

#        term1 = (-1.0*(XX-x0)/_np.power(ss,2.0))
#        gvec = ModelGaussian._partial(XX, aa, **kwargs)

        d2gdx = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        d2gdx[0,:] = d2fdx2/AA
        d2gdx[1,:] = -(XX-x0)*prof/_np.power(ss, 4.0) + dfdx/_np.power(ss, 2.0) + (XX-x0)*d2fdx2/_np.power(ss, 2.0)
        d2gdx[2,:] = -2.0*d2fdx2/ss + 2.0*(XX-x0)/_np.power(ss, 3.0)*dfdx + _np.power(XX-x0, 2.0)*d2fdx2/_np.power(ss, 3.0)
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


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] = self.slope*aout[0]
        aout[1] = self.xslope*aout[1]+self.xoffset
        aout[2] = self.xslope*aout[2]
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


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] = self.slope*aout[0] + self.offset
        aout[1] = self.slope*aout[1]
        aout[2] = self.xslope*aout[2]+self.xoffset
        aout[3] = self.xslope*aout[3]
        return aout

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

def model_offsetnormal(XX, af, **kwargs):
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

    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] = self.slope*ain[0] + self.offset
        aout[1] = self.slope*self.xslope*ain[1]
        aout[2] = self.xslope*ain[2]+self.xoffset
        aout[3] = self.xslope*aout[3]
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
        nn = _np.sqrt(2.0*_np.pi*_np.power(ss, 2.0))
        return ModelGaussian._model(XX, aa, **kwargs)/nn

    @staticmethod
    def _deriv(XX, aa, **kwargs):
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
        return ModelGaussian._deriv(XX, aa, **kwargs)/nn

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)
            A = af[0]
            xo = af[1]
            ss = af[2]

        dfdx = -(x-xo)/ss^2 *f
        d2fdx2 > see model gaussian
        """
        AA, x0, ss = tuple(aa)
        nn = _np.sqrt(2.0*_np.pi*_np.power(ss, 2.0))
        return ModelGaussian._deriv2(XX, aa, **kwargs)/nn

    @staticmethod
    def _partial(XX, aa, **kwargs):
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
        gvec = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        prof = normal(XX, aa)
        gvec = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = prof/AA
        gvec[1,:] = ((XX-x0)/(ss**2.0))*prof
        gvec[1,:] = (((XX-x0)**2.0)/(ss**3.0)) * prof -prof/_np.abs(ss)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)

        dfdx = -(x-xo)/ss^2 *f

        dfdA = f/A
        dfdxo = (x-xo)/ss^2 * f
        dfdss =  (x-xo)^2/ss^3 * f + A*exp(-(x-xo)**2.0/(2.0*ss**2.0))*d/ds( sqrt(2*pi*ss^2.0)**-1 )
            d/ds( sqrt(2*pi*ss^2.0)**-1 ) = sqrt(2*pi)^-1 * d/ds( (ss^2.0)**-0.5 ) = sqrt(2*pi)^-1 * d/ds( abs(ss)^-1.0 )
                = sqrt(2*pi)^-1 * (2*ss)*-0.5 * (ss^2.0)^-1.5 = sqrt(2*pi)^-1 *-ss/(ss^3.0) = sqrt(2*pi)^-1 *1.0/ss^2
                = sqrt(2*pi*ss**2.0)^-1 *1.0/abs(ss)
        dfdss =  (x-xo)^2/ss^3 * f + 1.0/abs(ss) * f

        d2fdxdA = dfdx/A = -(x-xo)/ss^2 *dfdA
        d2fdxdxo = f/ss^2 -(x-xo)/ss^2 *dfdxo
        d2fdxds = 2*(x-xo)/ss^3 *f -(x-xo)/ss^2 *dfds

        """
        AA, x0, ss = tuple(aa)
        gvec = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        prof = normal(XX, aa)
        gvec = partial_normal(XX, aa)
        dgdx = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = -(XX-x0)/_np.power(ss, 2.0)*gvec[0,:]
        dgdx[1,:] = prof/_np.power(XX, 2.0) - ((XX-x0)/_np.power(ss, 2.0))*gvec[1,:]
        dgdx[1,:] = 2.0*(XX-x0)*prof/_np.power(XX, 3.0) - ((XX-x0)/_np.power(ss, 2.0))*gvec[2,:]
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a normalized Gaussian (unit area)
        f = A*exp(-(x-xo)**2.0/(2.0*ss**2.0))/sqrt(2*pi*ss^2.0)

        dfdx = -(x-xo)/ss^2 *f

        dfdA = f/A
        dfdxo = (x-xo)/ss^2 * f
        dfdss =  (x-xo)^2/ss^3 * f + A*exp(-(x-xo)**2.0/(2.0*ss**2.0))*d/ds( sqrt(2*pi*ss^2.0)**-1 )
            d/ds( sqrt(2*pi*ss^2.0)**-1 ) = sqrt(2*pi)^-1 * d/ds( (ss^2.0)**-0.5 ) = sqrt(2*pi)^-1 * d/ds( abs(ss)^-1.0 )
                = sqrt(2*pi)^-1 * (2*ss)*-0.5 * (ss^2.0)^-1.5 = sqrt(2*pi)^-1 *-ss/(ss^3.0) = sqrt(2*pi)^-1 *1.0/ss^2
                = sqrt(2*pi*ss**2.0)^-1 *1.0/abs(ss)
        dfdss =  (x-xo)^2/ss^3 * f + 1.0/abs(ss) * f

        d2fdxdA = dfdx/A = -(x-xo)/ss^2 *dfdA
        d2fdxdxo = f/ss^2 -(x-xo)/ss^2 *dfdxo
        d2fdxds = 2*(x-xo)/ss^3 *f -(x-xo)/ss^2 *dfds

        d3fdx2dA = -dfdA/ss^2 - (x-xo)/ss^2*d2fdAdx
        d3fdx2dxo = dfdx/ss^2-x/ss^2*dfdxo - (x-xo)/ss^2*d2fdxdxo
        d3fdx2dss = 2.0/ss^3*f + 2*(x-xo)/ss^3*dfdx - dfdss/ss^2 - (x-xo)/ss^2*dfdssdx
        """
        AA, x0, ss = tuple(aa)
        prof = ModelNormal._model(XX, aa, **kwargs)
        dfdx = ModelNormal._deriv(XX, aa, **kwargs)
        gvec = ModelNormal._partial(XX, aa, **kwargs)
        dgdx = ModelNormal._partial_deriv(XX, aa, **kwargs)

        d2gdx2 = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
        d2gdx2[0,:] = -gvec[0,:]/_np.power(ss,2.0)-(XX-x0)/_np.power(ss, 2.0)*dgdx[0,:]
        d2gdx2[1,:] = dfdx/_np.power(ss, 2.0) - XX*gvec[1,:]/_np.power(ss, 2.0) - ((XX-x0)/_np.power(ss, 2.0))*dgdx[1,:]

        d2gdx2[2,:] = 2.0*prof/_np.power(ss, 3.0) + 2.0*(XX-x0)*dfdx/_np.power(ss, 3.0) \
             - gvec[2,:]/_np.power(ss,2.0)- ((XX-x0)/_np.power(ss, 2.0))*dgdx[2,:]
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


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] = self.slope*self.xslope*aout[0]
        aout[1] = self.xslope*aout[1]+self.xoffset
        aout[2] = self.xslope*aout[2]
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
    _af = _np.asarray([1.0, -0.3, 0.1], dtype=_np.float64)
    _LB = _np.asarray([ 1e-18, -_np.inf, -_np.inf], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf,  _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
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
        return (10.0/_np.log(10))*_np.abs(_np.log(AA)-_np.power(XX-x0, 2.0)/(2.0*_np.power(ss, 2.0)))

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

        d2fdx2 = -10.0/(ss**2.0 * _np.log(10)
        """
        AA, xo, ss = tuple(aa)
        return -10.0*(XX-xo)/(_np.power(ss, 2.0) * _np.log(10))

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

        d2fdx2 = -10.0/(ss**2.0 * _np.log(10)
        """
        AA, xo, ss = tuple(aa)
        return -10.0/(_np.power(ss, 2.0) * _np.log(10))

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
        gvec[1,:] = 10.0*((XX-x0)/(_np.power(ss, 2.0)*_np.log(10.0)))
        gvec[2,:] = 10.0*_np.power(XX-x0, 2.0)/(_np.power(ss, 3.0)*_np.log(10.0))
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
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

        dgdx = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        dgdx[1,:] = 10.0/(_np.log(10.0)*_np.power(ss,2.0))
        dgdx[2,:] = 20.0*(XX-x0)/(_np.log(10)*_np.power(ss,3.0))
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
        d2gdx2[2,:] = 20.0/(_np.log(10)*_np.power(ss,3.0))
        return d2gdx2


#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] = _np.exp( _np.log(10.0)*self.offset/10.0 + self.slope*_np.log(_np.abs(aout[0])))
        aout[1] = self.xslope*aout[1]+self.xoffset
        aout[2] = self.xslope*aout[2]*_np.sqrt( _np.log(10.0)/(10.0*self.slope))
        return aout

    # ====================================== #

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

        dfdx = -16*A*ss*(x-xo)/( ss^2 +4*(xo-x)^2)^2 / pi
        """
        AA, x0, ss = tuple(aa)
        return -1.0*AA*16.0*(XX-x0)*ss/(_np.pi*_np.power(4.0*(XX-x0)*(XX-x0)+ss*ss, 2.0))

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        ´Derivative of a Lorentzian normalization such that integration equals AA (af[0])
            f = 0.5*A*ss / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi

        dfdx = -16*A*ss*(x-xo)/( ss^2 +4*(xo-x)^2)^2 / pi
        d2fdx2 = -16*A*ss*( ss^2-12*(x-xo)^2 )/( ss^2 +4*(x-xo)^2)^3 / pi
        """
        AA, x0, ss = tuple(aa)
        return -1.0*AA*16.0*ss*(ss*ss-12.0*(XX-x0)*(XX-x0))/(_np.pi*_np.power(4.0*(XX-x0)*(XX-x0)+ss*ss, 3.0))

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        Jacobian of a Lorentzian normalization such that integration equals AA (af[0])
            f = 0.5*A*ss / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi

        dfdx = -16*A*ss*(x-xo)/( ss^2 +4*(xo-x)^2)^2 / pi
        d2fdx2 = -16*A*ss*( ss^2-12*(x-xo)^2 )/( ss^2 +4*(x-xo)^2)^3 / pi

        dfdA = f/A
        dfdxo = A*ss*(x-xo)/( ss^2/4.0 + (x-xo)^2 )^2  / pi
        dfds = -2*A*(ss^2-4*(x-xo)^2)/( b^2 + 4*(x-xo)^2 )^2 / pi
        """
        AA, x0, ss = tuple(aa)

        gvec = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = 0.5*ss/((XX-x0)*(XX-x0)+0.25*ss*ss)/_np.pi
        gvec[1,:] = AA*ss*(XX-x0)/_np.power((XX-x0)*(XX-x0)+0.25*ss*ss, 2.0 )/_np.pi
        gvec[2,:] = -2.0*AA*(ss*ss-4.0*(XX-x0)*(XX-x0))/_np.power(ss*ss+4.0*(XX-x0)*(XX-x0), 2.0)/_np.pi
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        Jacobian of the derivative of a Lorentzian normalization such that
        integration equals AA (af[0])
            f = 0.5*A*ss / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi

        dfdx = -16*A*ss*(x-xo)/( ss^2 +4*(xo-x)^2)^2 / pi
        d2fdx2 = -16*A*ss*( ss^2-12*(x-xo)^2 )/( ss^2 +4*(x-xo)^2)^3 / pi

        dfdA = f/A
        dfdxo = A*ss*(x-xo)/( ss^2/4.0 + (x-xo)^2 )^2  / pi
        dfds = -2*A*(ss^2-4*(x-xo)^2)/( b^2 + 4*(x-xo)^2 )^2 / pi

        d2fdxdA = -16*ss*(x-xo)/( ss^2 +4*(xo-x)^2)^2 / pi
        """
        AA, x0, ss = tuple(aa)

        dgdx = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = -1.0*16.0*(XX-x0)*ss/(_np.pi*_np.power(4.0*(XX-x0)*(XX-x0)+ss*ss, 2.0))
        dgdx[1,:] = 16.0*AA*ss*(ss*ss-12.0*(XX-x0)*(XX-x0))/_np.power(ss*ss+4*_np.power(XX-x0, 2.0),3.0)/_np.pi
        dgdx[2,:] = 16.0*AA*(XX-x0)*(3.0*ss*ss-4.0*(XX*XX-2.0*XX*x0+x0*x0))/(_np.pi*_np.power(ss*ss+4.0*(XX*XX-2.0*XX*x0+x0*x0), 3.0))
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        Jacobian of the 2nd derivative of a Lorentzian normalization such that
        integration equals AA (af[0])
            f = 0.5*A*ss / ( (x-xo)**2.0 + 0.25*ss**2.0 ) / pi

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
        """
        AA, x0, ss = tuple(aa)

        d2gdx2 = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        d2gdx2[0,:] = -1.0*16.0*ss*( ss*ss-12.0*(XX-x0)*(XX-x0) )/_np.power( ss*ss +4.0*(XX-x0)*(XX-x0), 3.0) / _np.pi
        d2gdx2[1,:] =768.0*AA*ss*(x0-XX)*(ss*ss-4.0*_np.power(XX-x0,4.0))/_np.power(ss*ss+4.0*_np.power(XX-x0,2.0), 4.0)/_np.pi
        d2gdx2[2,:] = 48.0*AA*(_np.power(ss,4.0)-24.0*ss*ss*_np.power(XX-x0,2.0)+16.0*_np.power(XX-x0,4.0))/_np.power(ss*ss+4.0*_np.power(XX-x0,2.0), 4.0)/_np.pi
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] = self.slope*self.xslope*aout[0]
        aout[1] = self.xslope*aout[1]+self.xoffset
        aout[2] = self.xslope*aout[2]
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
        return super(ModelLorentzian, self).scalings(xdat, ydat, **kwargs)

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
    def _model(XX, aa, **kwargs):
        return (aa[0]*ModelLorentzian._model(XX, aa[1:4], **kwargs)
                (1.0-aa[0])*ModelNormal._model(XX, aa[4:], **kwargs))

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        return (aa[0]*ModelLorentzian._deriv(XX, aa[1:4], **kwargs)
                (1.0-aa[0])*ModelNormal._deriv(XX, aa[4:], **kwargs))

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        return (aa[0]*ModelLorentzian._deriv2(XX, aa[1:4], **kwargs)
                (1.0-aa[0])*ModelNormal._deriv2(XX, aa[4:], **kwargs))

    @staticmethod
    def _partial(XX, aa, **kwargs):
        gvec = _np.concatenate( (ModelLorentzian._model(XX, aa[1:4], **kwargs)
                                - ModelNormal._model(XX, aa[4:], **kwargs),
                            aa[0]*ModelLorentzian._partial(XX, aa[1:4], **kwargs),
                           (1.0-aa[0])*ModelNormal._partial(XX, aa[4:], **kwargs)), axis=0)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        dgdx = _np.concatenate( (ModelLorentzian._deriv(XX, aa[1:4], **kwargs)
                                - ModelNormal._deriv(XX, aa[4:], **kwargs),
                                 aa[0]*ModelLorentzian._partial_deriv(XX, aa[1:4], **kwargs),
                           (1.0-aa[0])*ModelNormal._partial_deriv(XX, aa[4:], **kwargs)), axis=0)
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        d2gdx2 = _np.concatenate( (ModelLorentzian._deriv2(XX, aa[1:4], **kwargs)
                                - ModelNormal._deriv2(XX, aa[4:], **kwargs),
                                 aa[0]*ModelLorentzian._partial_deriv2(XX, aa[1:4], **kwargs),
                           (1.0-aa[0])*ModelNormal._partial_deriv2(XX, aa[4:], **kwargs)), axis=0)
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain):
        """
        A combination of lorentzian and normal models
        y = a * L + (1-a)*N
            Note that x-shift and xy-scaling works for Normal/Lorentzian models
            but that y-shifting does not
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)

        LM = ModelLorentzian(None)
        LM.slope = self.slope
        LM.xslope = self.xslope
        LM.xoffset = self.xoffset
        aout[1:4] = LM.unscaleaf(aout[1:4])

        NM = ModelNormal(None)
        NM.slope = self.slope
        NM.xslope = self.xslope
        NM.xoffset = self.xoffset
        aout[4:] = NM.unscaleaf(aout[4:])

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
#        return 10.0*_np.log(lorentzian(XX, aa))/_np.log(10)
        AA, x0, ss = tuple(aa)
        return ( 10.0*_np.log10(0.5) + 10.0*_np.log10(AA)  + 10.0*_np.log10(ss) - 10.0*_np.log10(_np.pi)
               - 10.0*_np.log10(_np.abs(_np.power(XX-x0, 2.0) + 0.25*_np.power(ss, 2.0) ) ) )

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
        return -20.0*(XX-x0)/( _np.log(10) *( _np.power(XX-x0, 2.0)+0.25*_np.power(ss, 2.0)) )

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
        return -80.0*(ss*ss-4.0*_np.power(XX-x0,2.0))/( _np.log(10) *_np.power( 4.0*_np.power(XX-x0, 2.0)+_np.power(ss, 2.0), 2.0) )

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

        gvec = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] = 10.0/(_np.log(10.0)*AA)
        gvec[1,:] = 20.0*(XX-x0)/( _np.log(10.0)*(_np.power(XX-x0, 2.0) + 0.25*_np.power(ss, 2.0 )))
        gvec[2,:] = 10.0/(_np.log(10.0)*ss) - 5.0*ss/( _np.log(10.0)*(_np.power(XX-x0, 2.0) + 0.25*_np.power(ss, 2.0 )))
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

        dgdx = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        dgdx[1,:] = 80.0*(ss-2.0*(XX-x0))*(ss+2.0*(XX-x0))/( _np.log(10)*_np.power( 4.0*_np.power(XX-x0, 2.0) + _np.power(ss, 2.0) , 2.0) )
        dgdx[2,:] = 10.0*ss*(XX-x0)/(_np.log(10.0)*_np.power(_np.power(XX-x0, 2.0)+0.25*_np.power(ss, 2.0), 2.0) )
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

        d2gdx2 = _np.zeros( (3,_np.size(XX)), dtype=_np.float64)
        d2gdx2[1,:] = 640.0*(XX-x0)*(3.0*ss*ss-4.0*_np.power(XX-x0,2.0))/(_np.log(10.0)*_np.power(ss*ss+4.0*_np.power(XX-x0,2.0), 3.0))
        d2gdx2[2,:] = 160.0*ss*(ss*ss-12.0*_np.power(XX-x0,2.0))/(_np.log(10.0)*_np.power(ss*ss+4.0*_np.power(XX-x0,2.0), 3.0))
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] = _np.exp( _np.log(10.0)*self.offset/10.0 - 2.0*self.slope*_np.log(_np.abs(self.xslope))+self.slope*_np.log(_np.abs(aout[0])))
        aout[1] = self.xslope*aout[1]+self.xoffset
        aout[2] = self.xslope*aout[2]
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

    def unscaleaf(self, ain):
        """
        A combination of Lorentzian and Normal models.
        Both can be xy-scaled, and x-shifted but not y-shifted
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        a0, a1, a2 = _parse_noshift(aout, model_order=self.model_order)

        NM = ModelNormal(None)
        NM.slope = self.slope
        NM.xoffset = self.xoffset
        NM.xslope = self.xslope
        aout = NM.unscaleaf(a0)
        if self.model_order>0:
            LM = ModelLorentzian(None)
            LM.slope = self.slope
            LM.xoffset = self.xoffset
            LM.xslope = self.xslope
            aout = _np.concatenate(aout, LM.unscaleaf(a1), axis=0)
        if self.model_order>1:
            aout = _np.concatenate(aout, NM.unscaleaf(a2), axis=0)
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
    def __init__(self, XX, af=None, **kwargs):
        model_order = kwargs.setdefault('model_order', 2)
        noshift = kwargs.setdefault('noshift', True)
        self._af[0] = 1e-3
        self._LB[0] = -_np.inf
        self._UB[0] = 10.0
        if model_order>0:
            self._af[3] = 1
            self._LB[3] = -_np.inf
            self._UB[3] = 10.0
            if noshift: self._fixed[4] = 1 # end if
        if model_order>1:
            self._af[6] = 1e-3
            self._LB[6] = -_np.inf
            self._UB[6] = 10.0
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
        return 10.0*_np.log10(ModelDoppler._model(XX, aa, **kwargs))

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
        return (10.0/_np.log(10.0))*(deriv2/prof-_np.power(deriv/prof, 2.0))

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
        """
        kwargs.setdefault('model_order', 2)
        prof = ModelDoppler._model(XX, aa, **kwargs)
        gvec = ModelDoppler._partial(XX, aa, **kwargs)
        dgdx = ModelDoppler._partial_deriv(XX, aa, **kwargs)
        dlngdx = _np.zeros_like(gvec)
        for ii in range(len(aa)):
            dlngdx[ii,:] = dgdx[ii,:]/prof -gvec[ii,:]/(prof*prof)
        # end for
        return 10.0*dlngdx/_np.log(10)

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        2nd Derivative of the Jacobian of the Doppler model
        d2fdxdai = d2/dxdai(NormalizedGaussian1 + Lorentzian + NormalizedGaussain2 )

        d3fdx2da = d/dx d(df/dx)/da
                = 10.0/ln(10) * ( d3ydx2da/y - d2ydxda*dydx/y2 + 2*dydx*dyda/y^3 - d2ydxda/y^2 )
        """
        kwargs.setdefault('model_order', 2)
        y = ModelDoppler._model(XX, aa, **kwargs)
        dydx = ModelDoppler._deriv(XX, aa, **kwargs)
        dyda = ModelDoppler._partial(XX, aa, **kwargs)
        d2ydxda = ModelDoppler._partial_deriv(XX, aa, **kwargs)
        d3ydx2da = ModelDoppler._partial_deriv2(XX, aa, **kwargs)

        y = _np.atleast_2d(y)
        dydx = _np.atleast_2d(dydx)

        d2gdx2 = _np.zeros_like(dyda)
        for ii in range(len(aa)):
            d2gdx2[ii,:] = (d3ydx2da[ii,:]/y - d2ydxda[ii,:]*dydx/_np.power(y, 2.0)
                    + 2.0*dydx*dyda[ii,:]/_np.power(y, 3.0) - d2ydxda[ii,:]/_np.power(y,2.0))
        # end for
        return 10.0*d2gdx2/_np.log(10)

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #

    def unscaleaf(self, ain):
        """
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        a0, a1, a2 = _parse_noshift(aout, model_order=self.model_order)

        NM = ModelNormal(None)
        NM.slope = self.slope
        NM.xoffset = self.xoffset
        NM.xslope = self.xslope
        aout = ModelNormal.unscaleaf(a0)
        if self.model_order>0:
            LM = ModelLorentzian(None)
            LM.slope = self.slope
            LM.xoffset = self.xoffset
            LM.xslope = self.xslope
            aout = _np.concatenate(aout, LM.unscaleaf(a1), axis=0)
        if self.model_order>1:
            aout = _np.concatenate(aout, NM.unscaleaf(a2), axis=0)
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.offset = 0.0
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
    _af = _np.asarray([1.0, 2.0, 1.0], dtype=_np.float64)
    _LB = _np.asarray([  1e-18,   1e-18, -_np.inf], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
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
        c, d = tuple(aa)
        prof = _np.zeros_like(XX)
#        prof = _np.power(1.0-_np.power(_np.abs(XX), c), d)
        prof[XX<1] = _np.power(1.0-_np.power(_np.abs(XX[XX<1]), c), d)
        prof[XX>=1] = -1.0*_np.power(1.0-_np.power(_np.abs(XX[XX>=1]), -c), d)
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
    def _deriv(XX, aa, **kwargs):
        """
            f = a*(1.0 - x^c)^d
         dfdx = -a*c*d*x^(c-1)*(1.0 - x**c)**(d-1)
              = -c*d*x^(c-1)*_twopower(x, [c, d-1])
       """
        XX = _np.copy(_np.abs(XX))
        a, c, d = tuple(aa)
        dfdx = -1.0*c*d*_np.power(XX, c-1.0)*_ModelTwoPower._model(XX, [a, c, d-1.0])
    #    return -1.0*a*c*d*_np.power(XX, c-1.0)*_np.power(1.0-_np.power(XX,c), d-1.0)
        return dfdx

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
            f = a*(1.0 - x^c)^d
         dfdx = -a*c*d*x^(c-1)*(1.0 - x**c)**(d-1)
              = -c*d*x^(c-1)*_twopower(x, [c, d-1])
         d2fdx2 = -a*c*d*(c-1)*x^(c-2)*(1.0 - x**c)**(d-1)
                  +a*c^2*d*(d-1)*x^(2c-2)*(1.0 - x**c)**(d-2)

       """
        XX = _np.copy(_np.abs(XX))
        a, c, d = tuple(aa)
        d2fdx2 = c*c*d*(d-1)*_np.power(XX, 2*c-2.0)*_ModelTwoPower._model(XX, [a, c, d-2.0])
        d2fdx2 -= c*d*(c-1)*_np.power(XX, c-2.0)*_ModelTwoPower._model(XX, [a, c, d-1.0])
        return d2fdx2

    @staticmethod
    def _partial(XX, aa, **kwargs):
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
    def _partial_deriv(XX, aa, **kwargs):
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

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
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
        d2fdx2 = _ModelTwoPower._deriv2(XX, aa)

        d2gdx2 = _np.zeros((3,_np.size(XX)), dtype=_np.float64)
        d2gdx2[0,:] = d2fdx2/a
        d2gdx2[1,:] = d2fdx2*(_np.power(c, -1.0) + _np.log(_np.abs(XX))
            - (d-1.0)*_np.power(XX, c)*_np.log(_np.abs(XX))*_ModelTwoPower._model(XX, [1.0, c, -1.0]))
        d2gdx2[1,:] += dfdx*(1.0/XX
            - (d-1.0)*c*_np.power(XX, c-1.0)*_np.log(_np.abs(XX))*_ModelTwoPower._model(XX, [1.0, c, -1.0])
            - (d-1.0)*_np.power(XX, c)*1.0/XX*_ModelTwoPower._model(XX, [1.0, c, -1.0])
            - (d-1.0)*_np.power(XX, c)*_np.log(_np.abs(XX))*_ModelTwoPower._deriv(XX, [1.0, c, -1.0]))

        d2gdx2[2,:] = d2fdx2*(_np.power(d, -1.0) + _np.log(_np.abs(_ModelTwoPower._model_base(XX, [c, d-1.0]))))
        d2gdx2[2,:] += dfdx*(_ModelTwoPower._deriv(XX, [c, d-1.0])/(a*_ModelTwoPower._model_base(XX, [c, d-1.0])))
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain):
        """
         y-scaling is possible, but everything else is non-linear
                     f = a*(1.0 - x^c)^d
        y' = y/ys
        y = a*ys*(1-x^c)^d
        a = ys*a'
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] = self.slope*ain[0]
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
    _af = _np.asarray([1.0, 0.0, 12.0, 3.0], dtype=_np.float64)
    _LB = _np.asarray([1e-18, 0.0, 1e-18, -20.0], dtype=_np.float64)
    _UB = _np.asarray([   20, 1.0,  20.0,  20.0], dtype=_np.float64)
    _fixed = _np.zeros( (4,), dtype=int)
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
    def _partial(XX, aa, **kwargs):
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

        nx = _np.size(XX)
        gvec = _np.zeros((4, nx), dtype=_np.float64)
        gvec[0, :] = ModelTwoPower._model(XX, [1.0, b, c, d])
        gvec[1, :] = a-_ModelTwoPower._model(XX, [a, c, d])
        gvec[2, :] = -d*_np.log(_np.abs(XX))*_np.power(_np.abs(XX), c)*ModelTwoPower._model(XX, [a, b, c, d-1.0])
        gvec[3, :] = (1.0-b)*_ModelTwoPower._model(XX, [a, c, d])*_np.log(_np.abs(_ModelTwoPower._model(XX, [1.0, c, 1.0])))
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
        dfdx = ModelTwoPower._deriv(XX, aa)

        nx = _np.size(XX)
        dgdx = _np.zeros((4, nx), dtype=_np.float64)
        dgdx[0, :] = dfdx/a
        dgdx[1, :] = -dfdx/(1.0-b)
        dgdx[2, :] = dfdx*( 1.0/c + _np.log(_np.abs(XX))
                - (d-1.0)*_np.power(XX, c)*_np.log(_np.abs(XX))/_ModelTwoPower._model_base(XX,[c, 1.0]))
        dgdx[3, :] = dfdx*( 1.0/d + _np.log(_np.abs(_ModelTwoPower._model_base(XX, [c, 1.0]))) )
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        a, b, c, d = tuple(aa)
        dfdx = ModelTwoPower._deriv(XX, aa)
        d2fdx2 = ModelTwoPower._deriv2(XX, aa)

        nx = _np.size(XX)
        dgdx = _np.zeros((4, nx), dtype=_np.float64)
        dgdx[0, :] = d2fdx2/a
        dgdx[1, :] = -d2fdx2/(1.0-b)
        dgdx[2, :] = d2fdx2*( 1.0/c + _np.log(_np.abs(XX))
                - (d-1.0)*_np.power(XX, c)*_np.log(_np.abs(XX))/_ModelTwoPower._model_base(XX,[c, 1.0]))
        dgdx[2, :] += dfdx*( 1.0/XX
                - (d-1.0)*c*_np.power(XX, c-1.0)*_np.log(_np.abs(XX))/_ModelTwoPower._model_base(XX,[c, 1.0])
                - (d-1.0)*_np.power(XX, c)*1.0/XX/_ModelTwoPower._model_base(XX,[c, 1.0])
                + (d-1.0)*_np.power(XX, c)*_np.log(_np.abs(XX))*a*_ModelTwoPower._deriv(XX,[c, 1.0])/_np.power(_ModelTwoPower._model(XX,[c, 1.0]), 2.0))
        dgdx[3, :] = d2fdx2*( 1.0/d + _np.log(_np.abs(_ModelTwoPower._model_base(XX, [c, 1.0]))) )
        dgdx[3, :] += dfdx*( _ModelTwoPower._deriv(XX, [c, 1.0])/_ModelTwoPower._model(XX, [c, 1.0]))
        return dgdx

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        yo = self.offset
        ys = self.slope
        aout[0] = yo + ys*ain[0]
        aout[1] = 1.0 - ys*(1-ain[1])*ain[0]/aout[0]
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
    _af = _np.asarray([2.0, 1.0], dtype=_np.float64)
    _LB = _np.asarray([   1e-18, 1e-18], dtype=_np.float64)
    _UB = _np.asarray([ _np.inf, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (2,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelExpEdge, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _exp(XX, aa, **kwargs):
        e, h = tuple(aa)
        return _np.exp(-_np.power(XX, 2.0)/_np.power(h, 2.0))

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
#        return e*(1.0-_np.exp(-_np.power(XX, 2.0)/_np.power(h, 2.0)))

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
        return (2.0*XX*e/_np.power(h, 2.0))*ModelExpEdge._exp(XX, aa, **kwargs)

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
#        dfdx = (2.0*XX*e/_np.power(h, 2.0))*ModelExpEdge._exp(XX, aa, **kwargs)
        d2fdx2 = (2.0*e/_np.power(h, 2.0))*ModelExpEdge._exp(XX, aa, **kwargs)
        d2fdx2 += -(4.0*XX*XX*e/_np.power(h, 4.0))*ModelExpEdge._exp(XX, aa, **kwargs)
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
        gvec = _np.zeros((2,_np.size(XX)), dtype=_np.float64)
        gvec[0,:] =  1.0 - ModelExpEdge._exp(XX, aa, **kwargs)
        gvec[1,:] = (-2.0*e*_np.power(XX, 2.0)/_np.power(h, 3.0))*ModelExpEdge._exp(XX, aa, **kwargs)
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
        dgdx = _np.zeros((2,_np.size(XX)), dtype=_np.float64)
        dgdx[0,:] = (2.0*XX/_np.power(h, 2.0))*ModelExpEdge._exp(XX, aa, **kwargs)
        dgdx[1,:] = (-4.0*e*XX/_np.power(h,5.0))*(h-XX)*(h+XX)*ModelExpEdge._exp(XX, aa, **kwargs)
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
        d2gdx2 = _np.zeros((2,_np.size(XX)), dtype=_np.float64)
        d2gdx2[0,:] = (2.0/_np.power(h, 2.0))*ModelExpEdge._exp(XX, aa, **kwargs)
        d2gdx2[0,:] += -(4.0*XX*XX/_np.power(h, 4.0))*ModelExpEdge._exp(XX, aa, **kwargs)


        d2gdx2[1,:] = (-4.0*e/_np.power(h,5.0))*(h-XX)*(h+XX)*ModelExpEdge._exp(XX, aa, **kwargs)
        d2gdx2[1,:] += (-4.0*e*XX/_np.power(h,5.0))*(-1.0)*(h+XX)*ModelExpEdge._exp(XX, aa, **kwargs)
        d2gdx2[1,:] += (-4.0*e*XX/_np.power(h,5.0))*(h-XX)*(1.0)*ModelExpEdge._exp(XX, aa, **kwargs)
        d2gdx2[1,:] += (8.0*e*XX*XX/_np.power(h,7.0))*(h-XX)*(h+XX)*ModelExpEdge._exp(XX, aa, **kwargs)
        return d2gdx2


#    @staticmethod
#    def _hessian(XX, aa, **kwargs):
#        return None
#        raise NotImplementedError

    # ====================================== #


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] = self.slope*aout[0]
        aout[1] = self.xslope*aout[1]
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xoffset = 0.0
        self.offset = 1.0
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
#    d2pdx2 = aa[3]*(aa[2]**2.0)*(aa[3]-1.0)*(1.0+aa[4]-aa[1])*_np.power(_np.abs(XX), 2.*aa[2]-2.0)*(1-_np.power(_np.abs(XX),aa[2]))**(aa[3]-2.0)
#    d2pdx2 -= (aa[2]-1.0)*aa[2]*aa[3]*(1.0+aa[4]-aa[1])*_np.power(_np.abs(XX), aa[2]-2.0)*_np.power(1-_np.power(_np.abs(XX), aa[2]), aa[3]-1.0)
#    d2pdx2 += (2.0*aa[4]*_np.exp(-_np.power(XX,2.0)/_np.power(aa[5], 2.0)))/_np.power(aa[5], 2.0)
#    d2pdx2 -= (4*aa[4]*_np.power(XX, 2.0)*_np.exp(-_np.power(XX, 2.0)/_np.power(aa[5], 2.0)))/_np.power(aa[5], 4.0)
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
#    gvec[1,:] = -p*q*Y0*_np.power(_np.abs(XX), p-2.0)*_np.power(1.0-_np.power(_np.abs(XX), p), q-2.0)*(p*(q*_np.power(_np.abs(XX), p)-1.0)-_np.power(XX, p)+1.0)
#    gvec[2,:] = p*_np.log(_np.abs(XX))*(p*(_np.power(q, 2.0)*_np.power(_np.abs(XX), 2.0*p)-3.0*q*_np.power(_np.abs(XX), p)+_np.power(_np.abs(XX), p)+1.0)-(_np.power(XX, p)-1.0)*(q*_np.power(_np.abs(XX), p)-1.0))
#    gvec[2,:] += (_np.power(_np.abs(XX), p)-1.0)*(2.0*p*(q*_np.power(_np.abs(XX), p)-1.0)-_np.power(_np.abs(XX), p)+1.0)
#    gvec[2,:] *= q*Y0*(g-h-1.0)*_np.power(_np.abs(XX), p-2.0)*(_np.power(1.0-_np.power(_np.abs(XX), p), q-3.0))
#    gvec[3,:] = p*Y0*(-(g-h-1.0))*_np.power(_np.abs(XX), p-2.0)*_np.power(1.0-_np.power(_np.abs(XX), p), q-2.0)*(p*(2.0*q*_np.power(_np.abs(XX), p)-1.0)+q*(p*(q*_np.power(_np.abs(XX), p)-1.0)-_np.power(_np.abs(XX), p)+1.0)*_np.log(_np.abs(1.0-_np.power(_np.abs(XX)**p)))-_np.power(_np.abs(XX), p)+1.0)
#    gvec[4,:] = Y0*(p*q*_np.power(_np.abs(XX), p-2.0)*_np.power(1.0-_np.power(_np.abs(XX), p), q-2.0)*(p*(q*_np.power(_np.abs(XX), p)-1.0)-_np.power(_np.abs(XX), p)+1.0)+(2.0*_np.exp(-_np.power(XX, 2.0)/_np.power(w, 2.0))*(_np.power(w, 2.0)-2.0*_np.power(_np.abs(XX), 2.0)))/_np.power(w, 4.0))
#    gvec[5,:] = -(4.0*h*Y0*_np.exp(-_np.power(XX, 2.0)/_np.power(w, 2.0))*(_np.power(w, 4.0)-5*_np.power(w, 2.0)*_np.power(_np.abs(XX), 2.0)+2.0*_np.power(_np.abs(XX), 4.0)))/_np.power(w, 7.0)
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

    aa[0] - Y0 - function value on-axis
    aa[1] - gg - Y1/Y0 - function value at edge over core
    aa[2],aa[3]-  pp, qq - power scaling parameters
    aa[4],aa[5]-  hh, ww - hole depth and width

    f/a = prof1/a + prof2/a

        prof1 = a*( b+(1-b)*(1-XX^c)^d )    # ModelTwoPower
    {x element R: (d>0 and c=0 and x>0) or (d>0 and x=1)
            or (c>0 and 0<=x<1) or (c<0 and x>1) }

        prof2 = e*(1-exp(-XX^2/h^2))        # EdgePower
   {x element R}

        nohollow = True
        af = _np.hstack((af,0.0))
        af = _np.hstack((af,1.0))

        Y0 = core
        YO*aa[1] = edge
    """
    _af = _np.asarray([1.0, 0.002, 2.0, 0.7, -0.24, 0.30], dtype=_np.float64)
    _LB = _np.asarray([1e-18, 0.0,  1e-18,-10,-1, 0], dtype=_np.float64)
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
    def _model(XX, aa, **kwargs):
        a, b, c, d, e, f = tuple(aa)
        prof = ( ModelTwoPower._model(XX, [a, b-e, c, d])
                 + a*ModelExpEdge._model(XX, [e, f]) )
        return prof

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        a, b, c, d, e, f = tuple(aa)
        return ( ModelTwoPower._deriv(XX, [a, b-e, c, d])
                 + a*ModelExpEdge._deriv(XX, [e, f]) )

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        a, b, c, d, e, f = tuple(aa)
        return ( ModelTwoPower._deriv2(XX, [a, b-e, c, d])
                 + a*ModelExpEdge._deriv2(XX, [e, f]) )

    @staticmethod
    def _partial(XX, aa, **kwargs):
        a, b, c, d, e, f = tuple(aa)
        gvec = _np.zeros((6,_np.size(XX)), dtype=_np.float64)
        gvec[0, :] = ModelQuasiParabolic._model(XX, aa)/a
        gvec[1:4,:] = ModelTwoPower._partial(XX, [a, b-e, c, d])[1:,:]
        gvec[4:, :] = a*ModelExpEdge._partial(XX, [e, f])
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        a, b, c, d, e, f = tuple(aa)
        dgdx = _np.zeros((6,_np.size(XX)), dtype=_np.float64)
        dgdx[0, :] = ModelQuasiParabolic._deriv(XX, aa)/a
        dgdx[1:4,:] = ModelTwoPower._partial_deriv(XX, [a, b-e, c, d])[1:,:]
        dgdx[4:,:] = a*ModelExpEdge._partial_deriv(XX, [e, f])
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        a, b, c, d, e, f = tuple(aa)
        d2gdx2 = _np.zeros((6,_np.size(XX)), dtype=_np.float64)
        d2gdx2[0, :] = ModelQuasiParabolic._deriv2(XX, aa)/a
        d2gdx2[1:4,:] = ModelTwoPower._partial_deriv2(XX, [a, b-e, c, d])[1:,:]
        d2gdx2[4:,:] = a*ModelExpEdge._partial_deriv2(XX, [e, f])
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain):
        """
        rescaling: y'=(y-yo)/ys
            y' = a0'*( a1'-a4'+(1-a1'+a4')*(1-x^a2')^a3' + a4'*(1-exp(x^2/a5'^2)))
               = a0'*a1'-a0'*a4' + a0'*(1-a1'+a4')*(...) + a0'*a4'*(...)
               = -a4' + (1+a4')*(...) + a4'*(...)

            y = yo+a0'*(...)
                by inspection
                (1)
            constants :    a0*a1-a0*a4 = yo+a0'*a1'-a0'*a4'

            and (2)
            (1-x^a2)^a3 :   a0*(1-a1+a4) = a0'*(1-a1'+a4')
                     a0-a1*a0+a0*a4 = a0'-a0'*a1'+a0'*a4'
            and (3)
            (1-exp(x^2/a5^2)) : a0*a4 = a0'*a4'
                                    --------
                    use (3) in (1):  a0*a1 = yo+a0'*a1'
                        (3) in (2):  a0-a1*a0 = a0'-a0'*a1' ->
                                    a0*(1-a1) = a0'*(1-a1')
                                    --------
                  a0 = a0'*(1-a1)/(1-a1') = (yo+a0'*a1')/a1
                   a0'*(1-a1)/(1-a1') = (yo+a0'*a1')/a1
                        a0'*(1-a1)*a1 = (1-a1')*(yo+a0'*a1')
                        a1-a1^2 = (1-a1')*(yo/a0'+a1')
                        a1-a1^2 = yo/a0'+a1'-yo*a1'/a0'-a1'^2

                            a1 = yo/a0' + a1' - yo*a1'/a0'
                            a0 = a0'*(1-a1)/(1-a1')
                            a4 = a0'*a4'/a0

        rescaling: x'=(x-xo)/xs
            is not possible because of the non-linearities in the x-functions
                (1-x^a2)^a3 = (1-(x-xo)^a2'/xs^a2')^a3'

          and   exp(x^2/a5^2) = exp((x-xo)^2/(xs*a5')^2)
                exp(x/a5) = exp((x-xo)/(xs*a5'))
                    x/a5 = (x-xo)/(xs*a5')
                    x*xs*a5' = x*a5-xo*a5
                    independence of x means that x*(xs*a5'-a5) = 0, a5 = xs*a5'
                    but x*xs*a5' = x*xs*a5'-xo*a5 requires that xo == 0

                (1-x^a2)^a3 = (1-x^a2'/xs^a2')^a3'
                a3*ln(1-x^a2) = a3'*ln(1-x^a2'/xs^a2')
                    assume a3 == a3'
                    x^a2 = x^a2'/xs^a2'
                    a2*ln(x) = a2'*ln(x)-a2'*ln(xs)
                        a2 = a2'*(1-ln(xs)/ln(x))  ... only works if xs = 1.0

        a1 = yo/a0' + a1' - yo*a1'/a0'
        a0 = a0'*(1-a1)/(1-a1')
        a4 = a0'*a4'/a0
        iff xs = 1.0, xo = 0.0
        """
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        yo = self.offset
        aout[1] = yo/ain[0] + ain[1] - yo*ain[1]/ain[0]
        aout[0] = ain[0]*(1-aout[1])/(1-ain[1])
        aout[4] = ain[0]*ain[4]/aout[0]
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xoffset = 0.0
        self.xslope = 1.0
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
        af[1] - b - determines the gradient location
        af[2] - c - the gradient steepness
    The profile is constant near the plasma center, smoothly descends into a
    gradient near (x/b)=1 and tends to zero for (x/b)>>1
    """
    _af = _np.asarray([1.0, 0.4, 5.0], dtype=_np.float64)
    _LB = _np.asarray([1e-18, 0.0, 1.0], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, 1.0, _np.inf], dtype=_np.float64)
    _fixed = _np.zeros( (3,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelFlattop, self).__init__(XX, af, **kwargs)
    # end def __init__

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        A flat-top plasma parameter profile with three free parameters:
            a, b, c
        prof ~ f(x) = a / (1 + (x/b)^c)

        """
        a, b, c = tuple(aa)
        temp = _np.power(XX/b, c)
        return a / (1.0 + temp)

    @staticmethod
    def _deriv(XX, aa, **kwargs):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)
        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)
        """
        a, b, c = tuple(aa)
        temp = _np.power(XX/b, c)
        return -1.0*a*c*temp/(XX*_np.power(1.0+temp,2.0))

    @staticmethod
    def _deriv2(XX, aa, **kwargs):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)
        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)
             = -a*c*(x/b)^c/(x/b^(2c)*(b^c+x^c)^2)
             = -a*c*(x*b)^c/(x*(b^c+x^c)^2)
        d2fdx2 = -a*c^2*(x/b)^(c-1)/(x^2*(1+(x/b)^c)^2)
                + a*c*(x/b)^c/(x*(1+(x/b)^c)^2)
        """
        a, b, c = tuple(aa)
        temp = _np.power(XX/b, c)
        return -1.0*a*c*temp/(XX*_np.power(1.0+temp,2.0))

    @staticmethod
    def _partial(XX, aa, **kwargs):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)
        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)

        dfda = 1.0/(1+_np.power(x/b, c))
        dfdb = a*c*(x/b)^c/(b*(1+(x/b)^c)^2)
        dfdc = a*_np.log(x/b)*(x/b)^c / (1+(x/b)^c)^2
        """
        nx = _np.size(XX)
        a, b, c = tuple(aa)
        temp = _np.power(XX/b, c)
        prof = flattop(XX, aa)

        gvec = _np.zeros((3, nx), dtype=_np.float64)
        gvec[0, :] = prof / a
        gvec[1, :] = a*c*temp / (b*_np.power(1.0+temp,2.0) )
        gvec[2, :] = a*temp*_np.log(_np.abs(XX/b)) / _np.power(1.0+temp, 2.0)
        return gvec

    @staticmethod
    def _partial_deriv(XX, aa, **kwargs):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)
        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)

        dfda = 1.0/(1+_np.power(x/b, c))
        dfdb = a*c*(x/b)^c/(b*(1+(x/b)^c)^2)
        dfdc = a*_np.log(x/b)*(x/b)^c / (1+(x/b)^c)^2

        d2fdxda = -1 * c*(x/b)^c / ( x* ( 1+(x/b)^c )^2 )
                = dfdx / a
        d2fdxdb = -1 * a * c^2 * (x/b)^(c-1) * ((x/b)^c-1.0) / (b^2*( 1+(x/b)^c )^3 )
                = prof * -1 * c^2 * (x/b)^(c-1) * ((x/b)^c-1.0) / (b^2*( 1+(x/b)^c )^2 )
                = prof * x*dprofdx * c/a * (x/b)^(-1) * ((x/b)^c-1.0) / (b^2)
                = prof * dprofdx * c/a * ((x/b)^c-1.0) / b
        d2fdxdc = a * (x/b)^c * ( -(x/b)^c + c*(x/b)^c*ln|x/b| - c*ln|x/b| - 1.0  )
                 / ( x*( (x/b)^c + 1 )^3  )
                = prof*dfdx/(a*c) * ( -(x/b)^c + c*(x/b)^c*ln|x/b| - c*ln|x/b| - 1.0  )

        """
        nx = _np.size(XX)
        a, b, c = tuple(aa)
        temp = _np.power(XX/b, c)
        prof = flattop(XX, aa)
        dprofdx = deriv_flattop(XX, aa)

        dgdx = _np.zeros((3, nx), dtype=_np.float64)
        dgdx[0, :] = dprofdx / a
        dgdx[1, :] = prof * dprofdx * (c/a)*(temp-1.0)/b
        dgdx[2, :] = prof*dprofdx/(a*c) *(
            -1.0*temp + c*temp*_np.log(_np.abs(XX/b)) - c*_np.log(_np.abs(XX/b)) - 1.0 )
#        dgdx[2, :] = dprofdx/c
#        dgdx[2, :] += dprofdx*_np.log(_np.abs(XX/b))
#        dgdx[2, :] -= 2.0*(dprofdx**2.0)*(_np.log(_np.abs(XX/b))/prof)
        return dgdx

    @staticmethod
    def _partial_deriv2(XX, aa, **kwargs):
        """
        prof ~ f(x) = a / (1 + (x/b)^c)
        dfdx = -1*c*x^(c-1)*b^-c* a/(1+(x/b)^c)^2
             = -a*c*(x/b)^c/(x*(1+(x/b)^c)^2)

        dfda = 1.0/(1+_np.power(x/b, c))
        dfdb = a*c*(x/b)^c/(b*(1+(x/b)^c)^2)
        dfdc = a*_np.log(x/b)*(x/b)^c / (1+(x/b)^c)^2

        d2fdxda = -1 * c*(x/b)^c / ( x* ( 1+(x/b)^c )^2 )
                = dfdx / a
        d2fdxdb = -1 * a * c^2 * (x/b)^(c-1) * ((x/b)^c-1.0) / (b^2*( 1+(x/b)^c )^3 )
                = prof * -1 * c^2 * (x/b)^(c-1) * ((x/b)^c-1.0) / (b^2*( 1+(x/b)^c )^2 )
                = prof * x*dprofdx * c/a * (x/b)^(-1) * ((x/b)^c-1.0) / (b^2)
                = prof * dprofdx * c/a * ((x/b)^c-1.0) / b
        d2fdxdc = a * (x/b)^c * ( -(x/b)^c + c*(x/b)^c*ln|x/b| - c*ln|x/b| - 1.0  )
                 / ( x*( (x/b)^c + 1 )^3  )
                = prof*dfdx/(a*c) * ( -(x/b)^c + c*(x/b)^c*ln|x/b| - c*ln|x/b| - 1.0  )

        """
        nx = _np.size(XX)
        a, b, c = tuple(aa)
        temp = _np.power(XX/b, c)
        dtdx = c*_np.power(b, -1.0*c)*_np.power(XX, c-1.0)

        prof = ModelFlattop._model(XX, aa, **kwargs)
        dprofdx = ModelFlattop._deriv(XX, aa, **kwargs)
        d2profdx2 = ModelFlattop._deriv2(XX, aa, **kwargs)

        d2gdx2 = _np.zeros((3, nx), dtype=_np.float64)
        d2gdx2[0, :] = d2profdx2 / a
        d2gdx2[1, :] = (dprofdx * dprofdx * (c/a)*(temp-1.0)/b
                       + prof * d2profdx2 * (c/a)*(temp-1.0)/b
                       + prof * dprofdx * (c/a)*(dtdx)/b )

        d2gdx2[2, :] = dprofdx*dprofdx/(a*c) *(
            -1.0*temp + c*temp*_np.log(_np.abs(XX/b)) - c*_np.log(_np.abs(XX/b)) - 1.0 )
        d2gdx2[2, :] += prof*d2profdx2/(a*c) *(
            -1.0*temp + c*temp*_np.log(_np.abs(XX/b)) - c*_np.log(_np.abs(XX/b)) - 1.0 )
        d2gdx2[2, :] += prof*dprofdx/(a*c) *(
            -1.0*dtdx + c*dtdx*_np.log(_np.abs(XX/b))+c*temp*b/XX - c*b/XX)

#        d2gdx2[2, :] = dprofdx/c
#        d2gdx2[2, :] += dprofdx*_np.log(_np.abs(XX/b))
#        d2gdx2[2, :] -= 2.0*(dprofdx**2.0)*(_np.log(_np.abs(XX/b))/prof)
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] *= self.slope
        aout[1] *= self.xslope
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


def massberg(XX, aa):
    return ModelMassberg._model(XX, aa)

def deriv_massberg(XX, aa):
    return ModelMassberg._deriv(XX, aa)

def partial_massberg(XX, aa):
    return ModelMassberg._partial(XX, aa)

def partial_deriv_massberg(XX, aa):
    return ModelMassberg._partial_deriv(XX, aa)

def model_massberg(XX=None, af=None, **kwargs):
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
    _LB = _np.asarray([1e-18, 0.0, 1.0, -1], dtype=_np.float64)
    _UB = _np.asarray([_np.inf, 1.0, _np.inf, 1], dtype=_np.float64)
    _fixed = _np.zeros( (4,), dtype=int)
    def __init__(self, XX, af=None, **kwargs):
        super(ModelMassberg, self).__init__(XX, af, **kwargs)
    # end def __init__

#    @staticmethod
#    def _model(XX, aa, **kwargs):
#        a, b, c, h = tuple(aa)
#        return a*(1.0-h*(XX/b)) / (1+_np.power(XX/b, c))

    @staticmethod
    def _model(XX, aa, **kwargs):
        """
        f = a*(1-h*x/b)/(1+(x/b)^c)
          = flattop*(1-h*x/b)
        """
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
        a, b, c, h = tuple(aa)
    #    prof = massberg(XX, af)
#        dprofdx = deriv_massberg(XX, aa)

        temp = XX/b

        prft = flattop(XX, [a, b, c])
        dflatdx = deriv_flattop(XX, [a, b, c])
        gft = partial_flattop(XX, [a, b, c])
        dgft = partial_deriv_flattop(XX, [a, b, c])

        dgdx = _np.zeros((4, nx), dtype= float)
        dgdx[0, :] = dgft[0,:]*(1.0-h*temp) - gft[0,:]*h/b
        dgdx[1, :] = ( dgft[1,:]*(1.0-h*temp) - gft[1,:]*h/b
                     + dflatdx*h*XX/_np.power(b,2.0) + prft*h/_np.power(b,2.0) )
        dgdx[2, :] = dgft[2,:]*(1.0-h*temp) - gft[2,:]*h/b
        dgdx[3, :] = dflatdx*(-1.0*XX/b) - prft/b

#        dgdx[1,:] = dgft[1,:]*(1.0-h*temp) + dflatdx * h*XX/_np.power(b, 2.0)
#        dgdx[1,:] += prft*h/_np.power(b, 2.0) - (h/b)*gft[1,:]
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
        a, b, c, h = tuple(aa)
    #    prof = massberg(XX, af)
#        dprofdx = deriv_massberg(XX, aa)

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
                     + d2flatdx2*h*XX/_np.power(b,2.0) + 2.0*dflatdx*h/_np.power(b,2.0))

        d2gdx2[2, :] = d2gfdx2[2,:]*(1.0-h*temp)+dgft[2,:]*(-h*dtdx) - dgft[2,:]*h/b
        d2gdx2[3, :] = d2flatdx2*(-1.0*XX/b) + dflatdx*(-1.0/b) - dflatdx/b
        return d2gdx2

#    @staticmethod
#    def _hessian(XX, aa):

    # ====================================== #


    def unscaleaf(self, ain):
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
        ain = _np.copy(ain)
        aout = _np.copy(ain)
        aout[0] *= self.slope
        aout[1] *= self.xslope
        return aout

    def scalings(self, xdat, ydat, **kwargs):
        self.xoffset = 0.0
        self.offset = 0.0
        return super(ModelMassberg, self).scalings(xdat, ydat, **kwargs)

    # ====================================== #

#    def checkbounds(self, dat):
#        return super(ModelMassberg, self).checkbounds(dat, self.aa, mag=None)

    # ====================================== #
# end def ModelMassberg

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
                       - 0.5*zz*_np.log(_np.abs(bb))*(bb**af[ii])*(1 - temp**2))

#        #indice of transitions
#        bx = _np.floor(1+bb/(XX[2]-XX[1]))
#        gvec[num_fit-1,ba-1:bx-1] = (zz*_np.log(_np.abs(bb))*(-bb)**af[num_fit-1]
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

    mod = ModelLine().test_numerics()
    mod = ModelSines().test_numerics()
    mod = ModelSines().test_numerics(nfreqs=5, fmod=5.0, shape='square', duty=0.40)
    mod = ModelFourier().test_numerics() # broken
    mod = ModelFourier.test_numerics(nfreqs=5, shape='square', duty=0.40)
    mod = ModelPoly.test_numerics(npoly=1)
    mod = ModelPoly.test_numerics(npoly=2)
    mod = ModelPoly.test_numerics(npoly=5)
    mod = ModelPoly.test_numerics(npoly=12)
    mod = ModelProdExp.test_numerics(npoly=1)
    mod = ModelProdExp.test_numerics(npoly=2)
    mod = ModelProdExp.test_numerics(npoly=5)
    mod = ModelProdExp.test_numerics(npoly=12)
    mod = ModelEvenPoly.test_numerics(npoly=1)
    mod = ModelEvenPoly.test_numerics(npoly=2)
    mod = ModelEvenPoly.test_numerics(npoly=3)
    mod = ModelEvenPoly.test_numerics(npoly=4)
    mod = ModelEvenPoly.test_numerics(npoly=8)
    mod = ModelPowerLaw.test_numerics(npoly=2)
    mod = ModelPowerLaw.test_numerics(npoly=3)
    mod = ModelPowerLaw.test_numerics(npoly=4)
    mod = ModelPowerLaw.test_numerics(npoly=8)
    mod = ModelParabolic.test_numerics()
    mod = ModelExponential.test_numerics()
    mod = ModelGaussian.test_numerics()
    mod = ModelOffsetGaussian.test_numerics()
    mod = ModelOffsetNormal.test_numerics()
    mod = ModelNormal.test_numerics()
    mod = ModelLogGaussian.test_numerics()
    mod = ModelLorentzian.test_numerics()
    mod = ModelPseudoVoigt.test_numerics()
    mod = ModelLogLorentzian.test_numerics()
    mod = ModelDoppler.test_numerics()
    mod = ModelLogDoppler.test_numerics()
    mod = _ModelTwoPower.test_numerics()
    mod = ModelTwoPower.test_numerics()
    mod = ModelExpEdge.test_numerics()
    mod = ModelQuasiParabolic.test_numerics()
    mod = ModelFlattop.test_numerics()
    mod = ModelMassberg.test_numerics()
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


