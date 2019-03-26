# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:23:49 2018

@author: gawe
"""

# ======================================================================== #
# ======================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

# ========================================================================== #
# ========================================================================== #

import numpy as _np
import matplotlib.pyplot as _plt
from pybaseutils import utils as _ut
from pybaseutils.Struct import Struct
from FIT import model_spec as _ms, derivatives as _dd

# Ideally, we'll use the straight python implementation of LMFIT.
# This is mostly python agnostic and has better option controls
try:
    __import__('mpfit')
    __mpfitonpath__ = True
    from mpfit import mpfit as LMFIT
#    __mpfitonpath__ = True
except:
    __mpfitonpath__ = False
# end try

# There are annoying differences in the context between scipy version of
# leastsq and curve_fit, and the method least_squares doesn't exist before 0.17
import scipy.version as _scipyversion
from scipy.optimize import curve_fit, leastsq

# Make a version flag for switching between least squares solvers and contexts
_scipyversion = _scipyversion.version
try:
    _scipyversion = _np.float(_scipyversion[0:4])
except:
    _scipyversion = _np.float(_scipyversion[0:3])
# end try
if _scipyversion >= 0.17:
    print("Using a new version of scipy")
#    from scipy.optimize import least_squares
else:
    print("Using an older version of scipy")
# endif

__metaclass__ = type

# ======================================================================== #
# ======================================================================== #


def gen_random_init_conds(func, **fkwargs):
#    af0 = fkwargs.pop('af0', None)
#    varaf0 = fkwargs.pop('varaf0', None)

    # Call the wrapper function to get reasonable bounds
    temp = func(XX=None, **fkwargs)
    LB = _np.copy(temp.Lbounds)
    UB = _np.copy(temp.Ubounds)
    fixed = _np.copy(temp.fixed)

    # Differentiate base data set from default starting positions by randomly
    # setting the profile to be within the upper / lower bounds (or reasonable values)
    # aa = _ms.randomize_initial_conditions(LB=temp.Lbounds, UB=temp.Ubounds, af0=af0, varaf0=varaf0)
    aa = fkwargs.setdefault('af0', _np.copy(temp.af))
#    aa += 0.5*aa*_np.random.uniform(0.0, 1.0, len(aa))
    for ii in range(len(aa)):
        if fixed[ii] == 0:
#            if _np.isinf(LB[ii]):     LB[ii] = -10.0     # end if
#            if _np.isinf(UB[ii]):     UB[ii] =  10.0     # end if
#            _np.random.seed()
#            aa[ii] += 0.1*aa[ii]*_np.random.normal(0.0, 1.0, 1)
            aa[ii] += 0.5*aa[ii]*_np.random.uniform(low=-1.0, high=1.0, size=1)
#            aa[ii] = _np.random.uniform(low=LB[ii], high=UB[ii], size=1)
            if aa[ii]<LB[ii]:                aa[ii] = LB[ii] + 0.5
            if aa[ii]>UB[ii]:                aa[ii] = UB[ii] - 0.5

#            if _np.isinf(LB[ii]):    LB[ii] = -10.0      # end if
#            if _np.isinf(UB[ii]):    UB[ii] = 10.0       # end if
#            aa[ii] = _np.random.uniform(low=LB[ii], high=UB[ii], size=1)  # fails sometimes
        # end if
    # end for
    return aa

# ========================================================================== #
# ========================================================================== #

# ============================================= #
# ---------- Generalized Curve Fitting -------- #
# ============================================= #

# Fitting using the Levenberg-Marquardt algorithm.    #
def modelfit(x, y, ey, XX, func, fkwargs={}, **kwargs):
    kwargs = _default_mpfit_kwargs(**kwargs)
    return fit_mpfit(x, y, ey, XX, func, fkwargs, **kwargs)

def _default_mpfit_kwargs(**kwargs):
    kwargs.setdefault('xtol', 1e-16)  # 1e-16
    kwargs.setdefault('ftol', 1e-16)
    kwargs.setdefault('gtol', 1e-16)
    kwargs.setdefault('damp', 0)
    kwargs.setdefault('maxiter', 1600)
#    kwargs.setdefault('factor', 100)  # 100
    kwargs.setdefault('nprint', 10) # 100
#    kwargs.setdefault('iterfunct', 'default')
#    kwargs.setdefault('iterkw', {})
#    kwargs.setdefault('nocovar', 0)
#    kwargs.setdefault('rescale', 0
#    kwargs.setdefault('autoderivative', 1)
    kwargs.setdefault('autoderivative', 0)
    kwargs.setdefault('quiet', 1)
#    kwargs.setdefault('diag', None)
#    kwargs.setdefault('epsfcn', max((_np.nanmean(_np.diff(x.copy())),1e-2))) #5e-4) #1e-3
#    kwargs.pop('epsfcn')
    kwargs.setdefault('debug', 0)
#    kwargs.setdefault('debug', 1)
    return kwargs

def _clean_mpfit_kwargs(kwargs):
    mwargs = {}
    if 'xtol' in kwargs:        mwargs['xtol'] = kwargs['xtol']   # end if
    if 'ftol' in kwargs:        mwargs['ftol'] = kwargs['ftol']   # end if
    if 'gtol' in kwargs:        mwargs['gtol'] = kwargs['gtol']   # end if
    if 'damp' in kwargs:        mwargs['damp'] = kwargs['damp']   # end if
    if 'maxiter' in kwargs:     mwargs['maxiter'] = kwargs['maxiter']   # end if
    if 'factor' in kwargs:      mwargs['factor'] = kwargs['factor']   # end if
    if 'nprint' in kwargs:      mwargs['nprint'] = kwargs['nprint']   # end if
    if 'iterfunct' in kwargs:   mwargs['iterfunct'] = kwargs['iterfunct']   # end if
    if 'iterkw' in kwargs:      mwargs['iterkw'] = kwargs['iterkw']   # end if
    if 'nocovar' in kwargs:     mwargs['nocovar'] = kwargs['nocovar']   # end if
    if 'rescale' in kwargs:     mwargs['rescale'] = kwargs['rescale']   # end if
    if 'autoderivative' in kwargs:  mwargs['autoderivative'] = kwargs['autoderivative']   # end if
    if 'quiet' in kwargs:       mwargs['quiet'] = kwargs['quiet']   # end if
    if 'diag' in kwargs:        mwargs['diag'] = kwargs['diag']   # end if
    if 'epsfcn' in kwargs:      mwargs['epsfcn'] = kwargs['epsfcn']   # end if        :
    if 'debug' in kwargs:       mwargs['debug'] = kwargs['debug']   # end if
    return mwargs

def fit_mpfit(x, y, ey, XX, func, fkwargs={}, **kwargs):

    if (ey == 0).any():
        print('Input error meant for weights includes zeros! this is not allowed')
        raise Exception('input weights (err) to fit_mpfit cannot have 0 value!')
#        return None
    # subfunction kwargs

    # This is incredibly slow, but improves the convergence for some problems.
    # Also messes up the error propagation, so you have to rerun the minimizer
    # and set the initial conditions to match the output of the first run.
    use_perpendicular_distance = kwargs.setdefault('perpchi2', True)

    # fitter kwargs
    LB = kwargs.setdefault('LB', None)
    UB = kwargs.setdefault('UB', None)
    p0 = kwargs.setdefault('af0', None)
    fixed = kwargs.setdefault('fixed', None)

    # check for alternative names
    if LB is None: LB = kwargs.setdefault('Lbounds', None)  # end if
    if UB is None: UB = kwargs.setdefault('Ubounds', None)  # end if
    if p0 is None: p0 = kwargs.pop('af', None)       # end if

#    skipwithnans = kwargs.pop('PassBadFit', False)
    plotit = kwargs.pop('plotit', False)

    # default initial conditions come directly from the model functions
    _, _, info = func(_np.copy(XX), af=None, **fkwargs)
    info.success = False
    info.af = gen_random_init_conds(func, af0=info.af, **fkwargs)
    info.update()
    if p0 is None:        p0 = info.af          # end if
    if LB is None:        LB = info.Lbounds     # end if
    if UB is None:        UB = info.Ubounds     # end if
    if fixed is None:     fixed = info.fixed    # end if
    numfit = len(p0)

    if numfit != LB.shape[0]:
        print('oops')
    # end if

    scale_by_data = kwargs.setdefault('scale_problem', True)
    if scale_by_data:
        x, y, vy = info.scaledat(_np.copy(x), _np.copy(y), _np.power(_np.copy(ey), 2.0))
        ey = _np.sqrt(vy)

        # Scale the initial guess and the bounds as well
        p0 = info.scaleaf(p0)
        LB = info.scaleaf(LB)
        UB = info.scaleaf(UB)
        LB = _np.where(_np.isnan(LB), -_np.inf, LB)
        UB = _np.where(_np.isnan(UB), _np.inf, UB)
    # end if

    # ============================================= #

    autoderivative = kwargs.setdefault('autoderivative', 1)
    def mymodel(p, fjac=None, x=None, y=None, err=None, nargout=1):
        # Parameter values are passed in "p"
        # If fjac==None then partial derivatives should not be
        # computed.  It will always be None if MPFIT is called with default
        # flag.
        # Non-negative status value means MPFIT should continue, negative means
        # stop the calculation.
        status = 0
        if _np.isnan(p).any():
            print('NaN in model parameters!')
            status = -2
            raise Exception('Nan in model parameters!')
        elif _np.isnan(p).all():
            print('All the model parameters are NaNs!')
            status = -3
        # end if

#        model, gvec = info.update_minimal(x, p)
        model, gvec, temp = func(x, p, **fkwargs)

        if nargout>1:
            return model, gvec, temp

        if _np.isnan(model).any():
            print('NaN in model!')
            status = -1

        if "errx" in kwargs:
            errx = kwargs["errx"]
        elif use_perpendicular_distance:
            errx = _np.ones_like(err)  # no weighting
        else:
            errx = _np.zeros_like(err)  # no perpendicular distance
        if use_perpendicular_distance or (errx !=0).any():    #perp_distance
            weights = _np.sqrt(err*err + temp.dprofdx*temp.dprofdx*(errx*errx))
#            weights = _np.sqrt(err**2.0 + (errx*temp.dprofdx)**2.0)  # effective variance method
        else:
            # vertical distance
            weights = err
        # end if
        weights = _np.where(weights==0, 1.0, weights)
        residual = (y-model)/weights

        if _np.isnan(residual).any():
            print('NaN in residual!')
            status = -4

#        if _np.isnan(residual).any():
#            raise Exception('Nan detected in chi2. Check model parameters and bounds \n %s'%(str(residual),))

        if autoderivative == 0 and nargout == 1:
            fjac = gvec.copy().T   # use analytic jacobian  # transpose is from weird reshape in mpfit

            # analytic jacobian of the resiudal function
            p1s = _np.ones( (1,len(p)), dtype=fjac.dtype)
            fjac = -1.0*fjac/(_np.atleast_2d(weights).T*p1s)
            if use_perpendicular_distance or (errx != 0).any():
                # The jacobian of the analytic residual is modified by the derivative in the weights
                fjac = fjac - temp.dgdx.T*(_np.atleast_2d(residual*errx*errx*temp.dprofdx/(weights*weights)).T*p1s)

            fjac = -1.0*fjac  # result is 90 deg. out-of-phase?
#            if _np.isnan(fjac).any():
#                raise Exception('Nan detected in jacobian. Check model parameters and bounds \n %s'%(str(fjac),))
            return {'status':status, 'residual':residual, 'jacobian':fjac}
        elif autoderivative == 1 and nargout == 1:
            fjac = None
            return {'status':status, 'residual':residual}
        # end if
    # end def mymodel

    # Settings for each parameter of the fit.
    #   'value' is the initial value that will be updated by mpfit
    #   'fixed' is a boolean: 0 vary this parameter, 1 do not vary this parameter
    #   'limited' is a pair of booleans in a list [Lower bound, Upper bound]:
    #       ex//   [0, 0] -> no lower or upper bounds on parameter
    #       ex//   [0, 1] -> no lower bound on parameter, but create an upper bound
    #       ex//   [1, 1] -> lower and upper bounds on parameter
    #   'limits' lower and upper bound values matching boolean mask in 'limited'
    def getparinfo():
        parinfo = [{'value':p0[ii], 'fixed':fixed[ii], 'limited':[1,1], 'limits':[LB[ii],UB[ii]],
                'mpside':[2]} for ii in range(numfit)]
        return parinfo

    # Pass data into the solver through keywords
    fa = {'x':x, 'y':y, 'err':ey}

    # Call mpfit
#    kwargs['nprint'] = kwargs.get('nprint',10)
    mpwargs = _clean_mpfit_kwargs(kwargs)
#    m = LMFIT(mymodel, p0, parinfo=parinfo, residual_keywords=fa, **kwargs)
    if 'damp' in mpwargs and mpwargs['damp']:
#        mpwargs['damp'] = 0.1*_np.max(y)/(ey[_np.argmax(y)])
        parinfo = getparinfo()
        m = LMFIT(mymodel, p0, parinfo=parinfo, residual_keywords=fa, **mpwargs)
        p0 = m.params
        mpwargs['damp'] = 0
    if use_perpendicular_distance:
        parinfo = getparinfo()
        m = LMFIT(mymodel, p0, parinfo=parinfo, residual_keywords=fa, **mpwargs)
        p0 = m.params
        use_perpendicular_distance = False
    # end if
    parinfo = getparinfo()
    m = LMFIT(mymodel, p0, parinfo=parinfo, residual_keywords=fa, **mpwargs)
    #  m - object
    #   m.status   - there are more than 12 return codes (see mpfit documentation)
    #   m.errmsg   - a string error or warning message
    #   m.fnorm    - value of final summed squared residuals
    #   m.covar    - covariance matrix
    #           set to None if terminated abnormally
    #   m.nfev     - number of calls to fitting function
    #   m.niter    - number if iterations completed
    #   m.perror   - formal 1-sigma uncertainty for each parameter (0 if fixed or touching boundary)
    #           .... only meaningful if the fit is weighted.  (errors given)
    #   m.params   - outputs!
    print(m.statusString())

    # Store the optimization information / messages and a boolean indicating success
    info.mpfit = m         # store the problem that was actually solved
    info.success = True

    if (m.status <= 0):
        info.success = False
        print('error message = ', m.errmsg)
        raise ValueError('The NL fitter failed to converge: see fit_mpfit/modelfit in fitNL')
        return info
    # end error checking

    # check covariance matrix
    if m.covar is None:
        info.success = False
        errmsg = 'error in calculation (no covariance returned) => %s'%(m.errmsg,)
        if 0:
#        if skipwithnans:
            print(errmsg)
            m.params = _np.nan*_np.ones_like(p0)
            m.perror = _np.nan*_np.ones_like(p0)
            m.covar = _np.nan*_np.ones( (numfit,numfit))
            m.fnorm = _np.nan
            pass
        else:
            raise ValueError(errmsg)
        # end if
    # end if
    info.errmsg = m.errmsg
    info.af = _np.copy(m.params)

    # Final function evaluation
    info.update(af=info.af)
    out = mymodel(info.af, x=x, y=y, err=ey)
    residual = out['residual']
#    residual = calc_chi2(info.prof, info.af, x, y, ey)

    # ====== Post-processing ====== #

    # degrees of freedom in fit
    info.dof = len(x) - numfit # deg of freedom

    # scaled uncertainties in fitting parameters
    info.chi2 = _np.nansum(residual*residual)
    info.chi2_reduced = info.chi2/info.dof
#    info.chi2_reduced = _np.sqrt(m.fnorm / info.dof)
    info.perror = m.perror * info.chi2_reduced

    # Scaled covariance matrix
    info.covmat = info.chi2_reduced * _np.copy(m.covar)

    # ====== Error Propagation ====== #
    # Propagate uncertainties in fitting parameters (scaled covariance) into
    # the profile and it's derivative
    info.varprof = _ut.properror(info.XX, info.covmat, info.gvec)

    if hasattr(info, 'dgdx'):
        info.vardprofdx = _ut.properror(info.XX, info.covmat, info.dgdx)
    # endif

    # reclaculate parameter errors based on scaled covariance
    info.perror = _np.diagonal(info.covmat)

    # Calculate the hessian by finite differencing the analytic jacobian
#    info.hessian(XX, aa=info.af, **kwargs)

    # calculate correlation matrix
    info.cormat = _np.copy(info.covmat) * 0.0
    for ii in range(numfit):
        for jj in range(numfit):
            info.cormat[ii,jj] = _ms.divide(info.covmat[ii,jj], _np.sqrt(info.covmat[ii,ii]*info.covmat[jj,jj]))
        # end for
    # end for

    info.varprof = _ut.interp_irregularities(info.varprof, corezero=False)
    info.vardprofdx = _ut.interp_irregularities(info.vardprofdx, corezero=False)

    # unscale the problem if it has previously been scaled for domain reasons
    if scale_by_data:
        # note: the error propagation is automatically completed after unscaling
        # by an auto-update of the model
        x, y, vy = info.unscaledat(af=info.af)
        XX = info.XX
        ey = _np.sqrt(vy)

        if info._analytic_xscaling:
            # Scaling model parameters to reproduce original data
            info.af = info.unscaleaf(info.af)

            # unscale the input covariance if there is one from a fitter
#            if hasattr(self, 'covmat'):
#                self.covmat = self.unscalecov(self.covmat)
#            # end if

            # Update with the unscaled parameters
            info.update()
        # end if
    # end if

    # Actual fitting parameters
    info.params = _np.copy(info.af)

    if plotit:
        _plt.figure()
        _plt.errorbar(x, y, yerr=ey, fmt='ko', color='k')
        _plt.plot(XX, info.prof, 'k-')
        _plt.fill_between(XX, info.prof-_np.sqrt(info.varprof), info.prof+_np.sqrt(info.varprof),
                          interpolate=True, color='k', alpha=0.3)
#        ylim = _plt.ylims()
        _plt.ylim((_np.min((0, 0.8*_np.min(info.prof), 1.2*_np.min(info.prof))),
                   _np.max((   0.8*_np.max(info.prof), 1.2*_np.max(info.prof)))))
    # end if
    return info


# ========================================================================== #

def bootstrapfit(xdat, ydat, ey, XX, func, fkwargs={}, nmonti=30, **kwargs):

    weightit = kwargs.setdefault('weightit', False)
    gvecfunc = kwargs.setdefault('gvecfunc', None)

    # =============== #

    niterate = 1
    if nmonti > 1:
        niterate = nmonti
        # niterate *= len(x)
    # endif

    # nch = len(x)
    numfit = len(info.af0)
    vary = ey**2.0

    xsav = xdat.copy()
    ysav = ydat.copy()
    vsav = vary.copy()

    af = _np.zeros((niterate, numfit), dtype=_np.float64)
    covmat = _np.zeros((niterate, numfit, numfit), dtype=_np.float64)
    vaf = _np.zeros((niterate, numfit), dtype=_np.float64)
    chi2 = _np.zeros((niterate,), dtype=_np.float64)

    nx = len(XX)
    mfit = _np.zeros((niterate, nx), dtype=_np.float64)
    dprofdx = _np.zeros_like(mfit)

    if gvecfunc is not None:
        gvec, _, dgdx = gvecfunc(info.af0, XX)
    # end if
    vfit = _np.zeros((niterate, nx), dtype=_np.float64) # end if
    vdprofdx = _np.zeros_like(vfit)

    for mm in range(niterate):
        _np.random.seed()
        ydat = ysav.copy() + _np.sqrt(vsav)*_np.random.normal(0.0,1.0,_np.shape(ysav))
        vary = (ydat-ysav)**2
#            vary = vsav.copy()
#            vary = vsav.copy()*_np.abs((ydat-ysav)/ysav)
#            vary = vsav.copy()*(1 + _np.abs((ydat-ysav)/ysav))
#            cc = 1+_np.floor((mm-1)/nmonti)
#            if nmonti > 1:
#                ydat[cc] = ysav[cc].copy()
#                ydat[cc] += _np.sqrt(vsav[cc]) * _np.random.normal(0.0,1.0,_np.shape(vsav[cc]))
#                vary[cc] = (ydat[cc]-ysav[cc])**2
#                # _np.ones((1,nch), dtype=_np.float64)*
#            # endif
#            print(mm, niterate)
#            res = self.run()
#            af[mm, :], _ = res
#            if verbose:
        if mm % 10 == 0:
            print('%i of %i'%(mm,niterate))
        # end if
        af[mm, :], covmat[mm,:,:] = run()
        vaf[mm, :] = perror.copy()**2.0

        # chi2[mm] = _np.sum(self.chi2)/(numfit-nch-1)
        mfit[mm, :] = func(af[mm,:].copy(), xvec.copy())
        chi2[mm] = _np.sum(chi2_reduced.copy())
        # mfit[mm, :] = self.yf.copy()

        if gvecfunc is not None:
            gvec, tmp, dgdx = gvecfunc(af[mm,:].copy(), XX.copy())
            dprofdx[mm,:] = tmp.copy()
            if gvec is not None:
                vfit[mm,:] = properror(XX, gvec)
            if dgdx is not None:
                vdprofdx[mm,:] = properror(XX, dgdx)
            # end if
        # end if
#            if dgdx is not None:
#                vdfdx[mm,:] = properror(XX, dgdx)
        # end if
    # endfor
    xdat = xsav
    ydat = ysav
    vary = vsav

#        _plt.figure()
#        _plt.plot(xvec, mfit.T, 'k-')
#        _plt.plot(xvec, (mfit+_np.sqrt(vfit)).T, 'k--')
#        _plt.plot(xvec, (mfit-_np.sqrt(vfit)).T, 'k--')

    if weightit:
        # # weighted mean and covariance
        # aw = 1.0/(1.0-chi2) # chi2 close to 1 is good, high numbers good in aweights
        # # aw[_np.isnan(aw)*_np.isinf(aw)]
        # Weighting by chi2
        aw = 1.0/chi2
#            aw = 1.0/_np.sqrt(chi2)

        # self.af = _np.sum( af*aw, axis=0) / _np.sum(aw, axis=0)
#            self.perror, self.af = _ut.nanwvar(af, statvary=None, systvary=None, weights=aw, dim=0)
        perror, af = _ut.nanwvar(af.copy(), statvary=vaf, systvary=None, weights=aw, dim=0, nargout=2)
        perror = _np.sqrt(perror)
        covmat = _np.cov( af, rowvar=False, fweights=None, aweights=aw)

#        mfit = _np.nansum( mfit * 1.0/_np.sqrt(vfit) ) / _np.nansum( 1.0/_np.sqrt(vfit) )
#        vfit = _np.nanvar( mfit, axis=0)
        # weight by chi2 or by individual variances?
        if gvecfunc is not None:
            # vfit, mfit = _ut.nanwvar(mfit.copy(), statvary=vfit, systvary=None, weights=1.0/_np.sqrt(vfit), dim=0, nargout=2)
#           vfit, mfit = _ut.nanwvar(mfit.copy(), statvary=None, systvary=None, weights=1.0/_np.sqrt(vfit), dim=0, nargout=2)
            vfit, mfit = _ut.nanwvar(mfit.copy(), statvary=None, systvary=None, weights=1.0/vfit, dim=0, nargout=2)
            vdprofdx, dprofdx = _ut.nanwvar(dprofdx.copy(), statvary=None, systvary=None, weights=1.0/vdprofdx, dim=0, nargout=2)
        else:
            aw = aw.reshape((niterate,1))*_np.ones((1,nx),dtype=_np.float64)              # reshape the weights
#            self.vfit, self.mfit = _ut.nanwvar(mfit.copy(), statvary=vfit, systvary=None, weights=aw, dim=0)
            vfit, mfit = _ut.nanwvar(mfit.copy(), statvary=None, systvary=None, weights=aw, dim=0)
            vdprofdx, dprofdx = _ut.nanwvar(dprofdx.copy(), statvary=None, systvary=None, weights=aw, dim=0)
        # end if
    else:
        # straight mean and covariance
        af, perror = _ut.combine_var(af, statvar=None, systvar=None, axis=0)
        perror = _np.sqrt(perror)
        mfit, vfit = _ut.combine_var(mfit, statvar=None, systvar=None, axis=0)
        dprofdx, vdprofdx = _ut.combine_var(dprofdx, statvar=None, systvar=None, axis=0)

        covmat = _np.cov(af, rowvar=False)
    # end if
    return af, covmat, info

# ========================================================================== #


class fitNL_base(Struct):
    """
    To use this first generate a class that is a child of this one
    class Prob(fitNL)

        def __init__(self, xdata, ydata, yvar, **kwargs)
            # call super init
            super(fitNL, self).__init__(xdat, ydat, vary, af0, self.func, kwargs)
        # end def

        def func(self, af):
            return y-2*af[0]+af[1]

    """

    def __init__(self, xdat, ydat, vary, af0, func, fjac=None, **kwargs):
        if vary is None or (vary == 0).all():  vary = 1e-1*_np.nanmean(ydat)  # endif

        self.xdat = xdat
        self.ydat = ydat
        self.vary = vary
        self.af0 = af0
        self.func = func
        self.fjac = fjac

        # ========================== #

        solver_options = {}

        # ========================== #

        self.nmonti = kwargs.pop("nmonti", 600)
        self.af0 = kwargs.pop("af0", self.af0)
        self.LB = kwargs.pop("LB", -_np.Inf*_np.ones_like(self.af0))
        self.UB = kwargs.pop("UB",  _np.Inf*_np.ones_like(self.af0))
        self.fixed = kwargs.pop("fixed", _np.zeros(_np.shape(self.af0), dtype=int))
        self.use_perpendicular_distance = kwargs.pop('perpchi2', False)

        # ========================== #

        # Pull out the run data from the options dictionary
        #  possibilities include
        #   - LB, UB - Lower and upper bounds on fitting parameters (af)
        self.solver_options = solver_options
        self.__dict__.update(**kwargs)
        return kwargs
    # end def __init__

    # ========================== #

    def run(self, **kwargs):
        self.__dict__.update(kwargs)

        self.lmfit(**kwargs)
        return self.af, self.covmat
    # end def run

    # ========================== #

    def calc_chi2(self, af):
#        try:
        if self.use_perpendicular_distance:
#            # perpendicular distance
            xmp = _np.linspace(self.xdat.min()-1e-3, self.xdat.max()+1e-3, num=201)
            ymp = self.func(af, xmp)
            npts = len(self.ydat)
            self.chi2 = _np.zeros((npts,), dtype=_np.float64)
            for ii in range(npts):
                perp2 = _np.power(self.ydat[ii]-ymp,2) + _np.power(self.xdat[ii]-xmp,2)
                imin = _np.argmin( _np.sqrt(_np.abs(perp2)) )
                self.chi2[ii] = _np.sqrt(_np.abs(perp2[imin]))
                self.chi2[ii] *= _np.sign(self.ydat[ii]-ymp[imin])
                self.chi2[ii] /= _np.sqrt(_np.abs(self.vary[ii]))
            # end for
        else:
            # vertical distance
            self.chi2 = (self.func(af, self.xdat) - self.ydat)
            self.chi2 /= _np.sqrt(_np.abs(self.vary))
#        except:
#            pass
        if _np.isnan(self.chi2).any():
            raise Exception('Nan detected in chi2. Check model parameters and bounds \n %s'%(str(self.chi2),))
        return self.chi2

    def bootstrapper(self, xvec=None, **kwargs):
#        self.solver_options['nprint'] = 100    # debug info
        self.solver_options['quiet'] = 1 # debug info


        if not hasattr(self, 'nmonti'):  self.nmonti=400  # end if
        nmonti = kwargs.setdefault('nmonti', self.nmonti)

        if not hasattr(self, 'weightit'):  self.weightit = False  # end if
        weightit = kwargs.setdefault('weightit', self.weightit)

        if xvec is None and hasattr(self, 'xx'):  xvec = self.xx  # endif

#        if not hasattr(self, 'gvecfunc'):  self.gvecfunc = None  # end if
#        gvecfunc = kwargs.setdefault('gvecfunc', self.gvecfunc)

        self.__dict__.update(kwargs)
        # =============== #

#        nch = len(self.xdat)
        numfit = len(self.af0)
        xsav = self.xdat.copy()
        ysav = self.ydat.copy()
        vsav = self.vary.copy()

        # =============== #

        niterate = len(xsav)
        if nmonti > 1:
#            niterate = _np.copy(nmonti)
            niterate *= _np.copy(nmonti)
            # niterate *= len(self.xdat)
        # endif

        af = _np.zeros((niterate, numfit), dtype=_np.float64)
        covmat = _np.zeros((niterate, numfit, numfit), dtype=_np.float64)
        vaf = _np.zeros((niterate, numfit), dtype=_np.float64)
        chi2 = _np.zeros((niterate,), dtype=_np.float64)

#        nx = len(xvec)
#        self.mfit = self.func(self.af, xvec)
#        mfit = _np.zeros((niterate, nx), dtype=_np.float64)
#        dprofdx = _np.zeros_like(mfit)

#        if gvecfunc is not None:
#            gvec, _, dgdx = gvecfunc(self.af, xvec)
#        # end if
#        vfit = _np.zeros((niterate, nx), dtype=_np.float64) # end if
#        vdprofdx = _np.zeros_like(vfit)

        cc = -1
        for mm in range(niterate):
            _np.random.seed()
#            self.ydat = ysav.copy() + _np.sqrt(vsav)*_np.random.normal(0.0,1.0,_np.shape(ysav))
#            self.vary = (self.ydat-ysav)**2
#            self.vary = vsav.copy()
#            self.vary = vsav.copy()*_np.abs((self.ydat-ysav)/ysav)
#            self.vary = vsav.copy()*(1 + _np.abs((self.ydat-ysav)/ysav))
#            cc = 1+_np.floor((mm-1)/self.nmonti)
#            if self.nmonti > 1:
            cc += 1
            if (cc >= len(xsav)):
                cc = 0
            # end if
            if 1:
                self.ydat = ysav.copy()
                self.vary = vsav.copy()
                self.ydat[cc] += _np.sqrt(vsav[cc]) * _np.random.normal(0.0,1.0,_np.shape(vsav[cc]))
                self.vary[cc] = (self.ydat[cc]-ysav[cc])**2
                    # _np.ones((1,nch), dtype=_np.float64)*
            # endif
#            print(mm, niterate)
#            res = self.run()
#            af[mm, :], _ = res
#            if self.verbose:
            if mm % 10 == 0:
                print('%i of %i'%(mm,niterate))
            # end if
            af[mm, :], covmat[mm,:,:] = self.run()
            vaf[mm, :] = self.perror.copy()**2.0

            # chi2[mm] = _np.sum(self.chi2)/(numfit-nch-1)
#            mfit[mm, :] = (self.func(af[mm,:].copy(), xvec.copy())).copy()
            chi2[mm] = _np.sum(self.chi2_reduced.copy())
            # mfit[mm, :] = self.yf.copy()

            covmat[mm] = _np.copy(self.covmat)
#            if gvecfunc is not None:
#                gvec, tmp, dgdx = gvecfunc(af[mm,:].copy(), xvec.copy())
#                dprofdx[mm,:] = tmp.copy()
#                if gvec is not None:
#                    vfit[mm,:] = (self.properror(xvec, gvec)).copy()
#                if dgdx is not None:
#                    vdprofdx[mm,:] = (self.properror(xvec, dgdx)).copy()
#                # end if
            # end if
#            if dgdx is not None:
#                vdfdx[mm,:] = self.properror(self.xx, dgdx)
            # end if
        # endfor
        self.xdat = xsav.copy()
        self.ydat = ysav.copy()
        self.vary = vsav.copy()

#        _plt.figure()
#        _plt.plot(xvec, mfit.T, 'k-')
#        _plt.plot(xvec, (mfit+_np.sqrt(vfit)).T, 'k--')
#        _plt.plot(xvec, (mfit-_np.sqrt(vfit)).T, 'k--')

        if weightit:
            # # weighted mean and covariance
            # aw = 1.0/(1.0-chi2) # chi2 close to 1 is good, high numbers good in aweights
            # # aw[_np.isnan(aw)*_np.isinf(aw)]
            # Weighting by chi2
            aw = 1.0/chi2
#            aw = 1.0/_np.sqrt(chi2)

            # self.af = _np.sum( af*aw, axis=0) / _np.sum(aw, axis=0)
#            self.perror, self.af = _ut.nanwvar(af, statvary=None, systvary=None, weights=aw, dim=0)
            self.perror, self.af = _ut.nanwvar(af.copy(), statvary=vaf, systvary=None, weights=aw, dim=0, nargout=2)
            self.perror = _np.sqrt(self.perror)

            # sum over covariance matrix elements ... combine variances statistically
            self.covmat = _np.cov( af, rowvar=False, fweights=None, aweights=aw)

#            self.mfit = _np.nansum( mfit * 1.0/_np.sqrt(vfit) ) / _np.nansum( 1.0/_np.sqrt(vfit) )
#            self.vfit = _np.nanvar( mfit, axis=0)
            # weight by chi2 or by individual variances?
#            if gvecfunc is not None:
#                 self.vfit, self.mfit = _ut.nanwvar(mfit.copy(), statvary=vfit, systvary=None, weights=aw, dim=0, nargout=2)
#                 self.vdprofdx, self.dprofdx = _ut.nanwvar(dprofdx.copy(), statvary=vdprofdx, systvary=None, weights=aw, dim=0, nargout=2)
##                self.vfit, self.mfit = _ut.nanwvar(mfit.copy(), statvary=None, systvary=None, weights=1.0/_np.sqrt(vfit), dim=0, nargout=2)
##                self.vfit, self.mfit = _ut.nanwvar(mfit.copy(), statvary=None, systvary=None, weights=1.0/vfit, dim=0, nargout=2)
##                self.vdprofdx, self.dprofdx = _ut.nanwvar(dprofdx.copy(), statvary=None, systvary=None, weights=1.0/vdprofdx, dim=0, nargout=2)
#            else:
#                aw = aw.reshape((niterate,1))*_np.ones((1,nx),dtype=_np.float64)              # reshape the weights
#    #            self.vfit, self.mfit = _ut.nanwvar(mfit.copy(), statvary=vfit, systvary=None, weights=aw, dim=0)
#                self.vfit, self.mfit = _ut.nanwvar(mfit.copy(), statvary=None, systvary=None, weights=aw, dim=0)
#                self.vdprofdx, self.dprofdx = _ut.nanwvar(dprofdx.copy(), statvary=None, systvary=None, weights=aw, dim=0)
#            # end if
        else:
            # straight mean and covariance
            self.af, self.perror = _ut.combine_var(af.copy(), statvar=vaf.copy(), systvar=None, axis=0)
#            self.af, self.perror = _ut.combine_var(af, statvar=None, systvar=None, axis=0)
            self.perror = _np.sqrt(self.perror)
#            self.mfit, self.vfit = _ut.combine_var(mfit.copy(), statvar=vfit.copy(), systvar=None, axis=0)
##            self.mfit, self.vfit = _ut.combine_var(mfit, statvar=None, systvar=None, axis=0)
#            self.dprofdx, self.vdprofdx = _ut.combine_var(dprofdx, statvar=vdprofdx.copy(), systvar=None, axis=0)

            # sum over covariance matrix elements ... combine variances statistically
            self.covmat = _np.cov(af, rowvar=False)
        # end if
        return self.af, self.covmat

    # ========================== #

    def properror(self, xvec=None, gvec=None):  # (x-positions, gvec = dqparabda)
        if gvec is None: gvec = self.gvec.copy()  # endif
        if xvec is None: xvec = self.xx.copy()    # endif
        self.vfit = _ut.properror(xvec, covmat=self.covmat, fjac=gvec)
        return self.vfit

    # ========================== #
# end class fitNL

# ======================================================================== #
# ======================================================================== #


class fitNL(fitNL_base):

    def __init__(self, xdat, ydat, vary=None, af0=[], func=None, **kwargs):
        """
        Initialize the class for future calculations.  You have to call the
        obj.run_calc() method to actually get a result
        """
        kwargs = super(fitNL, self).__init__(xdat=xdat, ydat=ydat, vary=vary, af0=af0, func=func, **kwargs)
        self.__dict__.update(**kwargs)
        self.lmfit = self.__use_mpfit  # alias the fitter
    # end __init__

    def run(self):
        if not hasattr(self, 'solver_options'):  self.solver_options = {}  # end if
        self.solver_options = _default_mpfit_kwargs(**self.solver_options)
        super(fitNL, self).run(**self.solver_options)
        return self.af, self.covmat

        # ========================================== #
        if self.covmat is None:
            print("oops! The least squares solver didn't find a solution")
        # endif
    # end def

    def __use_mpfit(self, **kwargs):
        """
        """

        def model_ud(p, fjac=None, x=None, y=None, err=None):
            # Parameter values are passed in "p"
            # If FJAC!=None then partial derivatives must be comptuer.
            # FJAC contains an array of len(p), where each entry
            # is 1 if that parameter is free and 0 if it is fixed.
            chi2 = self.calc_chi2(p)
#            model = self.func(p, x)
            # Non-negative status value means MPFIT should continue, negative means
            # stop the calculation.
            status = 0
            self.gvec = self.fjac(p, x)
            pderiv = self.gvec.T  # pderiv[xx, pp]
            pderiv /= _np.atleast_2d(err).T*_np.ones( (1,len(p)), dtype=fjac.dtype)
            return {'status':status, 'residual':chi2, 'jacobian':pderiv}

        def model_ad(p, fjac=None, x=None, y=None, err=None):
            chi2 = self.calc_chi2(p)

            # Non-negative status value means MPFIT should continue, negative means
            # stop the calculation.
            status = 0
            if _np.isnan(p).all():
                status = -3
            elif _np.isnan(p).any():
                status = -2
            # end if
            self.gvec = None
            return {'status':status, 'residual':chi2}

        if hasattr(self, 'autoderivative') and (not self.autoderivative):
            mymodel = model_ud
        else:
            mymodel = model_ad
        # end if
        # default initial conditions come directly from the model functions
        self.success = False
        p0 = _np.atleast_1d(self.af0)

        LB = self.LB
        UB = self.UB
        fixed = self.fixed
        numfit = len(p0)

        # Settings for each parameter of the fit.
        #   'value' is the initial value that will be updated by mpfit
        #   'fixed' is a boolean: 0 vary this parameter, 1 do not vary this parameter
        #   'limited' is a pair of booleans in a list [Lower bound, Upper bound]:
        #       ex//   [0, 0] -> no lower or upper bounds on parameter
        #       ex//   [0, 1] -> no lower bound on parameter, but create an upper bound
        #       ex//   [1, 1] -> lower and upper bounds on parameter
        #   'limits' lower and upper bound values matching boolean mask in 'limited'
        parinfo = [{'value':p0[ii], 'fixed':fixed[ii], 'limited':[1,1], 'limits':[LB[ii],UB[ii]],
                    'mpside':[2]} for ii in range(numfit)]

        # Pass data into the solver through keywords
        fa = {'x':self.xdat, 'y':self.ydat, 'err':_np.sqrt(self.vary)}

        m = LMFIT(mymodel, p0, parinfo=parinfo, residual_keywords=fa, **kwargs)

        if (m.status <= 0):
            print('error message = ', m.errmsg)
            return
        # end error checking

        # Store the optimization information / messages and a boolean indicating success
        self.mpfit = m
        self.success = True

        # Actual fitting parameters
        self.af = m.params

        # ====== Error Propagation ====== #
        # Make a final call to the fitting function to update object values
        self.calc_chi2(self.af)

        # degrees of freedom in fit
        # self.dof = len(self.xdat) - numfit # deg of freedom
        self.dof = m.dof

        # scaled uncertainties in fitting parameters
#        self.chi2 = m.fnorm
#        self.chi2_reduced = _np.sqrt(m.fnorm/m.dof)
        self.chi2 = _np.sum(_np.power(self.chi2, 2.0))
        self.chi2_reduced = self.chi2/self.dof
        self.perror = m.perror * self.chi2_reduced

        # calculate correlation matrix
        self.covmat = m.covar       # Covariance matrix

        # Scaled covariance matrix
        self.covmat = self.chi2_reduced * self.covmat
        try:
            self.cormat = self.covmat * 0.0
        except:
            raise
        # end if
        for ii in range(numfit):
            for jj in range(numfit):
                self.cormat[ii,jj] = self.covmat[ii,jj]/_np.sqrt(self.covmat[ii,ii]*self.covmat[jj,jj])
            # end for
        # end for

        return self.af, self.covmat
    # end def __use_mpfit

    # ===================================================================== #

    @property
    def xx(self):
        return self._x
    @xx.setter
    def xx(self, value):
        self._x = _np.asarray(value)
        self.yf = self.func(self.af, self._x)
#        if not self.autoderivative:
#            self.properror()
    @xx.deleter
    def xx(self):
        del self._x

    # ===================================================================== #

    def plot(self):
        if not hasattr(self, "xx") or not hasattr(self, "yf"):
            xx = _np.linspace(self.xdat.min(), self.xdat.max(), num=100)
            yf = self.func(self.af, xx)
            vary = _np.zeros_like(yf)
        else:
            xx = self.xx
            yf = self.yf
            vary = self.vary
        # end if
        yerr = _np.sqrt(vary)

        _plt.figure()
        _plt.errorbar(self.xdat, self.ydat, yerr=yerr, fmt='ko')
        _plt.plot(xx, yf, 'b-')
        if hasattr(self, 'vfit'):
            _plt.fill_between(xx, yf-_np.sqrt(self.vfit), yf+_np.sqrt(self.vfit),
                              interpolate=True, color='b', alpha=0.3)
#            _plt.plot(xx, yf+_np.sqrt(self.vfit), 'b--')
#            _plt.plot(xx, yf-_np.sqrt(self.vfit), 'b--')
        _plt.title('Input data and fitted model')
        _plt.show()
    # ===================================================================== #

# end class fitNL

# ========================================================================== #
# ========================================================================== #

# ============================================ #
# ---------- Specific Profile Fitting -------- #
# ============================================ #

def profilefit(x, y, ey, XX, func, fkwargs={}, **kwargs):
    """"
    For profile fitting, we generally end up with models that cannot be
    analytically scaled to a feasible problem / domain space by shifting/scaling
    parameter inputs.
        This is because we end up with terms like a*(1-x^2) or even (1-x^c)^d
        that cannot handle an x-shift with constant coefficients
            ==> introducing new x-dependency

    Here we handle the scaling problem by scaling input data, and then
    unscaling the resulting fits/derivatives. This means that we cannot output
    the fitting parameters because they solve the scaled problem.

    If you want the output parameters of the fit, then set the flag "scale_problem"
    to false, and the problem will not be scaled before fitting. Default is true.

    Note:  scaling is important because the domain of the problem is important!
        a*(1-x^c)^d is only a valid profile shape for x<1!
        (otherwise you need to vary c across the boundary)
    """
    bootstrapit = kwargs.pop('bootstrapit', 0)
    kwargs.setdefault('perpchi2', False)
    # kwargs.pop('errx')   # this prevents the effective variance from cominng into the problem
    kwargs.setdefault('scale_problem', False)
    kwargs.setdefault("scalex", True)
    kwargs.setdefault("scaley", False)

    xt, yt, eyt, XXt = _np.copy(x), _np.copy(y), _np.copy(ey), _np.copy(XX)
    if 'ex' in kwargs:
        ex = _np.copy(kwargs.pop('ex'), 0.0)
        kwargs['errx'] = ex
    # end if

    if bootstrapit:
        info = bootstrapprofilefit(xt, yt, eyt, XXt, func, fkwargs, nmonti=bootstrapit, **kwargs)
    else:

        slope = 1.0
        offset = 0.0
        xslope = 1.0
        xoffset = 0.0

        if kwargs['scaley']:
            slope = _np.nanmax(yt)-_np.nanmin(yt)   # maximum 1.0
            offset = _np.nanmin(yt)                # minimum 0.0

            if slope == 0:    slope = 1.0   # end if
            yt = (yt.copy()-offset)/slope
            eyt = eyt.copy()/(slope)
        # end if
        if kwargs['scalex']:
            xslope = 1.05*_np.nanmax((_np.nanmax(xt), _np.nanmax(XXt))) # shrink so maximum is less than 1.0
            xoffset = -1e-4  # prevent 0 from being in problem
            if xslope == 0:    xslope = 1.0   # end if

            XXt = (XXt.copy()-xoffset)/xslope
            xt = (xt.copy()-xoffset)/xslope
            if 'errx' in kwargs:
                errx = _np.copy(kwargs['errx'])
                errx = errx/xslope
            # end if
        # end if

        if 'errx' in kwargs:
            kwargs['errx'] = errx
        info = modelfit(xt, yt, eyt, XXt, func, fkwargs, **kwargs)

        if hasattr(info, 'xdat'):            info.xdat = xslope*info.xdat+xoffset
        if hasattr(info, 'vxdat'):           info.vxdat = info.vxdat*(xslope*xslope)
        if hasattr(info, 'pdat'):            info.pdat = slope*info.pdat+offset
        if hasattr(info, 'vdat'):            info.vdat = info.vdat*(slope*slope)

        info.XX = info.XX*xslope+xoffset
        info.prof = slope*info.prof+offset
        info.varprof = slope*slope*info.varprof
        info.dprofdx *= (slope/xslope)
        info.vardprofdx *= slope*slope/(xslope*xslope)
        info.d2profdx2 *= slope*slope/(xslope*xslope)
#        info.vard2profdx2 *= (slope*slope/(xslope*xslope))**2.0
        # end if
    # end if
    return info

def bootstrapprofilefit(xdat, ydat, ey, XX, func, fkwargs={}, nmonti=30, **kwargs):
    weightit = kwargs.setdefault('weightit', False)

    # =============== #
    # TODO!:   save the covariance matrices and then do the error propagation at end

    niterate = 1
    if nmonti > 1:
        niterate = nmonti
        # niterate *= len(x)
    # endif
    vary = ey**2.0

    xsav = xdat.copy()
    ysav = ydat.copy()
    vsav = vary.copy()

    chi2_reduced = _np.zeros((niterate,), dtype=_np.float64)

    nx = len(XX)
    mfit = _np.zeros((niterate, nx), dtype=_np.float64)
    dprofdx = _np.zeros_like(mfit)

    vfit = _np.zeros((niterate, nx), dtype=_np.float64) # end if
    vdprofdx = _np.zeros_like(vfit)

    nn = 0
    _np.random.seed()
    for mm in range(niterate):
        ydat = ysav.copy() + _np.sqrt(vsav)*_np.random.normal(0.0,1.0,_np.shape(ysav))
#        ydat[ydat<0] *= -1.0
        ydat[ydat<0] = 0.0
        vary = (ydat-ysav)*(ydat-ysav)
        vary = _np.where(_np.sqrt(vary)<1e-3*_np.nanmin(ydat), 1.0, vary)
        vary = _np.where(vary==0, _np.nanmean(vary), vary)
        if mm % 10 == 0:
            print('%i of %i'%(mm,niterate))
        # end if
        try:
#        if 1:
            temp = profilefit(xdat, ydat, _np.sqrt(vary), XX, func, fkwargs, **kwargs)

            print(r"%i: $\chi_%i^2$=%4.1f"%(mm, temp.dof, temp.chi2_reduced))
            if temp.chi2_reduced>10.0:
                continue
            kwargs['af0'] = temp.af.copy()
            mfit[nn, :] = _np.copy(temp.prof)
            vfit[nn, :] = _np.copy(temp.varprof)
            dprofdx[nn,:] = _np.copy(temp.dprofdx)
            vdprofdx[nn,:] = _np.copy(temp.vardprofdx)
            chi2_reduced[nn] = _np.copy(temp.chi2_reduced)
            nn += 1
        except:
            pass
        # end try
    # endfor
    mfit = mfit[:nn, :]
    vfit = vfit[:nn, :]
    dprofdx = dprofdx[:nn, :]
    vdprofdx = vdprofdx[:nn, :]
    chi2_reduced = chi2_reduced[:nn]

    xdat = xsav
    ydat = ysav
    vary = vsav

    _plt.figure()
    _plt.plot(XX, mfit.T, 'k--')
#    _plt.plot(XX, (mfit+_np.sqrt(vfit)).T, 'k--')
#    _plt.plot(XX, (mfit-_np.sqrt(vfit)).T, 'k--')

    if 0 and weightit:
        vfit, mfit = _ut.nanwvar(mfit.copy(), statvary=None, systvary=None, weights=1.0/vfit, dim=0, nargout=2)
        vdprofdx, dprofdx = _ut.nanwvar(dprofdx.copy(), statvary=None, systvary=None, weights=1.0/vdprofdx, dim=0, nargout=2)
    else:
        vfit = _np.nanvar(mfit, axis=0)
        vfit += _np.nansum(vfit, axis=0)/(nn*nn)
        mfit = _np.nanmean(mfit, axis=0)

        vdprofdx = _np.nanvar(dprofdx, axis=0)
        vdprofdx += _np.nansum(vdprofdx, axis=0)/(nn*nn)
        dprofdx = _np.nanmean(dprofdx, axis=0)

#        mfit, vfit = _ut.combine_var(mfit, statvar=vfit, systvar=None, axis=0)
#        dprofdx, vdprofdx = _ut.combine_var(dprofdx, stat♣var=vdprofdx, systvar=None, axis=0)
#        mfit, vfit = _ut.combine_var(mfit, statvar=None, systvar=None, axis=0)
#        dprofdx, vdprofdx = _ut.combine_var(dprofdx, statvar=None, systvar=None, axis=0)
    # end if
    info = temp
    info.update()

    numfit = len(temp.af)
    info.residuals = (ydat-_ut.interp(XX, mfit, ei=None, xo=xdat))/_np.where(ey==0, 1.0, ey)
    info.chi2_reduced = _np.sum(info.residuals*info.residuals)/(len(xdat)-numfit)
    info.prof = mfit
    info.varprof = vfit
    info.dprofdx = dprofdx
    info.vardprofdx = vdprofdx
    return info

# ========================================================================== #


def qparabfit(x, y, ey, XX, **kwargs):
    """
    This is a wrapper for using the MPFIT LM-solver with a quasi-parabolic model.
    This was written before the general fitting model above and is deprecated (obviously)
    in favor of the more general function.  This is still here purely because I
    see no reason to remove it yet.
    """
    # subfunction kwargs
    nohol = kwargs.pop("nohollow", False)

    # fitter arguments
    setedge = kwargs.setdefault('setedge', False)
    setcore = kwargs.setdefault('setcore', False)
    kwargs.setdefault("scale_problem", False)
    kwargs.setdefault("scalex", True)
    kwargs.setdefault("scaley", False)
    kwargs.setdefault("bootstrapit", 300)
#    kwargs.setdefault('perpchi2', True)
#    kwargs.setdefault('errx', 0.02)
    kwargs.setdefault('errx', 0)

    # plotting kwargs
    onesided = kwargs.pop('onesided', True)
    plotit = kwargs.pop('plotit', False)
    xlbl = kwargs.pop('xlabel', r'$r/a$')
    ylbl1 = kwargs.pop('ylabel1', r'T$_e$ [KeV]')
    ylbl2 = kwargs.pop('ylabel2', r'$-a\nabla\rho\nabla T_e/T_e$')
    titl = kwargs.pop('title', r'Quasi-parabolic profile fit')
    clr = kwargs.pop('color', 'r')
    alph = kwargs.pop('alpha', 0.3)
    hfig = kwargs.pop('hfig', None)
    agradrho = kwargs.pop('agradrho', 1.00)
    xlims = kwargs.pop('xlim', None)
    ylims1 = kwargs.pop('ylim1', None)
    ylims2 = kwargs.pop('ylim2', None)
    fs = kwargs.pop('fontsize', _plt.rcParams['font.size'])
    fn = kwargs.pop('fontname', _plt.rcParams['font.family'])

    fontdict = {'fontsize':fs, 'fontname':fn}

    if len(_np.atleast_1d(agradrho))==1:
        agradrho = agradrho * _np.ones_like(XX)
    # end if
    if _np.atleast_1d(agradrho).all() != 1.0 and ylbl2 == r'-$\nabla$ T$_e$/T$_e$':
        ylbl2 = r'a/L$_{Te}$'
    # endif

    # ============================= #

    xin = x.copy()
    yin = y.copy()
    eyin = ey.copy()

    xin = _np.abs(xin)
    isort = _np.argsort(xin)
    xin = xin[isort]
    yin = yin[isort]
    eyin = eyin[isort]

    # force flat profile in the middle either by cylindrical symmetry or
    # adding a ficticious point
    if (_np.min(xin) != 0) and setcore:
        xin = _np.insert(xin,0,0.0)
        yin = _np.insert(yin,0,yin[0])
        eyin = _np.insert(eyin,0,eyin[0])
    # end if
    # force the profile to go to zero somewhere in the edge if it doesn't already
    if (_np.nanmin(yin)>0.01*_np.nanmax(yin)) and setedge:
        # First try linearly interpolating the last few points to 0
        xedge, vedge = _ut.interp(yin[-3:], xin[-3:], ei=_np.sqrt(eyin[-3:]), xo=0.0)
        if (xedge < xin[-1]) or xedge>1.15:
            xedge = 1.05*_np.max((_np.nanmax(xin), 1.10))
            vedge = eyin[-1]
        xin = _np.insert(xin,-1, xedge)
        yin = _np.insert(yin,-1, 0.0)
        eyin = _np.insert(eyin,-1,vedge)
    isort = _np.argsort(_np.abs(xin))
    xin = xin[isort]
    yin = yin[isort]
    eyin = eyin[isort]
#
#    if xin[0] != -1*xin[-1] and xin[1] != -1*xin[-2]:
#        _, eyin = _ut.cylsym(xin, eyin)
#        xin, yin = _ut.cylsym(xin, yin)
##        xin = _ut.cylsym_odd(xin)   # repeated zeros at axis
##        yin = _ut.cylsym_even(yin)
##        eyin = _ut.cylsym_even(eyin)
#    # end if

    # ============================= #

    def myqparab(x=None, af=None, **kwargs):
        if 'XX' in kwargs: kwargs.pop('XX')
        return _ms.model_qparab(x, af=af, **kwargs)

    info = myqparab(None) #, nohollow=nohollow)
    af0 = info.af.copy()
    if not kwargs['scale_problem']:
        af0[0] = y[_np.argmin(x)].copy()
        af0[1] = 1e-18
    kwargs.setdefault('af0',af0)

    # Call mpfit
#    info = fit_mpfit(xin, yin, eyin, XX, myqparab, fkwargs={"nohollow":nohol}, **kwargs)
    info = profilefit(xin, yin, eyin, XX, myqparab, fkwargs={"nohollow":nohol}, **kwargs)

    # ================================= #

    info.dlnprofdx = info.dprofdx/info.prof
    info.vardlnprofdx = info.dlnprofdx**2.0 * (info.varprof/info.prof**2.0 + info.vardprofdx/info.dprofdx**2.0)

    info.aoverL = -1.0*agradrho*info.dlnprofdx
    info.var_aoverL = info.aoverL**2.0
    info.var_aoverL *= info.vardlnprofdx

    agr = _ut.interp(XX, agradrho, ei=None, xo=x)
    dydx_fd, vardydx_fd = _dd.findiff1d(x.copy(), y.copy(), ey.copy()**2.0)
    info.aoverL_fd = -1.0*agr*dydx_fd/y
    info.var_aoverL_fd = info.aoverL_fd**2.0
    info.var_aoverL_fd *= ( (ey/y)**2.0 + vardydx_fd/(dydx_fd**2.0))

    if plotit:
        if hfig is None:
            hfig = _plt.figure()
        else:
            _plt.figure(hfig.number)
        # endif
        ax1 = _plt.subplot(2,1,1)
        ax2 = _plt.subplot(2,1,2, sharex=ax1)

        if onesided:
            ax1.errorbar(x[x>0], y[x>0], yerr=ey[x>0], fmt=clr+'o', color=clr )
        else:
            ax1.errorbar(x, y, yerr=ey, fmt=clr+'o', color=clr )
        # end if
        ax1.plot(XX, info.prof, '-', color=clr, lw=2)
        ax1.fill_between(XX, info.prof-_np.sqrt(info.varprof),
                              info.prof+_np.sqrt(info.varprof),
                         interpolate=True, color=clr, alpha=alph)

        # ====== #
        ax2.plot(XX, info.aoverL, '-', color=clr, lw=1)
        if onesided:
            ax2.errorbar(x[x>0], info.aoverL_fd[x>0], yerr=_np.sqrt(info.var_aoverL_fd[x>0]), fmt=clr+'o', color=clr )
        else:
            ax2.errorbar(x, info.aoverL_fd, yerr=_np.sqrt(info.var_aoverL_fd), fmt=clr+'o', color=clr )
        # end if
        ax2.fill_between(XX, info.aoverL-_np.sqrt(info.var_aoverL),
                              info.aoverL+_np.sqrt(info.var_aoverL),
                         interpolate=True, color=clr, alpha=alph)
        # endif
        if xlims is None:    xlims = ax1.get_xlim()  # endif
        if ylims1 is None:   ylims1 = ax1.get_ylim()  # endif
        if ylims2 is None:
#            ylims2 = ax2.get_ylim()
            ylims2 = (_np.min(( -1.0, 1.2*_np.min(info.aoverL_fd), 1.2*_np.min(info.aoverL))),
                      _np.max((  1.0, 1.2*_np.max(info.aoverL_fd), 1.2*_np.max(info.aoverL))))
#                      _np.min((1.2*_np.max(info.aoverL_fd), _np.diff(ylims1)/(0.1*_np.diff(xlims)))))
        # endif
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims1)
        ax2.set_ylim(ylims2)
        ax1.set_title(titl, fontdict)
        ax1.set_ylabel(ylbl1, fontdict)
        ax2.set_ylabel(ylbl2, fontdict)
        ax2.set_xlabel(xlbl, fontdict)
        _plt.tight_layout()
        ax1.grid()
        ax2.grid()
    # end if plotit

    return info

# ========================================================================== #
# ========================================================================== #

# =========================== #
# ---------- testing -------- #
# =========================== #

def test_fit(func=_ms.model_qparab, **fkwargs):
    if 'Fs' in fkwargs:
        Fs = fkwargs.pop('Fs', 10.0e3)
        tstart, tend = tuple(fkwargs.pop('tbnds', [0.0, 6.0/33.0]))
        numpts = fkwargs.pop('num', int((tend-tstart)*Fs))
        xdat = _np.linspace(tstart, tend, num=numpts)
        XX = _np.linspace(tstart, tend, num=numpts)
    else:
        numpts = 21
        xdat = _np.linspace(-0.05, 1.25, numpts)
#        XX = _np.linspace( -1.0, 1.3, 99)
        XX = _np.linspace( 0.0, 1.3, 99)
#        XX = _np.linspace( 1e-6, 1.0, 99)
    # end if

    # Model to fit
    yxx, _, temp = func(XX=XX, **fkwargs)
    aa =temp.af
    ydat, _, _, _ = temp.update(xdat)

#    if (yxx<0.0).any():
#        offset = yxx.min()
#        ydat -= offset
#        yxx -= offset
#    # end if
    kwargs = temp.dict_from_class()
    kwargs.pop('XX')
    kwargs.pop('af')

    # Initial conditions for model
    if 'shape' in fkwargs: fkwargs.pop('shape')
    kwargs['af0'] = gen_random_init_conds(func, af0=aa, **fkwargs)

    # ====================== #

    yerr = 0.1*(_np.nanmean(ydat*ydat) - _np.nanmean(ydat)*_np.nanmean(ydat))
    yerr = yerr*_np.ones_like(ydat)
#    yerr = 0.1*ydat
    yerr = _np.max((0.1*ydat, yerr))

    kwargs.setdefault('plotit', False)
    info1 = modelfit(xdat, ydat, yerr, XX, func, fkwargs, **kwargs)
#    assert _np.allclose(info1.params, aa)
#    assert info.dof==len(xdat)-len(aa)

    # ====================== #
    _np.random.seed()
#    ydat = ydat + 0.1*_np.nanmean(ydat)*_np.random.normal(0.0, 1.0, len(ydat))
    ydat = ydat + yerr*_np.random.normal(0.0, 1.0, len(ydat))
    info2 = modelfit(xdat, ydat, yerr, XX, func, fkwargs, **kwargs)

    _plt.figure()
    ax1 = _plt.subplot(3, 1, 1)
    if numpts<50:
        ax1.errorbar(xdat, ydat, yerr=yerr, fmt='kx')
    else:
        ax1.plot(xdat, ydat, 'k-')
    ax1.plot(XX, yxx, 'k-', linewidth=1.0)
    ax1.plot(XX, info1.prof, 'b-')
    ax1.plot(XX, info1.prof-_np.sqrt(info1.varprof), 'b--')
    ax1.plot(XX, info1.prof+_np.sqrt(info1.varprof), 'b--')
    ax1.plot(XX, info2.prof, 'r-')
    ax1.plot(XX, info2.prof-_np.sqrt(info2.varprof), 'r--')
    ax1.plot(XX, info2.prof+_np.sqrt(info2.varprof), 'r--')

    # ax1.fill_between(XX, info.prof-_np.sqrt(info.varprof))
#    isolid = _np.where(yerr/ydat<0.3)[0]
#    if len(isolid)>0:
#        ax1.set_ylim((_np.nanmin(_np.append(0.0, _np.asarray([0.8,1.2])*_np.nanmin(ydat[isolid]-yerr[isolid]))),
#                      _np.nanmax(_np.append(0.0, _np.asarray([0.8,1.2])*_np.nanmax(ydat[isolid]+yerr[isolid]))) ))
    ax1.set_title(type(info1).__name__)
    xlims = ax1.get_xlim()
    ylims = ax1.get_ylim()
    ax1.text(x=0.6*(xlims[1]-xlims[0])+xlims[0], y=0.6*(ylims[1]-ylims[0])+ylims[0],
             s=r'\chi_\nu^2=%4.1f'%(info2.chi2_reduced,), fontsize=16)
#    ylims = ax1.get_ylim()
#    xlims = ax1.get_xlim()
#    print(ylims)

    ax2 = _plt.subplot(3, 1, 2, sharex=ax1)
    if numpts<50:
        ax2.plot(xdat, temp.dprofdx, 'kx')
    else:
        ax2.plot(xdat, temp.dprofdx, 'k-')
    ax2.plot(XX, info1.dprofdx, 'b-')
    ax2.plot(XX, info1.dprofdx-_np.sqrt(info1.vardprofdx), 'b--')
    ax2.plot(XX, info1.dprofdx+_np.sqrt(info1.vardprofdx), 'b--')
    ax2.plot(XX, info2.dprofdx, 'r-')
    ax2.plot(XX, info2.dprofdx-_np.sqrt(info2.vardprofdx), 'r--')
    ax2.plot(XX, info2.dprofdx+_np.sqrt(info2.vardprofdx), 'r--')
#    ax2.set_ylim((_np.min((0,1.2*_np.min(info.dprofdx))), 1.2*_np.max(info.dprofdx)))
#    isolid = _np.where(_np.sqrt(info.vardprofdx)/info.dprofdx<0.3)[0]
#    if len(isolid)>0:
#        ax2.set_ylim((_np.nanmin(_np.append( 0.0, _np.asarray([0.8, 1.2])*_np.nanmin(info.dprofdx[isolid]-_np.sqrt(info.vardprofdx[isolid])) )),
#                      _np.nanmax(_np.append( 0.0, _np.asarray([0.8, 1.2])*_np.nanmax(info.dprofdx[isolid]+_np.sqrt(info.vardprofdx[isolid])) )) ))
#                  #, (_np.nanmax(ydat)-_np.nanmin(ydat))/(0.1*_np.diff(xlims))))))
##    print(ax2.get_ylim())

    ax3 = _plt.subplot(3, 1, 3)
    ax3.bar(left=_np.asarray(range(len(aa))), height=100.0*(1.0-info2.params/aa), width=1.0)
    ax3.set_title('perc. diff.')
    _plt.show()
#    print(ydat)
# end

# ========================================================================== #

def test_profile_fit(func=_ms.model_qparab, **fkwargs):
    numpts = 21
    xdat = _np.linspace(0.01, 1.05, numpts)
#    xdat = _np.linspace(0.01, 1.25, numpts)
    XX = _np.linspace( 0.0, 1.3, 99)

    # Model to fit
#    aa = gen_random_init_conds(_ms.model_qparab, **fkwargs)
#    yxx, _, temp = _ms.model_qparab(XX=XX, af=aa, **fkwargs)
    aa = gen_random_init_conds(func, **fkwargs)
    yxx, _, temp = func(XX=XX, af=aa, **fkwargs)
#    yxx, _, temp = func(XX=XX, **fkwargs)
#    aa =temp.af
    ydat, _, _, _ = temp.update(xdat)
    offset = _np.min((yxx.min(), ydat.min(), 0.0))
    yxx = yxx - offset
    ydat = ydat - offset

    # ====================== #

    _, _, tempf = func(XX=XX, **fkwargs)
    kwargs = tempf.dict_from_class()
    if 'XX' in kwargs: kwargs.pop('XX')
    if 'af' in kwargs: kwargs.pop('af')

    # Initial conditions for model
    if 'shape' in fkwargs: fkwargs.pop('shape')
    kwargs['af0'] = gen_random_init_conds(func, **fkwargs)
#    kwargs['af0'] = gen_random_init_conds(func, af0=aa, **fkwargs)

    # ====================== #

#    yerr = 0.1*_np.mean(ydat)
    yerr = 0.10*ydat.max()
    yerr = yerr*_np.ones_like(ydat)
    yerr = _np.sqrt(yerr*yerr + (0.10*0.10*ydat*ydat))
    if (yerr==0).any():
        yerr[yerr==0] = 0.2*ydat.max()
    if (yerr<0).any():
        yerr[yerr<0] = _np.abs(yerr[yerr<0])

    kwargs.setdefault('plotit', False)
    info1 = profilefit(xdat, ydat, yerr, XX, func, fkwargs, **kwargs)
#    assert _np.allclose(info1.params, aa)
#    assert info.dof==len(xdat)-len(aa)

    # ====================== #
    _np.random.seed()
#    ydat = ydat + 0.1*_np.nanmean(ydat)*_np.random.normal(0.0, 1.0, len(ydat))
    ydat = ydat + yerr*_np.random.normal(0.0, 1.0, len(ydat))
#    ydat += 0.3*_np.nanmean(ydat)*_np.abs(_np.sin(2*_np.pi*xdat/1.0))

    info2 = profilefit(xdat, ydat, yerr, XX, func, fkwargs, **kwargs)

    _plt.figure()
    if len(aa) == len(info2.params):
        ax1 = _plt.subplot(3, 1, 1)
        ax2 = _plt.subplot(3, 1, 2, sharex=ax1)
        ax3 = _plt.subplot(3, 1, 3)
    else:
        ax1 = _plt.subplot(2, 1, 1)
        ax2 = _plt.subplot(2, 1, 2, sharex=ax1)
    # end if

    if numpts<50:
        ax1.errorbar(xdat, ydat, yerr=yerr, fmt='kx')
    else:
        ax1.plot(xdat, ydat, 'k-')
    ax1.plot(XX, yxx, 'k-', linewidth=1.0)
    ax1.plot(XX, info1.prof, 'b-')
    ax1.plot(XX, info1.prof-_np.sqrt(info1.varprof), 'b--')
    ax1.plot(XX, info1.prof+_np.sqrt(info1.varprof), 'b--')
    ax1.plot(XX, info2.prof, 'r-')
    ax1.plot(XX, info2.prof-_np.sqrt(info2.varprof), 'r--')
    ax1.plot(XX, info2.prof+_np.sqrt(info2.varprof), 'r--')

    ax1.set_title(type(info1).__name__)
    xlims = ax1.get_xlim()
    ylims = ax1.get_ylim()
    ax1.text(x=0.6*(xlims[1]-xlims[0])+xlims[0], y=0.6*(ylims[1]-ylims[0])+ylims[0],
             s=r'\chi_\nu^2=%4.1f'%(info2.chi2_reduced,), fontsize=16)

    if numpts<50:
        ax2.plot(xdat, temp.dprofdx, 'kx')
    else:
        ax2.plot(xdat, temp.dprofdx, 'k-')
    ax2.plot(XX, info1.dprofdx, 'b-')
    ax2.plot(XX, info1.dprofdx-_np.sqrt(info1.vardprofdx), 'b--')
    ax2.plot(XX, info1.dprofdx+_np.sqrt(info1.vardprofdx), 'b--')
    ax2.plot(XX, info2.dprofdx, 'r-')
    ax2.plot(XX, info2.dprofdx-_np.sqrt(info2.vardprofdx), 'r--')
    ax2.plot(XX, info2.dprofdx+_np.sqrt(info2.vardprofdx), 'r--')

    if len(aa) == len(info2.params):
        ax3.bar(left=_np.asarray(range(len(aa))), height=100.0*(1.0-info2.params/aa), width=1.0)
        ax3.set_title('perc. diff.')
    _plt.show()
# end

# ========================================================================== #

def test_fourier_fit(func=_ms.model_sines, **fkwargs):
    plotit = fkwargs.pop('plotit', True)
    fmod = fkwargs.setdefault('fmod', 113.0)
    Fs = fkwargs.pop('Fs', 10.0e3)
    tstart, tend = tuple(fkwargs.pop('tbnds', [0.0, 6.0/fmod]))
    numpts = fkwargs.pop('num', int((tend-tstart)*Fs))
    xdat = _np.linspace(tstart, tend, num=numpts)
    XX = _np.linspace(tstart, tend, num=numpts)
    Fs = len(xdat)/(xdat[-1]-xdat[0])

    # Model to fit
    af0 = gen_random_init_conds(func, **fkwargs)
    yxx, _, temp = func(XX=XX, af=af0, **fkwargs)
#    yxx, _, temp = func(XX=XX, **fkwargs)
    aa =_np.copy(temp.af)
    ydat, _, _, _ = temp.update(xdat)
    kwargs = temp.dict_from_class()
    kwargs.pop('XX')
    kwargs.pop('af')
    if 'LB'  in kwargs: kwargs.pop('LB')   # end if
    if 'Lbounds'  in kwargs: kwargs.pop('Lbounds')   # end if
    if 'UB'  in kwargs: kwargs.pop('UB')   # end if
    if 'Ubounds'  in kwargs: kwargs.pop('Ubounds')   # end if
    if 'fixed'  in kwargs: kwargs.pop('fixed')   # end if

    kwargs['scale_problem'] = True
    kwargs['perpchi2'] = True   # This greatly improves the convergence.

#    tmp = _np.cos(2.0*_np.pi*fmod*XX)

    # ====================== #

    # Estimate of noise on data
    yerr = 0.1*_np.ones_like(ydat)
    yerr = 0.5*_np.abs(_np.nanmax(ydat)-_np.nanmin(ydat))*yerr

    # Noise on data
    _np.random.seed()
    ytmp = ydat + 0.1*_np.abs(_np.nanmax(ydat)-_np.nanmin(ydat))*_np.random.normal(0.0, 1.0, len(ydat))

    # ====================== #
    # Loop through frequencies and build the results 1 frequency at a time
    nfreqs = fkwargs.pop('nfreqs', 1)

    # Use the auto-power spectrum to select initial guessed for the model
    from FFT.fft_analysis import getNpeaks
    afps = getNpeaks(nfreqs, xdat, ytmp, ytmp, plotit=plotit, Navr=1, windowoverlap=0.0,
                     detrend_style=1, windowfunction='box', fmin=0.5*fmod, minsep=0.4*fmod)
    aoffset = 0.0
    poffset = 2.0*_np.pi*fmod*xdat[_np.argmax(ytmp[:int(0.5*Fs/fmod)])]
#    aoffset = 2.0*_np.nanmean(ytmp)

#    fkwargs['offset'] = 0.0  # used in scaling
    if temp._params_per_freq == 3:  # model sines
        af0 = _np.asarray([aoffset]+afps[0])
        af0[-1] = poffset
#        kwargs['fixed'] = _np.zeros(af0.shape, dtype=int)
#        kwargs['fixed'][0] = 1  # offset fixed for looped runs
    elif temp._params_per_freq == 2:  # fourier model
        af0 = _np.asarray([fmod]+(_ms.ModelSines._convert2fourier(afps[0])).tolist())
#        kwargs['fixed'] = _np.zeros(af0.shape, dtype=int)
#        kwargs['fixed'][1] = 1
    # end if

    info1, info2 = None, None
    for ii in range(nfreqs):
        fkwargs['nfreqs'] = ii+1
        # Initial conditions for model
#        af0 = gen_random_init_conds(func, **fkwargs)
#        itmp = func(XX=None, **fkwargs)
#        af0 = _np.copy(itmp._af)

#        kwargs['maxiter'] = 200 if ii<(nfreqs-1) else 1200
        if ii>0:
            af0 = _np.copy(info1.af)
#            af0 = _np.copy(info2.af)
            if temp._params_per_freq == 3:  # model sines
                af0 = _np.asarray(af0.tolist() + afps[ii], dtype=_np.float64)
                # if the input cross-phase is zero, initialize with the solved
                # for cross-phase from the previous step.
                if af0[-1] == 0:
#                    af0[-1] = _np.copy(info2.af[-1])
                    af0[-1] = poffset
                # end if
            elif temp._params_per_freq == 2:  # fourier model
                af0 = _np.asarray(af0.tolist()+(_ms.ModelSines._convert2fourier(afps[ii])).tolist())
            # end if
#            af0 = _np.asarray( info1.af.tolist() + af0[-info1._params_per_freq:].tolist() )
            kwargs['fixed'] = _np.hstack(( _np.ones_like(info1.fixed),
                                _np.zeros((info1._params_per_freq,), dtype=int)))
            kwargs['Lbounds'] = _np.asarray(
                (info1.Lbounds).tolist()+info1.Lbounds[-info1._params_per_freq:].tolist())
            kwargs['Ubounds'] = _np.asarray(
                (info1.Ubounds).tolist()+info1.Ubounds[-info1._params_per_freq:].tolist())
#            kwargs['Lbounds'] = _np.asarray(
#                (info1.af-0.5*_np.abs(info1.af)).tolist()+info1.Lbounds[-info1._params_per_freq:].tolist())
#            kwargs['Ubounds'] = _np.asarray(
#                (info1.af+0.5*_np.abs(info1.af)).tolist()+info1.Ubounds[-info1._params_per_freq:].tolist())
        # end if
        kwargs['af0'] = _np.copy(af0)

        kwargs.setdefault('plotit', False)
        info1 = modelfit(xdat, ydat, yerr, XX, func, fkwargs, **kwargs)
        info2 = modelfit(xdat, ytmp, yerr, XX, func, fkwargs, **kwargs)
    # end if
#    kwargs['fixed'] = _np.zeros_like(info1.fixed)
##    kwargs['Lbounds'] = info1.af-0.1*_np.abs(info1.af)
##    kwargs['Ubounds'] = info1.af+0.1*_np.abs(info1.af)
#    kwargs['af0'] = info1.af
#    info1 = modelfit(xdat, ydat, yerr, XX, func, fkwargs, **kwargs)
#    kwargs['af0'] = info2.af
#    info2 = modelfit(xdat, ytmp, yerr, XX, func, fkwargs, **kwargs)

    _plt.figure()
    ax1 = _plt.subplot(3, 1, 1)
    if numpts<50:
        ax1.errorbar(xdat, ydat, yerr=yerr, fmt='kx')
    else:
        ax1.plot(xdat, ydat, 'k-')
    ax1.plot(XX, yxx, 'k-', linewidth=1.0)
    ax1.plot(XX, info1.prof, 'b-')
    ax1.plot(XX, info1.prof-_np.sqrt(info1.varprof), 'b--')
    ax1.plot(XX, info1.prof+_np.sqrt(info1.varprof), 'b--')
    ax1.plot(XX, info2.prof, 'r-')
    ax1.plot(XX, info2.prof-_np.sqrt(info2.varprof), 'r--')
    ax1.plot(XX, info2.prof+_np.sqrt(info2.varprof), 'r--')

    # ax1.fill_between(XX, info.prof-_np.sqrt(info.varprof))
#    isolid = _np.where(yerr/ydat<0.3)[0]
#    if len(isolid)>0:
#        ax1.set_ylim((_np.nanmin(_np.append(0.0, _np.asarray([0.8,1.2])*_np.nanmin(ydat[isolid]-yerr[isolid]))),
#                      _np.nanmax(_np.append(0.0, _np.asarray([0.8,1.2])*_np.nanmax(ydat[isolid]+yerr[isolid]))) ))
    ax1.set_title(type(info1).__name__)
    xlims = ax1.get_xlim()
    ylims = ax1.get_ylim()
    ax1.text(x=0.6*(xlims[1]-xlims[0])+xlims[0], y=0.6*(ylims[1]-ylims[0])+ylims[0],
             s=r'\chi_\nu^2=%4.1f'%(info2.chi2_reduced,), fontsize=16)
#    ylims = ax1.get_ylim()
#    xlims = ax1.get_xlim()
#    print(ylims)

    ax2 = _plt.subplot(3, 1, 2, sharex=ax1)
    if numpts<50:
        ax2.plot(xdat, temp.dprofdx, 'kx')
    else:
        ax2.plot(xdat, temp.dprofdx, 'k-')
    ax2.plot(XX, info1.dprofdx, 'b-')
    ax2.plot(XX, info1.dprofdx-_np.sqrt(info1.vardprofdx), 'b--')
    ax2.plot(XX, info1.dprofdx+_np.sqrt(info1.vardprofdx), 'b--')
    ax2.plot(XX, info2.dprofdx, 'r-')
    ax2.plot(XX, info2.dprofdx-_np.sqrt(info2.vardprofdx), 'r--')
    ax2.plot(XX, info2.dprofdx+_np.sqrt(info2.vardprofdx), 'r--')
#    ax2.set_ylim((_np.min((0,1.2*_np.min(info.dprofdx))), 1.2*_np.max(info.dprofdx)))
#    isolid = _np.where(_np.sqrt(info.vardprofdx)/info.dprofdx<0.3)[0]
#    if len(isolid)>0:
#        ax2.set_ylim((_np.nanmin(_np.append( 0.0, _np.asarray([0.8, 1.2])*_np.nanmin(info.dprofdx[isolid]-_np.sqrt(info.vardprofdx[isolid])) )),
#                      _np.nanmax(_np.append( 0.0, _np.asarray([0.8, 1.2])*_np.nanmax(info.dprofdx[isolid]+_np.sqrt(info.vardprofdx[isolid])) )) ))
#                  #, (_np.nanmax(ydat)-_np.nanmin(ydat))/(0.1*_np.diff(xlims))))))
##    print(ax2.get_ylim())

    ax3 = _plt.subplot(3, 1, 3)
    ax3.bar(left=_np.asarray(range(len(aa))), height=100.0*(1.0-info2.params/aa), width=1.0)
    ax3.set_title('perc. diff.')
    _plt.show()
#    print(ydat)
# end

# ========================================================================== #


def test_qparab_fit(nohollow=False):

    aa= _np.asarray([0.30, 0.002, 2.0, 0.7, -0.24, 0.30], dtype=float)
    if nohollow:
        aa[4] = 0.0
        aa[5] = 1.0
    # endif
    aa = gen_random_init_conds(_ms.model_qparab, af=aa.copy())

    xdat = _np.linspace(-0.05, 1.15, 21)
#    xdat = _np.linspace(-1.15, 1.2, 20)
    XX = _np.linspace( 0.0, 1.05, 101)

#    _np.random.seed()
    xdat += 0.02*_np.random.normal(0.0, 1.0, len(xdat))

    xoffset = -1e-4
    xslope = 1.05*_np.max((_np.nanmax(_np.abs(xdat)), _np.nanmax(XX)))
    ydat, _, temp = _ms.model_qparab((_np.abs(xdat).copy()-xoffset)/xslope, aa, nohollow=nohollow)
    temp.dprofdx = temp.dprofdx/(xslope)

    yxx, _, _ = _ms.model_qparab((XX.copy()-xoffset)/xslope, aa)

    # ============================================== #

    solver_options = {}  # end if

    # ============================================== #

#    yerr = 0.10*ydat
#    info = qparabfit(xdat, ydat, yerr, XX, nohollow=nohollow, **solver_options)
#
#    assert _np.allclose(info.params, aa)
#    assert info.dof==len(xdat)-len(aa)

#    _np.random.seed()
#    ydat += yerr*_np.random.normal(0.0, 1.0, len(xdat))
    ydat += 0.1*ydat*_np.random.normal(0.0, 1.0, len(xdat))
    yerr = 0.05*ydat + 0.025*_np.nanmean(ydat)
    info = qparabfit(xdat, ydat, yerr, XX, nohollow=nohollow, **solver_options)

    _plt.figure()
    ax1 = _plt.subplot(3, 1, 1)
    _plt.errorbar(xdat, ydat, yerr=yerr, fmt='bo')
    _plt.plot(XX, yxx, 'b-')
    _plt.plot(XX, info.prof, 'r-')
#    _plt.plot(XX, info.prof-_np.sqrt(info.varprof), 'r--')
#    _plt.plot(XX, info.prof+_np.sqrt(info.varprof), 'r--')
    _plt.fill_between(XX, info.prof-_np.sqrt(info.varprof),
                          info.prof+_np.sqrt(info.varprof),
                      interpolate=True, color='r', alpha=0.3)

    _plt.ylim((0, 1.2*_np.max(ydat)))
    ylims = _plt.ylim()
    xlims = _plt.xlim()
    ax1.set_title('Quasiparabolic')
    ax1.text(x=0.6*(xlims[1]-xlims[0])+xlims[0], y=0.6*(ylims[1]-ylims[0])+ylims[0],
             s=r'$\chi_\nu^2$=%4.1f'%(info.chi2_reduced,), fontsize=16)

    _plt.subplot(3, 1, 2, sharex=ax1)
    _plt.plot(xdat, temp.dprofdx, 'bo')
    _plt.plot(XX, info.dprofdx, 'r-')
#    _plt.plot(XX, info.dprofdx-_np.sqrt(info.vardprofdx), 'r--')
#    _plt.plot(XX, info.dprofdx+_np.sqrt(info.vardprofdx), 'r--')
    _plt.fill_between(XX, info.dprofdx-_np.sqrt(info.vardprofdx),
                          info.dprofdx+_np.sqrt(info.vardprofdx),
                      interpolate=True, color='r', alpha=0.3)
#    _plt.ylim((_np.min((0,1.2*_np.min(info.dprofdx))), 1.2*_np.max(info.dprofdx)))
    _plt.ylim((_np.min(( -1.0, 1.2*_np.min(info.dprofdx))),
               _np.max(( 1.0, 1.2*_np.max(info.dprofdx), _np.diff(ylims)/(0.1*_np.diff(xlims))))))

    _plt.subplot(3, 1, 3)
    tst, _, _ = _ms.model_qparab((_np.abs(xdat).copy()-xoffset)/xslope, info.params)
    _plt.bar(left=_np.asarray(range(len(aa)))+0.5, height=aa-info.params, width=0.5)
    _plt.show()
# end

# ========================================================================== #

def test_dat(multichannel=True):
#    x = _np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#                  dtype=_np.float64)
#    y = _np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47,
#                   98.36, 112.25, 126.14, 140.03], dtype=_np.float64)

#    _np.random.seed()
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

# ========================================================================== #

def test_doppler_data(**kwargs):
    noshift = kwargs.setdefault('noshift', True)
    model_order = kwargs.setdefault('model_order', 2)
    Fs = kwargs.setdefault('Fs', 10e6)
    logdata = kwargs.setdefault('logdata', False)
    hopstep = kwargs.pop('hopstep', 10e-3)
    freq = kwargs.pop('freq', None)
    relvar = kwargs.pop('relvar', 0.01)    # relative variance in data
    skewness = kwargs.pop('skewness', 0.0)    # relative skewness in data noise
    af = kwargs.pop('af', None)

    if freq is None:
        if Fs == 1:
            freq = _np.linspace(0.0, Fs, num=int(18750)) -0.5*Fs
        else:
            hopstep = kwargs.setdefault('hopstep', 10.0e-3) # hop step
            Nwin = kwargs.setdefault('Nwin', 8)  # default number of windows at 50% overlap in welch's PSD
            freq = _np.linspace(0.0, Fs, num=int(Fs*hopstep*1.5/Nwin)) -0.5*Fs
        # end if
#        freq /= Fs
    # end if

    if af is None:        # guassian
#        A0 = 1.0e-6   # [dB], amplitude
#        x0 = -0.04   # [Hz/Fs], shift of Doppler peak
#        s0 = 0.005    # [Hz/Fs], width of Doppler peak
#        _np.random.seed()
        A0 = _np.random.uniform(low=5.0e-7, high=1.0e-5, size=1)[0]
        x0 = _np.random.uniform(low=-0.08, high=0.01, size=1)[0]
        s0 = _np.random.uniform(low= 0.002, high=0.010, size=1)[0]
        af = _np.asarray([A0, x0, s0], dtype=_np.float64)

        if model_order>0:        # centered lorentzian
#            A1 = 1.0e-3   # [dB], amplitude
#            x1 = 0.001
#            s1 = 0.01    # [Hz/Fs], width of Doppler peak
            A1 = _np.random.uniform(low=1.0e-4, high=1.0e-3, size=1)[0]
            x1 = _np.random.uniform(low=-0.001, high=0.001, size=1)[0]
            s1 = _np.random.uniform(low=0.001,  high=0.02,  size=1)[0]
            if noshift:
                x1 = 0.0   # [Hz/Fs], shift of Doppler peak
            # end if
            af = _np.asarray(af.tolist()+[A1, x1, s1], dtype=_np.float64)
        if model_order>1:        # second shifted gaussian
#            A2 =  0.5*_np.copy(A0)   # [dB], amplitude
#            x2 = -0.75*_np.copy(x0)   # [Hz/Fs], shift of Doppler peak
#            s2 = 2*_np.copy(s0)    # [Hz/Fs], width of Doppler peak
            A2 = _np.random.uniform(low=1.0e-7, high=5.0e-6, size=1)[0]
            x2 = _np.random.uniform(low=0.001, high=0.08, size=1)[0]
            s2 = _np.random.uniform(low= 0.002, high=0.010, size=1)[0]
            af = _np.asarray(af.tolist()+[A2, x2, s2], dtype=_np.float64)
        # end if

        # add some randomization to the data
#        af += 0.2*af*_np.random.normal(0.0, 1.0, len(af))

#        # randomize the sign of the doppler shift
#        af[1] *= _np.random.choice([1.0, -1.0])
#        if model_order>1:            af[7] *= _np.random.choice([1.0, -1.0])        # end if

        for ii in range(1,len(af)):
            if _np.mod(ii+1,3) != 0:
                af[ii] *= Fs
            # end if
        # end for
    # end if

    logdata = kwargs.pop('logdata', False)
    data, _, _ = _ms.model_doppler(XX=freq, af=af, logdata=False, **kwargs)
    _, gvec, info = _ms.model_doppler(XX=freq, af=af, logdata=logdata, **kwargs)

    # assume the data represents the PSD of noisy IQ data
#    _np.random.seed()
    err = _np.sqrt(relvar)*_np.random.normal(loc=skewness, scale=1.0, size=len(freq))
    info.vary = _np.power(_np.sqrt(relvar)*data, 2.0)
    data += err*data
    data[data<0] = 0.1*_np.min(data[data>0])

    # assume the data represents the PSD of IQ data
    if logdata:
        data = 10.0*_np.log10(data)
        info.vary = _np.power(10.0/_np.log(10.0), 2.0) * _np.power(err, 2.0)
    # end if

    info.freq = freq
    return data, info

def test_doppler_model(**kwargs):
    _, info = test_doppler_data(**kwargs)
    af = info.af
    model = info.func
    partial_model = info.gfunc
    return af, model, partial_model, _ms.model_doppler
# end def

# ========================================================================== #

def test_fitNL(test_qparab=True, scale_by_data=False):

    def test_line_data():
        af = [0.2353335600009, 3.1234563234]
        return af, _ms.line, _ms.line_gvec, _ms.model_line

    def test_qparab_data():
        af = _np.array([5.0, 0.002, 2.0, 0.7, -0.24, 0.30], dtype=_np.float64)
        return af, _ms.qparab, _ms.partial_qparab, _ms.model_qparab

    def mymodel(_a, _x):
        return model(_x, _a)
    def myjac(_a, _x):
        return pderivmodel(_x, _a)

    if test_qparab:
        af, model, pderivmodel, wrapper = test_qparab_data()
        x = _np.linspace(-0.8, 1.05, 21)
#        x = _np.linspace(-0.8, 0.95, 11)
    else:
        af, model, pderivmodel, wrapper = test_line_data()
        x = _np.linspace(-2, 15, 100)
    # endif
    af0 = _np.copy( _np.asarray(af, dtype=_np.float64) )
#    _np.random.seed()
    af0 += 0.5*af0*_np.random.normal(0.0, 1.0, len(af))

    y = mymodel(af, x)
#    y += 0.05*_np.sin(0.53 * 2*_np.pi*x/(_np.max(x)-_np.min(x)))
    y += 0.1*_np.mean(y)*_np.random.normal(0.0, 1.0, len(y))
#    vary = None
#    vary = _np.zeros_like(y)
    vary = _np.power(0.2*_np.mean(y), 2.0)
#    _np.random.seed()
    vary += ( 0.05*_np.mean(y)*_np.random.normal(0.0, 1.0, len(y)) )**2.0

    if test_qparab:
        import pybaseutils as _pyb   # TODO:  SORTING IS THIS ISSUE WITH HFS DATA STUFF, it isflipping upside down and backwards
        isort = _np.argsort(_np.abs(x))
        x = _np.abs(x[isort])
#        isort = _np.argsort(x)
#        x = x[isort]
        y = y[isort]
        vary = vary[isort]

#        _, vary = _pyb.utils.cylsym(x, vary)
#        x, y = _pyb.utils.cylsym(x, y)
##        x = _pyb.utils.cylsym_odd(x)
##        y = _pyb.utils.cylsym_even(_np.copy(y))
##        vary = _pyb.utils.cylsym_even(_np.copy(vary))
    else:
        isort = _np.argsort(x)
        x = x[isort]
        y = y[isort]
        vary = vary[isort]
    # endif

    info = wrapper(XX=None)

    if scale_by_data:
        y, vary, slope, offset = _ms.rescale_problem(_np.copy(y), _np.copy(vary))
    # end if

    options = {}
#    options = {'fjac':myjac, 'UB':UB, 'LB':LB}
#    options = {'fjac':myjac}
    options = {'UB':info.Ubounds, 'LB':info.Lbounds, 'fixed':info.fixed}
    ft = fitNL(xdat=x, ydat=y, vary=vary, af0=af0, func=mymodel, **options)
    ft.fjac = myjac
    ft.run()
    ft.xx = _np.linspace(x.min(), x.max(), num=100)
    ft.properror(xvec=ft.xx, gvec=myjac(ft.af, ft.xx))   # use with analytic jacobian
#    ft.bootstrapper(ft.xx)   # use with no analytic jacobian
#    ft.yf = ft.func(ft.af, ft.xx)
#    a, b = ft.af

    if scale_by_data:
#        self.af
#        self.chi2_reduced
#        self.perror
#        self.covmat

        ft.pdat = ft.ydat
        ft.vdat = ft.vary
        ft.prof = ft.yf
        ft.varprof = ft.vfit

        ft.slope = slope
        ft.offset = offset
        ft.unscaleaf = info.unscaleaf
        ft = _ms.rescale_problem(info=ft, nargout=1)
        ft.ydat = ft.pdat
        ft.vary = ft.vdat
        ft.vfit = ft.varp
        ft.yf = ft.prof
    # end if
    ft.plot()

#    a, b, Var_a, Var_b = linreg(x, y, verbose=True, varY=vary, varX=None, cov=False, plotit=True)
    print(ft.af-af)
    return ft, af
# end def test_fitNL


def doppler_test(scale_by_data=False, noshift=True, Fs=10.0e6, model_order=2, logdata=False, hopstep=10.0e-3, af=None, af0=None, varaf0=None, fmax=1.0e6):
#    fit = _fl.modelfit(x=freq, y=10.0*_np.log10(_np.abs(PS)),
#              ey=_np.sqrt((1.0/(_np.abs(PS)*_np.log(10.0))*_np.abs(PS)/_np.sqrt(Navr))**2.0),
#              XX=freq, func=model_doppler, fkwargs={}, scale_problem=False)
    y, tmp = test_doppler_data(noshift=noshift, model_order=model_order, Fs=Fs, logdata=logdata, hopstep=hopstep, af=af)
    vary = tmp.vary.copy()
    af = tmp.af.copy()
    x = tmp.freq
    if fmax is not None:
        vary = vary[_np.abs(x)<fmax]
        y = y[_np.abs(x)<fmax]
        x = x[_np.abs(x)<fmax]
    _, model, pderivmodel, wrapper = test_doppler_model(noshift=noshift, Fs=Fs, model_order=model_order, logdata=logdata, af=af)

    if af0 is None:
        # randomize the initial conditions a little
#        af0 = gen_random_init_conds(wrapper, noshift=noshift, Fs=Fs, model_order=model_order)
#        af0 = gen_random_init_conds(wrapper, noshift=noshift, Fs=1.0, model_order=model_order, af0=af0, varaf0=varaf0)
        af0 = _np.copy(_np.asarray(af, dtype=_np.float64))
#        _np.random.seed()
        af0 += 0.3*af0*_np.random.normal(0.0, 1.0, len(af0))
        af0[1] *= _np.random.choice([1.0, -1.0])
        if model_order>1:
#            af0[7] = -1.0*af0[0]
            af0[7] *= _np.random.choice([1.0, -1.0])
#        # end if
#        af0[1] = -0.08*Fs
    else:     # if input is specified, don't do the randomization
        af0 = _np.copy(af0)
    # end if

    def mymodel(_a, _x):
        return model(_x, _a)
    def myjac(_a, _x):
        return pderivmodel(_x, _a)

    info = wrapper(XX=None, af=af0, noshift=noshift, model_order=model_order, Fs=Fs, logdata=logdata)
    if scale_by_data:
        y, vary, slope, offset = _ms.rescale_problem(_np.copy(y), _np.copy(vary))
    # end if

    options = {}
#    options = {'UB':info.Ubounds, 'LB':info.Lbounds, 'fixed':info.fixed}
    options = {'fjac':myjac, 'UB':info.Ubounds, 'LB':info.Lbounds, 'fixed':info.fixed}
    ft = fitNL(xdat=x, ydat=y, vary=vary, af0=af0, func=mymodel, **options)
    ft.run()
    ft.xx = _np.linspace(x.min(), x.max(), num=len(y))
    ft.properror(xvec=ft.xx, gvec=myjac(ft.af, ft.xx))   # use with analytic jacobian
#    ft.bootstrapper(ft.xx)   # use with no analytic jacobian
#    ft.yf = ft.func(ft.af, ft.xx)

    if scale_by_data:
#        self.af
#        self.chi2_reduced
#        self.perror
#        self.covmat

        ft.pdat = ft.ydat
        ft.vdat = ft.vary
        ft.prof = ft.yf
        ft.varp = ft.vfit
        ft.slope = slope
        ft.offset = offset
        ft.unscaleaf = info.unscaleaf
        ft = _ms.rescale_problem(info=ft, nargout=1)
        ft.ydat = ft.pdat
        ft.vary = ft.vdat
        ft.vfit = ft.varp
        ft.yf = ft.prof

#        x *= ascl
#        ft.x *= ascl
#        af[1:3] /= ascl
#        af[4:6] /= ascl
#        af[6:] /= ascl
    # end if
    ft.plot()

    figs = 2
    if model_order>1:
#        # switch the peak reported as a Doppler shift by greatest doppler shift
#        # call that the main component
#        if _np.abs(ft.af[7]) > _np.abs(ft.af[1]):
#            asave = _np.copy(ft.af)
#            asave[:3] = _np.copy(ft.af[6:])
#            asave[6:] = _np.copy(ft.af[:3])
#            ft.af = _np.copy(asave)
        # switch the peak reported as a Doppler shift by greatest modelled power
        if ft.af[6] > ft.af[0]:
            asave = _np.copy(ft.af)
            asave[:3] = _np.copy(ft.af[6:])
            asave[6:] = _np.copy(ft.af[:3])
            ft.af = asave
        # end if
    # end if

    data_integral = _np.trapz(ft.ydat, x=ft.xdat)
    if model_order>-1:
        percdiff = ['Perc. Diff in param %i: %4.1f%%\n'%(int(ii+1), 100.0*(ft.af[ii]-af[ii])/af[ii],) for ii in range(3)]
        fit0 = _ms.normal(ft.xdat, ft.af[:3])
        int1 = _np.trapz(fit0, x=ft.xdat)
        print((int1, ft.af[0]))
        fit_integral = _np.copy(int1)
        fits = _np.copy(ft.ydat)

        out = _np.asarray([ft.af[1], int1])
    if model_order>0:
        percdiff += ['Perc. Diff in param %i: %4.1f%%\n'%(int(ii+1), 100.0*(ft.af[ii]-af[ii])/af[ii],) for ii in range(3,6) if ii!=4 ]
        fit1 = _ms.lorentzian(ft.xdat, ft.af[3:6])
        fit_integral += _np.trapz(fit1, x=ft.xdat)
        fits -= _np.copy(fit1)
    if model_order>1:
        percdiff += ['Perc. Diff in param %i: %4.1f%%\n'%(int(ii+1), 100.0*(ft.af[ii]-af[ii])/af[ii],) for ii in range(6,9)]
        fit2 = _ms.normal(ft.xdat, ft.af[6:])
        int2 = _np.trapz(fit2, x=ft.xdat)
        print((int2, ft.af[6]))
        fit_integral += _np.copy(int2)

        if int2>0.10*int1: # and ft.af[6]>0.25*ft.af[0]:
            out2 = _np.asarray([ft.af[7], int2])
            out = _np.asarray(out.tolist()+out2.tolist())
        else:
            fits -= _np.copy(fit2)
            out2 = None
    # end if
    integral_perc_error = 100.0*(1.0-fit_integral/data_integral)
    ft.interr = _np.copy(integral_perc_error)
    print('Percent error in signal power:'+'%6.1f%%'%(integral_perc_error,))

    # ======================================= #

    _plt.figure()
    _ax1 = _plt.subplot(figs, 1, 1)
    _ax1.set_title('Input data and fitted model')
    _ax1.errorbar(ft.xdat, ft.ydat, yerr=_np.sqrt(ft.vary), fmt='k-', linewidth=0.25)
    _ax1.plot(ft.xdat, ft.ydat, 'k-', linewidth=1.0)
#    _ax1.fill_between(ft.xdat, ft.ydat-_np.sqrt(ft.vary), ft.ydat+_np.sqrt(ft.vary),
#          interpolate=True, color='k', alpha=0.3)
    _ax1.plot(ft.xx, ft.yf, 'b-', linewidth=1.0)
    _ax1.fill_between(ft.xx, ft.yf-_np.sqrt(ft.vfit), ft.yf+_np.sqrt(ft.vfit),
          interpolate=True, color='b', alpha=0.3)
    _ax1.set_ylabel('Doppler model')
    _ax1.set_ylim((_np.min(( 0, 0.8*_np.min(ft.yf), 1.2*_np.min(ft.yf))),
                   _np.max((    0.8*_np.min(ft.yf), 1.2*_np.max(ft.yf)))))
    ylims = _ax1.get_ylim()
#    xlims = _ax1.get_xlim()

    # ==== #

    _ax2 = _plt.subplot(figs, 1, 2, sharex=_ax1)
#    _ax2.errorbar(ft.xdat, fits, yerr=_np.sqrt(ft.vary), fmt='k-', linewidth=0.25)
    _ax2.plot(ft.xdat, fits, 'k-', linewidth=1.0)
    if model_order>1:
        if out2 is not None:
            _ax2.axvline(x=ft.af[7], ymin=ylims[0], ymax=ylims[1], color='r', linewidth=0.5, linestyle='--')
            _ax2.plot(ft.xdat, fit2+fit0, 'b-', linewidth=1.0)
        # end if
    else:
        _ax2.plot(ft.xdat, fit0, 'b-', linewidth=1.0)
    # end if
    _ax2.axvline(x=ft.af[1], ymin=ylims[0], ymax=ylims[1], color='r', linewidth=0.5, linestyle='--')
    _ax2.set_xlabel('f')
    _ax2.set_ylabel('Doppler peak')
    _ax2.set_ylim(ylims)

#    _plt.show()

    if logdata is False:
        ylims = [-70,0.0]
        _plt.figure()
        _ax10 = _plt.subplot(2,1,1)
        _ax10.plot(ft.xdat, 10*_np.log10(ft.ydat), 'k-', linewidth=1.0)
        _ax10.plot(ft.xx, 10*_np.log10(ft.yf), 'b-', linewidth=1.0)
        _ax10.set_ylim(ylims)
        _ax20 = _plt.subplot(2,1,2, sharex=_ax10)
        _ax20.plot(ft.xdat, 10*_np.log10(fit0), 'b-', linewidth=1.0)
        if model_order>1 and out2 is not None:
            _ax20.plot(ft.xdat, 10*_np.log10(fit2), 'b-', linewidth=1.0)
        _ax20.plot(ft.xdat, 10*_np.log10(fits), 'k-', linewidth=0.25)
        _ax20.set_ylim(ylims)
#        _ax20.set_ylim([1.1*min((_np.min(10.0*_np.log10(fit0)),_np.min(10.0*_np.log10(fits)))),
#                        1.1*max((_np.max(10.0*_np.log10(fit0)),_np.max(10.0*_np.log10(fits))))])
    # end if

    print(''.join(percdiff))
    return _np.asarray(out), ft
# end def


# ========================================================================== #
# ========================================================================== #

if __name__=="__main__":

#    for ii in range(5):
##        out, ft = doppler_test(scale_by_data=False, logdata=False, Fs=10.0e6, fmax=1.0e6)
##        out, ft = doppler_test(scale_by_data=False, logdata=False, Fs=1.0, fmax=0.1)
#        out, ft = doppler_test(scale_by_data=False, logdata=False, Fs=1.0, fmax=None)
#    print(out)

    test_qparab_fit(nohollow=False)
#    test_qparab_fit(nohollow=True)
#    ft = test_fitNL(False)
#    ft = test_fitNL(True)  # issue with interpolation when x>1 and x<0?
#
#    test_fit(_ms.model_line, scale_problem=True)  # scaling and shifting works
#
#    test_fourier_fit(_ms.model_sines, nfreqs=1,  Fs=100e3, numpts=int(6.0*1e3/33.0), fmod=33.0)
#    test_fourier_fit(_ms.model_sines, nfreqs=1,  Fs=10e3, fmod=33.0)
#    test_fourier_fit(_ms.model_sines, nfreqs=3, Fs=10e3, fmod=33.0)
#    test_fourier_fit(_ms.model_sines, nfreqs=7, Fs=10e3, fmod=33.0, shape='square', duty=0.66667)
#    # =====  #
##    test_fourier_fit(_ms.model_fourier, nfreqs=3, Fs=10e3, numpts=int(6.0*2e3/33.0), fmod=33.0)
##    test_fourier_fit(_ms.model_fourier, nfreqs=14, Fs=10e3, numpts=int(6.0*2e3/33.0), fmod=33.0, shape='square')
##    test_fourier_fit(_ms.model_fourier, nfreqs=6, Fs=10e3, numpts=int(6.0*2e3/33.0), fmod=33.0, shape='square', duty=0.66667)
##    test_fourier_fit(_ms.model_fourier, nfreqs=10, Fs=10e3, numpts=int(6.0*2e3/33.0), fmod=33.0, shape='square', duty=0.66667)
##    test_fourier_fit(_ms.model_fourier, nfreqs=14, Fs=10e3, numpts=int(6.0*2e3/33.0), fmod=33.0, shape='square', duty=0.66667)
#    # =====  #
#
#    test_fit(_ms.model_poly, npoly=2)
#    test_fit(_ms.model_poly, npoly=3)
#    test_fit(_ms.model_poly, npoly=6)
#    test_fit(_ms.model_ProdExp, npoly=2)
#    test_fit(_ms.model_ProdExp, npoly=3)
#    test_fit(_ms.model_ProdExp, npoly=4)
#    test_fit(_ms.model_evenpoly, npoly=2)
#    test_fit(_ms.model_evenpoly, npoly=3)
#    test_fit(_ms.model_evenpoly, npoly=6)
#    test_fit(_ms.model_evenpoly, npoly=10)
#    test_fit(_ms.model_PowerLaw, npoly=4)   #
#    test_fit(_ms.model_PowerLaw, npoly=5)   #
#    test_fit(_ms.model_PowerLaw, npoly=6)   #
#    test_fit(_ms.model_Exponential)
#    test_fit(_ms.model_parabolic)
###
###    test_profile_fit(_ms.model_evenpoly, npoly=2)
###
#    test_fit(_ms.model_gaussian, Fs=10e3, numpts=int(10e3*6*1.2/33.0))
#    test_fit(_ms._model_offsetgaussian, Fs=10e3, numpts=int(10e3*6*1.2/33.0))
#    test_fit(_ms.model_normal, Fs=10e3, numpts=int(10e3*6*1.2/33.0))
#    test_fit(_ms.model_offsetnormal, Fs=10e3, numpts=int(10e3*6*1.2/33.0))
#    test_fit(_ms.model_loggaussian, Fs=10e3, numpts=int(10e3*6*1.2/33.0))
#    test_fit(_ms.model_lorentzian, Fs=10e3, numpts=int(10e3*6*1.2/33.0))
#    test_fit(_ms.model_pseudovoigt, Fs=10e3, numpts=int(10e3*6*1.2/33.0))
#    test_fit(_ms.model_loglorentzian, Fs=10e3, numpts=int(10e3*6*1.2/33.0))
#    test_fit(_ms.model_doppler, noshift=1, Fs=1.0, model_order=0, tbnds=[-0.5,0.5], num=18750)
#    test_fit(_ms.model_doppler, noshift=1, Fs=1.0, model_order=1, tbnds=[-0.5,0.5], num=18750)
#    test_fit(_ms.model_doppler, noshift=1, Fs=1.0, model_order=2, tbnds=[-0.5,0.5], num=18750)
#    test_fit(_ms.model_doppler, noshift=0, Fs=1.0, model_order=2, tbnds=[-0.5,0.5], num=18750)
#    test_fit(_ms.model_doppler, logdata=True, noshift=1, Fs=1.0, model_order=0, tbnds=[-0.5,0.5], num=18750)
#    test_fit(_ms.model_doppler, logdata=True, noshift=1, Fs=1.0, model_order=1, tbnds=[-0.5,0.5], num=18750)
#    test_fit(_ms.model_doppler, logdata=True, noshift=1, Fs=1.0, model_order=2, tbnds=[-0.5,0.5], num=18750)
#    test_fit(_ms.model_doppler, logdata=True, noshift=0, Fs=1.0, model_order=2, tbnds=[-0.5,0.5], num=18750)
#    test_fit(_ms._model_twopower)
#    test_fit(_ms.model_twopower)
#    test_fit(_ms.model_expedge)
#    test_profile_fit(_ms.model_qparab, nohollow=False)
#    test_profile_fit(_ms.model_qparab, nohollow=True)
#    test_profile_fit(_ms.model_flattop)
#    test_profile_fit(_ms.model_slopetop)
###
#    test_profile_fit(_ms.model_qparab, bootstrapit=30)
#    test_profile_fit(_ms.model_flattop, bootstrapit=30)
#    test_profile_fit(_ms.model_slopetop, bootstrapit=30)

#    # need to be reformatted still (2 left!)
##    test_fit(_ms.model_Heaviside, npoly=3, rinits=[0.30, 0.35])
##    test_fit(_ms.model_Heaviside, npoly=3, rinits=[0.12, 0.27])
##    test_fit(_ms.model_Heaviside, npoly=4, rinits=[0.30, 0.35])
##    test_fit(_ms.model_Heaviside, npoly=4, rinits=[0.12, 0.27])
##    test_fit(_ms.model_StepSeries, npoly=3)
##    test_fit(_ms.model_StepSeries, npoly=4)
##    test_fit(_ms.model_StepSeries, npoly=5)
# endif


# ======================================================================= #
# ======================================================================= #


#    def __use_least_squares(self, **kwargs):
#        """
#        Wrapper around the scipy least_squares function
#        """
#        lsqfitmethod = kwargs.get("lsqfitmethod", 'lm')
#        self.af, chi2, resid, jac = \
#            least_squares(self.calc_chi2, self.af, bounds=(self.LB, self.UB),
#                          method=lsqfitmethod)
#
#        # Make a final call to the fitting function to update object values
#        self.calc_chi2(self.af)
#
#        # Estimate of covariance in solution
#        # jac = _np.full(jac) #Sparse matrix form
#        self.covmat = (resid*_np.eye[self.numfit]) / self.numfit / \
#            _np.dot(jac[:, 0:self.numfit].T, jac[:, 0:self.numfit])
#
#        return self.af, self.covmat
#    # end def __use_least_squares
#
#    # ===================================================================== #
#
#    def __use_curvefit(self, **kwargs):
#        """
#        Wrapper around scipy's curve_fit function
#        """
#        lsqfitmethod = kwargs.pop("lsqfitmethod", 'lm')
#        def calcchi2(xdat, *af):
#            af = _np.asarray(af)
#            return self.calc_chi2(af)
#        # end def calcchi2
#
#        pfit, pcov = fit_curvefit(p0=self.af, xdat=self.xdat, ydat=self.ydat,
#                                  func=self.calcchi2, yerr=_np.sqrt(self.vary),
#                                  epsfcn=0.0001, lsqmethod=lsqfitmethod,
#                                  bounds=(self.LB, self.UB), **kwargs)
#        if _scipyversion >= 0.17:
#            pfit, pcov = \
#                curve_fit(calcchi2, self.xdat, self.ydat, p0=self.af,
#                          sigma=_np.sqrt(self.vary), epsfcn=0.0001,
#                          absolute_sigma=True, bounds=(self.LB, self.UB),
#                          method=lsqfitmethod, **kwargs)
#        else:
#            pfit, pcov = \
#                curve_fit(calcchi2, self.xdat, self.ydat, p0=self.af,
#                          sigma=_np.sqrt(self.vary), **kwargs)
#        # end if
#
#        self.af = _np.asarray(pfit)
#        if _np.isfinite(pcov) == 0:
#            print('FAILED in curvefitting!')
#        # end if
#        self.covmat = _np.asarray(pcov)
#        return self.af, self.covmat
#    # end def __use_curvefit
#
#    # ===================================================================== #
#
#    def __use_leastsq(self, **kwargs):
#        """
#        Wrapper for the leastsq function from scipy
#        """
#        lsqfitmethod = kwargs.get("lsqfitmethod", 'lm')
#        if _scipyversion >= 0.17:
#            af, pcov, infodict, errmsg, success = \
#                leastsq(self.calc_chi2, self.af, full_output=1, ftol=1e-8,
#                        xtol=1e-8, maxfev=1e3, epsfcn=0.0001,
#                        method=lsqfitmethod)
#
#            # self.covmat = (resid*_np.eye[self.numfit]) / self.numfit \
#            #     / _np.dot(jac[:, 0:self.numfit].T, jac[:, 0:self.numfit])
#        else:
#            pfit, pcov, infodict, errmsg, success = \
#                leastsq(self.calc_chi2, x0=self.af, full_output=1, **kwargs)
#        # end if
#
#        self.af = _np.asarray(pfit, dtype=_np.float64)
#        if (len(self.ydat) > len(self.af)) and pcov is not None:
#            pcov = pcov * ((self.calc_chi2(self.af)**2).sum()
#                           / (len(self.ydat)-len(self.af)))
#        else:
#            pcov = _np.inf
#        # endif
#
#        self.covmat = _np.asarray(pcov, dtype=_np.float64)
#        return self.af, self.covmat
    # end def __use_leastsq

    # ===================================================================== #


# ======================================================================= #


#class fitNLold(Struct):
#    """
#    To use this first generate a class that is a chil of this one
#    class Prob(fitNL)
#
#        def __init__(self, xdata, ydata, yvar, options, **kwargs)
#            # call super init
#            super(fitNL, self).__init__(xdat, ydat, vary, af0, self.func, options, kwargs)
#        # end def
#
#        def func(self, af):
#            return y-2*af[0]+af[1]
#
#    """
#
#    def __init__(self, xdat, ydat, vary, af0, func, fjac=None, **kwargs):
#
#        options = {}
#        options.update(**kwargs)
#
#        self.xdat = xdat
#        self.ydat = ydat
#        self.vary = vary
#        self.af0 = af0
#        self.func = func
#        self.fjac = fjac
#
#        # ========================== #
#
#        options["nmonti"] = options.get("nmonti", 300)
#        options["af0"] = options.get("af0", self.af0)
#        options["LB"] = options.get("LB", -_np.Inf*_np.ones_like(self.af0))
#        options["UB"] = options.get("UB",  _np.Inf*_np.ones_like(self.af0))
#
#        # 1) Least-squares, 2) leastsq, 3) Curve_fit
#        options["lsqfitmethod"] = options.get("lsqfitmethod", 'lm')
#        if _scipyversion >= 0.17:
#            options["lsqmethod"] = options.get("lsqmethod", int(1))
#        else:
#            options["lsqmethod"] = options.get("lsqmethod", int(2))
#        # end if
#
#        # ========================== #
#
#        # Pull out the run data from the options dictionary
#        #  possibilities include
#        #   - lsqfitmetod - from leastsquares - 'lm' (levenberg-marquardt,etc.)
#        #   - LB, UB - Lower and upper bounds on fitting parameters (af)
#        self.__dict__.update(options)
#
#    # end def __init__
#
#    # ========================== #
#
#    def run(self, **kwargs):
#        self.__dict__.update(kwargs)
#
#        if self.lsqmethod == 1:
#            self.__use_least_squares(**kwargs)
#        elif self.lsqmethod == 2:
#            self.__use_leastsq(**kwargs)
#        elif self.lsqmethod == 3:
#            self.__use_curvefit(**kwargs)
#        return self.af, self.covmat
#    # end def run
#
#    # ========================== #
#
#    def calc_chi2(self, af):
#        self.chi2 = (self.func(af, self.xdat) - self.ydat)
#        self.chi2 = self.chi2 / _np.sqrt(self.vary)
#        return self.chi2
#
#    # ========================== #
#
#    def __use_least_squares(self, **options):
#        """
#        Wrapper around the scipy least_squares function
#        """
#        lsqfitmethod = options.get("lsqfitmethod", 'lm')
#
#        if _np.isscalar(self.af0):
#            self.af0 = [self.af0]
#        self.numfit = len(self.af0)
#
#        res = least_squares(self.calc_chi2, self.af0, bounds=(self.LB, self.UB),
#                          method=lsqfitmethod, **options)
#                          # args=(self.xdat,self.ydat,self.vary), kwargs)
#        self.af = res.x
#        # chi2
#        #resid = res.fun
#        jac = res.jac
#
#        # Make a final call to the fitting function to update object values
#        self.calc_chi2(self.af)
#
#        # Estimate of covariance in solution
#        # jac = _np.full(jac) #Sparse matrix form
#        # resid*
#        self.covmat = (_np.eye(self.numfit)) / self.numfit / \
#            _np.dot(jac[:, 0:self.numfit].T, jac[:, 0:self.numfit])
#
#        return self.af, self.covmat
#    # end def __use_least_squares
#
#    # ========================== #
#
#    def __use_curvefit(self, **kwargs):
#        """
#        Wrapper around scipy's curve_fit function
#        """
#        lsqfitmethod = kwargs.get("lsqfitmethod", 'lm')
#        def calcchi2(xdat, *af):
#            af = _np.asarray(af)
#            return self.calc_chi2(af)
#        # end def calcchi2
#
#        if _scipyversion >= 0.17:
#            pfit, pcov = \
#                curve_fit(calcchi2, self.xdat, self.ydat, p0=self.af0,
#                          sigma=_np.sqrt(self.vary), epsfcn=0.0001,
#                          absolute_sigma=True, bounds=(self.LB, self.UB),
#                          method=lsqfitmethod, **kwargs)
#        else:
#            pfit, pcov = \
#                curve_fit(calcchi2, self.xdat, self.ydat, p0=self.af0,
#                          sigma=_np.sqrt(self.vary), **kwargs)
#        # end if
#
#        self.af = _np.asarray(pfit)
#        if _np.isfinite(pcov) == 0:
#            print('FAILED in curvefitting!')
#        # end if
#        self.covmat = _np.asarray(pcov)
#        return self.af, self.covmat
#    # end def __use_curvefit
#
#    # ========================== #
#
#    def __use_leastsq(self, **kwargs):
#        """
#        Wrapper for the leastsq function from scipy
#        """
#        lsqfitmethod = kwargs.get("lsqfitmethod", 'lm')
#        if _scipyversion >= 0.17:
#            pfit, pcov, infodict, errmsg, success = \
#                leastsq(self.calc_chi2, self.af0, full_output=1, ftol=1e-8,
#                        xtol=1e-8, maxfev=1e3, epsfcn=0.0001,
#                        method=lsqfitmethod)
#
#            # self.covmat = (resid*_np.eye[self.numfit]) / self.numfit \
#            #     / _np.dot(jac[:, 0:self.numfit].T, jac[:, 0:self.numfit])
#        else:
#            pfit, pcov, infodict, errmsg, success = \
#                leastsq(self.calc_chi2, x0=self.af0, full_output=1, **kwargs)
#        # end if
#
#        self.af = _np.asarray(pfit, dtype=_np.float64)
#        if (len(self.ydat) > len(self.af)) and pcov is not None:
#            pcov = pcov * ((self.calc_chi2(self.af)**2).sum()
#                           / (len(self.ydat)-len(self.af)))
#        else:
#            pcov = _np.inf
#        # endif
#
#        self.covmat = _np.asarray(pcov, dtype=_np.float64)
#        return self.af, self.covmat
#    # end def __use_leastsq
#
#    # ========================== #
#    # ========================== #
#
#    def bootstrapper(self, xvec, **kwargs):
#        self.__dict__.update(kwargs)
#
#        niterate = 1
#        if self.nmonti > 1:
#            niterate = self.nmonti
#            # niterate *= len(self.xdat)
#        # endif
#
#        nch = len(self.xdat)
#        numfit = len(self.af0)
#        xsav = self.xdat.copy()
#        ysav = self.ydat.copy()
#        vsav = self.vary.copy()
#        af = _np.zeros((niterate, numfit), dtype=_np.float64)
#        chi2 = _np.zeros((niterate,), dtype=_np.float64)
#
#        nx = len(xvec)
#        self.mfit = self.func(self.af, xvec)
#        mfit = _np.zeros((niterate, nx), dtype=_np.float64)
#        for mm in range(niterate):
#
#            self.ydat = ysav.copy()
#            self.vary = vsav.copy()
#            _np.random.seed()
#            self.ydat += _np.sqrt(self.vary)*_np.random.normal(0.0,1.0,_np.shape(self.ydat))
#            self.vary = (self.ydat-ysav)**2
#
##            cc = 1+_np.floor((mm-1)/self.nmonti)
##            if self.nmonti > 1:
##                self.ydat[cc] = ysav[cc].copy()
##                self.ydat[cc] += _np.sqrt(vsav[cc]) * _np.random.normal(0.0,1.0,_np.shape(vsav[cc]))
##                self.vary[cc] = (self.ydat[cc]-ysav[cc])**2
##                    # _np.ones((1,nch), dtype=_np.float64)*
##            # endif
#
#            af[mm, :], _ = self.run()
#            chi2[mm] = _np.sum(self.chi2)/(numfit-nch-1)
#
#            mfit[mm, :] = self.func(af[mm,:], xvec)
#        # endfor
#        self.xdat = xsav
#        self.ydat = ysav
#        self.vary = vsav
#
#        self.vfit = _np.var(mfit, axis=0)
#        self.mfit = _np.mean(mfit, axis=0)
#
#        # straight mean and covariance
#        self.covmat = _np.cov(af, rowvar=False)
#        self.af = _np.mean(af, axis=0)
#
##        # weighted mean and covariance
##        aw = 1.0/(1.0-chi2) # chi2 close to 1 is good, high numbers good in aweights
##        covmat = _np.cov( af, rowvar=False, aweights=aw)
#
#        # Weighting by chi2
#        # chi2 = _np.sqrt(chi2)
#        # af = _np.sum( af/(chi2*_np.ones((1,numfit),dtype=_np.float64)), axis=0)
#        # af = af/_np.sum(1/chi2, axis=0)
#
#        # self.covmat = covmat
#        # self.af = af
#        return self.af, self.covmat
#
#    # ========================== #
#
#    def properror(self, xvec, gvec):  # (x-positions, gvec = dqparabda)
#        if gvec is None: gvec = self.fjac # endif
#        sh = _np.shape(xvec)
#
#        nx = len(xvec)
#        self.mfit = self.func(self.af, xvec)
#
#        self.vfit = _np.zeros(_np.shape(xvec), dtype=_np.float64)
#        for ii in range(nx):
#            # Required to propagate error from model
#            self.vfit[ii] = _np.dot(_np.atleast_2d(gvec[:,ii]), _np.dot(self.covmat, gvec[:,ii]))
#        # endfor
#        self.vfit = _np.reshape(self.vfit, sh)
#        return self.vfit
#
#    # ========================== #
#
## end class fitNL



# ======================================================================== #
# ======================================================================== #




























# ======================================================================== #
# ======================================================================== #



