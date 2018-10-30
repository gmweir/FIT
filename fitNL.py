# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:23:49 2018

@author: gawe
"""

# ======================================================================== #
# ======================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals


import numpy as _np
import matplotlib.pyplot as _plt
from pybaseutils.Struct import Struct
from pybaseutils import utils as _ut
from FIT import model_spec as _ms

# Ideally, we'll use the straight python implementation of LMFIT.
# This is mostly python agnostic and has better option controls
try:
    __import__('mpfit')
    __mpfitonpath__ = True
    from mpfit import mpfit as LMFIT
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


#def bootstrapfit(xdat, ydat, ey, XX, func, fkwargs={}, nmonti=300, **kwargs):
#    kwargs.setdefault('xtol', 1e-14)
#    kwargs.setdefault('ftol', 1e-14)
#    kwargs.setdefault('gtol', 1e-14)
#    kwargs.setdefault('damp', 0.)
#    kwargs.setdefault('maxiter', 600)
#    kwargs.setdefault('factor', 100)  # 100
#    kwargs.setdefault('nprint', 10) # 100
#    kwargs.setdefault('iterfunct', 'default')
#    kwargs.setdefault('iterkw', {})
#    kwargs.setdefault('nocovar', 0)
#    kwargs.setdefault('rescale', 0)
#    kwargs.setdefault('autoderivative', 1)
#    kwargs.setdefault('quiet', 1)
#    kwargs.setdefault('diag', 0)
#    kwargs.setdefault('epsfcn', max((_np.nanmean(_np.diff(xdat)),1e-2))) #5e-4) #1e-3
#    kwargs.setdefault('debug', 0)
#
#    weightit = kwargs.setdefault('weightit', False)
#    gvecfunc = kwargs.setdefault('gvecfunc', None)
#
#    # =============== #
#
#    niterate = 1
#    if nmonti > 1:
#        niterate = nmonti
#        # niterate *= len(x)
#    # endif
#
#    # nch = len(x)
#    numfit = len(info.af0)
#    vary = ey**2.0
#
#    xsav = xdat.copy()
#    ysav = ydat.copy()
#    vsav = vary.copy()
#
#    af = _np.zeros((niterate, numfit), dtype=_np.float64)
#    covmat = _np.zeros((niterate, numfit, numfit), dtype=_np.float64)
#    vaf = _np.zeros((niterate, numfit), dtype=_np.float64)
#    chi2 = _np.zeros((niterate,), dtype=_np.float64)
#
#    nx = len(XX)
#    mfit = _np.zeros((niterate, nx), dtype=_np.float64)
#    dprofdx = _np.zeros_like(mfit)
#
#    if gvecfunc is not None:
#        gvec, _, dgdx = gvecfunc(info.af0, XX)
#    # end if
#    vfit = _np.zeros((niterate, nx), dtype=_np.float64) # end if
#    vdprofdx = _np.zeros_like(vfit)
#
#    for mm in range(niterate):
#        ydat = ysav.copy() + _np.sqrt(vsav)*_np.random.normal(0.0,1.0,_np.shape(ysav))
#        vary = (ydat-ysav)**2
##            vary = vsav.copy()
##            vary = vsav.copy()*_np.abs((ydat-ysav)/ysav)
##            vary = vsav.copy()*(1 + _np.abs((ydat-ysav)/ysav))
##            cc = 1+_np.floor((mm-1)/nmonti)
##            if nmonti > 1:
##                ydat[cc] = ysav[cc].copy()
##                ydat[cc] += _np.sqrt(vsav[cc]) * _np.random.normal(0.0,1.0,_np.shape(vsav[cc]))
##                vary[cc] = (ydat[cc]-ysav[cc])**2
##                # _np.ones((1,nch), dtype=_np.float64)*
##            # endif
##            print(mm, niterate)
##            res = self.run()
##            af[mm, :], _ = res
##            if verbose:
#        if mm % 10 == 0:
#            print('%i of %i'%(mm,niterate))
#        # end if
#        af[mm, :], covmat[mm,:,:] = run()
#        vaf[mm, :] = perror.copy()**2.0
#
#        # chi2[mm] = _np.sum(self.chi2)/(numfit-nch-1)
#        mfit[mm, :] = func(af[mm,:].copy(), xvec.copy())
#        chi2[mm] = _np.sum(chi2_reduced.copy())
#        # mfit[mm, :] = self.yf.copy()
#
#        if gvecfunc is not None:
#            gvec, tmp, dgdx = gvecfunc(af[mm,:].copy(), XX.copy())
#            dprofdx[mm,:] = tmp.copy()
#            if gvec is not None:
#                vfit[mm,:] = properror(XX, gvec)
#            if dgdx is not None:
#                vdprofdx[mm,:] = properror(XX, dgdx)
#            # end if
#        # end if
##            if dgdx is not None:
##                vdfdx[mm,:] = properror(XX, dgdx)
#        # end if
#    # endfor
#    xdat = xsav
#    ydat = ysav
#    vary = vsav
#
##        _plt.figure()
##        _plt.plot(xvec, mfit.T, 'k-')
##        _plt.plot(xvec, (mfit+_np.sqrt(vfit)).T, 'k--')
##        _plt.plot(xvec, (mfit-_np.sqrt(vfit)).T, 'k--')
#
#    if weightit:
#        # # weighted mean and covariance
#        # aw = 1.0/(1.0-chi2) # chi2 close to 1 is good, high numbers good in aweights
#        # # aw[_np.isnan(aw)*_np.isinf(aw)]
#        # Weighting by chi2
#        aw = 1.0/chi2
##            aw = 1.0/_np.sqrt(chi2)
#
#        # self.af = _np.sum( af*aw, axis=0) / _np.sum(aw, axis=0)
##            self.perror, self.af = _ut.nanwvar(af, statvary=None, systvary=None, weights=aw, dim=0)
#        perror, af = _ut.nanwvar(af.copy(), statvary=vaf, systvary=None, weights=aw, dim=0, nargout=2)
#        perror = _np.sqrt(perror)
#        covmat = _np.cov( af, rowvar=False, fweights=None, aweights=aw)
#
##        mfit = _np.nansum( mfit * 1.0/_np.sqrt(vfit) ) / _np.nansum( 1.0/_np.sqrt(vfit) )
##        vfit = _np.nanvar( mfit, axis=0)
#        # weight by chi2 or by individual variances?
#        if gvecfunc is not None:
#            # vfit, mfit = _ut.nanwvar(mfit.copy(), statvary=vfit, systvary=None, weights=1.0/_np.sqrt(vfit), dim=0, nargout=2)
##           vfit, mfit = _ut.nanwvar(mfit.copy(), statvary=None, systvary=None, weights=1.0/_np.sqrt(vfit), dim=0, nargout=2)
#            vfit, mfit = _ut.nanwvar(mfit.copy(), statvary=None, systvary=None, weights=1.0/vfit, dim=0, nargout=2)
#            vdprofdx, dprofdx = _ut.nanwvar(dprofdx.copy(), statvary=None, systvary=None, weights=1.0/vdprofdx, dim=0, nargout=2)
#        else:
#            aw = aw.reshape((niterate,1))*_np.ones((1,nx),dtype=_np.float64)              # reshape the weights
##            self.vfit, self.mfit = _ut.nanwvar(mfit.copy(), statvary=vfit, systvary=None, weights=aw, dim=0)
#            vfit, mfit = _ut.nanwvar(mfit.copy(), statvary=None, systvary=None, weights=aw, dim=0)
#            vdprofdx, dprofdx = _ut.nanwvar(dprofdx.copy(), statvary=None, systvary=None, weights=aw, dim=0)
#        # end if
#    else:
#        # straight mean and covariance
#        af, perror = _ut.combine_var(af, statvar=None, systvar=None, axis=0)
#        perror = _np.sqrt(perror)
#        mfit, vfit = _ut.combine_var(mfit, statvar=None, systvar=None, axis=0)
#        dprofdx, vdprofdx = _ut.combine_var(dprofdx, statvar=None, systvar=None, axis=0)
#
#        covmat = _np.cov(af, rowvar=False)
#    # end if
#    return af, covmat, info


# ======================================================================== #

# Fitting using the Levenberg-Marquardt algorithm.    #
def modelfit(x, y, ey, XX, func, fkwargs={}, **kwargs):
    kwargs.setdefault('xtol', 1e-14)
    kwargs.setdefault('ftol', 1e-14)
    kwargs.setdefault('gtol', 1e-14)
    kwargs.setdefault('damp', 0.)
    kwargs.setdefault('maxiter', 600)
    kwargs.setdefault('factor', 100)  # 100
    kwargs.setdefault('nprint', 10) # 100
    kwargs.setdefault('iterfunct', 'default')
    kwargs.setdefault('iterkw', {})
    kwargs.setdefault('nocovar', 0)
    kwargs.setdefault('rescale', 0)
    kwargs.setdefault('autoderivative', 1)
    kwargs.setdefault('quiet', 0)
    kwargs.setdefault('diag', 0)
    kwargs.setdefault('epsfcn', max((_np.nanmean(_np.diff(x.copy())),1e-3))) #5e-4) #1e-3
    kwargs.setdefault('debug', 0)
    return fit_mpfit(x, y, ey, XX, func, fkwargs, **kwargs)

def fit_mpfit(x, y, ey, XX, func, fkwargs={}, **kwargs):

    # subfunction kwargs
    scale_by_data = kwargs.pop('scale_problem',True)

    # fitter kwargs
    LB = kwargs.pop('LB', None)
    UB = kwargs.pop('UB', None)
    p0 = kwargs.pop('af0', None)

    skipwithnans = kwargs.pop('PassBadFit', False)

    # default initial conditions come directly from the model functions
    _, _, info = func(XX, af=None, **fkwargs)
    info.success = False
    if p0 is None:
        p0 = info.af
    if LB is None:
        LB = info.Lbounds
    if UB is None:
        UB = info.Ubounds
    # end if
    numfit = len(p0)

    if numfit != LB.shape[0]:
        print('oops')
    # end if

    if scale_by_data:
        y, ey2, slope, offset = _ms.rescale_problem(_np.copy(y), _np.copy(ey)**2.0)
        ey = _np.sqrt(ey2)
    # end if

    # ============================================= #

    def mymodel(p, fjac=None, x=None, y=None, err=None, nargout=1):
        # Parameter values are passed in "p"
        # If fjac==None then partial derivatives should not be
        # computed.  It will always be None if MPFIT is called with default
        # flag.
        model, gvec, info = func(x, p, **fkwargs)
        model = _ut.interp_irregularities(model, corezero=False)
        gvec = _ut.interp_irregularities(gvec, corezero=False)

        # Non-negative status value means MPFIT should continue, negative means
        # stop the calculation.
        status = 0
        if _np.isnan(p).any():
            print('NaN in model parameters!')
            status = -2
        elif _np.isnan(p).all():
            print('All the model parameters are NaNs!')
            status = -3
        # end if
        if kwargs['autoderivative'] == 0 and nargout == 1:
            fjac = gvec.copy()   # use analytic jacobian
            return {'status':status, 'residual':(y-model)/err, 'jacobian':fjac}
        elif kwargs['autoderivative'] == 1 and nargout == 1:
            fjac = None
            return {'status':status, 'residual':(y-model)/err}
        else:
            info.prof = model.copy()
            info.gvec = gvec.copy()
            info.dprofdx = _ut.interp_irregularities(info.dprofdx, corezero=False) # assumes cylindrical if True
            return model, gvec, info
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
    parinfo = [{'value':p0[ii], 'fixed':0, 'limited':[1,1], 'limits':[LB[ii],UB[ii]]}
                for ii in range(numfit)]

    # Pass data into the solver through keywords
    fa = {'x':x, 'y':y, 'err':ey}

    # Call mpfit
#    kwargs['nprint'] = kwargs.get('nprint',10)
    m = LMFIT(mymodel, p0, parinfo=parinfo, residual_keywords=fa, **kwargs)
    #  m - object
    #   m.status   - there are more than 12 return codes (see mpfit documentation)
    #   m.errmsg   - a string error or warning message
    #   m.fnorm    - value of final summed squared residuals
    #   m.covar    - covaraince matrix
    #           set to None if terminated abnormally
    #   m.nfev     - number of calls to fitting function
    #   m.niter    - number if iterations completed
    #   m.perror   - formal 1-sigma uncertainty for each parameter (0 if fixed or touching boundary)
    #           .... only meaningful if the fit is weighted.  (errors given)
    #   m.params   - outputs!

    # Store the optimization information / messages and a boolean indicating success
    info.mpfit = m
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
        if skipwithnans:
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

    # ====== Post-processing ====== #
    # Final function evaluation
    prof, fjac, info = mymodel(m.params, x=XX, nargout=3)
    info.prof = prof
    info.fjac = fjac

    # Actual fitting parameters
    info.params = m.params

    # degrees of freedom in fit
    info.dof = len(x) - numfit # deg of freedom

    # calculate correlation matrix
    info.covmat = m.covar       # Covariance matrix
    info.cormat = info.covmat * 0.0
    for ii in range(numfit):
        for jj in range(numfit):
            info.cormat[ii,jj] = info.covmat[ii,jj]/_np.sqrt(info.covmat[ii,ii]*info.covmat[jj,jj])
        # end for
    # end for

    # ====== Error Propagation ====== #

    # scaled uncertainties in fitting parameters
    info.chi2_reduced = _np.sqrt(m.fnorm / info.dof)
    info.perror = m.perror * info.chi2_reduced

    # Scaled covariance matrix
    info.covmat = info.chi2_reduced * info.covmat

    # Propagate uncertainties in fitting parameters (scaled covariance) into
    # the profile and it's derivative
    info.varprof = _ut.properror(XX, info.covmat, fjac)

    if hasattr(info, 'dgdx'):
        info.vardprofdx = _ut.properror(XX, info.covmat, info.dgdx)
    # endif

    info.varprof = _ut.interp_irregularities(info.varprof, corezero=False)
    info.vardprofdx = _ut.interp_irregularities(info.vardprofdx, corezero=False)

    if scale_by_data:
        info.varp = _np.copy(info.varprof)
        info.slope = slope
        info.offset = offset
        info = _ms.rescale_problem(info=info, nargout=1)
        info.varprof = info.varp
        y = y*slope+offset
        ey2 = ey2*(slope**2.0)
        ey = _np.sqrt(ey2)
    # end if

    return info

# ========================================================================== #


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

    method = kwargs.pop('lsqmethod','lm')
    epsfcn = kwargs.pop('epsfcn', None) #0.0001)
    bounds = kwargs.pop('bounds', None)
    if (_scipyversion >= 0.17) and (yerr is not None):
        pfit, pcov = curve_fit(func, xdat, ydat, p0=p0, sigma=yerr,
                               absolute_sigma = True, method=method,
                               epsfcn=epsfcn, bounds=bounds)
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


def fit_mcleastsq(p0, xdat, ydat, func, yerr_systematic=0.0, nmonti=300):
    """
    function [pfit,perr] = fit_mcleastsq(p0, xdat, ydat, func, yerr_systematic, nmonti)

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
    niterate = len(ydat)
    niterate *= nmonti
    cc = -1
    for ii in range(niterate):
        yy = ydat.copy()
        cc += 1
        if cc >= len(ydat):
            cc = 0
        # end if
        yy[cc] += _np.random.normal(0., sigma_err_total, 1)
#        yy = ydat + _np.random.normal(0., sigma_err_total, len(ydat))

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
        if vary is None or (vary == 0).all():  vary = 1e-3*_np.nanmean(ydat)  # endif

        self.xdat = xdat
        self.ydat = ydat
        self.vary = vary
        self.af0 = af0
        self.func = func
        self.fjac = fjac

        # ========================== #

        solver_options = {}
        solver_options['xtol'] = kwargs.pop('xtol', 1e-14) # 1e-10
        solver_options['ftol'] = kwargs.pop('ftol', 1e-14) # 1e-10
        solver_options['gtol'] = kwargs.pop('gtol', 1e-14) # 1e-10
        solver_options['damp'] = kwargs.pop('damp', 0)
        solver_options['maxiter'] = kwargs.pop('maxiter', 600) # 200
        solver_options['factor'] = kwargs.pop('factor', 100) # 100 without rescale, scales the chi2? or the parameters?
        solver_options['nprint'] = kwargs.pop('nprint', 10)    # debug info
        solver_options['iterfunct'] = kwargs.pop('iterfunct', 'default')
        solver_options['iterkw'] = kwargs.pop('iterkw', {})
        solver_options['nocovar'] = kwargs.pop('nocovar', 0)
        solver_options['rescale'] = kwargs.pop('rescale', 0) # 0
#        solver_options['autoderivative'] = kwargs.pop('autoderivative', 1)  # if 0, then you must supply gvec
        solver_options['quiet'] = kwargs.pop('quiet', 0)
        solver_options['diag'] = kwargs.pop('diag', 0) # with rescale: positive scale factor for variables
        solver_options['epsfcn'] = kwargs.pop('epsfcn', max((_np.nanmean(_np.diff(xdat.copy())),1e-2))) # 0.001
        solver_options['debug'] = kwargs.pop('debug', 0)
#        if fjac is None:
        if 1:  # the other way doesn't work yet
            # default to finite differencing in mpfit for jacobian
            solver_options['autoderivative'] = kwargs.pop('autoderivative', 1)
        else:
            # use the user-defined function to calculate analytic partial derivatives
            solver_options['autoderivative'] = kwargs.pop('autoderivative', 0)
        # end if

        # ========================== #

        self.nmonti = kwargs.pop("nmonti", 600)
        self.af0 = kwargs.pop("af0", self.af0)
        self.LB = kwargs.pop("LB", -_np.Inf*_np.ones_like(self.af0))
        self.UB = kwargs.pop("UB",  _np.Inf*_np.ones_like(self.af0))

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
        if 1:
            self.chi2 = (self.func(af, self.xdat) - self.ydat)
            self.chi2 /= _np.sqrt(self.vary)
#        except:
#            pass
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

        _np.random.seed(1)
        cc = -1
        for mm in range(niterate):
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

#        self.mfit = self.func(self.af, xvec)
        self.vfit = _ut.properror(xvec, covmat=self.covmat, fjac=gvec)

#        sh = _np.shape(xvec)
#        nx = len(xvec)
#
#        self.vfit = _np.zeros(_np.shape(xvec), dtype=_np.float64)
#        for ii in range(nx):
#            # Required to propagate error from model
#            self.vfit[ii] = _np.dot(_np.atleast_2d(gvec[:,ii]), _np.dot(self.covmat, gvec[:,ii]))
#        # endfor
#        self.vfit = _np.reshape(self.vfit, sh)
        return self.vfit

    # ========================== #
# end class fitNL

# ======================================================================== #
# ======================================================================== #

#def MinMaxScaler(ydat, ymin, ymax, forward=True):
#    if forward:
#        return (ydat-ymin)/(ymax-ymin)
#    else:
#        return (ymax-ymin)*ydat+ymin
#    # end if
## end def

class fitNL(fitNL_base):

#    def __init__(self, xdat, ydat, vary=None, af0=[], func=None, options={}, **kwargs):
    def __init__(self, xdat, ydat, vary=None, af0=[], func=None, **kwargs):
        """
        Initialize the class for future calculations.  You have to call the
        obj.run_calc() method to actually get a result
        """
        # Update the run options with any extra input values or overwrites
#        solver_options.update(kwargs)
#        self.__dict__.update(**kwargs)
#        if not __mpfitonpath__:
#            if self.lsqmethod == 1:
#               self.lmfit = self.__use_least_squares
#            elif self.lsqmethod == 2:
#               self.lmfit = self.__use_leastsq
#            elif self.lsqmethod == 3:
#                self.lmfit = self.__use_curvefit
#            # end if
#        else:
#            self.lmfit = LMFIT
#        # end if
#        super(fitNL, self).__init__(xdat=xdat, ydat=ydat, vary=vary, af0=af0, func=func, options=solver_options)
        kwargs = super(fitNL, self).__init__(xdat=xdat, ydat=ydat, vary=vary, af0=af0, func=func, **kwargs)
        self.__dict__.update(**kwargs)
        self.lmfit = self.__use_mpfit  # alias the fitter
    # end __init__

    def run(self):
        if not hasattr(self, 'solver_options'):  self.solver_options = {}  # end if
        self.solver_options.setdefault('xtol', 1e-14) # 1e-14
        self.solver_options.setdefault('ftol', 1e-14) # 1e-14
        self.solver_options.setdefault('gtol', 1e-14) # 1e-14
        self.solver_options.setdefault('damp', 0.)
        self.solver_options.setdefault('maxiter', 600)
        self.solver_options.setdefault('factor', 100)  # 100
        self.solver_options.setdefault('nprint', 10)
        self.solver_options.setdefault('iterfunct', 'default')
        self.solver_options.setdefault('iterkw', {})
        self.solver_options.setdefault('nocovar', 0)
        self.solver_options.setdefault('rescale', 0)
        self.solver_options.setdefault('autoderivative', 1)
        self.solver_options.setdefault('quiet', 0)
        self.solver_options.setdefault('diag', 0)
        self.solver_options.setdefault('epsfcn', max((_np.nanmean(_np.diff(self.xdat.copy())),1e-2))) #5e-4) #1e-3
        self.solver_options.setdefault('debug', 0)
#        # default to finite differencing in mpfit for jacobian
#        self.solver_options.setdefault('autoderivative', 1)

        super(fitNL, self).run(**self.solver_options)
#        self.__use_mpfit(**self.solver_options)
        return self.af, self.covmat

        # ========================================== #
        if self.covmat is None:
            print("oops! The least squares solver didn't find a solution")
        # endif
    # end def

    def __use_mpfit(self, **kwargs):
        """
        """
#        lsqfitmethod = kwargs.get("lsqfitmethod", 'lm')
#        self.af, chi2, resid, jac = \
#            mpfit(self.calc_chi2, self.af, bounds=(self.LB, self.UB),
#                          method=lsqfitmethod)
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

        if self.autoderivative:
            mymodel = model_ad
        else:
            mymodel = model_ud
        # end if
        # default initial conditions come directly from the model functions
        self.success = False
        p0 = _np.atleast_1d(self.af0)

        LB = self.LB
        UB = self.UB
        numfit = len(p0)

        # Settings for each parameter of the fit.
        #   'value' is the initial value that will be updated by mpfit
        #   'fixed' is a boolean: 0 vary this parameter, 1 do not vary this parameter
        #   'limited' is a pair of booleans in a list [Lower bound, Upper bound]:
        #       ex//   [0, 0] -> no lower or upper bounds on parameter
        #       ex//   [0, 1] -> no lower bound on parameter, but create an upper bound
        #       ex//   [1, 1] -> lower and upper bounds on parameter
        #   'limits' lower and upper bound values matching boolean mask in 'limited'
        parinfo = [{'value':p0[ii], 'fixed':0, 'limited':[1,1], 'limits':[LB[ii],UB[ii]]}
                    for ii in range(numfit)]

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

        # degrees of freedom in fit
        # self.dof = len(self.xdat) - numfit # deg of freedom
        self.dof = m.dof
        self.chi2 = m.fnorm

        # calculate correlation matrix
        self.covmat = m.covar       # Covariance matrix
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

        # ====== Error Propagation ====== #
        # scaled uncertainties in fitting parameters
        self.chi2_reduced = _np.sqrt(m.fnorm/m.dof)
        self.perror = m.perror * self.chi2_reduced

        # Scaled covariance matrix
        self.covmat = self.chi2_reduced * self.covmat

        # Make a final call to the fitting function to update object values
        self.calc_chi2(self.af)

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


def gen_random_init_conds(func, **fkwargs):

    _, _, temp = func(1.0, af=None, **fkwargs)

    # Differentiate base data set from default starting positions by randomly
    # setting the profile to be within the upper / lower bounds (or reasonable values)
    aa = temp.af
    aa += 0.5*aa*_np.random.normal(0.0, 1.0, len(aa))

    # Doesn't work super well:
#    aa = _np.zeros_like(temp.af)
#    for ii in range(len(temp.af)):
#        LB = temp.Lbounds[ii]
#        UB = temp.Ubounds[ii]
#        if _np.isinf(LB):   LB = 0.1*_np.sign(LB)*temp.af[ii]            # endif
#        if _np.isinf(UB):   UB = 0.1*_np.sign(UB)*temp.af[ii]            # endif
#
#        aa[ii] = temp.af[ii] + _np.random.normal(0.5*LB+0.5*UB, _np.abs(UB-LB), 1)
#    # end for
    return aa

# ========================================================================== #


def test_fit(func=_ms.model_qparab, **fkwargs):

    aa = gen_random_init_conds(func, **fkwargs)

    xdat = _np.linspace(0.05, 0.95, 10)
    XX = _np.linspace( 0.0, 1.0, 99)
    ydat, _, temp = func(xdat, af=aa, **fkwargs)

#    yerr = 0.00*ydat
    yerr = 0.000001*_np.ones_like(ydat)
    yxx, _, _ = func(XX, af=aa, **fkwargs)

#    info = fit_mpfit(xdat, ydat, yerr, XX, func)
    info = fit_mpfit(xdat, ydat, yerr, XX, func, fkwargs)

#    assert _np.allclose(info.params, aa)
#    assert info.dof==len(xdat)-len(aa)

    ydat += yerr*_np.random.normal(0.04, 0.99, len(xdat))
#    info = fit_mpfit(xdat, ydat, yerr, XX, func)
    info = fit_mpfit(xdat, ydat, yerr, XX, func, fkwargs)

    _plt.figure()
    ax1 = _plt.subplot(3, 1, 1)
    _plt.errorbar(xdat, ydat, yerr=yerr, fmt='bo')
    _plt.plot(XX, yxx, 'b-')
    _plt.plot(XX, info.prof, 'r-')
    _plt.plot(XX, info.prof-_np.sqrt(info.varprof), 'r--')
    _plt.plot(XX, info.prof+_np.sqrt(info.varprof), 'r--')
    #_plt.fill_between(XX, info.prof-_np.sqrt(info.varprof))
    _plt.ylim((0, 1.2*_np.max(ydat)))

    _plt.subplot(3, 1, 2, sharex=ax1)
    _plt.plot(xdat, temp.dprofdx, 'bo')
    _plt.plot(XX, info.dprofdx, 'r-')
    _plt.plot(XX, info.dprofdx-_np.sqrt(info.vardprofdx), 'r--')
    _plt.plot(XX, info.dprofdx+_np.sqrt(info.vardprofdx), 'r--')
#    _plt.ylim((_np.min((0,1.2*_np.min(info.dprofdx))), 1.2*_np.max(info.dprofdx)))

    _plt.subplot(3, 1, 3)
    tst, _, _ = _ms.model_ProdExp(xdat, info.params)
    _plt.bar(left=_np.asarray(range(len(aa))), height=aa-info.params, width=1.0)
# end

# ========================================================================== #


def qparabfit(x, y, ey, XX, **kwargs):
    """
    This is a wrapper for using the MPFIT LM-solver with a quasi-parabolic model.
    This was written before the general fitting model above and is deprecated (obviously)
    in favor of the more general function.  This is still here purely because I
    see no reason to remove it yet.
    """
    # subfunction kwargs
    nohollow = kwargs.pop("nohollow", False)
    scale_by_data = kwargs.pop('scale_problem',True)

    # solver kwargs
    kwargs.setdefault('maxiter',600)
    kwargs.setdefault('epsfcn', max((_np.nanmean(_np.diff(x.copy())),1e-2)))
    kwargs.setdefault('factor',100)
    kwargs.setdefault('autoderivative',1)

    # plotting kwargs
    onesided = kwargs.pop('onesided', True)
    plotit = kwargs.pop('plotit', True)
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
    if _np.atleast_1d(agradrho).all() != 1.0 and ylbl2 == r'-$\nabla$ T$_e$/T$_e$':
        ylbl2 = r'a/L$_{Te}$'
    # endif

    def myqparab(x, af=None, nohollow=False, infoin=None):
        prune = False
        if nohollow:            prune = True        # end if
        return _ms.model_qparab(x, af=af, nohollow=nohollow, prune=prune, info=infoin)

    info = myqparab(None)
    af0 = info.af.copy()
    if scale_by_data:
        y, ey2, slope, offset = _ms.rescale_problem(_np.copy(y), _np.copy(ey)**2.0)
        ey = _np.sqrt(ey2)
        af0[0] = 1.0
        af0[1] = 0.0
    # end if
#    af0[0] = y[_np.argmin(x)].copy()
    kwargs.setdefault('af0',af0)

    # Call mpfit
    info = fit_mpfit(x, y, ey, XX, myqparab, fkwargs={"nohollow":nohollow}, **kwargs)

    if scale_by_data:
        # slope = _np.nanmax(pdat)-_np.nanmin(pdat)
        # offset = _np.nanmin(pdat)
#        info.prof = _np.copy(prof)
        info.varp = _np.copy(info.varprof)
#        info.dprofdx = _np.copy(dprofdx)
#        info.vardprofdx = _np.copy(vardprofdx)
#        info.af = _np.copy(af)
        info.slope = slope
        info.offset = offset
        info = _ms.rescale_problem(info=info, nargout=1)
        info.varprof = info.varp
        y = y*slope+offset
        ey2 = ey2*(slope**2.0)
        ey = _np.sqrt(ey2)
    # end if
    if nohollow:
        # add back in the two parameters after fitting with hollowness
        _, _, info = myqparab(XX, af=info.params, nohollow=False, infoin=info)
        info.params = info.af.copy()
        info.LB = info.Lbounds.copy()
        info.UB = info.Ubounds.copy()
    # endif

    # ================================= #

    info.aoverL = -1.0*agradrho*info.dprofdx/info.prof
    info.var_aoverL = info.aoverL**2.0
    info.var_aoverL *= ( info.varprof/info.prof**2.0 + info.vardprofdx/info.dprofdx**2.0)

    if plotit:
        if hfig is None:
            hfig = _plt.figure()
        else:
            _plt.figure(hfig.number)
        # endif
        ax1 = _plt.subplot(2,1,1)
        ax2 = _plt.subplot(2,1,2, sharex=ax1)

        ax1.set_title(titl, fontdict)
        ax1.set_ylabel(ylbl1, fontdict)
        ax2.set_ylabel(ylbl2, fontdict)
        ax2.set_xlabel(xlbl, fontdict)

        if onesided:
            ax1.errorbar(x[x>0], y[x>0], yerr=ey[x>0], fmt=clr+'o', color=clr )
        else:
            ax1.errorbar(x[x>0], y[x>0], yerr=ey[x>0], fmt=clr+'o', color=clr )
        # end if
        ax1.plot(XX, info.prof, '-', color=clr, lw=2)
#        ax1.plot(XX, (info.prof+_np.sqrt(info.varprof)), '--', color=clr, lw=1)
#        ax1.plot(XX, (info.prof-_np.sqrt(info.varprof)), '--', color=clr, lw=1)

        ax1.fill_between(XX, info.prof-_np.sqrt(info.varprof),
                              info.prof+_np.sqrt(info.varprof),
                          interpolate=True, color=clr, alpha=alph)

        # ====== #
        ax2.plot(XX, info.aoverL, '-', color=clr, lw=1)
#        ax2.plot(XX, info.aoverL+_np.sqrt(info.var_aoverL), '--', color=clr, lw=1)
#        ax2.plot(XX, info.aoverL-_np.sqrt(info.var_aoverL), '--', color=clr, lw=1)

        ax2.fill_between(XX, info.aoverL-_np.sqrt(info.var_aoverL),
                              info.aoverL+_np.sqrt(info.var_aoverL),
                          interpolate=True, color=clr, alpha=alph)
        # endif
        _plt.tight_layout()

        if xlims is None:   xlims = ax1.get_xlim()  # endif
        if ylims1 is None:   ylims1 = ax1.get_ylim()  # endif
        if ylims2 is None:   ylims2 = ax2.get_ylim()  # endif
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims1)
        ax2.set_ylim(ylims2)
        ax1.grid()
        ax2.grid()
    # end if plotit

    return info

# ========================================================================== #


def test_qparab_fit(nohollow=False):

    aa= _np.asarray([0.30, 0.002, 2.0, 0.7, -0.24, 0.30], dtype=float)
    if nohollow:
        aa[4] = 0.0
        aa[5] = 1.0
    # endif

    xdat = _np.linspace(0.05, 0.95, 10)
    ydat, _, temp = _ms.model_qparab(xdat, aa)
    yerr = 0.10*ydat

    # ============================================== #

    solver_options = {}  # end if
    solver_options.setdefault('xtol', 1e-14)
    solver_options.setdefault('ftol', 1e-14)
    solver_options.setdefault('gtol', 1e-14)
    solver_options.setdefault('damp', 0.)
    solver_options.setdefault('maxiter', 600)
    solver_options.setdefault('factor', 100)  # 100
    solver_options.setdefault('nprint', 10)
    solver_options.setdefault('iterfunct', 'default')
    solver_options.setdefault('iterkw', {})
    solver_options.setdefault('nocovar', 0)
    solver_options.setdefault('rescale', 0)
    solver_options.setdefault('autoderivative', 1)
    solver_options.setdefault('quiet', 0)
    solver_options.setdefault('diag', 0)
    solver_options.setdefault('epsfcn', max((_np.nanmean(_np.diff(xdat.copy())),1e-2))) #5e-4) #1e-3
    solver_options.setdefault('debug', 0)
    # default to finite differencing in mpfit for jacobian
    # solver_options.setdefault('autoderivative', 1)

    # ============================================== #


    XX = _np.linspace( 0.0, 1.0, 99)
    yxx, _, _ = _ms.model_qparab(XX, aa)

    info = qparabfit(xdat, ydat, yerr, XX, nohollow=False, **solver_options)

#    assert _np.allclose(info.params, aa)
#    assert info.dof==len(xdat)-len(aa)

    ydat += yerr*_np.random.normal(0.0, 1.0, len(xdat))
    info = qparabfit(xdat, ydat, yerr, XX, nohollow=False, **solver_options)

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

    _plt.subplot(3, 1, 2, sharex=ax1)
    _plt.plot(xdat, temp.dprofdx, 'bo')
    _plt.plot(XX, info.dprofdx, 'r-')
#    _plt.plot(XX, info.dprofdx-_np.sqrt(info.vardprofdx), 'r--')
#    _plt.plot(XX, info.dprofdx+_np.sqrt(info.vardprofdx), 'r--')
    _plt.fill_between(XX, info.dprofdx-_np.sqrt(info.vardprofdx),
                          info.dprofdx+_np.sqrt(info.vardprofdx),
                      interpolate=True, color='r', alpha=0.3)
    _plt.ylim((_np.min((0,1.2*_np.min(info.dprofdx))), 1.2*_np.max(info.dprofdx)))

    _plt.subplot(3, 1, 3)
    tst, _, _ = _ms.model_qparab(xdat, info.params)
    _plt.bar(left=_np.asarray(range(len(aa))), height=aa-info.params, width=1.0)
# end

# ========================================================================== #
# ========================================================================== #

# =========================== #
# ---------- testing -------- #
# =========================== #

def test_dat(multichannel=True):
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


def test_fitNL(test_qparab=True, scale_by_data=True):
    try:
        import model_spec as _ms
    except:
        from . import model_spec as _ms
    # end try
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
        x = _np.linspace(-0.8, 1.05, 11)
    else:
        af, model, pderivmodel, wrapper = test_line_data()
        x = _np.linspace(-2, 15, 100)
    # endif
    af0 = 0.1*_np.asarray(af, dtype=float)

    y = mymodel(af, x)
    y += 0.05*_np.sin(0.53 * 2*_np.pi*x/(_np.max(x)-_np.min(x)))
    y += 0.3*_np.mean(y)*_np.random.normal(0.0, 1.0, len(y))
#    vary = None
#    vary = _np.zeros_like(y)
    vary = 0.05*_np.mean(y)
    vary += ( 0.2*_np.mean(y)*_np.random.normal(0.0, 1.0, len(y)) )**2.0

    if test_qparab:
        import pybaseutils as _pyb
        x = _pyb.utils.cylsym_odd(x)
        y = _pyb.utils.cylsym_even(y)
        vary = _pyb.utils.cylsym_even(vary)
    # endif
    info = wrapper(XX=None)

    if scale_by_data:
        y, vary, slope, offset = _ms.rescale_problem(_np.copy(y), _np.copy(vary))
    # end if

    options = {}
#    options = {'fjac':myjac, 'UB':UB, 'LB':LB}
#    options = {'fjac':myjac}
    options = {'UB':info.Ubounds, 'LB':info.Lbounds}
#    ft = fitNL(xdat=x, ydat=y, vary=vary, af0=af0, func=mymodel, fjac=myjac)
#    ft = fitNL(xdat=x, ydat=y, vary=vary, af0=af0, func=mymodel)
    ft = fitNL(xdat=x, ydat=y, vary=vary, af0=af0, func=mymodel, **options)
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
        ft.varp = ft.vfit
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
    return ft
# end def test_fitNL


# ========================================================================== #
# ========================================================================== #

if __name__=="__main__":
    test_qparab_fit(nohollow=False)
    test_qparab_fit(nohollow=True)
    ft = test_fitNL(True)
    ft = test_fitNL(False)

#    test_fit(_ms.model_qparab, nohollow=False)
#    test_fit(_ms.model_qparab, nohollow=True)
#    test_fit(_ms.model_ProdExp, npoly=2)
#    test_fit(_ms.model_ProdExp, npoly=3)
#    test_fit(_ms.model_ProdExp, npoly=4)
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
#
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



