# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:38:02 2016

@author: gawe
"""

#Test the models in stelltran/model_spec

from FIT.model_spec import *
import numpy as _np
import matplotlib.pyplot as _plt

XX = _np.linspace(0, 1, 200)
npoly = 3
model_number = 1
#    af = [0.1,0.1,0.1]
af = None
#    af = [2.0,0.1,0.1,0.5]

[chi_eff, gvec, info] = \
    model_chieff(XX=XX, af=af, model_number=model_number, npoly=npoly,
                 nargout = 3)
#[chi_eff, gvec, info] = \
#    model_profile(XX=XX, af=af, model_number=model_number, npoly=npoly,
#                  nargout = 3)
#
# [chi_eff, gvec, info] = model_parabolic(XX, af)
# [chi_eff, gvec, info] = model_flattop(XX, af)
# [chi_eff, gvec, info] = model_massberg(XX, af)
#info.dchidx = info.dprofdx

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