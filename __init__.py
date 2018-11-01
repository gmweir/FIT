# -*- coding: utf-8 -*-
"""
@author: gawe
"""

# ===================================================================== #
# ===================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

# ========================================================================== #
# ========================================================================== #

__version__ = "2018.11.01.15"
__all__ = ['fitting_dev', 'model_spec', 'derivatives']

try:
    __import__('mpfit')
    __mpfitonpath__ = True
    __all__.append('fitNL')
except:
    __mpfitonpath__ = False
# end try


from . import model_spec as models # analysis:ignore

#if __mpfitonpath__:
#    from . import fitting_mpfit as fitting
#else:
#    from . import fitting_scipy as fitting
## end try

# # from . import fitting_dev, model_spec # analysis:ignore
from . import fitting_dev as fitting # analysis:ignore
from . import derivatives as derivatives # analysis:ignore
from . import fitNL as fitNL

# ===================================================================== #
from .derivatives import findiff1d, interp_profile, deriv_bsgaussian # analysis:ignore
from .fitting_dev import linreg, savitzky_golay, spline, pchip, spline_bs # analysis:ignore

if __mpfitonpath__:
    from .fitNL import fitNL, modelfit, qparabfit # analysis:ignore
    from .fitting_dev import weightedPolyfit, fit_TSneprofile, fit_TSteprofile # analysis:ignore
# end if
# ===================================================================== #
# ===================================================================== #