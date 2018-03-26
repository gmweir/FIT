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

__version__ = "2017.05.22.17"
__all__ = ['fitting_dev', 'model_spec']

try:
    __import__('mpfit')
    __mpfitonpath__ = True
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


# ===================================================================== #
# ===================================================================== #