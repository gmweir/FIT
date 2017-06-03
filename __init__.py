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

# from . import fitting_dev, model_spec # analysis:ignore
from . import fitting_dev as fitting # analysis:ignore
from . import model_spec as models # analysis:ignore

# ===================================================================== #
# ===================================================================== #