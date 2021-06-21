"""
=======================================
Interpolation (`gradiompy.interpolate`)
=======================================

Univariate interpolation
========================

   DiscontinuousInterpolator
   
"""
from ._polyint import *
from ._cubic import *

__all__ = [s for s in dir() if not s.startswith('_')]