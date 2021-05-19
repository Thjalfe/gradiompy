"""
===================================
Integration (`gradiompy.integrate`)
===================================

Integrating functions, given fixed samples
==========================================

   composite            -- Use composite rule of given order to compute integral.
   cumulative_composite -- Use composite rule of given order to cumulatively compute integral.
   
"""
from ._quadrature import (
    composite, cumulative_composite)

__all__ = [
    'composite', 'cumulative_composite']