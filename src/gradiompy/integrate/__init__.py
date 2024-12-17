"""
===================================
Integration (`gradiompy.integrate`)
===================================

Integrating functions, given fixed samples
==========================================

   composite
   trapezoid
   simpson
   cumulative_composite
   cumulative_trapezoid
   cumulative_simpson
   
"""
from ._quadrature import (
    composite, cumulative_composite, trapezoid, cumulative_trapezoid, simpson, cumulative_simpson)

__all__ = [
    'composite', 'cumulative_composite', 'trapezoid', 'cumulative_trapezoid', 'cumulative_trapezoid', 'simpson', 'cumulative_simpson']