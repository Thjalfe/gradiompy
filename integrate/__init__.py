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
    composite, cumulative_composite, trapezoid, cumulative_trapezoid, simpson, cumulative_simpson)

__all__ = [
    'composite', 'cumulative_composite', 'trapezoid', 'cumulative_trapezoid', 'cumulative_trapezoid', 'simpson', 'cumulative_simpson']