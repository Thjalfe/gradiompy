# gradiompy
GradiomPy is an open-source software for mathematics, science, and engineering. Subpackages and function names follow SciPy's spirit.

Subpackages
-----------
Using any of these subpackages requires an explicit import. For example, ``import gradiompy.integrate``.

- **integrate** --- Integration routines for regularly sampled functions:
  - `trapezoid` --- Integration based on interpolating polynomials of order 1
  - `simpson` --- Integration based on interpolating polynomials of order 3
  - `composite` --- Integration based on interpolating polynomials of order 1 to 7
  - `cumulative_trapezoid` --- Cumulative version of `trapezoid`
  - `cumulative_simpson` --- Cumulative version of `simpson`
  - `cumulative_composite` --- Cumulative version of `composite`
  
- **interpolate** --- Univariate interpolation functions and classes:
  - `DiscontinuousInterpolator` --- Cubic polynomial interpolator handling discontinuities
