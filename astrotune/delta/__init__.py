"""Post-hoc learned uncertainties for the harmonized stellar parameters.

After the bijector is trained, the harmonized estimates of a star from two
catalogs should agree to within their (propagated) uncertainties. They do not
quite: for some catalog pairs the quoted errors are too small, for others too
large, and the residuals have heavy tails. This package

* builds a table of pairs of harmonized estimates (:mod:`.pairs`),
* measures how well calibrated the quoted errors are (:mod:`.diagnostics`),
* learns an additional, diagonal covariance ``exp(ln_delta_cov)`` per catalog
  entry as a function of the entry's own catalog parameters, reported
  uncertainties and catalog (:mod:`.model`, :mod:`.train`).

``pairs`` and ``diagnostics`` need only NumPy/SciPy/Matplotlib; ``model`` and
``train`` need TensorFlow and Sonnet.
"""
