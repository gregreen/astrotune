"""Post-hoc learned uncertainties for the harmonized stellar parameters.

After the bijector is trained, the harmonized estimates of a star from different
catalogs should agree to within their (propagated) uncertainties. They do not
quite: for some catalogs the quoted errors are too small, for others too large, and
the residuals have heavy tails. This package

* builds a table of every observation of every star (:mod:`.stars`),
* measures how well calibrated the quoted errors are (:mod:`.diagnostics`),
* learns an additional, diagonal covariance ``exp(ln_delta_cov)`` per catalog entry
  as a function of the entry's own catalog parameters, reported uncertainties and
  catalog (:mod:`.model`, :mod:`.train`).

The statistical frame is the one the latent-``y`` training uses: a star has one
unknown harmonized value, every catalog entry is a measurement of it with its own
covariance, and the star's value is integrated out. One likelihood term per star,
not per pair.

``stars`` and ``diagnostics`` need only NumPy/SciPy/Matplotlib; ``model`` and
``train`` need TensorFlow and Sonnet.
"""
