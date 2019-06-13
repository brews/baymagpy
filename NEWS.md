# v0.0.1a4

## Enhancements

* Update model design and parameter draws to match latest (as of May 2019) upstream development (Issue #8).
* Clean up public API and docstrings.

## Bug fixes


# v0.0.1a3

## Bug fixes

* Clean up docstring and argument logic for `predict_mgca`. There are changes to the arg order for `predict_mgca`. It is now `seatemp`, `cleaning`, `spp`, `seasonal_seatemp`, `omega`, `latlon`, `depth`, `sw_age`, and `drawsfun`. The `latlon` and `depth` args are now optional when `omega` is given. `spp` is now optional as well, defaulting to None. (Issue #6)


# v0.0.1a2

## Enhancements

* Add Mg/Ca seawater correction to `predict_mgca` with `sw_age` arg for Deep Time prediction (Issue #3).

* Replace internal `carbion` code with port of `fetch_omega` from upstream MATLAB (Issue #4). 
This reduces package size and drops some heavy dependencies.

* Updated model parameters and removed pH term from `predict_mgca` (Issue #5).

* Vectorize `predict_mgca`, should be faster.


# v0.0.1a1

## Enhancements

* Add optional arguments to specify pH and omega for `predict_mgca()`.

* `baymag.omega.carbion()` and `predict_mgca` now have optional distance threshold arg.

* Default distance to raise `DistanceThresholdError` when finding nearest gridpoints is now 20000 km.

* Updated MCMC model parameters. Still very experimental.

## Bug fixes

* Clarify cleaning protocol arguments in `predict_mgca()` (Issue #2).

* Estimate pH from sea surface instead of sample depth (Issue #1).


# v0.0.1a0

* Initial release.