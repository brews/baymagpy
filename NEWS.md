# v0.0.1a2

## Enhancements

* Add Mg/Ca seawater correction to `predict_mgca` with `sw_age` arg for Deep Time prediction (Issue #3)


## Bug fixes


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