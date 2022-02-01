# 1D Gaussian process


As before, the plots show some diagnostics in a table at the top.
Then, from top to bottom we show

* measurements and underlying states,
* the scaled basis function weights,
* the unit basis function weights and
* a pair-plot of the Gaussian process length scale and marginal standard deviation.

Black generally corresponds to the true values, i.e.

* the measurements,
* the latent states,
* the true basis function weights and
* the true values of the Gaussian process length scale and marginal standard deviation.

In the second and third row, the shaded gray area corresponds to 0 +/- SD, where the
standard deviations of the scaled weights depends on the index of the basis
function weight and uses the "actual" Gaussian process length scale and marginal standard deviation.

For the `strongly_informative` configuration, the data are so informative that
the basis function coefficients are very well determined and a centered parametrization
would have been more appropriate. For the `informative` or `weakly_informative`
configuration, this is less clear.
