"""
# 1D Gaussian process

Please see `stan/1d_gp.stan` for model details.

As before, the plots show some diagnostics in a table at the top.
Then, from top to bottom we show

* measurements and underlying states,
* the (scaled) basis function weights and
* a pairplot of the Gaussian process lengthscale and sigma[^1].

Black generally corresponds to the true values, i.e.

* the measurements,
* the latent states,
* the true basis function weights and
* the true values of the Gaussian process length scale and "sigma".

In the second row, the shaded grey area corresponds to 0 +/- SD, where the
standard deviations of the scaled weights depends on the index of the basis
function weight and uses the "actual" Gaussian process lengthscale and sigma.

For the `strongly_informative` configuration, the data are so informative that
the basis function coefficients are very well determined and a centered parametrization
would have been more appropriate. For the `informative` or `weakly_informative`
configuration, this is less clear.

[^1]: forgot the correct name for this.
"""


# Please see `linear_ode.md` for initialization documentation.
#
from xstanpy.incremental import *
from xstanpy.plot import *
np.random.seed(0)

os.environ['CMDSTAN'] = '../cmdstan'


# While the following looks identical to the `linear_ode.md` example,
# under the hood there is one important difference. While for the linear ODE
# example the dimensionality of the posterior was independent of the
# `slice_variables` or the `refinement_variable`, for the Gaussian process model
# the value of `gp_no_basis_functions` influences the dimensionality of the posterior.
# This does not affect Stan's regular warm-up/sampling or the `Incremental` warm-up,
# but it forces us to change the `Adaptive` warm-up.
# **Currently, the below only works because there is some
# [prolongation](https://en.wikipedia.org/wiki/Multigrid_method) happening
# under the hood. For almost any other model, simply specifying a data variable
# which infuences the dimensionality of the posterior as the `refinement_variable`
# will not work!**
model = Model(
    'stan/1d_gp.stan',
    slice_variables={'no_observations': ('xi_observed', 'y_observed')},
    refinement_variable='gp_no_basis_functions'
)
model.compilation_process.debug()
#~ There are two main obstacles to be able to use the `Adaptive` warm-up to reconfigure
# posteriors for which refinement of the used approximation increases the dimensionality
# of the approximate posterior:
#
# * First, we have to be able to take draws from the lower dimensional posterior
# and "prolongate" them onto the higher dimensional posterior.
# * Second, to be able to do PSIS, we have to figure out what the appropriate
# (log) posterior density of these prolongated draws is.
#
# For the prolongation, there are several options, but two appear to me most plausible
# in the current setting:
#
# * Fill the values of the "extra" basis function weights with zero or
# * fill the values of the "extra" basis function weights with draws from the prior.
#
# If we fill the extra values with zero, then the states of the Gaussian process
# will stay the same after prolongation. If on the other hand we fill the extra
# values with draws from the prior, the states of the Gaussian process will generally
# not stay the same.
#
# While the first option may sound reasonable, it is the wrong choice[^2]
# in this setting because the raw IS-weights passed to PSIS would all be identical.
# For the second option on the other hand we can reinterpret the draws from the lower
# dimensional posterior to stem from a model in the higher dimensional posterior
# where the **basis functions** of the new **basis function weights** are identically zero.
# **The IS-weights are then fully determined by the likelihood.**
#
# Implementation wise, the following happens currently:
#
# * The class `Adaptive` calls the function `Adaptive.data_reconfiguration` for each
# intermediate posterior.
# * There, `DrawsPool.psis` gets called.
# * This passes the prolongation bucket
# via `DrawsPool.draws_for` and `PosteriorDraws.draws_for` to `Model.draws_for`.
# * There, if the posteriors have the same dimensionality, nothing special happens.
# If however the dimensionality of the target posterior is larger than the dimensionality
# of the source posterior, the unconstrained parameter values get padded with draws
# from a standard normal. **This is very non-portable and has to be changed. For
# the current model this works because only the dimensionality of the `unit_weights`
# change, which in fact have a standard normal prior.**
# * Back in `DrawsPool.psis`, the prolongated draws are used to compute the
# log-likelihood difference.
# * The only extra work happens in `HMC.adaptation_for` via `HMC.pooled_metric_for`,
# where the metric from the lower dimensional posterior gets extended by an identity
# matrix in the new part of the diagonal. **This is of course also not portable**.
#
# [^2]: Or maybe not?



# What follows now is equivalent to what happens for the `linear_ode.md` example, with the
# only difference that different configurations have a different number of observations
# instead of a different number of (ODE) states.
configs = dict(
    weakly_informative=10,
    informative=100,
    strongly_informative=1000
)
for config_name, no_observations in configs.items():
    config = Data(dict(no_observations=no_observations), name=config_name)
    x_observed = np.cumsum(np.random.uniform(size=no_observations))
    x_observed = x_observed / x_observed[-1]
    xi_observed = 2 * x_observed - 1

    prior_data = dict(
        no_observations=no_observations,
        xi_observed=xi_observed,
        y_observed=np.zeros(no_observations),
        gp_support_fraction=5/6,
        gp_no_basis_functions=256,
        likelihood=0
    )
    prior = Posterior(model, prior_data)
    prior_fit = HMC.stan_regular(prior, warmup_no_draws=100, sampling_no_draws=100)
    #~ Here we want to ensure that something interesting is happening in the data,
    # so we select the draw from the prior with the lowest Gaussian process lengthscale.
    idx = np.argmin(prior_fit.samples.gp_unit_lengthscale.array)
    posterior = prior.updated(dict(
        y_observed=prior_fit.samples.y_generated.array[idx],
        likelihood=1
    ))
    #~ This again looks deceptively similar to what is done for the linear ODE example,
    # **While with the proper preparation the call signature should stay the same,
    # this preparation generally still has to happen in a custom `Model` subclass.**
    adaptive = Adaptive(
        posterior,
        callback=lambda sequence: print(sequence[-1].posterior.integer_data),
    )
    #~ This is again just the same as for the linear ODE example.
    #
    fits = {
        'adaptive': adaptive,
    }
    # To lower the runtime we only look at the following configuration
    for warmup_no_draws in [200]:
        key = f'{warmup_no_draws:04d}'
        fits[f'stan_{key}'] = HMC.stan_regular(
            posterior,
            warmup_no_draws=warmup_no_draws,
            sampling_no_draws=100
        )
    df = pd.DataFrame({
        name: fit.information for name, fit in fits.items()
    }).T.fillna(dict(relative_efficiency=1, pareto_shape_estimate=-np.inf))
    suptitle = df.to_string(float_format='{:.2f}'.format)
    print(suptitle)
    Figure([
        [
            Ax([
                LinePlot(prior_fit.samples.y_generated.array[idx], color='black', marker='x', linestyle='', label='measurements'),
            ] + [
                FillPlot(fit.samples.y_computed.array, alpha=.5, label=key)
                for key, fit in fits.items()
            ] + [
                LinePlot(prior_fit.samples.y_computed.array[idx], color='black', label='true values'),
            ],
                title='values & measurements',
                show_legend=False
            )
        ],
        [
            Ax([
                FillPlot(fit.samples.scaled_weights.array, alpha=.5, label=key)
                for key, fit in fits.items()
            ] + [
                FillPlot([
                    -prior_fit.samples.gp_sigma.array[idx],
                    prior_fit.samples.gp_sigma.array[idx]
                ], alpha=.25, color='black', label='|z|<=1'),
                LinePlot(prior_fit.samples.scaled_weights.array[idx], color='black', label='true values'),
            ],
                title='scaled basis function weights',
                xlabel='basis function index',
                ylabel='scaled weight',
                xlim=[0, 2*adaptive.sequence[-1].posterior.data['gp_no_basis_functions']],
                show_legend=False
            )
        ],
        [
            Ax([
                ScatterPlot(
                    fit.samples.gp_unit_lengthscale.array,
                    fit.samples.gp_unit_sigma.array,
                    label=key
                )
                for key, fit in fits.items()
            ] + [
                VerticalLine(
                    prior_fit.samples.gp_unit_lengthscale.array[idx], color='black'
                ),
                HorizontalLine(
                    prior_fit.samples.gp_unit_sigma.array[idx], color='black'
                )
            ],
                xlabel='gp_unit_lengthscale', ylabel='gp_unit_sigma',
                xscale='log', yscale='log')
        ]
    ],
        col_width=16,
        suptitle=suptitle,
        suptitle_family='monospace',
        show_legend='row'
    ).save(
        f'figs/{model.name}/{config.name}/{posterior.hash}.png'
    )
# print(stan.information)
