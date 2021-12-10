"""
# Linear ODE

A sample visualization for a medium sized configuration (a linear ODE with 8 states)
can be seen here:
![](figs/linear_ode/medium/262322ed0482f5fa5d8f94f1e56b65c2.png)

The table at the top of the figure compares among other things HMC wall times
(my current implementation has additional overhead due to the extra I/O needed e.g. for PSIS).
The plots show measurements, true ODE states and mean posterior ODE-states.

`adaptive` uses PSIS to adaptively refine an approximate solution to the ODE
while `stan_xxxx` just uses the "exact" solution
(obtained via one matrix exponential per time step).
`xxxx` corresponds to the total number of warm-up iterations.
The model itself only has two parameters (the ODE matrix is fixed), a time scale
parameter and a measurement noise parameter.
"""


# Needed for adaptive warm-up
from xstanpy.incremental import *
# Needed for plotting
from xstanpy.plot import *

# For proper compilation, $CMDSTAN has to point to our modified cmdstan version.
os.environ['CMDSTAN'] = '../cmdstan'


# A `Model` object takes care of compilation and stores additional information about
# the model, e.g.
# * how to slice its data for use by `Incremental` objects via their
# `Incremental.slice_update` and `Incremental.data_reconfiguration` methods
# * and how to refine a posterior's approximation via `Adaptive` objects using their
# `Adaptive.refinement_update` and their adapted `Adaptive.data_reconfiguration` methods.
#
# The models `Model.compilation_process` is a `Command` object which handles
# communication with the spawned subprocess. Its `Command.debug` method
# raises an error if the process has a non-zero returncode.
model = Model(
    'stan/linear_ode.stan',
    # The "incremental" part of the warm-up incrementally doubles the data
    # that we condition on by successively doubling the (integer) values
    # of the keys of the `slice_variables` dictionary and
    # slicing along the first axis of the corresponding values.
    slice_variables={'no_observations': ('x_observed', 'y_observed')},
    # The "adaptive"  part of the warm-up doubles the specified `refinement_variable`
    # until the pareto smoothed estimate of the relative efficiency of the importance sampling
    # is above the threshold `relative_efficiency_goal` (default: .5)
    refinement_variable='no_steps'
)
# This raises an error if model compilation fails and prints stdout and stderr.
model.compilation_process.debug()


# Our Stan model `stan/linear_ode.stan` solves a linear ordinary differential equation
#   u'(t) = A u(t)
# with a prespecified matrix A and assumes lognormal measurement,
# implying that our states have to remain positive for all times.
# For our tests, we will generate random matrices.
# To ensure positivity of the ODE states, we restrict our random matrices
# to have only non-negative off-diagonal entries.
# Furthermore, as a simple sufficient condition for the existence of a well-behaved
# stable steady state we restrict the columns to sum to zero, which enforces something
# akin to mass conservation.
def random_flow_matrix(no_dimensions):
    rv = np.random.lognormal(size=(no_dimensions, no_dimensions))
    rv -= np.diag(np.diag(rv))
    rv -= np.diag(np.sum(rv, axis=0))
    return rv


# We are interested in how the two approaches scale with the size of the system[^1].
# For this we look at systems with 2, 4, or 8 states. As we increase the number of states,
# the cost of each matrix exponential grows (cubically?)
# and begins to dominate the overall computational cost,
# such that our adaptive approximate approach quickly overtakes Stan's regular warm-up,
# if the latter uses the "exact" solution approach.
#
# Running tests for larger systems takes a little bit longer, which is why we do not
# do this here.
#
# [^1]: Slightly more honestly, my approach performs worse than Stan's defaults
# for small systems, which if left to stand alone would make for a bad first impression.
# I *think* due to the low-dimensionality of the posterior, Stan's default works well
# and my warm-up introduces quite some overhead. The situation should be different for
# higher-dimensional and geometrically more challenging posteriors.
configs = dict(
    smaller=2,
    small=4,
    medium=8,
    # large=16,
    # larger=32
)
# We'll keep the number of observations constant
no_observations = 16
# For equidistant measurements computing a single matrix exponential and
# then reusing it at every step would be "exact", so we change it up a bit.
# x_observed will be a series of somewhat randomly increasing positive reals
# with the last value equal to one.
x_observed = np.cumsum(np.random.uniform(size=no_observations))
x_observed = x_observed / x_observed[-1]
# We save the fit information for all configurations in this DataFrame
full_df = pd.DataFrame()
# Loop over all configurations
for config_name, no_dimensions in configs.items():
    # Allows us to access {config.name}
    config = Data(dict(no_dimensions=no_dimensions), name=config_name)
    # Set seed to zero for each configuration for reproducibility
    np.random.seed(0)
    # We save the fit information for the current configuration in this DataFrame
    config_df = pd.DataFrame()
    # These are just the data that are passed to the prior sampling
    prior_data = dict(
        no_observations=no_observations,
        no_dimensions=no_dimensions,
        # scale our observation times by the number of ODE states, as larger
        # systems tend to approach equilibrium quicker
        x_observed=x_observed/no_dimensions,
        # One randomly generated matrix per configuration
        ode_matrix=random_flow_matrix(no_dimensions),
        # `y_observed` is irrelevant for prior sampling
        y_observed=np.zeros((no_observations, no_dimensions)),
        # no_steps = 0 means always use matrix exponentials, i.e. the "exact" solution
        no_steps=0,
        # To sample from the prior we have to turn off the likelihood
        likelihood=0
    )
    #~ A `Posterior` object is defined by specifying a `Model` object and
    # a `dict` of data. The `Posterior` object handles metadata such as
    # * `Posterior.constrained_parameter_names`: the names of the constrained parameters,
    # * `Posterior.no_constrained_parameters`: the number of constrained parameters,
    # * `Posterior.no_unconstrained_parameters`: the number of *unconstrained* parameters and
    # * `Posterior.column_info`: parameterwise slice and shape data for easy access of properly reshaped
    # objects
    prior = Posterior(model, prior_data)
    #~ An `HMC` object saves the CmdStan's `sample` method output for several chains (default: 6).
    # Among other thing it handles spawning several subproccess, accessible either via
    # * `HMC.raw_commands`, if one wants more fine grained control, e.g. to specify a timeout, or
    # * `HMC.commands`, which if accessed waits for all processes to finish and raises an error
    # if any of the subprocesses encountered an error.
    # 
    # The `HMC` class provides several other convenience functions, such as
    # * `HMC.samples` to access an object's SAMPLES (excluding WARM-UP draws),
    # * `HMC.draws` to access an object's draws (INCLUDING warm-up draws)
    # * `HMC.stan_regular` calls CmdStan's `sample` method with Stan's default arguments
    prior_fit = HMC.stan_regular(prior, warmup_no_draws=100, sampling_no_draws=100)
    #~ Both an `HMC` object's `HMC.samples` and `HMC.draws` properties
    # return `DrawsPool` objects, which allow for some convenience functionalities.
    # They allow for intuitive slicing across chains and chain properties.
    # This is achieved by forwarding any attribute access which fails
    # directly on the `DrawsPool` object to its subelements via its inherited
    # `Pool.__getattr__` method. In addition, `DrawsPool` objects inherit the following
    # convenience properties from the `Pool` class:
    # * `Pool.array`: returns the `numpy` concatenation of the object's subelements
    # * `Pool.tensor`: returns the result of `Pool.array` reshaped such that its first axis
    # has the same length as the original `Pool` object.
    # In practice, this enables us to e.g. access all the constrained parameter values
    # of the `HMC.samples`
    # * via `prior_samples.constrained.array` as an `no_chains * no_draws x no_constrained_parameters`
    # `numpy` array or
    # * via `prior_samples.constrained.tensor` as an `no_chains x no_draws x no_constrained_parameters`
    # `numpy` array.
    # In addition, the same is possible for any variable defined in the
    # `parameters`, `transformed parameters` or `generated quantities` block.
    # For this model (`stan/linear_ode.stan`) we should be able to access e.g.
    # * `prior_fit.samples.k.array` (`no_chains * no_draws`),
    # * `prior_fit.samples.sigma.tensor` (`no_chains x no_draws`),
    # * `prior_fit.samples.y_computed.array` (`no_chains * no_draws x no_observations x no_dimensions`),
    # * `prior_fit.samples.y_generated.tensor` (`no_chains x no_draws x no_observations x no_dimensions`).
    prior_samples = prior_fit.samples
    for idx in np.random.choice(len(prior_samples.constrained.array), size=2):
        y_true = prior_samples.y_computed.array[idx]
        y_observed = prior_samples.y_generated.array[idx]
        posterior = prior.updated(dict(
            y_observed=y_observed,
            likelihood=1,
            no_steps=0
        ))
        adaptive = Adaptive(
            posterior,
            callback=lambda sequence: print(sequence[-1].posterior.integer_data),
        )

        fits = {
            'adaptive': adaptive,
        }
        for warmup_no_draws in [200, 400, 800, 1000]:
            key = f'{warmup_no_draws:04d}'
            fits[f'stan_{key}'] = HMC.stan_regular(
                posterior,
                warmup_no_draws=warmup_no_draws,
                sampling_no_draws=100
            )

        df = pd.DataFrame({
            name: fit.information for name, fit in fits.items()
        }).T.fillna(dict(relative_efficiency=1, pareto_shape_estimate=-np.inf))

        config_df = pd.concat([config_df, pd.concat({posterior.hash: df}, names=['posterior'])])
        suptitle = df.to_string(float_format='{:.2f}'.format)
        print(suptitle)
        axes = [
            Ax([
                LinePlot(y_true[:, i], color='black', label='true values'),
                LinePlot(y_observed[:, i], marker='x', color='black', label='observations'),
            ] + [
                LinePlot(np.mean(fit.samples.y_computed.array[:,:,i], axis=0), label=key)
                for key, fit in fits.items()
            ], yscale='log', show_legend=False)
            for i in range(no_dimensions)
        ]
        Figure(
            axes,
            suptitle=suptitle,
            suptitle_family='monospace',
            show_legend='row',
            legend_title='mean(y)'
        ).save(
            f'figs/{model.name}/{config.name}/{posterior.hash}.png'
        )
    full_df = pd.concat([full_df, pd.concat({config.name: config_df}, names=['config'])])
    report_path = pathlib.Path(f'figs/{model.name}/{config.name}.md')
    with open(report_path, 'w') as fd:
        fd.write(f"""
# {model.name} / {config.name} ({config.data})

## Aggregated data (mean)

{config_df.groupby(level=1).mean().to_markdown()}

## Separate data

{config_df.to_markdown()}
""")

report_path = pathlib.Path(f'figs/{model.name}.md')
with open(report_path, 'w') as fd:
    fd.write(f"""
# {model.name}

## Aggregated data (mean)

{full_df.groupby(level=[0,2]).mean().to_markdown()}

## Separate data

{full_df.to_markdown()}
""")
