# Needed for adaptive warm-up
from xstanpy.incremental import *
# Needed for plotting
from xstanpy.plot import *

# For proper compilation, $CMDSTAN has to point to our modified cmdstan version.
os.environ['CMDSTAN'] = '../cmdstan'

# A `Model` object takes care of compilation and stores additional information about
# the model, e.g. how to slice its data and how to refine its approximation.
model = Model(
    'stan/linear_ode.stan',
    slice_variables={'no_observations': ('x_observed', 'y_observed')},
    refinement_variable='no_steps'
)
# This raises an error if model compilation fails and prints stdout and stderr.
model.compilation_process.debug()

# To ensure positivity of the ODE states we restrict our random matrices
# to have only non-negative off-diagonal entries and for "conservation of mass"
# we restrict the columns to sum to zero.
def random_flow_matrix(no_dimensions):
    rv = np.random.lognormal(size=(no_dimensions, no_dimensions))
    rv -= np.diag(np.diag(rv))
    rv -= np.diag(np.sum(rv, axis=0))
    return rv

# We look at the following different numbers of dimensions:
configs = dict(
    smaller=2,
    small=4,
    medium=8,
)

# We save the fit information for all fits in this DataFrame
full_df = pd.DataFrame()
for config_name, no_dimensions in configs.items():
    # Allows us to access {config.name}
    config = Object(name=config_name)
    # Set seed to zero for each configuration for reproducibility
    np.random.seed(0)
    no_observations = 32
    # We save the fit information for each configuration in this DataFrame
    config_df = pd.DataFrame()
    # These are just the data that are passed to the prior sampling
    prior_data = dict(
        no_observations=no_observations,
        no_dimensions=no_dimensions,
        x_observed=np.linspace(0,1/no_dimensions,no_observations+1)[1:],
        ode_matrix=random_flow_matrix(no_dimensions),
        y_observed=np.zeros((no_observations, no_dimensions)),
        # no_steps = 0 means always use matrix exponentials, i.e. the "exact" solution
        no_steps=0,
        likelihood=0
    )
    # A `Posterior` object is defined via model+data
    prior = Posterior(model, prior_data)
    # An `HMC` object saves the CmdStan.sample output for several chains (default: 6)
    # `HMC.stan_regular` calls CmdStan.sample with Stan's default arguments
    prior_fit = HMC.stan_regular(prior, sampling_no_draws=100)
    # An `HMC` object's `.samples` handles the "raw" double output for the SAMPLES (excluding WARM-UP)
    prior_samples = prior_fit.samples
    # `prior_samples` is a `DrawsPool` object which manages the `PosteriorDraws`
    # for several chains.
    # `prior_samples.constrained.array` concatenates the chainwise constrained parameter values
    # such that its shape is (no_chains * no_draws x no_parameters)
    # `prior_samples.constrained.tensor` reshapes this array such that
    # such that its shape is (no_chains x no_draws x no_parameters)
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
            'adaptive': adaptive
        }
        for warmup_no_draws in [200, 400, 800, 1000]:
            key = f'stan_{warmup_no_draws:04d}'
            fits[key] = fit = HMC.stan_regular(
                posterior,
                warmup_no_draws=warmup_no_draws,
                sampling_no_draws=100
            )
            print(key, fit.configuration)
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
# {model.name} / {config.name}

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
