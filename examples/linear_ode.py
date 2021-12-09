from xstanpy.incremental import *
from xstanpy.plot import *

os.environ['CMDSTAN'] = '../cmdstan'

model = Model(
    'stan/linear_ode.stan',
    slice_variables={'no_observations': ('x_observed', 'y_observed')},
    refinement_variable='no_steps'
)
model.compilation_process.debug()

def random_flow_matrix(no_dimensions):
    rv = np.random.lognormal(size=(no_dimensions, no_dimensions))
    rv -= np.diag(np.diag(rv))
    rv -= np.diag(np.sum(rv, axis=0))
    return rv

configs = dict(
    smaller=2,
    small=4,
    medium=8,
)

full_df = pd.DataFrame()
for config_name, no_dimensions in configs.items():
    config = Object(name=config_name)
    np.random.seed(0)
    no_observations = 32
    config_df = pd.DataFrame()
    prior_data = dict(
        no_observations=no_observations,
        no_dimensions=no_dimensions,
        x_observed=np.linspace(0,1/no_dimensions,no_observations+1)[1:],
        ode_matrix=random_flow_matrix(no_dimensions),
        y_observed=np.zeros((no_observations, no_dimensions)),
        no_steps=0,
        likelihood=0
    )
    prior = Posterior(model, prior_data)
    prior_fit = HMC.stan_regular(prior, sampling_no_draws=100)
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
