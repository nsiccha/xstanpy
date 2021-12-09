import os
import time
import pathlib
import subprocess
import json
import hashlib
import numpy as np
import numpy.linalg as la
import pandas as pd
import arviz as az
from cached_property import cached_property as cproperty
from xstanpy import psis

class cached_command:
    def __init__(self, func, getter=None):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func
        if getter is not None: self.get_output = getter

    @classmethod
    def with_getter(cls, getter): return lambda func: cls(func, getter)

    def get_command(self, obj):
        cname = self.func.__name__ + '_command'
        if cname not in obj.__dict__:
            obj.__dict__[cname] = self.func(obj)
        return obj.__dict__[cname]

    def get_output(self, cmd): return cmd.output

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.get_output(
            self.get_command(obj)
        )
        return value

def logit(val): return np.log(val/(1-val))
def inv_logit(val): return np.exp(val) / (1 + np.exp(val))
def read_csv(path, *args, **kwargs): return pd.read_csv(*args, **kwargs, comment='#')
def log_sum_exp(*args): return psis.sumlogs(*args)

def Configuration(config_map):
    @cproperty
    def configuration(self):
        cmap = config_map
        if isinstance(cmap, str):
            cmap = getattr(self, cmap)
        if not isinstance(cmap, dict):
            cmap = dict(zip(cmap, cmap))
        rv = dict()
        for key, name_or_value in cmap.items():
            if isinstance(name_or_value, str):
                if not hasattr(self, name_or_value): continue
                name_or_value = getattr(self, name_or_value)
            rv[key] = name_or_value
        return rv
    return configuration

class Object:
    arg_names = tuple()
    config_names = tuple()
    info_names = tuple()
    configuration = Configuration('config_names')
    information = Configuration('info_names')

    def post_init(self): pass

    def __init__(self, *args, **kwargs):
        kwargs.update(dict(zip(self.arg_names, args)))
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.init_names = tuple(kwargs.keys())
        self.post_init()

    def __str__(self):
        args = ', '.join([
            f'{arg_name}={getattr(self, arg_name)}'
            for arg_name in self.init_names
        ])
        return f'{self.__class__.__name__}({args})'

    def __repr__(self): return str(self)

    @cproperty
    def init_kwargs(self):
        return {
            name: getattr(self, name) for name in self.init_names
        }

    @cproperty
    def hash(self):
        return Data(dict(self.init_kwargs, __class__=self.__class__.__name__)).hash

    def shallow_copy(self, **kwargs):
        return self.__class__(**dict(self.init_kwargs, **kwargs))

class Command(Object):
    arg_names = ('cmd', )
    cwd = '.'
    execute = True

    def post_init(self):
        if self.execute: return self.process
    @cproperty
    def process(self):
        self.__start = time.perf_counter()
        return subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    @cproperty
    def communication(self):
        rv = self.process.communicate()
        self.__end = time.perf_counter()
        return rv
    @cproperty
    def stdout(self): return self.communication[0]
    @cproperty
    def stderr(self): return self.communication[1]
    @cproperty
    def status(self):
        self.communication
        return self.process.returncode == 0

    def debug(self):
        print(' '.join(map(str, self.cmd)))
        print(self.stdout)
        if self.stderr:
            print("ERRORS:")
            print(self.stderr)
        else:
            print("No errors.")

    @cproperty
    def wall_time(self):
        assert(self.status)
        return self.__end - self.__start


class Model(Object):
    arg_names = ('stan_path', )
    exe_suffix = ''

    def post_init(self):
        if isinstance(self.stan_path, str):
            self.stan_path = pathlib.Path(self.stan_path)

    @cproperty
    def stanc_path(self):
        return pathlib.Path(os.environ['CMDSTAN']) / 'bin' / 'stanc'

    @cproperty
    def name(self): return self.stan_path.stem
    @cproperty
    def exe_path(self): return self.stan_path.with_suffix(self.exe_suffix)
    @cproperty
    def exe_info(self):
        return dict(map(
            lambda line: map(str.strip, line.split('=')),
            self.execute('info').stdout.splitlines()
        ))

    @cproperty
    def compilation_process(self):
        return Command(
            ['make', self.exe_path.resolve()],
            cwd=os.environ['CMDSTAN']
        )

    def execute(self, *args, **kwargs):
        assert(self.compilation_process.status)
        cmd = [self.exe_path]
        cmd.extend(map(str, args))
        cmd.extend(['='.join(map(str, item)) for item in kwargs.items()])
        return Command(cmd)

    @cproperty
    def src_info(self):
        return json.loads(
            Command([self.stanc_path, '--info', self.stan_path]).stdout
        )
    @cproperty
    def data_names(self): return tuple(self.src_info['inputs'])

    @cproperty
    def parameter_names(self): return tuple(self.src_info['parameters'])


class Data(Object):
    arg_names = ('data', )
    @cproperty
    def json_representation(self):
        if isinstance(self.data, dict):
            inner = ', '.join([
                f'"{key}": {Data(self.data[key]).json_representation}'
                for key in sorted(self.data)
            ])
            return f'{{{inner}}}'
        if np.shape(self.data):
            inner = ', '.join([
                Data(cell).json_representation for cell in self.data
            ])
            return f'[{inner}]'
        return {True: '1', False: '0'}.get(self.data, str(self.data))

    @cproperty
    def hash(self):
        return hashlib.md5(self.json_representation.encode('utf-8')).hexdigest()

    def dump(self, path, lazy=True):
        if lazy and path.exists(): return
        with open(path, 'w') as fd:
            fd.write(self.json_representation)


class Distribution(Object):
    pass


class Posterior(Distribution):
    arg_names = ('model', 'data')
    @cproperty
    def Data(self): return Data(self.data)
    @cproperty
    def path(self): return pathlib.Path('out') / self.model.name / self.Data.hash

    @cproperty
    def data_path(self):
        rv = self.path / 'data.json'
        self.Data.dump(rv)
        return rv

    @cproperty
    def constrained_parameter_names(self):
        return SingleCompute(self, constrained_parameters=True).columns

    @cproperty
    def unconstrained_parameter_names(self):
        return SingleCompute(self, unconstrained_parameters=True).columns

    @cproperty
    def transformed_parameters_names(self):
        return SingleCompute(self, transformed_parameters=True).columns

    @cproperty
    def generated_quantities_names(self):
        return SingleCompute(self, generated_quantities=True).columns

    @cproperty
    def no_constrained_parameters(self):
        return len(self.constrained_parameter_names)

    @cproperty
    def no_unconstrained_parameters(self):
        return len(self.unconstrained_parameter_names)

    @cproperty
    def column_info(self):
        rv = dict()
        xnames = {
            'constrained': self.constrained_parameter_names,
            'transformed': self.transformed_parameters_names,
            'generated': self.generated_quantities_names
        }
        for xkey, names in xnames.items():
            for i, name in enumerate(names):
                sname = name.split('.')
                next_sname = [''] if i+1 == len(names) else names[i+1].split('.')
                if sname[0] == next_sname[0]: continue
                shape = tuple(map(int, sname[1:]))
                size = np.prod(shape, dtype=int)
                rv[sname[0]] = (xkey, slice(i+1-size,i+1), shape)
        return rv

    def updated(self, update):
        return self.shallow_copy(data=dict(self.data, **update))

    @cproperty
    def estimated_cost(self):
        rv = 1
        if hasattr(self.model, 'slice_variables'):
            rv *= np.prod([
                self.data[slice_variable]
                for slice_variable in self.model.slice_variables
            ])
        if hasattr(self.model, 'refinement_variable'):
            rv *= self.data[self.model.refinement_variable]
        return 1+rv

    @cproperty
    def integer_data(self):
        return {
            key: value for key, value in self.data.items()
            if isinstance(value, int)
        }

    @cproperty
    def reference_hmc(self):
        return HMC.stan_regular(
            self,
            warmup_no_draws=10000,
            sampling_no_draws=10000,
            thin=10
        )

    @cproperty
    def reference_samples(self): return self.reference_hmc.samples

class PosteriorDraws(Object):
    arg_names = ('posterior', )

    @cached_command
    def unconstrained(self):
        return SingleCompute(
            self.posterior,
            input=self.constrained,
            unconstrained_parameters=True
        )

    def compute_command(self, **kwargs):
        return SingleCompute(
            self.posterior,
            input=self.unconstrained,
            input_unconstrained=True,
            **kwargs
        )

    @cached_command
    def constrained(self):
        return self.compute_command(constrained_parameters=True)

    @cached_command
    def transformed(self):
        return self.compute_command(transformed_parameters=True)

    @cached_command
    def generated(self):
        return self.compute_command(generated_quantities=True)

    @cached_command.with_getter(lambda cmd: cmd.output[:,0])
    def constrained_lp(self):
        return self.compute_command(constrained_log_probability=True)

    @cached_command
    def constrained_lpg(self):
        return self.compute_command(constrained_log_probability_gradient=True)

    @cached_command.with_getter(lambda cmd: cmd.output[:,0])
    def unconstrained_lp(self):
        return self.compute_command(unconstrained_log_probability=True)

    @cached_command
    def unconstrained_lpg(self):
        return self.compute_command(unconstrained_log_probability_gradient=True)

    @cproperty
    def ess(self): return az.ess(self.constrained_lp)

    @cproperty
    def potential_scale_reduction_factor(self): return az.rhat(self.constrained_lp)

    def distance_from(self, posterior):
        from scipy.stats import wasserstein_distance
        ref = posterior.reference_samples
        return np.sum([
            wasserstein_distance(ref_row, own_row)
            for ref_row, own_row in zip(ref.constrained.array.T, self.constrained.T)
        ])

    @cproperty
    def distance_from_reference(self): return self.distance_from(self.posterior)


    sliceable_attributes = [
        'unconstrained', 'constrained', 'transformed', 'generated',
        'constrained_lp', 'constrained_lpg',
        'unconstrained_lp', 'unconstrained_lpg'
    ]

    def __getitem__(self, key):
        return self.__class__(
            self.posterior,
            **{
                name: getattr(self, name)[key]
                for name in self.sliceable_attributes
                if name in self.__dict__
            }
        )

    def __getattr__(self, key):
        if key[0] == '_': raise AttributeError(self, key)
        if hasattr(self.__class__, key): raise AttributeError(self, key)
        try:
            info = self.posterior.column_info.get(key, None)
            if info is None: raise AttributeError(self, key)
            xname, xslice, xshape = info
            val = np.transpose(
                np.reshape(
                    getattr(self, xname)[:, xslice],
                    (-1,) + tuple(reversed(xshape))
                ),
                (0, ) + tuple(reversed(1+np.arange(len(xshape))))
            )
            rv = self.__dict__[key] = val
            return rv
        except AttributeError as ex:
            raise ex

class ApproximateDraws(PosteriorDraws):
    arg_names = ('posterior', 'proposal')
    @cproperty
    def unconstrained(self): return self.proposal.unconstrained

    @cproperty
    def unconstrained_elbo(self):
        return (
            np.mean(self.unconstrained_lp)
            - np.mean(self.proposal.unconstrained_lp)
        )

class SingleCommand(Object):
    arg_names = ('posterior', )

    cmdstan_to_xstan = dict(
        method=['method', dict(
            sample=dict(
                num_samples='sampling_no_draws',
                num_warmup='warmup_no_draws',
                save_warmup='warmup_save',
                thin='thin',
                adapt=dict(
                    engaged='warmup_enabled',
                    gamma='adaptation_regularization_scale',
                    delta='adaptation_target_acceptance_statistic',
                    kappa='adaptation_relaxation_exponent',
                    t0='adaptation_iteration_offset',
                    init_buffer='init_buffer',
                    window='metric_window',
                    term_buffer='term_buffer'
                ),
                algorithm=['algorithm', dict(
                    hmc=dict(
                        engine=['engine', dict(
                            static=dict(int_time='integration_time'),
                            nuts=dict(max_depth='max_treedepth'),
                        )],
                        metric='metric_type',
                        metric_file='metric_path',
                        stepsize='step_size',
                    ),
                    fixed_param=dict()
                )],
                num_chains='no_chains',
            ),
            optimize=dict(
                algorithm=['algorithm', dict(
                    bfgs=dict(),
                    lbfgs=dict(
                        history_size='history_size',
                    ),
                    newton=dict()
                )],
                iter='max_no_iterations',
                save_iterations='save_iterations',
            ),
            variational=dict(
                algorithm='algorithm',
                iter='max_no_iterations',
                output_samples='no_draws',
            ),
            diagnose=dict(
                test=['test', dict(
                    gradient=dict(epsilon='epsilon', error='error')
                )]
            ),
            generate_quantities=dict(
                fitted_params='input_path'
            ),
            compute=dict(
                input_path='binary_input_path',
                input_unconstrained='input_unconstrained',
                output_path='binary_output_path',
                unconstrained_parameters='unconstrained_parameters',
                constrained_parameters='constrained_parameters',
                transformed_parameters='transformed_parameters',
                generated_quantities='generated_quantities',
                constrained_log_probability='constrained_log_probability',
                constrained_log_probability_gradient='constrained_log_probability_gradient',
                unconstrained_log_probability='unconstrained_log_probability',
                unconstrained_log_probability_gradient='unconstrained_log_probability_gradient',
            )
        )],
        id='chain_idx',
        data=dict(file='data_path'),
        init='init_arg',
        random=dict(seed='seed'),
        output=dict(
            file='output_path',
            diagnostic_file='diagnostic_path',
            sig_figs='sig_figs',
        ),
        num_threads='num_threads'
    )
    @cproperty
    def no_draws(self): return self.warmup_no_draws + self.sampling_no_draws
    @cproperty
    def path(self):
        return self.posterior.path / self.method / Data(self.init_kwargs).hash

    chain_idx = 1
    @cproperty
    def data_path(self): return self.posterior.data_path
    @cproperty
    def init_path(self): return self.path / 'init.json'
    init = 2
    @cproperty
    def init_arg(self):
        rv = self.init
        if np.shape(rv):#, (dict, pd.Series)):
            return self.init_path
        return rv
    seed = 0
    @cproperty
    def stdout_path(self): return self.path / 'stdout'
    @cproperty
    def stderr_path(self): return self.path / 'stderr'
    @cproperty
    def output_path(self): return self.path / 'output.csv'
    @cproperty
    def diagnostic_path(self): return self.path / 'diagnostic.csv'
    sig_figs = 18
    num_threads = 1

    def build_process_args(self, recipe):
        rv = []
        for cmdstan_name, xstan_arg in recipe.items():
            xstan_name = xstan_value = subrecipe_mapping = subrecipe = None
            if isinstance(xstan_arg, str):
                xstan_name = xstan_arg
            elif isinstance(xstan_arg, dict):
                subrecipe = xstan_arg
            else:
                xstan_name, subrecipe_mapping = xstan_arg
            if xstan_name is not None:
                xstan_value = getattr(self, xstan_name, None)
            if xstan_value is not None:
                cmdstan_value = {
                    False: 0,
                    True: 1,
                }.get(xstan_value, xstan_value)
                rv.append(f'{cmdstan_name}={cmdstan_value}')

            if subrecipe_mapping is not None:
                rv.extend(self.build_process_args(subrecipe_mapping[cmdstan_value]))

            if subrecipe is not None:
                rv.append(cmdstan_name)
                rv.extend(self.build_process_args(subrecipe))
        return tuple(rv)

    @cproperty
    def process_args(self):
        return self.build_process_args(self.cmdstan_to_xstan)

    def prepare_input(self):
        self.path.mkdir(parents=True, exist_ok=True)

    @cproperty
    def process(self):
        self.prepare_input()
        return self.posterior.model.execute(*self.process_args)

    def dump(self, lazy=True):
        if not self.process.status:
            self.process.debug()
        assert(self.process.status)
        with open(self.stdout_path, 'w') as fd:
            fd.write(self.process.stdout)
        with open(self.stderr_path, 'w') as fd:
            fd.write(self.process.stderr)

    @cproperty
    def stdout(self):
        self.dump()
        with open(self.stdout_path, 'r') as fd:
            return fd.read()

    @cproperty
    def output_df(self):
        self.dump()
        return pd.read_csv(self.output_path, comment='#')

    @cproperty
    def columns(self): return self.output_df.columns

    @cproperty
    def output(self): return self.output_df.to_numpy()

    @cproperty
    def draws_column_slice(self):
        return slice(
            self.draws_column_offset,
            self.draws_column_offset+self.posterior.no_constrained_parameters
        )
    @cproperty
    def draws(self):
        return PosteriorDraws(
            self.posterior,
            constrained=self.output[:, self.draws_column_slice],
            constrained_lp=self.output[:, 0]
        )

    @cproperty
    def diagnostic_df(self):
        self.dump()
        return pd.read_csv(self.diagnostic_path, comment='#')

class SingleHMC(SingleCommand):
    method = 'sample'
    warmup_save = True
    @cproperty
    def warmup_no_draws(self):
        return self.init_buffer + self.metric_window + self.term_buffer
    thin = 1
    @cproperty
    def warmup_enabled(self): return self.warmup_no_draws > 0
    adaptation_regularization_scale = .05
    adaptation_target_acceptance_statistic = .8
    adaptation_relaxation_exponent = .75
    adaptation_iteration_offset = 10.
    algorithm = 'hmc'
    engine = 'nuts'
    max_treedepth = 10
    metric = None
    @cproperty
    def metric_type(self):
        metric = self.metric
        if metric is None: return 'dense_e'
        return {1: 'diag_e', 2: 'dense_e'}[len(np.shape(metric))]
    @cproperty
    def metric_path(self):
        return None if self.metric is None else (self.path / 'metric.json')
    step_size = 1
    no_chains = 1

    def prepare_input(self):
        super().prepare_input()

        init = self.init
        if np.shape(init) and not isinstance(init, (pd.Series, dict)):
            init = pd.Series(index=self.posterior.constrained_parameter_names, data=init)
        if isinstance(init, pd.Series):
            init = {
                name: init.filter(regex=f'^{name}(\.|$)')
                for name in self.posterior.model.parameter_names
            }
        if isinstance(init, dict):
            Data(init).dump(self.init_path)
        metric = self.metric
        if metric is not None:
            Data(dict(inv_metric=metric)).dump(self.metric_path)

    draws_column_offset = 7
    @cproperty
    def sampling_no_thinned_draws(self): return self.sampling_no_draws // self.thin
    @cproperty
    def samples(self): return self.draws[-self.sampling_no_thinned_draws:]
    @cproperty
    def hmc_df(self): return self.output_df.iloc[:, :self.draws_column_offset]
    @cproperty
    def sampling_hmc_df(self):
        return self.hmc_df.iloc[-self.sampling_no_thinned_draws:]

    @cproperty
    def window_sizes(self):
        rv = []
        if self.init_buffer: rv.append(self.init_buffer)
        if self.metric_window:
            remaining = self.warmup_no_draws - self.init_buffer - self.term_buffer
            window = self.metric_window
            assert(remaining >= window)
            while remaining > 0:
                if 2 * window > remaining: window = remaining
                rv.append(window)
                remaining -= window
                window = 2 * window
        if self.term_buffer: rv.append(self.term_buffer)
        if self.sampling_no_draws: rv.append(self.sampling_no_thinned_draws)
        return tuple(rv)

    @cproperty
    def window_slices(self):
        rv = []
        start = 0
        for size in self.window_sizes:
            rv.append(slice(start, start+size))
            start += size
        return tuple(rv)

    @cproperty
    def windowed_draws(self):
        return DrawsPool([
            PosteriorDraws(
                self.posterior,
                constrained=self.draws.constrained[window_slice]
            )
            for window_slice in self.window_slices
        ])

    @cproperty
    def last_window_draws(self): return self.windowed_draws[-1]

    @cproperty
    def metric_window_draws(self):
        return self.windowed_draws[
            -1
            - int(self.sampling_no_draws > 0)
            - int(self.term_buffer > 0)
        ]

    @cproperty
    def last_step_size(self): return self.hmc_df.stepsize__.iloc[-1]
    @cproperty
    def dual_averaged_step_size(self):
        if self.sampling_no_draws: return self.hmc_df.stepsize__.iloc[-1]
        # https://github.com/stan-dev/stan/blob/fc3fe7970d264818e3e948109d9c24f7abea5655/src/stan/mcmc/stepsize_adaptation.hpp#L60-L68
        df = self.hmc_df.iloc[-self.window_sizes[-1]:]
        no_draws = len(df.index)
        x = np.log(df.stepsize__.to_numpy())
        x_eta = (1+np.arange(no_draws)) ** -self.adaptation_relaxation_exponent
        x_bar = 0
        for i in range(no_draws):
            x_bar = (1 - x_eta[i]) * x_bar + x_eta[i] * x[i]
        return np.exp(x_bar)


    @cproperty
    def adaptation_metric(self): return np.atleast_2d(np.cov(self.draws.unconstrained.T))

    @cproperty
    def adaptation(self):
        return dict(
            step_size=self.dual_averaged_step_size,
            metric=self.adaptation_metric,
            init=self.draws.constrained[-1]
        )

    @cproperty
    def estimated_cost(self):
        return self.posterior.estimated_cost * self.hmc_df.n_leapfrog__.sum()

class SingleOptimize(SingleCommand):
    method = 'optimize'
    algorithm = 'lbfgs'
    history_size = 5
    max_no_iterations = 2000
    save_iterations = True
    unconstrained = False

    @cproperty
    def no_iterations(self):
        return len(self.output_df.index)

    draws_column_offset = 1

class SingleVariational(SingleCommand):
    method = 'variational'
    algorithm = 'meanfield'
    max_no_iterations = 10000
    no_draws = 1000


class SingleCompute(SingleCommand):
    method = 'compute'

    @cproperty
    def binary_input_path(self):
        return (self.path / 'input') if hasattr(self, 'input') else ''

    @cproperty
    def binary_output_path(self): return self.path / 'output'

    def prepare_input(self):
        super().prepare_input()
        if hasattr(self, 'input'):
            self.input.tofile(self.binary_input_path)

    @cproperty
    def output_df(self):
        self.dump()
        df = pd.read_csv(self.output_path, comment='#')
        if not hasattr(self, 'input'): return df
        data = np.fromfile(self.binary_output_path)
        no_columns = len(df.columns)
        return pd.DataFrame(
            data=np.reshape(data, (-1, no_columns)),
            columns=df.columns
        )

class Pool(Object):
    arg_names = ('elements', )
    def __len__(self): return len(self.elements)

    def __getattr__(self, key):
        if key[0] == '_': raise AttributeError(self, key)
        if hasattr(self.__class__, key): raise AttributeError(self, key)
        if len(self) == 0: return Pool([])

        try:
            cls_value = getattr(self.elements[0].__class__, key, None)
            if isinstance(cls_value, cached_command):
                commands = CommandPool(cls_value.get_command, [
                    dict(obj=element) for element in self.elements
                    if key not in element.__dict__
                ])
                assert(commands.status)
            return Pool([
                getattr(element, key) for element in self.elements
            ])
        except AttributeError as ex:
            raise ex

    def __call__(self, *args, **kwargs):
        return Pool([
            element(*args, **kwargs) for element in self.elements
        ])

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = key,
        if len(key) > 1:
            if isinstance(key[0], int):
                return self[key[0]][key[1:]]
            return Pool([
                element[key[1:]] for element in self.elements[key[0]]
            ])
        if isinstance(key[0], int):
            return self.elements[key[0]]

        return Pool([
            element for element in self.elements[key[0]]
        ])

    def __array__(self, dtype=None): return self.tensor

    @cproperty
    def tensor(self): return np.array(self.elements)

    @cproperty
    def array(self): return np.concatenate(self.elements)

    @cproperty
    def df(self): return pd.concat(self.elements)

class CommandPool(Pool):
    arg_names = ('command', 'configurations')
    @cproperty
    def elements(self):
        self.__start = time.perf_counter()
        return tuple([
            self.command(**configuration)
            for configuration in self.configurations
        ])

    @cproperty
    def processes(self): return self.process

    @cproperty
    def status(self):
        rv = all(self.processes.status)
        self.__end = time.perf_counter()
        return rv

    @cproperty
    def wall_time(self):
        assert(self.status)
        return self.__end - self.__start

class ImportanceSampling(Object):
    arg_names = ('log_weights', )
    @cproperty
    def no_samples(self): return len(self.log_weights)
    @cproperty
    def effective_sample_size(self):
        return np.exp(
            2*log_sum_exp(self.log_weights) - log_sum_exp(2*self.log_weights)
        )
    @cproperty
    def relative_efficiency(self):
        return  self.effective_sample_size / self.no_samples



class ParetoSmoothedImportanceSampling(ImportanceSampling):
    arg_names = ('raw_log_weights', )

    @cproperty
    def raw_psis(self): return psis.psislw(self.raw_log_weights)
    @cproperty
    def log_weights(self): return self.raw_psis[0]
    @cproperty
    def pareto_shape_estimate(self): return self.raw_psis[1]

class DrawsPool(Pool):
    @cproperty
    def posterior(self): return self[0].posterior
    @cproperty
    def ess(self): return az.ess(self.constrained_lp.tensor)
    @cproperty
    def potential_scale_reduction_factor(self): return az.rhat(self.constrained_lp.tensor)
    def distance_from(self, posterior):
        return PosteriorDraws(
            self.posterior, constrained=self.constrained.array
        ).distance_from(posterior)
    @cproperty
    def distance_from_reference(self): return self.distance_from(self.posterior)
    @cproperty
    def metric(self): return np.atleast_2d(np.cov(self.unconstrained.array.T))
    def psis(self, target_posterior):
        if isinstance(target_posterior, dict):
            target_posterior = self.posterior.updated(target_posterior)
        target_draws = self.__class__([
            PosteriorDraws(
                posterior=target_posterior,
                unconstrained=unconstrained
            )
            for unconstrained in self.unconstrained
        ])
        return ParetoSmoothedImportanceSampling(
            target_draws.constrained_lp.array - self.constrained_lp.array
        )

class HMC(Object):
    arg_names = ('posterior', )
    config_names = (
        'init_buffer', 'metric_window', 'term_buffer', 'warmup_no_draws',
        'sampling_no_draws', 'thin',
        'step_size', 'metric', 'init'
    )
    info_names = (
        'hmc_wall_time', 'ess', 'sampling_no_divergences', 'potential_scale_reduction_factor',
    )
    command = SingleHMC
    first_idx = 1
    no_chains = 6
    @cproperty
    def chain_idxs(self): return self.first_idx + np.arange(self.no_chains)

    @cproperty
    def configurations(self):
        rv = tuple([
            dict(
                self.configuration,
                posterior=self.posterior,
                chain_idx=chain_idx
            )
            for chain_idx in self.chain_idxs
        ])
        for i, rvi in enumerate(rv):
            if 'init' in rvi:
                rvi['init'] = rvi['init'][i]
        return rv

    @cproperty
    def raw_commands(self): return CommandPool(self.command, self.configurations)

    @cproperty
    def commands(self):
        assert(self.raw_commands.status)
        return self.raw_commands

    @cproperty
    def draws(self): return DrawsPool(self.commands.draws)

    @cproperty
    def samples(self): return DrawsPool(self.commands.samples)

    @cproperty
    def output_df(self): return self.commands.output_df.df

    @cproperty
    def hmc_df(self): return self.commands.hmc_df.df

    @cproperty
    def sampling_hmc_df(self): return self.commands.sampling_hmc_df.df

    @cproperty
    def pooled_step_size(self):
        return max(self.commands.dual_averaged_step_size)

    @cproperty
    def metric_window_draws(self):
        return DrawsPool(self.commands.metric_window_draws)

    @cproperty
    def pooled_metric(self):
        if not self.commands[0].metric_window: return self.commands[0].metric
        return self.metric_window_draws.metric

    def adaptation_for(self, new_posterior):
        if new_posterior is self.posterior:
            pooled_init = self.draws.constrained[:,-1].tensor
        else:
            new_draws = Pool([
                draws.shallow_copy(posterior=new_posterior) for draws in self.draws
            ])
            lw = self.draws.constrained_lp.array - new_draws.constrained_lp.array
            psislw, k = psis.psislw(lw)
            weights = np.exp(psislw - log_sum_exp(psislw))
            idxs = np.random.choice(
                len(lw),
                size=self.no_chains,
                replace=True,
                p=weights
            )
            pooled_init = self.draws.constrained.array[idxs]
        return dict(
            step_size=self.pooled_step_size,
            metric=self.pooled_metric,
            init=pooled_init
        )

    @cproperty
    def adaptation(self): return self.adaptation_for(self.posterior)

    @cproperty
    def estimated_cost(self): return sum(self.commands.estimated_cost)

    @cproperty
    def status(self): return self.commands.status

    @cproperty
    def hmc_wall_time(self): return self.commands.wall_time

    @cproperty
    def ess(self): return self.samples.ess

    @cproperty
    def sampling_no_divergences(self): return self.sampling_hmc_df.divergent__.sum()

    @cproperty
    def potential_scale_reduction_factor(self): return self.samples.potential_scale_reduction_factor

    def distance_from(self, posterior): return self.samples.distance_from(posterior)

    @cproperty
    def distance_from_reference(self): return self.samples.distance_from_reference

    @classmethod
    def stan_regular(cls, posterior, **kwargs):
        return cls(posterior, **dict(
            dict(
                init_buffer=75,
                metric_window=25,
                term_buffer=50,
                warmup_no_draws=1000,
                sampling_no_draws=1000,
            ),
            **kwargs
        ))
