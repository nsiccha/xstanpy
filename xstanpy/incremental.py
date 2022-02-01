from xstanpy.base import *
from xstanpy import psis

class ChainedHMC(Object):
    arg_names = ('posterior', 'configurations')
    information_names = HMC.information_names
    info_names = HMC.info_names
    initial_adaptation_configuration = Configuration(dict(
        init_buffer='init_buffer',
        metric_window=0,
        term_buffer=0,
        sampling_no_draws=0
    ))
    metric_adaptation_configuration = Configuration(dict(
        init_buffer=0,
        metric_window='metric_window',
        term_buffer=0,
        sampling_no_draws = 0
    ))
    final_adaptation_configuration = Configuration(dict(
        init_buffer=0,
        metric_window=0,
        term_buffer='term_buffer',
        sampling_no_draws = 0
    ))
    sampling_configuration = Configuration(dict(
        init_buffer=0,
        metric_window=0,
        term_buffer=0,
        sampling_no_draws='sampling_no_draws'
    ))
    @cproperty
    def configurations(self):
        return tuple([
            fit.init_kwargs for fit in self.sequence
        ])

    def callback(self, sequence):
        pass

    @cproperty
    def sequence(self):
        rv = []
        def append(fit):
            rv.append(fit)
            self.callback(rv)
        append(HMC(self.posterior, **self.configurations[0]))
        for configuration in self.configurations[1:]:
            append(rv[-1].shallow_copy(**configuration, **rv[-1].adaptation))
        return Pool(rv)

    @cproperty
    def sampling(self): return self.sequence[-1]

    def __getattr__(self, key):
        if key.startswith('_'): raise AttributeError(self, key)
        if hasattr(self.__class__, key): raise AttributeError(self, key)
        return getattr(self.sampling, key)

    # @cproperty
    # def samples(self): return self.sampling.samples

    @cproperty
    def estimated_cost(self): return sum(self.sequence.estimated_cost)

    @cproperty
    def status(self): return all(self.sequence.status)

    @cproperty
    def hmc_wall_time(self): return np.sum(self.sequence.hmc_wall_time)
    #
    # @cproperty
    # def ess(self): return self.sampling.ess
    #
    # @cproperty
    # def sampling_no_divergences(self):
    #     return self.sampling.sampling_no_divergences
    #
    # @cproperty
    # def avg_sampling_no_leapfrog_steps(self):
    #     return self.sampling.avg_sampling_no_leapfrog_steps
    #
    # @cproperty
    # def potential_scale_reduction_factor(self):
    #     return self.sampling.potential_scale_reduction_factor

    def distance_from(self, posterior): return self.sampling.distance_from(posterior)

    @cproperty
    def distance_from_reference(self): return self.distance_from(self.posterior)

class Incremental(ChainedHMC):
    arg_names = ('posterior', )
    init_buffer = 25
    metric_window = 100
    term_buffer = 25
    sampling_no_draws = 100

    hmc = HMC
    metric_adaptation_configuration = Configuration(dict(
        init_buffer='init_buffer',
        metric_window='metric_window',
        term_buffer=0,
        sampling_no_draws=0
    ))
    @cproperty
    def slice_variables(self): return self.posterior.model.slice_variables

    @cproperty
    def final_data(self): return self.posterior.data

    def slice_update(self, slice_idxs):
        rv = dict(slice_idxs)
        for slice_variable, slice_idx in slice_idxs.items():
            for sliced_variable in self.slice_variables[slice_variable]:
                rv[sliced_variable] = self.final_data[sliced_variable][:slice_idx]
        return rv

    @cproperty
    def initial_slice(self):
        return self.slice_update(
            dict(zip(self.slice_variables, [1]*len(self.slice_variables)))
        )

    @cproperty
    def initial_data(self): return dict(self.final_data, **self.initial_slice)

    def data_reconfiguration(self, sequence):
        last = sequence[-1]
        last_data = last.posterior.data
        for slice_variable in self.slice_variables:
            last_idx = last_data[slice_variable]
            max_idx = self.final_data[slice_variable]
            if last_idx >= max_idx:
                continue
            slice_idx = min(2 * last_idx, max_idx)
            return self.slice_update({slice_variable: slice_idx})

    def reconfiguration(self, sequence):
        data_reconfiguration = self.data_reconfiguration(sequence)
        if data_reconfiguration is None: return None
        last = sequence[-1]
        new_posterior = last.posterior.updated(data_reconfiguration)
        return dict(
            posterior=new_posterior,
            **last.adaptation_for(new_posterior)
        )

    @cproperty
    def sequence(self):
        rv = []
        def append(fit):
            rv.append(fit)
            self.callback(rv)
        append(self.hmc(
            self.posterior.shallow_copy(data=self.initial_data),
            **self.metric_adaptation_configuration
        ))
        while True:
            reconfiguration = self.reconfiguration(rv)
            if reconfiguration is None:
                break
            append(rv[-1].shallow_copy(**reconfiguration))
        append(rv[-1].shallow_copy(
            **self.final_adaptation_configuration,
            **rv[-1].adaptation
        ))
        append(rv[-1].shallow_copy(
            **self.sampling_configuration,
            **rv[-1].adaptation
        ))
        return Pool(rv)

class Adaptive(Incremental):
    relative_efficiency_goal = .5
    @cproperty
    def refinement_variable(self): return self.posterior.model.refinement_variable
    def refinement_update(self, data):
        return {
            self.refinement_variable: 2 * data[self.refinement_variable]
        }

    @cproperty
    def initial_refinement(self):
        return {self.refinement_variable: 1}

    @cproperty
    def initial_data(self):
        return dict(super().initial_data, **self.initial_refinement)

    def data_reconfiguration(self, sequence):
        last = sequence[-1]
        refinement_reconfiguration = self.refinement_update(last.posterior.data)
        refined_posterior = last.posterior.updated(refinement_reconfiguration)
        psis = last.draws.psis(refined_posterior)
        if psis.relative_efficiency < self.relative_efficiency_goal:
            return refinement_reconfiguration
        return super().data_reconfiguration(sequence)

    @cproperty
    def sampling_psis(self): return self.samples.psis(self.posterior)

    @cproperty
    def relative_efficiency(self): return self.sampling_psis.relative_efficiency

    @cproperty
    def pareto_shape_estimate(self): return self.sampling_psis.pareto_shape_estimate

class PooledStan(ChainedHMC):
    @cproperty
    def configurations(self):
        rv = []
        if self.init_buffer:
            rv.append(self.initial_adaptation_configuration)
        if self.metric_buffer:
            base = self.metric_adaptation_configuration
            remaining = 1 * self.metric_buffer
            window = 1 * self.metric_window
            assert(remaining >= window)
            while remaining > 0:
                if 2 * window > remaining: window = remaining
                rv.append(dict(base, metric_window=window))
                remaining -= window
                window *= 2
        if self.term_buffer:
            rv.append(self.final_adaptation_configuration)
        if self.sampling_no_draws:
            rv.append(self.sampling_configuration)
        return tuple(rv)
