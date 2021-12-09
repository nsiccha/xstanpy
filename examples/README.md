# Examples

* `linear_ode`: A linear ODE model with a fixed ODE matrix and two parameters
(time scale and measurement noise).
See [linear_ode.md](linear_ode.md) for details or [linear_ode.py](linear_ode.py)
for the semi-commented code.

This folder contains one `.py` file per example. Corresponding `.stan` files
are located in the `examples/stan` directory.

In general, figure outputs will be written to the `figs` directory and
`cmdstan` output will be written to the `out` directory.

## Running

Examples should be able to be run

* for file systems supporting symlinks (`examples/xstanpy` is a symlink to `xstanpy`)
* from within the `examples` directory
* with a recent enough python version and
* with the required python packages (among others
`numpy`, `pandas`, `arviz`, `cached_property` and `matplotlib`) installed.

## Expected output

When run via e.g. `python linear_ode.py`

* fit diagnostics get printed to `stdout`,
* figures will be generated for each posterior at `examples/figs/{model.name}/{config.name}/{posterior.hash}.png`,
* a short report will be generated for each configuration under
`examples/figs/{model.name}/{config.name}.md`
* and for each model under `examples/figs/{model.name}.md`.
