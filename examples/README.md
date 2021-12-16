# Examples

* [`linear_ode.{py,stan,md}`](examples/linear_ode.md): A simple linear ordinary differential equation.
* [`1d_gp.{py,stan,md}`](examples/1d_gp.md): A simple 1D Gaussian process.

This folder contains one `.py` file in the `examples/py` folder per example.
Corresponding `.stan` files are located in the `examples/stan` directory.

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

When run via e.g. `python py/linear_ode.py`

* fit diagnostics get printed to `stdout`,
* figures will be generated for each posterior at `examples/figs/{model.name}/{config.name}/{posterior.hash}.png`.
