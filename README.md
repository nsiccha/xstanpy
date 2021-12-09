# Custom python interface to xstan (a modified (cmd)stan)

Use at your own risk, currently everything is very brittle and will probably be changed
in the near future.

## Usage

Commented examples can be found in the `examples` directory and currently include

* `linear_ode.py`: A simple linear ordinary differential equation which gets fitted
either using a custom (pooled, incremental & adaptive) warm-up procedure or Stan's
default warm-up with a varying number of total warm-up iterations.

Examples should be able to be run from within the `examples` directory
with a recent enough python version and with the required python packages
(among others `numpy`, `pandas`, `arviz`, `cached_property` and `matplotlib`)
installed.
When run via e.g. `python linear_ode.py` fit diagnostics get printed to `stdout`
and figures named `{model.name}_{posterior.hash}.png`
get generated in the `examples/figs` directory.
