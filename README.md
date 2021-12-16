# Custom python interface to xstan (a modified (cmd)stan)

Use at your own risk, currently everything is very brittle and will probably be changed
in the near future.

## Download

Everything including modified submodules can be checked out via

`git clone --recurse-submodules -j8 https://github.com/nsiccha/xstanpy`

This should so far include

* the python code in `xstanpy`,
* a modified CmdStan version in `cmdstan`
* a modified stan version in `cmdstan/stan` and
* a so far unmodified math version in `cmdstan/stan/lib/stan_math`

## Usage

Commented examples can be found in the [`examples`](examples) directory and currently include

* [`linear_ode.{py,stan,md}`](examples/linear_ode.md): A simple linear ordinary differential equation.
* [`1d_gp.{py,stan,md}`](examples/1d_gp.md): A simple 1D Gaussian process.

Further instructions on how to run the examples can be found in
[`examples/README.md`](examples/README.md).


## Caching

Soon, results (model+data+command+config) will be automatically cached.

### Disclaimer

The only thing I do worse than documenting is naming things.
