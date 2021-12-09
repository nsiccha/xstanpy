# Custom python interface to xstan (a modified (cmd)stan)

Use at your own risk, currently everything is very brittle and will probably be changed
in the near future.

## Usage

Commented examples can be found in the `examples` directory and currently include

* `linear_ode.py`: A simple linear ordinary differential equation which gets fitted
either using a custom (pooled, incremental & adaptive) warm-up procedure or Stan's
default warm-up with a varying number of total warm-up iterations.

Further instructions on how to run the examples can be found in `examples/README.md`.


## Caching

Soon, results (model+data+command+config) will be automatically cached. 
