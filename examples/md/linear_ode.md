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
