
# linear_ode / small

## Aggregated data (mean)

|           |   hmc_wall_time |     ess |   sampling_no_divergences |   potential_scale_reduction_factor |   relative_efficiency |   pareto_shape_estimate |
|:----------|----------------:|--------:|--------------------------:|-----------------------------------:|----------------------:|------------------------:|
| adaptive  |        0.739412 | 285.312 |                         0 |                            1.0239  |              0.554251 |              -0.0931229 |
| stan_0200 |        0.925607 | 254.851 |                         0 |                            1.02102 |              1        |            -inf         |
| stan_0400 |        1.16601  | 278.968 |                         0 |                            1.01135 |              1        |            -inf         |
| stan_0800 |        1.85161  | 270.302 |                         0 |                            1.02063 |              1        |            -inf         |
| stan_1000 |        2.17598  | 283.933 |                         0 |                            1.01333 |              1        |            -inf         |

## Separate data

|                                                   |   hmc_wall_time |     ess |   sampling_no_divergences |   potential_scale_reduction_factor |   relative_efficiency |   pareto_shape_estimate |
|:--------------------------------------------------|----------------:|--------:|--------------------------:|-----------------------------------:|----------------------:|------------------------:|
| ('bdd7da7184b1b8ab2124ee03692c678a', 'adaptive')  |        0.698175 | 305.253 |                         0 |                            1.02447 |              0.261486 |               -0.309859 |
| ('bdd7da7184b1b8ab2124ee03692c678a', 'stan_0200') |        0.901209 | 225.414 |                         0 |                            1.0222  |              1        |             -inf        |
| ('bdd7da7184b1b8ab2124ee03692c678a', 'stan_0400') |        1.21255  | 261.59  |                         0 |                            1.01573 |              1        |             -inf        |
| ('bdd7da7184b1b8ab2124ee03692c678a', 'stan_0800') |        1.83691  | 217.749 |                         0 |                            1.03219 |              1        |             -inf        |
| ('bdd7da7184b1b8ab2124ee03692c678a', 'stan_1000') |        2.19889  | 283.205 |                         0 |                            1.01078 |              1        |             -inf        |
| ('97c0dd48c029975eb3fadfd64feecc17', 'adaptive')  |        0.780649 | 265.371 |                         0 |                            1.02333 |              0.847016 |                0.123613 |
| ('97c0dd48c029975eb3fadfd64feecc17', 'stan_0200') |        0.950006 | 284.288 |                         0 |                            1.01984 |              1        |             -inf        |
| ('97c0dd48c029975eb3fadfd64feecc17', 'stan_0400') |        1.11947  | 296.346 |                         0 |                            1.00697 |              1        |             -inf        |
| ('97c0dd48c029975eb3fadfd64feecc17', 'stan_0800') |        1.86631  | 322.855 |                         0 |                            1.00907 |              1        |             -inf        |
| ('97c0dd48c029975eb3fadfd64feecc17', 'stan_1000') |        2.15308  | 284.662 |                         0 |                            1.01588 |              1        |             -inf        |
