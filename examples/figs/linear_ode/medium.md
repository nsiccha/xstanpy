
# linear_ode / medium

## Aggregated data (mean)

|           |   hmc_wall_time |     ess |   sampling_no_divergences |   potential_scale_reduction_factor |   relative_efficiency |   pareto_shape_estimate |
|:----------|----------------:|--------:|--------------------------:|-----------------------------------:|----------------------:|------------------------:|
| adaptive  |         1.34041 | 356.582 |                         0 |                            1.01033 |              0.889832 |               -0.049166 |
| stan_0200 |         8.82346 | 352.349 |                         0 |                            1.0132  |              1        |             -inf        |
| stan_0400 |        13.4449  | 293.819 |                         0 |                            1.02037 |              1        |             -inf        |
| stan_0800 |        21.2477  | 329.721 |                         0 |                            1.00729 |              1        |             -inf        |
| stan_1000 |        26.1476  | 311.226 |                         0 |                            1.01482 |              1        |             -inf        |

## Separate data

|                                                   |   hmc_wall_time |     ess |   sampling_no_divergences |   potential_scale_reduction_factor |   relative_efficiency |   pareto_shape_estimate |
|:--------------------------------------------------|----------------:|--------:|--------------------------:|-----------------------------------:|----------------------:|------------------------:|
| ('262322ed0482f5fa5d8f94f1e56b65c2', 'adaptive')  |         1.52505 | 399.036 |                         0 |                            1.00192 |              0.858389 |               0.0296101 |
| ('262322ed0482f5fa5d8f94f1e56b65c2', 'stan_0200') |         8.04994 | 354.747 |                         0 |                            1.01365 |              1        |            -inf         |
| ('262322ed0482f5fa5d8f94f1e56b65c2', 'stan_0400') |        12.6052  | 236.158 |                         0 |                            1.03041 |              1        |            -inf         |
| ('262322ed0482f5fa5d8f94f1e56b65c2', 'stan_0800') |        20.2433  | 392.791 |                         0 |                            1.00413 |              1        |            -inf         |
| ('262322ed0482f5fa5d8f94f1e56b65c2', 'stan_1000') |        24.7922  | 320.207 |                         0 |                            1.01439 |              1        |            -inf         |
| ('3c1097005df578fa4461624e1a9203b8', 'adaptive')  |         1.15577 | 314.129 |                         0 |                            1.01873 |              0.921276 |              -0.127942  |
| ('3c1097005df578fa4461624e1a9203b8', 'stan_0200') |         9.59698 | 349.952 |                         0 |                            1.01275 |              1        |            -inf         |
| ('3c1097005df578fa4461624e1a9203b8', 'stan_0400') |        14.2845  | 351.479 |                         0 |                            1.01032 |              1        |            -inf         |
| ('3c1097005df578fa4461624e1a9203b8', 'stan_0800') |        22.252   | 266.652 |                         0 |                            1.01045 |              1        |            -inf         |
| ('3c1097005df578fa4461624e1a9203b8', 'stan_1000') |        27.503   | 302.246 |                         0 |                            1.01524 |              1        |            -inf         |
