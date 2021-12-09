
# linear_ode

## Aggregated data (mean)

|                          |   hmc_wall_time |     ess |   sampling_no_divergences |   potential_scale_reduction_factor |   relative_efficiency |   pareto_shape_estimate |
|:-------------------------|----------------:|--------:|--------------------------:|-----------------------------------:|----------------------:|------------------------:|
| ('medium', 'adaptive')   |        1.34041  | 356.582 |                         0 |                            1.01033 |              0.889832 |              -0.049166  |
| ('medium', 'stan_0200')  |        8.82346  | 352.349 |                         0 |                            1.0132  |              1        |            -inf         |
| ('medium', 'stan_0400')  |       13.4449   | 293.819 |                         0 |                            1.02037 |              1        |            -inf         |
| ('medium', 'stan_0800')  |       21.2477   | 329.721 |                         0 |                            1.00729 |              1        |            -inf         |
| ('medium', 'stan_1000')  |       26.1476   | 311.226 |                         0 |                            1.01482 |              1        |            -inf         |
| ('small', 'adaptive')    |        0.739412 | 285.312 |                         0 |                            1.0239  |              0.554251 |              -0.0931229 |
| ('small', 'stan_0200')   |        0.925607 | 254.851 |                         0 |                            1.02102 |              1        |            -inf         |
| ('small', 'stan_0400')   |        1.16601  | 278.968 |                         0 |                            1.01135 |              1        |            -inf         |
| ('small', 'stan_0800')   |        1.85161  | 270.302 |                         0 |                            1.02063 |              1        |            -inf         |
| ('small', 'stan_1000')   |        2.17598  | 283.933 |                         0 |                            1.01333 |              1        |            -inf         |
| ('smaller', 'adaptive')  |        0.467927 | 295.06  |                         0 |                            1.01948 |              0.770064 |              -0.384205  |
| ('smaller', 'stan_0200') |        0.132423 | 256.687 |                         0 |                            1.02069 |              1        |            -inf         |
| ('smaller', 'stan_0400') |        0.178983 | 353.601 |                         0 |                            1.00295 |              1        |            -inf         |
| ('smaller', 'stan_0800') |        0.27292  | 317.578 |                         0 |                            1.0094  |              1        |            -inf         |
| ('smaller', 'stan_1000') |        0.327122 | 208.196 |                         0 |                            1.0315  |              1        |            -inf         |

## Separate data

|                                                              |   hmc_wall_time |     ess |   sampling_no_divergences |   potential_scale_reduction_factor |   relative_efficiency |   pareto_shape_estimate |
|:-------------------------------------------------------------|----------------:|--------:|--------------------------:|-----------------------------------:|----------------------:|------------------------:|
| ('smaller', '79bae0b2b604edc25682ac84b1193b82', 'adaptive')  |        0.453781 | 263.512 |                         0 |                           1.01784  |              0.903215 |             -0.769906   |
| ('smaller', '79bae0b2b604edc25682ac84b1193b82', 'stan_0200') |        0.132298 | 260.774 |                         0 |                           1.01934  |              1        |           -inf          |
| ('smaller', '79bae0b2b604edc25682ac84b1193b82', 'stan_0400') |        0.181189 | 376.694 |                         0 |                           0.993527 |              1        |           -inf          |
| ('smaller', '79bae0b2b604edc25682ac84b1193b82', 'stan_0800') |        0.278589 | 295.876 |                         0 |                           1.01148  |              1        |           -inf          |
| ('smaller', '79bae0b2b604edc25682ac84b1193b82', 'stan_1000') |        0.323218 | 214.378 |                         0 |                           1.03187  |              1        |           -inf          |
| ('smaller', '90167a6b6b122efd849c420073b97442', 'adaptive')  |        0.482074 | 326.608 |                         0 |                           1.02112  |              0.636913 |              0.00149683 |
| ('smaller', '90167a6b6b122efd849c420073b97442', 'stan_0200') |        0.132548 | 252.6   |                         0 |                           1.02204  |              1        |           -inf          |
| ('smaller', '90167a6b6b122efd849c420073b97442', 'stan_0400') |        0.176776 | 330.509 |                         0 |                           1.01237  |              1        |           -inf          |
| ('smaller', '90167a6b6b122efd849c420073b97442', 'stan_0800') |        0.267251 | 339.28  |                         0 |                           1.00731  |              1        |           -inf          |
| ('smaller', '90167a6b6b122efd849c420073b97442', 'stan_1000') |        0.331025 | 202.013 |                         0 |                           1.03113  |              1        |           -inf          |
| ('small', 'bdd7da7184b1b8ab2124ee03692c678a', 'adaptive')    |        0.698175 | 305.253 |                         0 |                           1.02447  |              0.261486 |             -0.309859   |
| ('small', 'bdd7da7184b1b8ab2124ee03692c678a', 'stan_0200')   |        0.901209 | 225.414 |                         0 |                           1.0222   |              1        |           -inf          |
| ('small', 'bdd7da7184b1b8ab2124ee03692c678a', 'stan_0400')   |        1.21255  | 261.59  |                         0 |                           1.01573  |              1        |           -inf          |
| ('small', 'bdd7da7184b1b8ab2124ee03692c678a', 'stan_0800')   |        1.83691  | 217.749 |                         0 |                           1.03219  |              1        |           -inf          |
| ('small', 'bdd7da7184b1b8ab2124ee03692c678a', 'stan_1000')   |        2.19889  | 283.205 |                         0 |                           1.01078  |              1        |           -inf          |
| ('small', '97c0dd48c029975eb3fadfd64feecc17', 'adaptive')    |        0.780649 | 265.371 |                         0 |                           1.02333  |              0.847016 |              0.123613   |
| ('small', '97c0dd48c029975eb3fadfd64feecc17', 'stan_0200')   |        0.950006 | 284.288 |                         0 |                           1.01984  |              1        |           -inf          |
| ('small', '97c0dd48c029975eb3fadfd64feecc17', 'stan_0400')   |        1.11947  | 296.346 |                         0 |                           1.00697  |              1        |           -inf          |
| ('small', '97c0dd48c029975eb3fadfd64feecc17', 'stan_0800')   |        1.86631  | 322.855 |                         0 |                           1.00907  |              1        |           -inf          |
| ('small', '97c0dd48c029975eb3fadfd64feecc17', 'stan_1000')   |        2.15308  | 284.662 |                         0 |                           1.01588  |              1        |           -inf          |
| ('medium', '262322ed0482f5fa5d8f94f1e56b65c2', 'adaptive')   |        1.52505  | 399.036 |                         0 |                           1.00192  |              0.858389 |              0.0296101  |
| ('medium', '262322ed0482f5fa5d8f94f1e56b65c2', 'stan_0200')  |        8.04994  | 354.747 |                         0 |                           1.01365  |              1        |           -inf          |
| ('medium', '262322ed0482f5fa5d8f94f1e56b65c2', 'stan_0400')  |       12.6052   | 236.158 |                         0 |                           1.03041  |              1        |           -inf          |
| ('medium', '262322ed0482f5fa5d8f94f1e56b65c2', 'stan_0800')  |       20.2433   | 392.791 |                         0 |                           1.00413  |              1        |           -inf          |
| ('medium', '262322ed0482f5fa5d8f94f1e56b65c2', 'stan_1000')  |       24.7922   | 320.207 |                         0 |                           1.01439  |              1        |           -inf          |
| ('medium', '3c1097005df578fa4461624e1a9203b8', 'adaptive')   |        1.15577  | 314.129 |                         0 |                           1.01873  |              0.921276 |             -0.127942   |
| ('medium', '3c1097005df578fa4461624e1a9203b8', 'stan_0200')  |        9.59698  | 349.952 |                         0 |                           1.01275  |              1        |           -inf          |
| ('medium', '3c1097005df578fa4461624e1a9203b8', 'stan_0400')  |       14.2845   | 351.479 |                         0 |                           1.01032  |              1        |           -inf          |
| ('medium', '3c1097005df578fa4461624e1a9203b8', 'stan_0800')  |       22.252    | 266.652 |                         0 |                           1.01045  |              1        |           -inf          |
| ('medium', '3c1097005df578fa4461624e1a9203b8', 'stan_1000')  |       27.503    | 302.246 |                         0 |                           1.01524  |              1        |           -inf          |