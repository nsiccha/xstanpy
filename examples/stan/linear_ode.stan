/*

*/


functions {
#include linear_ode_functions.stan
}
data {
  int no_observations;
  int no_dimensions;
  array[no_observations] real x_observed;
  matrix[no_dimensions, no_dimensions] ode_matrix;
  array[no_observations] vector[no_dimensions] y_observed;

  int no_steps;
  real likelihood;
}
parameters {
  real<lower=0> k;
  real<lower=0> sigma;
}
transformed parameters {
  array[no_observations] vector[no_dimensions] y_computed = compute_y(
    no_steps, x_observed, k * ode_matrix
  );
}
model {
  k ~ lognormal(0,1);
  sigma ~ lognormal(-1,1);
  if(likelihood){
    for(observation_idx in 1:no_observations){
      y_observed[observation_idx] ~ lognormal(log(y_computed[observation_idx]), sigma);
    }

  }
}
generated quantities {
  array[no_observations] vector[no_dimensions] y_generated;
  for(observation_idx in 1:no_observations){
    y_generated[observation_idx] = to_vector(
      lognormal_rng(log(y_computed[observation_idx]), sigma)
    );
  }
}
