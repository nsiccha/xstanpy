functions {
  array[] vector exact_y(
      array[] real x_observed, matrix k_ode_matrix
    ){
    int no_observations = size(x_observed);
    int no_dimensions = rows(k_ode_matrix);
    array[no_observations] vector[no_dimensions] y_computed;
    real t = 0;
    vector[no_dimensions] y = rep_vector(1., no_dimensions);
    for(i in 1:no_observations){
      real dt = x_observed[i] - t;
      t += dt;
      y = matrix_exp(dt * k_ode_matrix) * y;
      y_computed[i] = y;
    }
    return y_computed;
  }

  array[] vector approximate_y(
      int no_steps, array[] real x_observed, matrix k_ode_matrix
    ){
    int no_observations = size(x_observed);
    int no_dimensions = rows(k_ode_matrix);
    array[no_observations] vector[no_dimensions] y_computed;
    real dt = x_observed[no_observations] / no_steps;
    real t = 0;
    vector[no_dimensions] last_y = rep_vector(1., no_dimensions);
    vector[no_dimensions] y = last_y;
    matrix[no_dimensions, no_dimensions] factor = matrix_exp(dt * k_ode_matrix);
    for(i in 1:no_observations){
      while(t < x_observed[i]){
        t += dt;
        last_y = y;
        y = factor * y;
      }
      real xi = (x_observed[i] - (t - dt)) / dt;
      y_computed[i] = (1 - xi) * last_y + xi * y;
    }
    return y_computed;
  }
  array[] vector compute_y(
      int no_steps, array[] real x_observed, matrix k_ode_matrix
    ){
      if(no_steps == 0){
        return exact_y(x_observed, k_ode_matrix);
      }else{
        return approximate_y(no_steps, x_observed, k_ode_matrix);
      }
    }
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
