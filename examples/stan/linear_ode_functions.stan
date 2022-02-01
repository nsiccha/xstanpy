array[] vector exact_y(
    array[] real x_observed, vector y_initial, matrix ode_matrix
  ){
  int no_observations = size(x_observed);
  int no_dimensions = rows(ode_matrix);
  array[no_observations] vector[no_dimensions] y_computed;
  real t = 0;
  vector[no_dimensions] y = y_initial;
  for(i in 1:no_observations){
    real dt = x_observed[i] - t;
    t += dt;
    y = matrix_exp(dt * ode_matrix) * y;
    y_computed[i] = y;
  }
  return y_computed;
}

array[] vector approximate_y(
    int no_steps, array[] real x_observed, vector y_initial, matrix ode_matrix
  ){
  int no_observations = size(x_observed);
  int no_dimensions = rows(ode_matrix);
  array[no_observations] vector[no_dimensions] y_computed;
  real dt = x_observed[no_observations] / no_steps;
  real t = 0;
  vector[no_dimensions] last_y = y_initial;
  vector[no_dimensions] y = last_y;
  matrix[no_dimensions, no_dimensions] factor = matrix_exp(dt * ode_matrix);
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
    int no_steps, array[] real x_observed, vector y_initial, matrix ode_matrix
  ){
  if(no_steps == 0){
    return exact_y(x_observed, y_initial, ode_matrix);
  }else{
    return approximate_y(no_steps, x_observed, y_initial, ode_matrix);
  }
}
