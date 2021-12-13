functions {
  vector range(int N){
    return linspaced_vector(N, 1, N);
  }
  vector eq_sqrt_lambdaf(int no_basis_functions){
    return range(no_basis_functions)*pi()/2;
  }
  matrix eq_phif(vector xi, int no_basis_functions){
    return sin(
      diag_post_multiply(
        rep_matrix(1+xi, no_basis_functions),
        eq_sqrt_lambdaf(no_basis_functions)
      )
    );
  }
  vector eq_unit_sqrt_spectral_densityf(real lengthscale, int no_basis_functions){
    return sqrt(
      sqrt(2*pi()) * lengthscale
    ) * exp(-square(lengthscale*eq_sqrt_lambdaf(no_basis_functions))/4);
  }
}
data {
  int no_observations;
  vector<lower=-1,upper=1>[no_observations] xi_observed;
  vector[no_observations] y_observed;

  real gp_support_fraction;
  int gp_no_basis_functions;
  real likelihood;
}
transformed data {
  matrix[no_observations, gp_no_basis_functions] gp_phi = eq_phif(
    gp_support_fraction * xi_observed, gp_no_basis_functions
  );
}
parameters {
  real<lower=0> gp_unit_lengthscale;
  real<lower=0> gp_unit_sigma;
  real<lower=0> sigma;

  vector[gp_no_basis_functions] unit_weights;
}
transformed parameters {
  real gp_lengthscale = 2 * gp_support_fraction * gp_unit_lengthscale;
  vector[gp_no_basis_functions] gp_sigma = gp_unit_sigma * eq_unit_sqrt_spectral_densityf(
    gp_lengthscale, gp_no_basis_functions
  );
  vector[gp_no_basis_functions] scaled_weights = gp_sigma .* unit_weights;
}
model {
  gp_unit_lengthscale ~ lognormal(0, 1);
  gp_unit_sigma ~ lognormal(0, 1);
  sigma ~ lognormal(0, 1);
  unit_weights ~ normal(0, 1);
  if(likelihood){
    y_observed ~ normal_id_glm(gp_phi, 0, scaled_weights, sigma);
  }
}
generated quantities {
  vector[no_observations] y_computed = gp_phi * scaled_weights;
  vector[no_observations] y_generated = to_vector(normal_rng(y_computed, sigma));
}
