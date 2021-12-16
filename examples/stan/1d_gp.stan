/*
We implement a simple Gaussian process (GP) regression using an approximate series expansion
of the covariance function as proposed in [SS2020] and picked up in [RM2020].
We loosely follow the code accompanying https://avehtari.github.io/casestudies/Birthdays/birthdays.html
as accessible via https://github.com/avehtari/casestudies/tree/master/Birthdays
and closely follow the notation in [R2020].

To estimate the noisily measured values of a
  scalar function f: (-1, +1) -> reals
given a GP prior with a squared exponential covariance function
of unknown
  lengthscale l (`gp_lengthscale`)
and unknown
  marginal standard deviation sigma, (`gp_unit_sigma * gp_sigma_multiplier`),
we approximate the
  latent values f(xi) (`y_computed`)
at the
  measurement points xi (`xi_observed`)
using a truncated series expansion
  f(xi) = phi_j(xi) * w_j
with m
  basis functions phi_j
and m
  basis function coefficients w_j (`scaled_weights`),
where summation over the repeated
  index j = 1,...,m
is implied.

As the
  number of basis functions m (`no_basis_functions`)
increases, our approximation is supposed to converge rapidly towards the corresponding
exact GP model. While the basis functions are normalized, the prior on the
basis function coefficients rapdily decays (with the index of the basis function)
towards a point mass at zero.

[SS2020]: [Solin and Särkkä (2020)](https://link.springer.com/article/10.1007/s11222-019-09886-w)
[R2020]: [Riutort-Mayol et al. (2020)](https://arxiv.org/abs/2004.11408)
*/


functions {
  // Returns a vector [1,...N]';
  vector range(int N){
    return linspaced_vector(N, 1, N);
  }
  // The square root of the eigenvalues.
  vector eq_sqrt_lambdaf(int no_basis_functions){
    return range(no_basis_functions)*pi()/2;
  }
  // The precomputable part of the matrix mapping the unit scale weights
  // onto the scaled basis function weights.
  matrix eq_phif(vector xi, int no_basis_functions){
    return sin(
      diag_post_multiply(
        rep_matrix(1+xi, no_basis_functions),
        eq_sqrt_lambdaf(no_basis_functions)
      )
    );
  }
  // The square root of the spectral density evaluated at the square root of the eigenvalues,
  // for the first `no_basis_functions` eigenvalues, given a `lengthscale`.
  vector eq_unit_sqrt_spectral_densityf(real lengthscale, int no_basis_functions){
    return sqrt(
      sqrt(2*pi()) * lengthscale
    ) * exp(-square(lengthscale*eq_sqrt_lambdaf(no_basis_functions))/4);
  }
}
data {
  // These are our observational data.
  int no_observations;
  vector<lower=-1,upper=1>[no_observations] xi_observed;
  vector[no_observations] y_observed;
  // To make our model more flexible, we allow a multiplier for the
  // Gaussian process lengthscale and sigma.
  real gp_lengthscale_multiplier;
  real gp_sigma_multiplier;
  // These are the configuration options for the basis function expansion
  // of the Gaussian process.
  real gp_support_fraction;
  int gp_no_basis_functions;
  // Simple flag to turn the likelihood on or off.
  // Needed for the adaptive refinement.
  real likelihood;
}
transformed data {
  matrix[no_observations, gp_no_basis_functions] gp_phi = eq_phif(
    gp_support_fraction * xi_observed, gp_no_basis_functions
  );
}
parameters {
  // To make things potentially easier for Stan,
  // we choose the parametrization such that the parameters have a standard normal
  // prior on their unconstrained scale.
  real<lower=0> sigma;
  real<lower=0> gp_unit_lengthscale;
  real<lower=0> gp_unit_sigma;
  // For more efficient posterior exploration, we might have to tune the "centeredness"
  // of our parametrization. For now and for the sake of ease of implementation,
  // we use a fixed non-centered parametrization.
  vector[gp_no_basis_functions] gp_unit_weights;
}
transformed parameters {
  real gp_lengthscale = gp_unit_lengthscale * gp_lengthscale_multiplier;
  real gp_sigma = gp_unit_sigma * gp_sigma_multiplier;
  vector[gp_no_basis_functions] gp_scaled_weights_sigma = gp_sigma
    * eq_unit_sqrt_spectral_densityf(
      2 * gp_support_fraction * gp_lengthscale,
      gp_no_basis_functions
    );
  vector[gp_no_basis_functions] gp_scaled_weights = gp_scaled_weights_sigma .* gp_unit_weights;
}
model {
  // Priors:
  // As mentioned before, these correspond to standard normal priors on the unconstrained scale.
  sigma ~ lognormal(0, 1);
  gp_unit_lengthscale ~ lognormal(0, 1);
  gp_unit_sigma ~ lognormal(0, 1);
  // "Non-centered" parametrization
  gp_unit_weights ~ normal(0, 1);
  // Likelihood:
  if(likelihood){
    // This is just y ~ normal(gp_phi * scaled_weights, sigma);
    // Allegedly this is more efficient than first computing `y_computed` as done below.
    y_observed ~ normal_id_glm(gp_phi, 0, gp_scaled_weights, sigma);
  }
}
generated quantities {
  // Recover the "true" latent values.
  vector[no_observations] y_computed = gp_phi * gp_scaled_weights;
  // Generate observations.
  vector[no_observations] y_generated = to_vector(normal_rng(y_computed, sigma));
}
