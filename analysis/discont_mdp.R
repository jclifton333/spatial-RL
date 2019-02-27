# Setup: two-stage MDP where Q-function is additive over 2 locations, and the treatment budget is 1 location.
# Stage-1 actions (1, 0), (0, 1) have expected rewards mu.1, mu.2 resp.  Call these actions a1, a2.  
# Expected payoff at stage 2 at each location is a linear function (with parameter \theta) of a 
# stage-1 covariate and the stage-1 action. 
# We want to evaluate the regret of the policy based on the estimator \theta, and in particular how this regret 
# varies as \theta approaches boundaries separating the value of different actions. 


solve.for.theta = function(x1, delta){
  x1.1 = x1[1]
  x1.2 = x1[2]
  Z = matrix(c(x1.1 - x1.2, x1.1, x1.2 - x1.1, x1.2), nrow=2, ncol=2, byrow=T)
  theta = solve(Z, delta)
  return(theta)
}

bvn.halfspace.intersection(mu, Sigma, v1, v2){
  # Compute the probability that a bivariate N(mu, Sigma) r.v. Y falls in the region defined by 
  # Y %*% v1 >= 0 and
  # Y %*% v2 >= 0.
  v1.1 = v1[1]
  v1.2 = v1[2]
  v2.1 = v2[1]
  v2.2 = v2[2]
  Sigma.det.inv = 1 / (Sigma[1,1]*Sigma[2,2] - Sigma[1,2]*Sigma[2,1])
  Sigma.inv = solve(Sigma)
  constant = 1 / sqrt(2 * pi) * sqrt(Sigma.det.inv)
  outer.function = function(y.2){
    lower = min(c(-y.2 * (v1.2 / v1.1), -y.2 * (v2.2 / v2.1)))
    inner.function = function(y.1){
      y = c(y.1, y.2)
      quad.term = -0.5 * (y - mu) %*% ((y - mu) %*% Sigma.inv)  
      return(constant * exp(quad.term))}
    return(integrate(inner.function, lower=lower, upper=Inf))
  }
  return(integrate(outer.function, lower=-Inf, upper=Inf))
}


prob.a1 = function(x1, mu.1, mu.2, delta, X1, A1, sigma.sq){
  # Get the probability of taking a1 at given state x1, where the randomness is over the sampling variability
  # of theta.hat.  
  # :param delta: vector controlling distance of theta from the decision boundary for x1.  (theta is solved
  #               for accordingly.)
  # :param X1: array of observed covariates; for estimating theta.
  # :param A1: array of observed actions; for estimating theta. 
  # :param sigma.sq: variance of the second-stage states conditional on first-stage state and action.  
  
  theta = solve.for.theta(x1, delta)
   
  # Get the probabilities of each of the half-space intersections corresponding to the pair of 
  # optimal actions at each location.  
  
  
  
  
  
} 

