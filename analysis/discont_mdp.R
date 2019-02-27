# THIS IS NOT RIGHT!  THETA SHOULD INDEX THE ESTIMATED MAX OF CONDITIONAL MEAN, NOT THE TRANSITION FUNCTION

#Setup: two-stage MDP where Q-function is additive over 2 locations, and the treatment budget is 1 location.
# Stage-1 actions (1, 0), (0, 1) have expected rewards mu.1, mu.2 resp.  Call these actions a1, a2.  
# Expected payoff at stage 2 at each location is a linear function (with parameter \theta) of a 
# stage-1 covariate and the stage-1 action. 
# We want to evaluate the regret of the policy based on the estimator \theta, and in particular how this regret 
# varies as \theta approaches boundaries separating the value of different actions. 
library(mvtnorm)


solve.for.theta = function(x1, delta){
  x1.1 = x1[1]
  x1.2 = x1[2]
  Z = matrix(c(x1.1 - x1.2, x1.1, x1.2 - x1.1, x1.2), nrow=2, ncol=2, byrow=T)
  theta = solve(Z, delta)
  return(theta)
}

# bvn.halfspace.intersection = function(mu, Sigma, v1, v2){
#   # Compute the probability that a bivariate N(mu, Sigma) r.v. Y falls in the region defined by 
#   # Y %*% v1 >= 0 and
#   # Y %*% v2 >= 0.
#   V = rbind(v1, v2)
#   mean = V %*% mu
#   Cov = V %*% (Sigma %*% t(V))
#   Cov.det.inv = 1 / (Cov[1,1]*Cov[2,2] - Cov[1,2]*Cov[2,1])
#   Cov.inv = solve(Cov)
#   constant = 1 / (2 * pi) * sqrt(Cov.det.inv)
#   outer.function = function(y.2){
#     inner.function = function(y.1){
#       y = c(y.1, y.2)
#       quad.term = -0.5 * (t(y - mean) %*% Cov.inv) %*% (y - mean)
#       return(constant * exp(quad.term))}
#     return(integrate(Vectorize(inner.function), lower=0, upper=Inf)$value)
#   }
#   return(integrate(Vectorize(outer.function), lower=0, upper=Inf)$value)
# }

mvn.prob(mu, Sigma, A, lower){
  # Compute the probability that 3-dimensional Y ~ N(mu, Sigma) satisfies
  # AY >= lower.
  A.mu = A %*% mu
  A.Sigma = A %*% (Sigma %*% t(A))
  prob = pmvnorm(lower=lower, upper=rep(Inf, 3), mean=A.mu, sigma=A.Sigma)
  return(prob) 
}

stage.two.value = function(stage.two.mean.vector){
  indicator = stage.two.mean.vector[1] >= stage.two.mean.vector[2]
  stage.two.value = stage.two.mean.vector[1]*indicator + stage.two.mean.vector[2]*(1 - indicator)
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
  design.matrix = cbind(X1, X1*A1)
  Sigma = sigma.sq * (t(design.matrix) %*% design.matrix)  # Covariance of theta.hat
   
  # Get the probabilities of each of the half-space intersections corresponding to the pair of 
  # optimal actions at each location.  
  x1.1 = x1[1]
  x1.2 = x1[2]
  x1.diff = x1.1 - x1.2
  mu.diff = mu.1 - mu.2
  lower = c(0, 0, mu.diff)
  
  A.11 = matrix(rbind(c(x1.diff, x1.1), c(-x1.diff, x1.2), c(x1.diff, x1.diff)), nrow=2, ncol=2, byrow=T)
  p.11 = mvn.prob(theta, Sigma, A.11, lower)
  A.10 = matrix(rbind(c(x1.diff, x1.1), -c(-x1.diff, x1.2), c(0, x1.1)), nrow=2, ncol=2, byrow=T)
  p.10 = mvn.prob(theta, Sigma, A.10, lower) 
  A.01 = matrix(rbind(-c(x1.diff, x1.1), c(-x1.diff, x1.2), c(0, -x1.2)), nrow=2, ncol=2, byrow=T)
  p.10 = mvn.prob(theta, Sigma, A.01, lower) 
  A.00 = matrix(rbind(-c(x1.diff, x1.1), -c(-x1.diff, -x1.2), c(-x1.diff, 0)), nrow=2, ncol=2, byrow=T)
  p.00 = mvn.prob(theta, Sigma, A.00, lower)
  
  p.a1 = p.11 + p.10 + p.01 + p.00 
  
  # Compute expected value under this policy.
  x1.a1 = rbind(c(x1.1, x1.1), c(x1.2, 0))
  x1.a2 = rbind(c(x1.1, 0), c(x1.2, x1.2))
  stage.two.mean.a1 = x1.a1 %*% theta 
  stage.two.mean.a2 = x1.a2 %*% theta
  stage.two.value.a1 = stage.two.value(stage.two.mean.a1) 
  stage.two.value.a2 = stage.two.value(stage.two.mean.a2)
  
  x1diff.x1.1 = c(x1.diff, x1.1) 
  p.a1.given.a1 = pnorm(0, mean=theta %*% x1diff.x1.1, sd=sqrt(x1diff.x1.1 %*% (Sigma %*% x1diff.x1.1)))
  x1diff.x1.2 = c(-x1.diff, x1.2)
  p.a1.given.a2 = pnorm(0, mean=theta %*% x1diff.x1.2, sd=sqrt(x1diff.x1.2 %*% (Sigma %*% x1diff.x1.2)))
} 

