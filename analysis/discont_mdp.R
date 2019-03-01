#Setup: two-stage MDP where Q-function is additive over 2 locations, and the treatment budget is 1 location.
# Stage-1 actions (1, 0), (0, 1) have expected rewards mu.1, mu.2 resp.  Call these actions a1, a2.  
# Expected payoff at stage 2 at each location is a linear function (with parameter \theta) of a 
# stage-1 covariate and the stage-1 action. 
# We want to evaluate the regret of the policy based on the estimator \theta, and in particular how this regret 
# varies as \theta approaches boundaries separating the value of different actions. 
library(mvtnorm)
library(MASS)


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

mvn.prob = function(mu, Sigma, A, lower){
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

prob.a1 = function(x1, mu.1, mu.2, delta, X1, A1, sigma.sq, mc.replicates=1000){
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
   
  # Generate many draws from conditional dbn
  conditional.means = design.matrix %*% theta
  X2 = mvrnorm(n=mc.replicates, mu=conditional.means, Sigma=sigma.sq*diag(length(conditional.means)))
  
  # Fit theta on each draw
  XprimeXinv.Xprime = solve(t(design.matrix) %*% design.matrix) %*% t(design.matrix)
  theta.hats = XprimeXinv.Xprime %*% t(X2) 
  
  # Get pseudo-outcomes for each draw
  X2.hats = design.matrix %*% theta.hats # Matrix with columns corresponding to predicted next X's for each draw
  n.sample = as.integer(nrow(X1) / 2)
  diff.matrix = matrix(0, nrow=n.sample, ncol=n.sample*2)
  for(i in 1:n.sample){
    diff.matrix[i, 2*i-1] = 1
    diff.matrix[i, 2*i] = -1
  }
  a1.mask.vector = diff.matrix %*% X2.hats >= 0
  a1.mask.matrix = kronecker(a1.mask.vector, c(1, 0))
  a2.mask.matrix = kronecker(1-a1.mask.vector, c(0, 1))
  pseudo.outcomes = X2.hats * (a1.mask.matrix + a2.mask.matrix) # Matrix with columns corresponding to pseudo-outcomes for each draw
  
  # Construct expanded features for pseudo-outcome model (with parameter eta)
  Phi1.ixn = matrix(, ncol=14, nrow=0)
  for(i in 1:n.sample){
    phi.i = design.matrix[(2*i-1):(2*i), ]  
    phi.i.ixn = ixn.features(phi.i)
    Phi1.ixn = rbind(Phi1.ixn, phi.i.ixn)
  }
  Phi.prime.Phi.inv.Phi.prime = solve(t(Phi1.ixn) %*% Phi1.ixn) %*% t(Phi1.ixn)
  eta.hats = Phi.prime.Phi.inv.Phi.prime %*% pseudo.outcomes
  
  # Get stage 1 action at each replicate 
  x1.a1.ixn = ixn.features(cbind(x1, c(1, 0)))
  x1.a2.ixn = ixn.features(cbind(x1, c(0, 1)))
  q.a1.hats = x1.a1.ixn %*% eta.hats  
  q.a2.hats = x1.a2.ixn %*% eta.hats
  a1.mask = colSums(q.a1.hats) > colSums(q1.a2.hats)
  
  expected.stage.1.reward = mu.1 * sum(a1.mask) + mu.2 * sum(1 - a1.mask)
  
  # Get expected stage 2 rewards
  stage.1.actions.mask = rbind(a1.mask, 1-a1.mask)
  stage.2.means = theta[1]*x1 + theta[2]*(x1 * stage.1.actions.mask)
  prob.stage.2.a1 = 1 - pnorm(0, mean=stage.2.means[1,] - stage.2.means[2,], sd=sqrt(2*sigma.sq)) # Prob of taking action 1 at stage 2
  stage.2.action.probs = rbind(prob.stage.2.a1, 1-prob.stage.2.a1)
  expected.stage.2.reward = sum(stage.2.action.probs * stage.2.means)
  
  return(expected.stage.1.reward + expected.stage.2.reward)
} 


q.opt = function(x1, theta, mu.1, mu.2, sigma.sq){
  # Value of stage 1 actions when optimal q is followed at next stage. 
  x1.1 = x1[1]
  x1.2 = x1[2]
  x1.diff = x1.1 - x1.2
  
  # Value of a1
  mean.a1 = theta %*% c(x1.diff, x1.1)
  var.a1 = sigma.sq * norm(c(x1.diff, x1.1))**2
  prob.opt.a1 = pnorm(0, mean=mean.a1, sd=sqrt(var.a1))
  
}

ixn.features = function(phi.i){
  # :param phi.i: array [x1 & a1*x1 \\ x2 & a2*x2] 
  phi.i.1 = phi.i[1,]
  phi.i.2 = phi.i[2,] 
  phi.i.flat = c(phi.i.1, phi.i.2)
  phi.ixn.1 = c(phi.i.1, phi.i, phi.i.1[1]*phi.i.flat, phi.i.1[2]*phi.i.flat)
  phi.ixn.2 = c(phi.i.2, phi.i, phi.i.2[1]*phi.i.flat, phi.i.2[2]*phi.i.flat)
  return(rbind(phi.ixn.1, phi.ixn.2))
}

n=15
mu.1 = 0
mu.2 = 1
sigma.sq = 1
A1 = as.matrix(c(rmultinom(n=n, size=1, prob=c(0.5, 0.5))))
X1 = as.matrix(c(mvrnorm(n=n, mu=c(mu.1, mu.2), Sigma=sigma.sq*diag(2))))
delta = c(0, 0.1)
x1 = as.matrix(c(1, 0.5))
prob.a1(x1, mu.1, mu.2, delta, X1, A1, sigma.sq, mc.replicates=1000)
  
