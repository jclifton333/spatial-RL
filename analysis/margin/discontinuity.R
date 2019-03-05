# Experimenting with the issue of discontinuity of p(s, argmax \sum p)
library(MASS)

compare.estimators = function(delta, n, mu.1=0.0){
  mu = c(mu.1, mu.1 + delta)
  x = mvrnorm(n=n, mu=mu, Sigma=diag(2))
  q.true = cbind(rep(0.0, n), rep(mu.1 + delta, n))
  q.hat.naive = naive.estimator(x)
  q.hat.smoothed = smoothed.estimator(x)
  q.hat.bsmoothed = bootstrapped.smoothed.estimator(x)
  naive.error = mean((q.true - q.hat.naive)**2)
  smoothed.error = mean((q.true - q.hat.smoothed)**2)
  bsmoothed.error = mean((q.true - q.hat.bsmoothed)**2)
  return(list(naive.error=naive.error, smoothed.error=smoothed.error, bsmoothed.error=bsmoothed.error))
}

compare.estimators.multiple.reps = function(delta, n, n.rep, mu.1=0.0){
  smoothed.errors = c()
  bsmoothed.errors = c()
  naive.errors = c()
  for(rep in 1:n.rep){
    errors = compare.estimators(delta, n, mu.1=mu.1)
    smoothed.errors[rep] = errors$smoothed.error
    bsmoothed.errors[rep] = errors$bsmoothed.error
    naive.errors[rep] = errors$naive.error
  }
  return(list(naive.mean=mean(naive.errors), naive.se=sd(naive.errors/n.rep), 
              smoothed.mean=mean(smoothed.errors), smoothed.se=sd(smoothed.errors/n.rep),
              bsmoothed.mean=mean(bsmoothed.errors), bsmoothed.se=sd(bsmoothed.errors/n.rep)))
}

naive.estimator = function(x){
  indicator = x[1] >= x[2]
  q.hat = c(x[1]*indicator, x[2]*(1-indicator))
  return(q.hat)
}

smoothed.estimator = function(x){
  n = nrow(x)
  sigma.sq.1 = sd(x[,1]) / sqrt(n)
  sigma.sq.2 = sd(x[,2]) / sqrt(n)
  xbar.1 = mean(x[,1])
  xbar.2 = mean(x[,2])
  p.1 = pnorm((xbar.1 - xbar.2) / sigma.sq.1)
  p.2 = pnorm((xbar.2 - xbar.1) / sigma.sq.2)
  p = cbind(rep(p.1, n), rep(p.2, n))
  x.bar = cbind(rep(xbar.1, n), rep(xbar.2, n))
  q.hat = x.bar * p
  return(q.hat)
}


bootstrapped.smoothed.estimator = function(x, B=2){
 n = nrow(x)   
 sigma.sq.1 = sd(x[,1]) / sqrt(n)
 sigma.sq.2 = sd(x[,2]) / sqrt(n)
 xbar.1 = mean(x[,1])
 xbar.2 = mean(x[,2])
 q.hat = cbind(rep(0, n), rep(0, n))
 for(b in 1:B){
   x.b = mvrnorm(n=n, mu=c(xbar.1, xbar.2), Sigma=diag(c(sigma.sq.1, sigma.sq.2)))
   q.hat.b = naive.estimator(x.b)
   q.hat = q.hat + q.hat.b 
 }
 q.hat = q.hat / B
 return(q.hat)
}

importance.sampling.weight = function(xbar.1.n, xbar.2.n, sigma1.n, sigma2.n, xbar.1.nmk, xbar.2.nmk, n, k){
  # Get probability of xbar.1.nmk, xbar.2.nmk under MVN( c(xbar.1.n, xbar.2.n), diag(sigma.1.n, sigma.2.n))
  Sigma.n.inv = Sigma.n.inv = diag(c(1/sigma1.n**2, 1/sigma2.n**2))
  xbar.n = c(xbar.1.n, xbar.2.n)
  xbar.nmk = c(xbar.1.nmk, xbar.2.nmk)
  normsq.Sigma.n = norm(Sigma.n.inv %*% (xbar.n - xbar.nmk))**2 
  # Double check formula
  is.weight = ((n-k)/n) * exp( -0.5 * (normsq.Sigma.n + ((n - k) / n) * normsq.Sigma.n))
  return(is.weight)
}


importance.sampling.estimator = function(xbar.1.n, xbar.2.n, sigma1.n, sigma2.n, xbar.1.history, 
                                         xbar.2.history, q.history){
  n = length(xbar.1.history) + 1
  xbar.n = c(xbar.1.n, xbar.2.n)
  qhat.naive = naive.estimator(xbar.n) 
  qhat = qhat.naive
    for(i in 1:(n-1)){
    xbar.1.nmk = xbar.1.history[i]
    xbar.2.nmk = xbar.2.history[i]
    qhat.nmk = q.history[i]
    xbar.nmk = c(xbar.1.nmk, xbar.2.nmk)
    k = n - i
    is.weight = importance.sampling.weight(xbar.1.n, xbar.2.n, sigma1.n, sigma2.n, xbar.1.nmk, xbar.2.nmk,
                                           n, k)
    qhat = qhat + is.weight*qhat.nmk/3
      } 
  qhat = qhat / n
  return(list(qhat=qhat, qhat.naive=qhat.naive))
}


xbar.multiplier.boot = function(x){
  w = rexp(n=length(x))
  xbar = (w %*% x) / sum(w)
  return(xbar)
}

online.estimation = function(delta, horizon, mu.1=0.0){
  mu = c(mu.1, mu.1 + delta)
  Sigma = diag(2)
  x = matrix(mvrnorm(n=1, mu=mu, Sigma=Sigma), nrow=1, ncol=2)
  xbar.1.history = c(x[,1])
  xbar.2.history = c(x[,2])
  q.history = c(naive.estimator(x))
  q.naive.history = c(naive.estimator(x))
  for(i in 2:horizon){
     # Get estimator at time i
     x.i = mvrnorm(n=1, mu=mu, Sigma=Sigma)
     x = rbind(x, x.i)
     xbar.1 = xbar.multiplier.boot(x[,1])
     xbar.2 = xbar.multiplier.boot(x[,2])
     sigma1 = sd(x[,1]) / sqrt(i)
     sigma2 = sd(x[,2]) / sqrt(i)
     
     q.hats = importance.sampling.estimator(xbar.1, xbar.2, sigma1, sigma2, xbar.1, xbar.2, 
                                           q.history)
     q.hat = q.hats$qhat
     q.hat.naive = q.hats$qhat.naive
     
     # Update histories 
     xbar.1.history = append(xbar.1.history, xbar.1)
     xbar.2.history = append(xbar.2.history, xbar.2)
     q.history = append(q.history, q.hat)
     q.naive.history = append(q.naive.history, q.hat.naive)
  }
  print(q.history)
  print(xbar.1.history)
  print(xbar.2.history)
  q.true = c(mu.1, mu.1 + delta)
  q.naive.error = mean((q.true - q.naive.history)**2)
  q.error = mean((q.true - q.history)**2)
  return(list(naive.error=q.naive.error, is.error=q.error))
}

online.estimation.multiple.reps = function(delta, horizon, n.rep){
  naive.errors = c()
  is.errors = c()
  for(rep in 1:n.rep){
    errors = online.estimation(delta, horizon)
    naive.errors = append(naive.errors, errors$naive.error)
    is.errors = append(is.errors, errors$is.error)
  }
  return(list(naive.error=mean(naive.errors), is.error=mean(is.errors)))
}


# print(online.estimation.multiple.reps(5, 30, 1000))







