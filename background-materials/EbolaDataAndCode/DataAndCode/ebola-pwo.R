library(Matrix)

makecirclemass <- function(dist, pop){
  n <- length(pop)
  circlemass <- matrix(NA, nrow=n, ncol=n)

  for(i in 1:n){
    for(j in 1:n){
      circlemass[i,j] <- sum(pop[which(dist[i,] <= dist[i,j])])
    }
  }

  diag(circlemass) <- rep(0, n)
  return(circlemass)
}

makepwomat <- function(cm, pop){
  N <- length(pop)
  M <- sum(pop)
  Oj <- rep(1, N) %*% t(pop)
  Sji <- t(cm)
  Aij <- Oj * ((1/Sji) - (1/M))
  diag(Aij) <- 0
  PWO <- (t(Oj) * Aij)/(rowSums(Aij) %*% t(rep(1, N)))
  return(PWO)
}

#Spatial decay function
decay<-function(x,pwo,beta0,beta1,beta2){
  return(tmp=1/(1+exp(beta0 + beta1*x + beta2*pwo)))
}

#returns probabilities that each node will be infected in the next time step
getprobs<-function(start,dist,mass,b,pwomat){

  #Check Trivial Cases
  if(all(start)){
    return(rep(1,length(start)))
  }
  if(!any(start)){
    return(rep(0,length(start)))
  }

  tmp=decay(dist[which(start),which(!start)], pwomat[which(start),which(!start)], b[1],b[2],b[3])
  tmp=tmp*(dist[which(start),which(!start)] != 0)

  #combine probabilities of infected nodes for each uninfected node
  if(!is.null(dim(tmp))){
    tmp=log(1-tmp)
    tmp=rep(1,dim(tmp)[1]) %*% tmp
    tmp=1-exp(tmp[1,])
  }else if(sum(!start)==1){
    tmp=1-exp(sum(log(1-tmp)))
  }

  #construct return vector
  ret=vector(length=length(start))
  ret[which(start)]=1
  ret[which(!start)]=tmp

  return(ret)
}

#calculate log likelihood for a single timestep
lglike<-function(p,d){
  for(i in 1:length(d)){
    if(!d[i]){
      p[i]=1-p[i]
    }
  }
  return(sum(log(p)))
}

#simulates infection spread over multiple time steps from a given starting point
simforeward<-function(dist, mass, beta, pwomat, start, first, last){
  ret=start
  for(i in first:last){
    #if all nodes are infected stop the simulation
    if(all(ret!=Inf)){
      return(ret)
    }

    #calculate probabilities of infection
    p=getprobs((ret<i),dist,mass,beta,pwomat)
    for(j in 1:length(ret)){
      #infect nodes with calculated probabilities
      if(start[j] == Inf & p[j] > runif(1,0,1)){
        ret[j] = i
      }
    }
    start=ret
  }
  return(ret)
}

#creates a sparse distance matrix
makedist<-function(coords, cutoff=Inf){
  dmat=sqrt(((coords[1,] %*% t(rep(1,length(coords[1,])))) - (rep(1,length(coords[1,])) %*% t(coords[1,])))**2 + ((coords[2,] %*% t(rep(1,length(coords[2,])))) - (rep(1,length(coords[2,])) %*% t(coords[2,])))**2)

  dmat=dmat*(dmat<cutoff)

  return(Matrix(dmat))
}

#calculates the negative log likelihood over all time steps (objective function for optimization)
objf<-function(beta, dist, mass, data, pwomat, days){
  ret=0

  for(i in 1:(days)){
    p=getprobs((data<i),dist,mass,beta,pwomat)
    ret=ret+lglike(p,(data<(i+1)))
  }
  
  return(-ret)
}

#find the best fit for beta by minimizing negative log likelihood
getbeta<-function(dist,mass,data,pwomat,start=c(5, 1, .0000001),days=max(data[which(data!=Inf)])){
  ret=optim(par=start,fn=objf,dist=dist, mass=mass, data=data, pwomat=pwomat, days=days)
  return(ret)
}

##
getdelta<-function(beta, dist, mass, data, pwomat, days=max(data[which(data!=Inf)])){
  require(mvtnorm)
  require(numDeriv)

  #Estimate Hessian at mle
  hess=hessian(objf, beta, dist=dist, mass=mass, data=data, pwomat=pwomat, days=days)
  #Invert Hessian to get Variance-Covariance Matrix
  A=solve(hess,diag(rep(1,length(beta))))
  #Calculate the 95th equicoordinate quantile
  q=qmvnorm(.95,mean=beta,sigma=A,tail='both',interval=c(5,15))$quantile
  #Get bounds
  ret=q*sqrt(diag(A))/length(beta)
  #Add names
  names=vector()
  for(i in 0:(length(beta)-1)){
    names=c(names,paste('delta',i,sep=''))
  }
  names(ret)<-names
  return(ret)
}

#Simulate multiple runs
makesims<-function(data, PID, pop, pwo, dist, beta, first=0, last=157, nsims=1000, seed=802){
  set.seed(seed)
  ret <- matrix(NA, nrow=length(PID), ncol=nsims+1)
  ret[,1]=PID
  names=rep(NA,(nsims+1))
  names[1]='PID'
  start=data
  start[which(start>first)]<-Inf
  for(i in 1:nsims){
    add=simforeward(dist=dist,mass=pop,beta=unlist(beta[i,]),pwomat=pwo,start=start,first=(first+1),last=last)
    ret[,(i+1)] <- add
    names[i+1] <- paste('Sim',i,sep='')
    print(i)
  }
  colnames(ret)<-names
  return(ret)
}
