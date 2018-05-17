library(Matrix) #For sparse matricies

#Spatial decay function
decay<-function(x,cross,beta0,beta1,beta2){

  return(c(1,beta2)[cross+1]/(1+exp(beta0 + (beta1*x))))
 
}

makecross <- function(regions){
  return(outer(regions, regions, get("!=")))
}

makecross.group <- function(regions, group=c(5, 8, 13)){
  v <- regions %in% group
  return(outer(v, v, get("!=")))
}

#Calculate Probability of Infection
getprobs<-function(start,dist,cross,b){
  
  #Check Trivial Cases
  if(all(start)){
    return(rep(1,length(start)))
  }
  if(!any(start)){
    return(rep(0,length(start)))
  }

  #calculate probability of infection from each infected node node to each uninfected node
  tmp=decay(dist[which(start),which(!start)],cross[which(start),which(!start)],b[1],b[2],b[3])

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
simforeward<-function(dist, cross, beta, start, first, last){
  ret=start
  for(i in first:last){
    #if all nodes are infected stop the simulation
    if(all(ret!=Inf)){
      return(ret)
    }

    #calculate probabilities of infection
    p=getprobs((ret<i),dist,cross,beta)

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
  
  #calculate distances (matrix operations used for efficiancy at expense of readability)
  dmat=sqrt(((coords[1,] %*% t(rep(1,length(coords[1,])))) - (rep(1,length(coords[1,])) %*% t(coords[1,])))**2 + ((coords[2,] %*% t(rep(1,length(coords[2,])))) - (rep(1,length(coords[2,])) %*% t(coords[2,])))**2)

  #Set any distance larger than the cutoff to zero
  dmat=dmat*(dmat<cutoff)
  return(Matrix(dmat))
}

#calculates the negative log likelihood over all time steps (objective function for optimization)
objf<-function(beta, dist, cross, data, days){
  ret=0;

  for(i in 1:(days)){
    p=getprobs((data<i),dist,cross,beta)
    ret=ret+lglike(p,(data<(i+1)))
  }
  
  return(-ret)
}

#find the best fit for beta by minimizing negative log likelihood
getbeta<-function(dist,cross,data,start=c(5,1,.5),days=max(data[which(data!=Inf)])){
  ret=optim(par=start,fn=objf,dist=dist, cross=cross, data=data, days=days)
  return(ret)
}

##
getdelta<-function(beta, dist, cross,data, days=max(data[which(data!=Inf)])){
  require(mvtnorm)
  require(numDeriv)

  #Estimate Hessian at mle
  hess=hessian(objf, beta, dist=dist, cross=cross, data=data, days=days)
  #Invert Hessian to get Variance-Covariance Matrix
  A=solve(hess,diag(rep(1,length(beta))))
  #Calculate the 95th equicoordinate quantile
  q=qmvnorm(.95,mean=beta,sigma=A,tail='both')$quantile
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
makesims<-function(data, PID, cross, dist, beta, first=0, last=157, nsims=1000, seed=802){
  set.seed(seed)
  ret <- matrix(NA, nrow=length(PID), ncol=nsims+1)
  ret[,1]=PID
  names=rep(NA,(nsims+1))
  names[1]='PID'
  start=data
  start[which(start>first)]<-Inf
  for(i in 1:nsims){
    add=simforeward(dist=dist,cross=cross,beta=unlist(beta[i,]),start=start,first=(first+1),last=last)
    ret[,(i+1)] <- add
    names[i+1] <- paste('Sim',i,sep='')
    print(i)
  }
  colnames(ret)<-names
  return(ret)
}
