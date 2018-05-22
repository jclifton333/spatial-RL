library(Matrix)

#returns probabilities that each node will be infected in the next time step
getprobs<-function(start,dist,mass,b){

  #Check Trivial Cases
  if(all(start)){
    return(rep(1,length(start)))
  }
  if(!any(start)){
    return(rep(0,length(start)))
  }

  S=which(!start)
  I=which(start)
  tmp <- rep(0,length(start))
  tmp[I] <- -Inf

  for(i in S){
    tmp[i] <- b[1]*mass[i]**b[2]*sum((mass[I]**b[3])/(dist[I,i]**b[4]))
  }

  ret=1-exp(tmp)
  ret[start] <- 1
  ret[which(ret<0)] <- 0

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

#simulates infection spread over multiple years from a given starting point
simforeward<-function(dist, mass, beta, start, first, last){
  ret=start
  for(i in first:last){
    #if all nodes are infected stop the simulation
    if(all(ret!=Inf)){
      return(ret)
    }

    #calculate probabilities of infection
    p=getprobs((ret<i),dist,mass,beta)
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

#plot node locations
plotcoords<-function(coords,data=0){
  if(sum(data)){
    plot(x=coords[1,which(as.logical(data))],y=coords[2,which(as.logical(data))],xlim=c(min(coords[1,]),max(coords[1,])),ylim=c(min(coords[2,]),max(coords[2,])), xlab='', ylab='',col='red')
    points(x=coords[1,which(!as.logical(data))],y=coords[2,which(!as.logical(data))],col='blue')
  }else{
    plot(x=coords[1,],y=coords[2,],xlab='',ylab='')
  }
}

#add node locations to existing plot
addnodes<-function(coords,data=0){
  if(sum(data)){
    points(x=coords[1,which(!data)],y=coords[2,which(!as.logical(data))],col='blue')
    points(x=coords[1,which(data)],y=coords[2,which(as.logical(data))],col='red')
  }else{
    points(x=coords[1,],y=coords[2,])
  }
}

#plot spread of infection over time
plotspread<-function(coords,data,add=FALSE){
  y=max(data[which(data!=Inf)])+1

  if(!add){
    plot(c(0,max(coords)),c(0,max(coords)),col='white', xlab='', ylab='')
  }

  for(i in 1:y){
    addnodes(coords,(data<i))
    readline("Press <Enter> to continue")
  }
}

#creates a sparse distance matrix
makedist<-function(coords, cutoff=Inf){
  dmat=sqrt(((coords[1,] %*% t(rep(1,length(coords[1,])))) - (rep(1,length(coords[1,])) %*% t(coords[1,])))**2 + ((coords[2,] %*% t(rep(1,length(coords[2,])))) - (rep(1,length(coords[2,])) %*% t(coords[2,])))**2)

  dmat=dmat*(dmat<cutoff)

  return(Matrix(dmat))
}

#calculates the negative log likelihood over all years (objective function for optimization)
objf<-function(beta, dist, mass, data, years){
  ret=0

  for(i in 1:(years)){
    p=getprobs((data<i),dist,mass,beta)
    ret=ret+lglike(p,(data<(i+1)))
  }
  
  return(-ret)
}

#find the best fit for beta by minimizing negative log likelihood
getbeta<-function(dist,mass,data,start=c(-10,1,-1,-1),years=max(data[which(data!=Inf)])){
  ret=optim(par=start,fn=objf,dist=dist, mass=mass, data=data, years=years)
  return(ret)
}

##
getdelta<-function(beta, dist, mass, data, years=max(data[which(data!=Inf)])){
  require(mvtnorm)
  require(numDeriv)

  #Estimate Hessian at mle
  hess=hessian(objf, beta, dist=dist, mass=mass, data=data, years=years)
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
makesims<-function(data, PID, pop, dist, beta, first=0, last=300, nsims=1000, seed=802){
  set.seed(seed)
  ret <- matrix(NA, nrow=length(PID), ncol=nsims+1)
  ret[,1]=PID
  names=rep(NA,(nsims+1))
  names[1]='PID'
  start=data
  start[which(start>first)]<-Inf
  for(i in 1:nsims){
    add=simforeward(dist=dist,mass=pop,beta=unlist(beta[i,]),start=start,first=(first+1),last=last)
    ret[,(i+1)] <- add
    names[i+1] <- paste('Sim',i,sep='')
    print(i)
  }
  colnames(ret)<-names
  return(ret)
}

