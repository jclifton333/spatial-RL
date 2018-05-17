library(Matrix)

#Spatial decay function
decay<-function(x,mass,cross,beta0,beta1,beta2,beta3){
  return(c(1,beta3)[cross+1]/(1+exp(beta0 + (beta1*x)/(mass**beta2))))
}

makecross <- function(regions){
  return(outer(regions, regions, get("!=")))
}

makecross.group <- function(regions, group=c(5, 8, 13)){
  v <- regions %in% group
  return(outer(v, v, get("!=")))
}

#returns probabilities that each node will be infected in the next time step
getprobs<-function(start,dist,mass,cross,b){

  #Check Trivial Cases
  if(all(start)){
    return(rep(1,length(start)))
  }
  if(!any(start)){
    return(rep(0,length(start)))
  }

  #calculate probability of infection from each infected node node to each uninfected node
  mass=mass%*%t(mass)
  tmp=decay(dist[which(start),which(!start)],mass[which(start),which(!start)],cross[which(start),which(!start)],b[1],b[2],b[3],b[4])
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

#simulates infection spread over multiple years from a given starting point
simforeward<-function(dist, mass, cross, beta, start, first, last){
  ret=start
  for(i in first:last){
    #if all nodes are infected stop the simulation
    if(all(ret!=Inf)){
      return(ret)
    }

    #calculate probabilities of infection
    p=getprobs((ret<i),dist,mass,cross,beta)
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
    points(x=coords[1,which(!data)],y=coords[2,which(!as.logical(data))],col='blue', pch=19)
    points(x=coords[1,which(data)],y=coords[2,which(as.logical(data))],col='red', pch=19)
  }else{
    points(x=coords[1,],y=coords[2,])
  }
}

#plot connectivity network
plotconnect<-function(coords,dist,mass,cross,beta,edgescale=1,add=FALSE){
  n=dim(coords)[2]
  mass=mass%*%t(mass)

  if(!add){
    plot(c(0,max(coords)),c(0,max(coords)),col='white', xlab='', ylab='')
  }
  
  for(i in 2:n){
    for(j in 1:(i-1)){
      if(dist[i,j] != 0){
        lines(coords[1,c(i,j)],coords[2,c(i,j)],lwd=edgescale*decay(dist[i,j],mass[i,j],cross[i,j],beta[1],beta[2],beta[3],beta[4]))
      }
    }
  }
}

#plot spread of infection over time
plotspread<-function(coords,data,add=FALSE){
  y=max(data[which(data!=Inf)])+1

  if(!add){
    plot(c(min(coords),max(coords)),c(min(coords),max(coords)),col='white', xlab='', ylab='')
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
objf<-function(beta, dist, mass, cross, data, years){
  ret=0

  for(i in 1:(years)){
    p=getprobs((data<i),dist,mass,cross,beta)
    ret=ret+lglike(p,(data<(i+1)))
  }
  
  return(-ret)
}

#find the best fit for beta by minimizing negative log likelihood
getbeta<-function(dist,mass,cross,data,start=c(5.1,127,.18,.2),years=max(data[which(data!=Inf)])){
  ret=optim(par=start,fn=objf,dist=dist, mass=mass, cross=cross, data=data, years=years)
  return(ret)
}

##
getdelta<-function(beta, dist, mass,cross,  data, years=max(data[which(data!=Inf)])){
  require(mvtnorm)
  require(numDeriv)

  #Estimate Hessian at mle
  hess=hessian(objf, beta, dist=dist, mass=mass, cross=cross, data=data, years=years)
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
makesims<-function(data, PID, pop, cross, dist, beta, first=0, last=300, nsims=1000, seed=802){
  set.seed(seed)
  ret <- matrix(NA, nrow=length(PID), ncol=nsims+1)
  ret[,1]=PID
  names=rep(NA,(nsims+1))
  names[1]='PID'
  start=data
  start[which(start>first)]<-Inf
  for(i in 1:nsims){
    add=simforeward(dist=dist,mass=pop,cross=cross,beta=unlist(beta[i,]),start=start,first=(first+1),last=last)
    ret[,(i+1)] <- add
    names[i+1] <- paste('Sim',i,sep='')
    #print(i)
  }
  colnames(ret)<-names
  return(ret)
}

