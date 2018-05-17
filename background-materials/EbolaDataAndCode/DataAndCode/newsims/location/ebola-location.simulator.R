library(snow)
source('ebola-GravNet-boarder.R')

collect.wa <- function(cutoff=TRUE){
  require(maptools)
  require(PBSmapping)

  polys <- readRDS("WestAfricaCountyPolygons.rds")
  centroids <- calcCentroid(SpatialPolygons2PolySet(polys), rollup=1)
  region <- as.integer(as.factor(polys$ISO))

  outbreak <- readRDS("OutbreakDateByCounty_Summer_AllCountries.rds")

  first <- as.integer(outbreak$first)
  first[which(is.na(first))] <- Inf
  first <- first - min(first)
  first.recover <- as.integer(outbreak$first.recover)
  first.recover[which(is.na(first.recover))] <- Inf
  first.recover <- first.recover - min(first.recover)
  second <- as.integer(outbreak$second)
  second[which(is.na(second))] <- Inf
  second <- second - min(second)
  if(cutoff){
    second[which(second>157)] <- Inf
  }
  second.recover <- as.integer(outbreak$second.recover)
  second.recover[which(is.na(second.recover))] <- Inf
  second.recover <- second.recover - min(second.recover)

  out <- data.frame(	county_names = outbreak$county_names, 
			PID = centroids$PID,
			region = region,
			x = centroids$X,
			y = centroids$Y,
			pop = polys$pop.size,
			first = first,
			first.recover=first.recover,
			second=second,
			second.recover=second.recover)

  return(out)
}

location.simulator <- function(location, PID, pop, cross, dist, beta=c(5.7924, 105.67, 0.18597, 0.15021), days=183, nsims=100, seed=802){
  start <- rep(Inf,290)
  start[location] <- 0

  b <- matrix(beta, nrow=nsims, ncol=length(beta), byrow=TRUE)

  sims <- makesims(data=start, PID=PID, pop=pop, cross=cross, dist=dist, beta=b, first=0, last=days, nsims=nsims, seed=seed)
  return(sims)
}



data <- collect.wa()
locations <- which(data$region %in% c(5,8,13))
dist <- makedist(rbind(data$x,data$y))
cross <- makecross.group(data$region)
beta <- c(5.7924, 105.67, 0.18597, 0.15021)

mycluster=makeCluster(12, type='SOCK')
clusterExport(mycluster, c('makesims', 'simforeward', 'getprobs', 'decay'))

#weird error on this line.  run again and works
location.sims <- clusterApplyLB(mycluster, locations, location.simulator, PID=data$PID, pop=data$pop, cross=cross, dist=dist)

stopCluster(mycluster)

