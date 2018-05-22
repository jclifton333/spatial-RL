collect.wa <- function(cutoff=TRUE){
  require(maptools)
  require(PBSmapping)

  polys <- readRDS("WestAfricaCountyPolygons.rds")
  centroids <- calcCentroid(SpatialPolygons2PolySet(polys), rollup=1)
  region <- as.integer(as.factor(polys$ISO))

  outbreak <- readRDS("OutbreakDateByCounty_Summer_AllCountries.rds")

  infectionDate <- as.integer(outbreak$infection_date)
  infectionDate[which(is.na(infectionDate))] <- Inf
  infectionDate <- infectionDate - min(infectionDate)
  if(cutoff){
    infectionDate[which(infectionDate>157)] <- Inf
  }

  out <- data.frame(	county_names = outbreak$county_names, 
			PID = centroids$PID,
			region = region,
			x = centroids$X,
			y = centroids$Y,
			pop = polys$pop.size,
			infectionDate=infectionDate)

  return(out)
}

get.mobility.wa <- function(dataset=c("civ", "ipums", "kenya", "sen")){
  table <- readRDS('MobilityDataIDs.rds')
  btwn <- read.csv('AdmUnits_WBtwn.csv', as.is=TRUE)
  polys <- readRDS("WestAfricaCountyPolygons.rds")
  table$county[which(table$county=="Gbapolu")] <- "Gbarpolu"
  table$county[which(table$county=="BafatÃ¡")]<-"Bafatá"
  table$county[which(table$county=="GabÃº")]<-"Gabú"

  n <- nrow(polys)
  ref <-rep(NA, n)
  for(i in 1:n){
    ref[i] <- table$loc[which((table$county == polys$county_names[i])&(table$country == polys$ISO[i]))]
  }

  model <- paste("predict_", match.arg(dataset), sep='')

  ret <- matrix(NA, n, n)
  for(i in 1:n){
    for(j in 1:n){
      if(i==j) next
      ret[i,j] <- btwn[[model]][which((btwn$from_loc == ref[i])&(btwn$to_loc == ref[j]))]
    }
  }

  diag(ret) <- 0

  return(ret)
}
