library(foreign)
library(sp)
library(maptools)
require(stats4)
library(spatstat)
library(rgeos)
library(geosphere)


##Read in the simulations from the initial conditions
wa.before.sims<-list()
wa.before.files<-list.files(path="newsims",pattern="sims.init.")
partial<-unlist(strsplit(wa.before.files,split="sims.init.",fixed=TRUE))
i<-1
partial1<-partial[1:(i+1)==(i+1)] #This elimates blanks from the "partial" list to get the file names
wa.before.models<-unlist(strsplit(partial1,split=".wa.RDS"))
for (i in 1:length(wa.before.files)) wa.before.sims[[i]]<-readRDS(paste("newsims/",wa.before.files[i],sep=""))

#Read in the simulations from the Oct 1st conditions
wa.after.sims<-list()
wa.after.files<-list.files(path="newsims",pattern="sims.after.")
partial<-unlist(strsplit(wa.after.files,split="sims.after.",fixed=TRUE))
i<-1
partial1<-partial[1:(i+1)==(i+1)] #This elimates blanks from the "partial" list to get the file names
wa.after.models<-unlist(strsplit(partial1,split=".wa.RDS"))
for (i in 1:length(wa.after.files)) wa.after.sims[[i]]<-readRDS(paste("newsims/",wa.after.files[i],sep=""))


#These are the numeric dates (after beginning of outbreak observed for each dataset)
infected.wa <- readRDS("OutbreakDateByCounty_Summer_AllCountries.rds"); observed.inf.wa <- infected.wa$second-min(infected.wa$second, na.rm=T)

###Geographic information for core counties 
core.counties <- readRDS("EbolaCountyPolygons.rds")
counties <- readRDS("WestAfricaCountyPolygons.rds")
dist.core<-spDists(core.counties, longlat=TRUE)
dist.wa<-spDists(counties, longlat=TRUE)
dist.Gueck.core<-dist.core[,which(counties$county_names=="Guéckédou")]
dist.Gueck.wa<-dist.wa[,which(counties$county_names=="Guéckédou")]
dist.Conakry.core<-dist.core[,which(counties$county_names=="Conakry")]
dist.Conakry.wa<-dist.wa[,which(counties$county_names=="Conakry")]
core.centroids1<-coordinates(core.counties); core.centroids<-SpatialPoints(core.centroids1,proj4string=CRS(proj4string(counties)))
centroids1<-coordinates(counties); centroids<-SpatialPoints(centroids1,proj4string=CRS(proj4string(counties)))

##The following code calculates summary statistics (total infected, max distance, median distance and convex hull) for outbreak and simulations
################## COUNT CUMULATIVE NUMBER OF COUNTIES ##################
####### OBSERVED
real.counties.wa <- NULL; dist.max.Gueck.wa <- NULL; dist.max.Conakry.wa <- NULL
dist.med.Gueck.wa <- NULL; dist.med.Conakry.wa <- NULL; chull.wa <- NULL
for (i in 1:157) {
  real.counties.wa[i] <- sum(observed.inf.wa<=i,na.rm=T)
  dist.max.Gueck.wa[i] <- max(dist.Gueck.wa[which(observed.inf.wa<=i)],na.rm=T)
  dist.max.Conakry.wa[i] <- max(dist.Conakry.wa[which(observed.inf.wa<=i)],na.rm=T)
  dist.med.Gueck.wa[i] <- median(dist.Gueck.wa[which(observed.inf.wa<=i)],na.rm=T)
  dist.med.Conakry.wa[i] <- median(dist.Conakry.wa[which(observed.inf.wa<=i)],na.rm=T)
  convex.hull<-gConvexHull(centroids[which(observed.inf.wa<=i),])
  if (real.counties.wa[i]>2){ 
    hull.pts<-rbind(convex.hull@polygons[[1]]@Polygons[[1]]@coords,convex.hull@polygons[[1]]@Polygons[[1]]@coords[1,])
    chull.wa[i]<-areaPolygon(hull.pts,r=6378137/1000) 
  }else {chull.wa[i]<-0}
}
observed.wa<-list(real.counties=real.counties.wa,dist.max.Gueck=dist.max.Gueck.wa,
                      dist.max.Conakry=dist.max.Conakry.wa,dist.med.Gueck=dist.med.Gueck.wa,
                      dist.med.Conakry=dist.med.Conakry.wa,chull.area=chull.wa)
saveRDS(observed.wa,"ObservedWaDataMarch.RDS")


####### SIMULATIONS
#count cumulative number of counties (by day)
cumulative <- function(infection.dates,period="before",model="summer"){
  num.sims<-dim(infection.dates)[2]-1
  if(period=="before"){
        max.time<-157
        cents<-centroids
        dists.Gueck<-dist.Gueck.wa
        dists.Conakry<-dist.Conakry.wa
  }else{
        max.time<-as.Date("06/30/2015","%m/%d/%Y")-min(infected.wa$second,na.rm=T)
        cents<-centroids
        dists.Gueck<-dist.Gueck.wa
        dists.Conakry<-dist.Conakry.wa
  }
  cumulative.counties.med<-NULL
  u.l.cumulative.counties<-NULL
  l.l.cumulative.counties<-NULL
  cumulative.counties<-matrix(nrow=max.time,ncol=num.sims)
  for (j in 1:num.sims){
    for (i in 1:max.time){
      cumulative.counties[i,j] <- sum(infection.dates[,j+1]<=i)
    }
  }
  for (i in 1:max.time) cumulative.counties.med[i]<-median(cumulative.counties[i,])
  for (i in 1:max.time) u.l.cumulative.counties[i]<-quantile(cumulative.counties[i,],0.975)
  for (i in 1:max.time) l.l.cumulative.counties[i]<-quantile(cumulative.counties[i,],0.025)
  max.distance.Gueck.med<-NULL
  u.l.max.distance.Gueck<-NULL
  l.l.max.distance.Gueck<-NULL
  max.distance.Gueck<-matrix(nrow=max.time,ncol=num.sims)
  for (j in 1:num.sims){
    for (i in 1:max.time){
      max.distance.Gueck[i,j] <- max(dists.Gueck[which(infection.dates[,j+1]<=i)],na.rm=T)
    }
  }
  for (i in 1:max.time) max.distance.Gueck.med[i]<-median(max.distance.Gueck[i,])
  for (i in 1:max.time) u.l.max.distance.Gueck[i]<-quantile(max.distance.Gueck[i,],0.975)
  for (i in 1:max.time) l.l.max.distance.Gueck[i]<-quantile(max.distance.Gueck[i,],0.025)
  
  max.distance.Conakry.med<-NULL
  u.l.max.distance.Conakry<-NULL
  l.l.max.distance.Conakry<-NULL
  max.distance.Conakry<-matrix(nrow=max.time,ncol=num.sims)
  for (j in 1:num.sims){
    for (i in 1:max.time){
      max.distance.Conakry[i,j] <- max(dists.Conakry[which(infection.dates[,j+1]<=i)],na.rm=T)
    }
  }
  for (i in 1:max.time) max.distance.Conakry.med[i]<-median(max.distance.Conakry[i,])
  for (i in 1:max.time) u.l.max.distance.Conakry[i]<-quantile(max.distance.Conakry[i,],0.975)
  for (i in 1:max.time) l.l.max.distance.Conakry[i]<-quantile(max.distance.Conakry[i,],0.025)
  
  med.distance.Gueck.med<-NULL
  u.l.med.distance.Gueck<-NULL
  l.l.med.distance.Gueck<-NULL
  med.distance.Gueck<-matrix(nrow=max.time,ncol=num.sims)
  for (j in 1:num.sims){
    for (i in 1:max.time){
      med.distance.Gueck[i,j] <- median(dists.Gueck[which(infection.dates[,j+1]<=i)],na.rm=T)
    }
  }
  for (i in 1:max.time) med.distance.Gueck.med[i]<-median(med.distance.Gueck[i,])
  for (i in 1:max.time) u.l.med.distance.Gueck[i]<-quantile(med.distance.Gueck[i,],0.975)
  for (i in 1:max.time) l.l.med.distance.Gueck[i]<-quantile(med.distance.Gueck[i,],0.025)
  
  med.distance.Conakry.med<-NULL
  u.l.med.distance.Conakry<-NULL
  l.l.med.distance.Conakry<-NULL
  med.distance.Conakry<-matrix(nrow=max.time,ncol=num.sims)
  for (j in 1:num.sims){
    for (i in 1:max.time){
      med.distance.Conakry[i,j] <- median(dists.Conakry[which(infection.dates[,j+1]<=i)],na.rm=T)
    }
  }
  for (i in 1:max.time) med.distance.Conakry.med[i]<-median(med.distance.Conakry[i,])
  for (i in 1:max.time) u.l.med.distance.Conakry[i]<-quantile(med.distance.Conakry[i,],0.975)
  for (i in 1:max.time) l.l.med.distance.Conakry[i]<-quantile(med.distance.Conakry[i,],0.025)
  
  chull.area.med<-NULL
  u.l.chull.area<-NULL
  l.l.chull.area<-NULL
  chull.area<-matrix(nrow=max.time,ncol=num.sims)
  for (j in 1:num.sims){
    for (i in 1:max.time){
      convex.hull<-gConvexHull(cents[which(infection.dates[,j+1]<=i),])
      if (sum(infection.dates[,j+1]<=i)>2){ 
        hull.pts<-rbind(convex.hull@polygons[[1]]@Polygons[[1]]@coords,convex.hull@polygons[[1]]@Polygons[[1]]@coords[1,])
        chull.area[i,j]<-areaPolygon(hull.pts,r=6378137/1000) 
      }else {chull.area[i,j]<-0}
    }
  }
  for (i in 1:max.time) chull.area.med[i]<-median(chull.area[i,])
  for (i in 1:max.time) u.l.chull.area[i]<-quantile(chull.area[i,],0.975)
  for (i in 1:max.time) l.l.chull.area[i]<-quantile(chull.area[i,],0.025)
  
  return(list(median.inf=cumulative.counties.med,u.l.inf=u.l.cumulative.counties,l.l.inf=l.l.cumulative.counties,
              max.dist.Gueck.med=max.distance.Gueck.med,u.l.max.dist.Gueck=u.l.max.distance.Gueck,l.l.max.dist.Gueck=l.l.max.distance.Gueck,
              max.dist.Conakry.med=max.distance.Conakry.med,u.l.max.dist.Conakry=u.l.max.distance.Conakry,l.l.max.dist.Conakry=l.l.max.distance.Conakry,
              med.dist.Gueck.med=med.distance.Gueck.med,u.l.med.dist.Gueck=u.l.med.distance.Gueck,l.l.med.dist.Gueck=l.l.med.distance.Gueck,
              med.dist.Conakry.med=med.distance.Conakry.med,u.l.med.dist.Conakry=u.l.med.distance.Conakry,l.l.med.dist.Conakry=l.l.med.distance.Conakry,
              chull.area.med=chull.area.med,u.l.chull.area=u.l.chull.area,l.l.chull.area=l.l.chull.area))
}

summary.wa.before.sims<-lapply(wa.before.sims,cumulative,period="before",model="wa")
names(summary.wa.before.sims)<-wa.before.models
saveRDS(summary.wa.before.sims,"summary.wa.before.RDS")
summary.wa.after.sims<-lapply(wa.after.sims,cumulative,period="after",model="wa")
names(summary.wa.after.sims)<-wa.after.models
saveRDS(summary.wa.after.sims,"summary.wa.after.RDS")

##Read above data (takes~4 days to aggregate from scratch)
#summary.wa.before.sims<-readRDS("summary.wa.before.RDS") 
#summary.wa.after.sims<-readRDS("summary.wa.after.RDS")

