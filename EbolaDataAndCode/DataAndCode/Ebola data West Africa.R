##Data import and visualization
library(sp)
library(shapefiles)
library(rgeos)
library(rgdal)
library(maptools)
library(raster)
library(plyr)
library(RColorBrewer)

##Adminstrative boundaries and population data

##First the adminstrative boundaries and population data
##Core countries of Guinea, Liberia and Sierra Leone are loaded first and cleaned to ensure agreement with reporting
load("WestAfrica GADM/GIN_adm2.RData"); guinea_prefec<-gadm; guinea_prefec$NAME_2<-ifelse(guinea_prefec$NAME_2=="Conarky","Conakry",guinea_prefec$NAME_2) ##Correct capital prefecture spelling
names(guinea_prefec)<-c(names(guinea_prefec[1:8]),"NL","VARNAME","TYPE","ENGTYPE")
guinea_prefec<-spChFIDs(guinea_prefec,as.character(guinea_prefec$PID))
guinea.afripop<-raster("Afripop/GIN14adjv1.tif")
guinea_prefec<-spTransform(guinea_prefec,CRS(projection(guinea.afripop)))
guinea.pop<-mask(guinea.afripop,guinea_prefec)
guinea.pop<-extract(guinea.pop,guinea_prefec,fun=sum,df=T,na.rm=T)
guinea_prefec<-spCbind(guinea_prefec,guinea.pop[,2]); names(guinea_prefec)[13]<-"pop.size"
load("WestAfrica GADM/LBR_adm1.RData"); liberia1<-gadm; liberia1$NAME_1<-ifelse(liberia1$NAME_1=="Gbapolu","Gbarpolu",liberia1$NAME_1) ##Corrected spelling to match official data
liberia<-spCbind(liberia1[,1:6],data.frame(ID_2=rep(NA,length(liberia1[,1])),NAME_2=rep(NA,length(liberia1[,1]))))
liberia<-spCbind(liberia,data.frame(liberia1[,7:10]))
names(liberia)<-c(names(liberia[1:8]),"NL","VARNAME","TYPE","ENGTYPE")
liberia<-spChFIDs(liberia,as.character(liberia$PID))
liberia.afripop<-raster("Afripop/LBR14adjv1.tif")
liberia<-spTransform(liberia,CRS(projection(liberia.afripop)))
liberia.pop<-mask(liberia.afripop,liberia)
liberia.pop<-extract(liberia.pop,liberia,fun=sum,df=T,na.rm=T)
liberia<-spCbind(liberia,liberia.pop[,2]); names(liberia)[13]<-"pop.size"
load("WestAfrica GADM/SLE_adm2.RData"); sierraleone<-gadm
names(sierraleone)<-c(names(sierraleone[1:8]),"NL","VARNAME","TYPE","ENGTYPE")
sierraleone<-spChFIDs(sierraleone,as.character(sierraleone$PID))
sierraleone.afripop<-raster("Afripop/SLE14adjv1.tif")
sierraleone<-spTransform(sierraleone,CRS(projection(sierraleone.afripop)))
sierraleone.pop<-mask(sierraleone.afripop,sierraleone)
sierraleone.pop<-extract(sierraleone.pop,sierraleone,fun=sum,df=T,na.rm=T)
sierraleone<-spCbind(sierraleone,sierraleone.pop[,2]); names(sierraleone)[13]<-"pop.size"
counties<-spRbind(sierraleone,spRbind(guinea_prefec,liberia))
county_names<-c(sierraleone$NAME_2,guinea_prefec$NAME_2,liberia$NAME_1)
counties<-spCbind(counties,county_names)

saveRDS(counties,"EbolaCountyPolygons.rds")

#Reload already constructed data
#counties<-readRDS("EbolaCountyPolygons.rds")

##The remaining countries in West Africa are appended
country.list<-c("BEN","BFA","CIV","GHA","GMB","GNB","MLI","NER","NGA","SEN","TGO")
country.list<-cbind(country.list,c(1,1,1,1,1,1,2,1,1,2,1))
for (j in 1:dim(country.list)[1]){
  load(paste("WestAfrica GADM/",country.list[j,1],"_adm",country.list[j,2],".RData",sep=""))

  pop.name<-ifelse(country.list[j,1]=="GMB",paste(country.list[j,1],"10adjv4.tif",sep=""),paste(country.list[j,1],"14adjv1.tif",sep=""))
  gadm.temp<-gadm
  if(country.list[j,2]==1){
    temp<-spCbind(gadm.temp[,1:6],data.frame(ID_2=rep(NA,length(gadm.temp[,1])),NAME_2=rep(NA,length(gadm.temp[,1]))))
    temp<-spCbind(temp,data.frame(gadm.temp[,7:10],pop.size=rep(NA,length(temp[,1])),county_names=rep(NA,length(temp[,1]))))
    names(temp)[9:12]<-c("NL","VARNAME","TYPE","ENGTYPE")
    temp<-spChFIDs(temp,as.character(temp$PID))

    temp$county_names<-temp$NAME_1 #This is the NAME_1 column that is the proper geographic division for this country
  } else {
    temp<-spCbind(gadm.temp,data.frame(pop.size=rep(NA,length(gadm.temp[,1])),county_names=rep(NA,length(gadm.temp[,1]))))
    names(temp)[9:12]<-c("NL","VARNAME","TYPE","ENGTYPE")
    temp<-spChFIDs(temp,as.character(temp$PID))

    temp$county_names<-temp$NAME_2 #This is the NAME_2 column that is the proper geographic division for this country
  }
  temp.afripop<-raster(paste("Afripop/",pop.name,sep=""))
  proj4string(temp.afripop)<-CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0")
  temp<-spTransform(temp,CRS(projection(temp.afripop)))
  temp.pop<-mask(temp.afripop,temp)
  temp.pop<-extract(temp.pop,temp,fun=sum,df=T,na.rm=T)
  temp$pop.size<-temp.pop[,2]
  counties<-spRbind(counties,temp)
}
outlines<-unionSpatialPolygons(counties,counties$ID_0)
MLI.correct.sp<-unionSpatialPolygons(counties[c(226:229),],rep(counties$PID[226],4)) #These are a single county in the mobility data
MLI.correct.df<-data.frame(counties[226,1:12],pop.size=sum(counties$pop.size[226:229]),county_names=counties$county_names[226]) #This is the associated data, using first county as the name
MLI.correct<-SpatialPolygonsDataFrame(MLI.correct.sp,MLI.correct.df)
counties <- spRbind(spRbind(counties[1:225,],MLI.correct),counties[-(1:229),]) #(rows 13,14,15,16 of Mali)
ids<-ids[!(ids[,2]=="Water body"),] ##Already removed this from "counties" dataframe
counties<-counties[!(counties$county_names=="Water body"),]##Removes "Water body" from Nigeria data, 
saveRDS(counties,"WestAfricaCountyPolygons.rds")

##Reload already constructed data
#counties<-readRDS("WestAfricaCountyPolygons.rds")
#core.counties<-readRDS("EbolaCountyPolygons.rds")

##Then add in the mobility data from Flowminder.org
mobility<-read.csv("mobility-data-1stSept14/AdmUnits_WBtwn.csv")
mobil.list<-c("BEN","BFA","CIV","LBR","GHA","GIN","GMB","GNB","MLI","NER","NGA","SEN","SLE","TGO")
mobil.ids<-NULL
for (i in mobil.list){
  if (i=="SLE") {shape<-readOGR(dsn=paste("Spatial_Data/AdminUnits/",i,sep=""),layer="sl_2004")
  }else{shape<-readOGR(dsn=paste("Spatial_Data/AdminUnits/",i,sep=""),layer=i)}
  if(i=="BEN"){ids<- cbind(as.character(shape$SP_ID),as.character(shape$NAME_1))
               ids[ids[,2]=="OuÃ©mÃ©",2]<-"Ouémé"
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="BFA"){ids<- cbind(as.character(shape$IPUMSID),as.character(shape$ADM2_NAME)) #These are relevant columns
               ids[ids[,2]=="Comoe",2]<-"Komoé" #Need to correct this spelling before reordering
               ids<-ids[order(ids[,2]),] #This puts in alphabetical order by department name
               ids<-ids[!(ids[,2]=="Noumbiel"),] ##This name isn't present in the current gadm even though maps draw the same
               ids[,2]<-counties$NAME_1[counties$ISO=="BFA"] ##This rectifies the spellings and accents, checked by eye
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="CIV"){ids<- cbind(as.character(shape$SP_ID),as.character(shape$NAME_1))
               ids<-ids[order(ids[,2]),] #This puts in alphabetical order by department name
               ids[,2]<-counties$NAME_1[counties$ISO=="CIV"] ##This rectifies the spellings and accents, checked by eye
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="LBR"){ids<- cbind(as.character(shape$SP_ID),as.character(shape$NAME_1))
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="GHA"){ids<- cbind(as.character(shape$IPUMSID),as.character(shape$gadm_name))
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="GIN"){ids<- cbind(as.character(shape$IPUMSID),as.character(shape$NAME_2))
               ids[,2]<-counties$NAME_2[counties$ISO=="GIN"] ##This rectifies the spellings and accents, checked by eye
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="GMB"){ids<- cbind(as.character(shape$SP_ID),as.character(shape$NAME_1))
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="GNB"){ids<- cbind(as.character(shape$SP_ID),as.character(shape$NAME_1))
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="MLI"){cent_ids<-signif(gCentroid(shape,byid=T)@coords,3) #get centroids because data file is missing labels
               cents<-signif(gCentroid(counties[counties$ISO=="MLI",],byid=T)@coords,3)
               test<-gDistance(SpatialPoints(cents),SpatialPoints(cent_ids),byid=T) #get the distance between centroids to match polygons
               MLI<-counties[counties$ISO=="MLI",]
               NAME_2<-rep(NA,length(shape))
               for (j in 1:length(shape)){
                 NAME_2[j]<-MLI$NAME_2[MLI$PID==colnames(test)[which.min(test[j,])]] #This makes a vector with the correct county name for each mobility polygon
               }
               ids<-cbind(shape$IPUMSID,NAME_2)
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="NER"){ids<- cbind(as.character(shape$SP_ID),as.character(shape$NAME_1))
               ids[,2]<-counties$NAME_1[counties$ISO=="NER"] ##This rectifies the spellings and accents, checked by eye
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }  
  if(i=="NGA"){ids<- cbind(as.character(shape$SP_ID),as.character(shape$NAME_1))
               ids<-ids[!(ids[,2]=="Water body"),] ##Already removed this from "counties" dataframe
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="SEN"){ids<- cbind(as.character(shape$IPUMSID),as.character(shape$NAME_2))
               ids[,2]<-counties$NAME_2[counties$ISO=="SEN"] ##This rectifies the spellings and accents, checked by eye
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="SLE"){ids<- cbind(as.character(shape$IPUMSID),as.character(shape$NAME_2))
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  if(i=="TGO"){ids<- cbind(as.character(shape$SP_ID),as.character(shape$NAME_1))
               ids<-data.frame(SID=ids[,1],county=ids[,2],loc=paste(ids[,1],i,sep=""),country=i)
  }
  mobil.ids<-rbind(mobil.ids,ids)
}
mobil.ids$county<-as.character(mobil.ids$county); mobil.ids$loc<-as.character(mobil.ids$loc); mobil.ids$country<-as.character(mobil.ids$country)
saveRDS(mobil.ids,"MobilityDataIDs.rds") ##This file is used to idenfity the mobility data used in the model fits

#Reload already constructed dataframe
#readRDS("MobilityDataIDs.rds")

#Infection dates for all countries
Infections<-read.csv("InfectedCountiesAllCountries.csv")  #This is Laura's data set, includes various start dates from first and second waves
Infections$First.infection<-as.Date(Infections$First.infection,"%m/%d/%Y")
Infections$Last.infection.spring<-as.Date(Infections$Last.infection,"%m/%d/%Y")

Infections$wave2.lastconf<-as.Date(Infections$wave2.lastconf,"%m/%d/%Y")
Infections$ForAnalysis<-as.Date(Infections$ForAnalysis,"%m/%d/%Y")

outbreak.date<-data.frame(county_names=counties$county_names,first=NA, first.recover=NA, second=NA, second.recover=NA)  ##The county_names matches the spatialpolygondataframe and should be used as the reference
for (i in 1:dim(outbreak.date)[1]){

  second<-ifelse(outbreak.date$county_names[i]%in%Infections$County,Infections$ForAnalysis[as.character(Infections$County)==as.character(outbreak.date$county_names[i])],NA)
  outbreak.date[i,2]<-c(second)
}
for(i in 2){
  outbreak.date[,i]<-as.Date(outbreak.date[,i],origin="1970-01-01") #Returns to correct format
}

##Version where second wave outbreak begins from Conakry and Gueckdou in April
outbreak.date[20,2]<-as.Date("2014-4-26")
outbreak.date[44,2]<-outbreak.date[20,4]
saveRDS(outbreak.date,"OutbreakDateByCounty_Summer_AllCountries.rds")

##Script that adds the WHO corrected, weekly dates to the data, these were released spring 2015
WHO.date<-read.csv("updated.infection.dates.csv")
colnames(WHO.date)<-c("Country","County","Analysis.date","WHO.susp","WHO.confirm","Agree","After.fit")

WHO.date$Analysis.date<-as.Date(new.dates$Analysis.date,"%m/%d/%Y")
WHO.date$WHO.confirm<-as.Date(new.dates$WHO.confirm,"%m/%d/%Y")
WHO.corrected<-NULL
for (i in 1:dim(outbreak.date)[1]){
  WHO.corrected[i]<- ifelse(outbreak.date$county_names[i]%in%WHO.date$County,WHO.date$WHO.confirm[as.character(WHO.date$County)==as.character(outbreak.date$county_names[i])],NA)
  if(i==196 | i==204) WHO.corrected[i]<-outbreak.date$second[i]
}

WHO.corrected<-as.Date(WHO.corrected,origin="1970-01-01") #Returns to correct format
weekly.WHO<-as.numeric(as.factor(WHO.corrected))
new.outbreak.date<-cbind(outbreak.date,WHO.corrected,weekly.WHO)
saveRDS(new.outbreak.date,"OutbreakDateByCounty_WHO_AllCountries.rds")

#These scripts can produce simple geographic visualizations of the spread
# spread.col<-colorRampPalette(c("darkorchid1","lightblue"))
# spread.col<-spread.col(6)
# spread.col<-heat.colors(6)
# month.infection<-as.numeric(format(outbreak.date$second,"%m"))
# 
# time.color<-rep("white",dim(counties)[1]); month<-NULL
# for (i in 1:dim(outbreak.date)[1]){
#     month[i]<-month.infection[i]  ##This assigns the month from outbreak.date
# 
#     if (outbreak.date$county_names[i]=="Conakry" | outbreak.date$county_names[i]=="Guéckédou") time.color[i]<-"red"
# 
#     if (!is.na(month[i]) & month[i]==5) time.color[i]<-spread.col[2]
#     if (!is.na(month[i]) & month[i]==6) time.color[i]<-spread.col[3]
#     if (!is.na(month[i]) & month[i]==7) time.color[i]<-spread.col[4]
#     if (!is.na(month[i]) & month[i]==8) time.color[i]<-spread.col[5]
#     if (!is.na(month[i]) & month[i]==9) time.color[i]<-spread.col[6]
# 
#   }
# 
# 
# ##Plot map with colors for different months
# png("West Africa spread map.png",width=6,height=5,units="in",res=300)
# plot(counties,col=time.color,main="Month of Infection")
# plot(outlines,border="black",lwd=2,add=T)
# 
# legend("topright",legend=c("April","May","June","July","August","September"),
#        fill=c("red",spread.col[2:6]))
# Dak.coord<-coordinates(counties[counties$county_names=="Dakar",]); Con.coord<-coordinates(counties[counties$county_names=="Conakry",])
# arrows(Dak.coord[1]+0.5,Dak.coord[2]+2.5,Dak.coord[1],Dak.coord[2],length=0.1,col=spread.col[5])
# text(x=Dak.coord[1]+0.5,y=Dak.coord[2]+2.5,"Dakar",col=spread.col[5],pos=3)
# arrows(Con.coord[1]-3,Con.coord[2]-1.5,Con.coord[1]-.2,Con.coord[2],length=0.1,col="red")
# text(x=Con.coord[1]-3,y=Con.coord[2]-1.5,"Conakry",col="red",pos=1)
# dev.off()
# 
# 
# if (!is.na(month[i]) & month[i]==5) time.color[i]<-"darkorange3"
# if (!is.na(month[i]) & month[i]==6) time.color[i]<-"darkorange"
# if (!is.na(month[i]) & month[i]==7) time.color[i]<-"orange"
# if (!is.na(month[i]) & month[i]==8) time.color[i]<-"gold2"
# if (!is.na(month[i]) & month[i]==9) time.color[i]<-"yellow"
# 
# 
# ##Plot labeled map for notes
# pdf("LabeledCountyMap.pdf",w=8,h=10.5)
# plot(core.counties)
# text(getSpPPolygonsLabptSlots(core.counties), labels=as.character(core.counties$county_names), cex=0.5)
# dev.off()


