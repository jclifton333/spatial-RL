 ##This script draws the expected number of infected counties after 6 months for each starting point
##The starting points are all the counties in the core countries
library(sp)

##Load the geographic data
counties<-readRDS('WestAfricaCountyPolygons.rds')
ord.counties<-data.frame(country=counties$ISO[1:63],pop.size=counties$pop.size[1:63],county_names=counties$county_names[1:63],PID=seq(1,63))

##Load the simulation data, a list with the day of infection for each WA county in each sim
##There are 100 sims for each
location.sims<-readRDS("newsims/locationsims.ggb.wa.RDS")


##Function to get number of counties infected, by sim, by origin
inf.sum<-function(data){
  data<-data[,-1] ##Remove the leading PID column
  data<-ifelse(data<Inf,1,0)
  return(colSums(data))
}
  
infect.sums<-sapply(location.sims,FUN=inf.sum)
#calculate quantiles for prediction intervals
q=apply(infect.sums,2,quantile,probs=c(.025,.25,.5,.75,.975))
max.inf<-max(q)
#Create data frame of quantiles
ord.counties<-data.frame(ord.counties,q2.5=q[1,],q25=q[2,],q50=q[3,],q75=q[4,],q97.5=q[5,])

##Reorder within each country by median number infected
order.sl<-sort(ord.counties$q50[1:14],decreasing=T,index.return=T)$ix
order.guin<-sort(ord.counties$q50[15:48],decreasing=T,index.return=T)$ix
order.lib<-sort(ord.counties$q50[49:63],decreasing=T,index.return=T)$ix
ord.counties1<-rbind(ord.counties[1:14,][order.sl,],ord.counties[15:48,][order.guin,],ord.counties[49:63,][order.lib,])


#Draw plot for manuscript
#pdf(width=7,height=4,"Fig 3_Originiation scenarios.pdf")
#jpeg(width=7,height=4,units="in",res=300,"Fig 3_Origination scenarios.jpg",quality=100)
png(width=7.5,height=4,units="in",res=300,"Fig 3_Origination scenarios.png")
#postscript(width=7.5,height=4,file="Figure 3 Origination scenarios.eps")

par(mar=c(6.5,3,0.25,0),las=1)
plot(NA,xlim=c(1,66),ylim=c(0,107),xlab='',ylab='',xaxt="n",main=NULL,bty='n',cex.axis=0.9)
bordercolor<-"white"
for(i in 1:63){
  if(i<15) {color<-"#3FBAE9" ##Colors match the country colors on the network map
            transp.color<-adjustcolor(color,alpha.f=0.2)
  }
  if(i>14 & i<49) {color<-"#568E57"    # Old color"#5E5E68"
                   transp.color<-adjustcolor(color,alpha.f=0.2)
  }
  if(i>48) {color<-"#F5B61E"
            transp.color<-adjustcolor(color,alpha.f=0.2)
  }
  rect((i-1),ord.counties1$q2.5[i],i,ord.counties1$q97.5[i],border=bordercolor,col=transp.color,lwd=0.5) # 95% confidance interval
  rect((i-1),ord.counties1$q25[i],i,ord.counties1$q75[i],border=bordercolor,col=color,lwd=0.5) # interquartile range
  rect((i-1),ord.counties1$q50[i],i,ord.counties1$q50[i],border=bordercolor,col=bordercolor,lwd=0.75) # median
}
highlight<-c(2,6,15,21,49)#These are Kenema, Western Urban, Conakry, Gueckedou, Montserrado
for(i in highlight){
  if(i<15) {color<-"#3FBAE9"
            transp.color<-adjustcolor(color,alpha.f=0.4)

  }
  if(i>14 & i<49) {color<-"#568E57"    #Old color "#5E5E68"
                   transp.color<-adjustcolor(color,alpha.f=0.4)

  }
  
  if(i>48) {color<-"#F5B61E"
            transp.color<-adjustcolor(color,alpha.f=0.4)

  }
  rect((i-1),ord.counties1$q2.5[i],i,ord.counties1$q97.5[i],border=bordercolor,col=transp.color,lwd=0.5) # 95% confidance interval
  rect((i-1),ord.counties1$q25[i],i,ord.counties1$q75[i],border=bordercolor,col=color,lwd=0.5) # interquartile range
  rect((i-1),ord.counties1$q50[i],i,ord.counties1$q50[i],border=bordercolor,col=bordercolor,lwd=0.75) # median
}
bordercolor<-"white"
rect(64.5,25.975,65.5,105,border=bordercolor,col=adjustcolor("#AED4B5",alpha.f=0.2),lwd=0.5) # 95% confidance interval
rect(64.5,43,65.5,62,border=bordercolor,col="#568E57",lwd=0.5) # interquartile range
rect(64.5,51,65.5,51,border=bordercolor,col=bordercolor,lwd=0.75) # median
# add actual nubmer infected as line, there were 56 infected counties on 10-22
points(x=65,y=56,pch=23,col="black",bg="white",cex=1.1)
text(64,73,col="#5E5E68",cex=0.66,"Actual infections\nby Oct. 24, 2014",adj=c(1,NA))
arrows(x0=62,x1=64,y0=67.5,y1=57.5,col="#5E5E68",length=0.05)
##x-axis
axis(1,at=seq(0.5,62.5,1),labels=FALSE)
axis(1,at=seq(64,66,1),labels=FALSE)
text(seq(1.5,63.5,1), -10, labels = ord.counties1$county_names, srt = 90, pos = 2, xpd = TRUE,cex=0.66)
text(66,-10,"Conakry + Guéckédou",srt=90, pos=2,xpd=TRUE,cex=0.66)
mtext(side=2,line=2,las=3,"Infected Counties",cex=0.9)
rect(xleft=1.25,xright=2.75,ybottom=90,ytop=93,col="#3FBAE9",border="white"); text(2.75,91.5,"Sierra Leone",col="#3FBAE9",cex=0.9,pos=4)
rect(xleft=25.5,xright=27,ybottom=90,ytop=93,col="#568E57",border="white"); text(26.5,91.5,"Guinea",col="#568E57",cex=0.9,pos=4)
rect(xleft=51.5,xright=53,ybottom=90,ytop=93,col="#F5B61E",border="white"); text(53,91.5,"Liberia",col="#F5B61E",cex=0.9,pos=4)
dev.off ()


##Accessory ANOVA to look at variance explained by initially infected unit
# library(reshape)
# colnames(infect.sums)<-counties$county_names[1:63]
# infect.data<-melt(infect.sums); colnames(infect.data)<-c("Sim","County","Infect.count")
# infect.var.exp<-lm(Infect.count~County,data=infect.data)
# summary(infect.var.exp)
