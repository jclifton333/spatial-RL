#Manuscript plots
library(xtable)

##Read in data (takes~4 days to aggregate from scratch)
##Each of these has 26 or 20 different model configurations
summary.wa.before.sims<-readRDS("summary.wa.before.RDS")
summary.wa.after.sims<-readRDS("summary.wa.after.RDS")
observed.wa<-readRDS("ObservedWaData.RDS")
outbreak <- readRDS("OutbreakDateByCounty_Summer_AllCountries.rds")
second <- as.integer(outbreak$second)
second[which(is.na(second))] <- Inf
second <- second - min(second)
infected <- which(second!=Inf)

#Infected colors from network map
colorfn <- colorRampPalette(c("#e71d0e","#f6e52d"))
colors <- colorfn(max(second[infected])+1)
colors <- paste(colors,"BF",sep='')
map.leg.colors<-colors[c(1,119,40,158,79)]
greyT<-rgb(204,204,204,alpha=100,m=255)
dataT<-rgb(239,148,32,alpha=100,m=255)

##Date labels for plots
label.wa<-as.Date(rep(NA,430))
for (i in 1:430) label.wa[i]<-as.Date("2014-04-25")+i



##########Now draw figures
#####Will have a set of figures for each data set

###West Africa data set, save all in pdf

#Set order of outputs, based on names(wa.before.models)
mod.order<-c(11,9,12,1,2,4,5,6,7,8,10,13,3,14,15,16)
#Official labels in the order of the models in the data, will be called in mod.order in script
mod.names<-c("Well-mixed","Constant","Mobility (Senegal) + diffusion","Diffusion","Diffusion + country borders",
             "Diffusion + core borders","Force of infection (unnormalized)","Force of infection (normalized)",
             "Gravity","Gravity + country borders","Gravity + core borders","Gravity + long distance",
             "Mobility (Senegal)","Diffusion + long distance","Population weighted opportunity","Radiation")
#Time axis
time<-seq(1,430)
pdf(file="Ebola simulation plots_West Africa_ms.pdf",width=3.504,height=7.08,onefile=T)
par(mfrow=c(4,1),mar=c(0,0,0,0),oma=c(5,5.5,0,0.4),xaxs="i",yaxs="i",las=1,mgp=c(2,0.5,0)) #Plot format options
for (i in mod.order){ ###########i = 11 is the gravity-group-border model used in manuscript###########
  before.data<-summary.wa.before.sims[[i]]
  after.data<-summary.wa.after.sims[[i]]
  ##First panel, cumulative infected units over time
  Counties.Fig<-plot(before.data$median.inf~time[1:157],
                     xlim=c(1,430),ylim=c(0,max(after.data$u.l.inf)+0.04*max(after.data$u.l.inf)),
                     type="l",bty="l",xaxt="n",cex.axis=0.9,ylab=NA,xlab=NA)
  polygon(x=c(1,157,157,1),
         y=c(0,0,max(after.data$u.l.inf)+0.04*max(after.data$u.l.inf),max(after.data$u.l.inf)+0.04*max(after.data$u.l.inf)),
         col=dataT,border=F) ##This outlines the time period that was used to fit model
  abline(v=1)
  lines(before.data$median.inf~time[1:157],lwd=2,col="black")
  axis(1,at=c(100,200,300,400),labels=NA)
  mtext("Total units infected",side=2,line=3.1,las=0,cex=0.8)
  lines(before.data$u.l.inf~time[1:157],lty=2,col="black")
  lines(before.data$l.l.inf~time[1:157],lty=2,col="black")
  points(observed.wa$real.counties~time[1:157],pch=19,cex=0.8,col=map.leg.colors[3])
  lines(after.data$median.inf[157:430]~time[157:430],lwd=2,col="grey")
  polygon(c(time[157:430],rev(time[157:430])),c(after.data$u.l.inf[157:430],rev(after.data$l.l.inf[157:430])),
          col=greyT,border=greyT) #Predictive interval outline
  text(x=2,y=270,"Data",col=map.leg.colors[3],pos=4)
  text(x=158,y=270,"Projection",col="grey",pos=4)
  ##Second panel, max distance over time, from Gueckodou
  Max.Fig<-plot(before.data$max.dist.Gueck.med~time[1:157],
                xlim=c(1,430),ylim=c(350,max(c(max(after.data$u.l.max.dist.Gueck)+0.04*max(after.data$u.l.max.dist.Gueck),
                                               max(observed.wa$dist.max.Gueck)+0.04*max(observed.wa$dist.max.Gueck)))),
                type="l",bty="l",xaxt="n",cex.axis=0.9,ylab=NA,xlab=NA)
  mtext("Maximum distance \n from Guéckédou (km)", side=2,line=2.5, las=0,cex=0.8)
  polygon(x=c(1,157,157,1),
         y=c(350,350,
             max(c(max(after.data$u.l.max.dist.Gueck)+0.04*max(after.data$u.l.max.dist.Gueck),
                           max(observed.wa$dist.max.Gueck)+0.04*max(observed.wa$dist.max.Gueck))),
             max(c(max(after.data$u.l.max.dist.Gueck)+0.04*max(after.data$u.l.max.dist.Gueck),
                   max(observed.wa$dist.max.Gueck)+0.04*max(observed.wa$dist.max.Gueck)))),
         col=dataT,border=F)
  abline(v=1)
  abline(h=max(c(max(after.data$u.l.max.dist.Gueck)+0.04*max(after.data$u.l.max.dist.Gueck),
                              max(observed.wa$dist.max.Gueck)+0.04*max(observed.wa$dist.max.Gueck))))
  lines(before.data$max.dist.Gueck.med~time[1:157],lwd=2,col="black")
  axis(1,at=c(100,200,300,400),labels=NA)
  lines(before.data$u.l.max.dist.Gueck~time[1:157],lty=2,col="black")
  lines(before.data$l.l.max.dist.Gueck~time[1:157],lty=2,col="black")
  points(observed.wa$dist.max.Gueck~time[1:157],pch=19,cex=0.8,col=map.leg.colors[3])
  lines(after.data$max.dist.Gueck.med[157:430]~time[157:430],lwd=2,col="grey")
  polygon(c(time[157:430],rev(time[157:430])),c(after.data$u.l.max.dist.Gueck[157:430],rev(after.data$l.l.max.dist.Gueck[157:430])),
          col=greyT,border=greyT)
  axis(1,at=c(100,200,300,400),labels=NA,cex.axis=0.8)

  ##Third panel median distance from Gueck
  Med.Fig<-plot(before.data$med.dist.Gueck.med~time[1:157],
                xlim=c(1,430),ylim=c(0,max(c(max(c(after.data$u.l.med.dist.Gueck, before.data$u.l.med.dist.Gueck))+0.04*max(c(after.data$u.l.med.dist.Gueck, before.data$u.l.med.dist.Gueck)),
                                             max(observed.wa$dist.med.Gueck)+0.04*max(observed.wa$dist.max.Gueck)))),
                type="l",bty="l",xaxt="n",cex.axis=0.9,ylab=NA,xlab=NA)
  mtext("Median distance \n from Guéckédou (km)", side=2,line=2.5, las=0,cex=0.8)
  polygon(x=c(1,157,157,1),
          y=c(0,0,
              max(c(max(c(after.data$u.l.med.dist.Gueck, before.data$u.l.med.dist.Gueck))+0.04*max(c(after.data$u.l.med.dist.Gueck, before.data$u.l.med.dist.Gueck)),
                    max(observed.wa$dist.med.Gueck)+0.04*max(observed.wa$dist.max.Gueck))),
              max(c(max(c(after.data$u.l.med.dist.Gueck, before.data$u.l.med.dist.Gueck))+0.04*max(c(after.data$u.l.med.dist.Gueck, before.data$u.l.med.dist.Gueck)),
                    max(observed.wa$dist.med.Gueck)+0.04*max(observed.wa$dist.max.Gueck)))),
          col=dataT,border=F)
  abline(v=1)
  abline(h=max(c(max(c(after.data$u.l.med.dist.Gueck, before.data$u.l.med.dist.Gueck))+0.04*max(c(after.data$u.l.med.dist.Gueck, before.data$u.l.med.dist.Gueck)),
                              max(observed.wa$dist.med.Gueck)+0.04*max(observed.wa$dist.max.Gueck))))
  lines(before.data$med.dist.Gueck.med~time[1:157],lwd=2,col="black")
  axis(1,at=c(100,200,300,400),labels=NA)
  lines(before.data$u.l.med.dist.Gueck~time[1:157],lty=2,col="black")
  lines(before.data$l.l.med.dist.Gueck~time[1:157],lty=2,col="black")
  points(observed.wa$dist.med.Gueck~time[1:157],pch=19,cex=0.8,col=map.leg.colors[3])
  lines(after.data$med.dist.Gueck.med[157:430]~time[157:430],lwd=2,col="grey")
  polygon(c(time[157:430],rev(time[157:430])),c(after.data$u.l.med.dist.Gueck[157:430],rev(after.data$l.l.med.dist.Gueck[157:430])),
          col=greyT,border=greyT)
  ##Fourth panel
  Chull.Fig<-plot(before.data$chull.area.med~time[1:157],
                  xlim=c(1,430),ylim=c(0,max(c(max(after.data$u.l.chull.area)+0.04*max(after.data$u.l.chull.area),
                                               max(observed.wa$chull.area)+0.04*max(observed.wa$chull.area)))),
                  type="l",bty="l",xaxt="n",cex.axis=0.9,ylab=NA,xlab=NA)
  mtext(expression("Area of convex hull" ~~(km^2)), side=2,line=3.1, las=0,cex=0.8)
  polygon(x=c(1,157,157,1),
          y=c(0,0,
              max(c(max(after.data$u.l.chull.area)+0.04*max(after.data$u.l.chull.area),
                    max(observed.wa$chull.area)+0.04*max(observed.wa$chull.area))),
              max(c(max(after.data$u.l.chull.area)+0.04*max(after.data$u.l.chull.area),
                    max(observed.wa$chull.area)+0.04*max(observed.wa$chull.area)))),
          col=dataT,border=F)
  abline(v=1)
  abline(h=max(c(max(after.data$u.l.chull.area)+0.04*max(after.data$u.l.chull.area),
                              max(observed.wa$chull.area)+0.04*max(observed.wa$chull.area)))); abline(h=0)
  lines(before.data$chull.area.med~time[1:157],lwd=2,col="black")
  axis(1,at=c(100,200,300,400),labels=NA)
  lines(before.data$u.l.chull.area~time[1:157],lty=2,col="black")
  lines(before.data$l.l.chull.area~time[1:157],lty=2,col="black")
  points(observed.wa$chull.area~time[1:157],pch=19,col=map.leg.colors[3])
  lines(after.data$chull.area.med[157:430]~time[157:430],lwd=2,col="grey")
  polygon(c(time[157:430],rev(time[157:430])),c(after.data$u.l.chull.area[157:430],rev(after.data$l.l.chull.area[157:430])),
          col=greyT,border=greyT)
  axis(1,at=c(100,200,300,400),labels=label.wa[c(100,200,300,400)],cex.axis=0.8)
  mtext(side=1,line=2,"Date",cex=0.9)
  mtext(side=1,outer=T,line=3.5,mod.names[i],font=2)
}
dev.off()


##This produces the correlation tables used in the manuscript
##Read in correlations
sp.corr.summer<-readRDS("spearman.correlation.summer.RDS")
sp.corr.second<-readRDS("spearman.correlation.second.RDS")
sp.corr.wa<-readRDS("spearman.correlation.wa.RDS")
pr.corr.summer<-readRDS("pearson.correlation.summer.RDS")
pr.corr.second<-readRDS("pearson.correlation.second.RDS")
pr.corr.wa<-readRDS("pearson.correlation.wa.RDS")

##Only want a subset of correlations: gravregionfrom, gravregionto, grav, expregionto, exp, rad, mobilityciv

###Table for West Africa, Spearman correlations
rownames(sp.corr.wa)<-colnames(sp.corr.wa)
selected<-c("GravGroupBorder","GravPow","Grav","GravBorder","Exp","Rad","MobilitySen")
pretty<-c("Gravity w/ core isolation","Gravity + long dispersal","Gravity","Gravity w/ borders","Diffusion","Radiation","Mobility model (SEN)")
wa.table<-NULL; temp.wa.table<-NULL
for (i in 1:length(selected)) temp.wa.table<-rbind(temp.wa.table,sp.corr.wa[selected[i],])
for (i in 1:length(temp.wa.table[,1])) wa.table<-cbind(wa.table,temp.wa.table[,selected[i]])
rownames(wa.table)<-pretty
colnames(wa.table)<-pretty
wa.table<-round(wa.table,2)
wa.table[upper.tri(wa.table,diag=T)]<-""
wa.table<-as.data.frame(wa.table)
wa.table<-wa.table[,1:length(wa.table)-1]
wa.table<-wa.table[2:length(selected),]
sp.wa.output<-xtable(wa.table,caption="Spearman rank correlation of transmission links - West Africa",
                         align=rep("c",7))
print(sp.wa.output,file="Spearman Wa Corr output.tex",caption.placement="top")


###Table for West Africa, Pearson correlations
rownames(pr.corr.wa)<-colnames(pr.corr.wa)
selected<-c("GravGroupBorder","GravPow","Grav","GravBorder","Exp","Rad","MobilitySen")
pretty<-c("Gravity w/ core isolation","Gravity + long dispersal","Gravity","Gravity w/ borders","Diffusion","Radiation","Mobility model (SEN)")
wa.table<-NULL; temp.wa.table<-NULL
for (i in 1:length(selected)) temp.wa.table<-rbind(temp.wa.table,pr.corr.wa[selected[i],])
for (i in 1:length(temp.wa.table[,1])) wa.table<-cbind(wa.table,temp.wa.table[,selected[i]])
rownames(wa.table)<-pretty
colnames(wa.table)<-pretty
wa.table<-round(wa.table,2)
wa.table[upper.tri(wa.table,diag=T)]<-""
wa.table<-as.data.frame(wa.table)
wa.table<-wa.table[,1:length(wa.table)-1]
wa.table<-wa.table[2:length(selected),]
pr.wa.output<-xtable(wa.table,caption="Pearson correlation of transmission links - West Africa",
                         align=rep("c",7))
print(pr.wa.output,file="Pearson WA Corr output.tex",caption.placement="top")






