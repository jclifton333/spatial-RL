#1e90ff - dodgerblue
#1c86ee - dodgerblue2
#1874cd - dodgerblue3
#104c8b - dodgerblue4

latebreak <- readRDS("OutbreakDateByCounty_Summer_AllCountries.rds")
late.second <- as.integer(latebreak$infection_date)
late.second[which(is.na(late.second))] <- Inf
late.second <- late.second - min(late.second)
infect.late <- which(late.second != Inf)
order <- order(late.second[infect.late])

region <- as.integer(as.factor(readRDS('WestAfricaCountyPolygons.rds')$ISO))

colormod <- c("#8D4652", "#AED4B5", "#D45C24", "#DF22B8", "#568E57", "#5E5E68", "#FFA3D5", "#F5B61E", "#AC784B", "#6262A2", "#D14747", "#AE86E2", "#3FBAE9", "#BEAEC3")

ylabels <- as.Date(rep(NA,5000))
for(i in 1:5000) ylabels[i] <- as.Date("04-25-14","%m-%d-%y")+i
ylabels1 <- format(ylabels,"%m/%d/%y")
ylabels2 <- format(ylabels, "%b-%d-%y")
xlabels <- as.character(latebreak$county_names)[infect.late][order]
colormod[region[infect.late]] -> colors

spellout <- c("Benin", "Burkina Faso", "Côte d'Ivoire", "Ghana", "Guinea", "Gambia", "Guinea-Bissau", "Liberia", "Mali", "Niger", "Nigeria", "Senegal", "Sierra Leone", "Togo")

#creates boxplots of infection time
gofPlot<-function(sims,data,lateindex=NULL,order=NULL,cutoff=NULL,region=NULL,colors='red',boxcolors=c("#BFBFBF", "#404040", "#F2F2F2","dodgerblue","dodgerblue3"), bordercolor='#000000', cutcolor="#888888", maxyear=100, psize=1.5, main='',all=F){
  d=dim(sims)
  d[2]=d[2]-1  #less 1 for FIPS column

  #calculate quantiles
  q=apply(sims[,2:d[2]],1,quantile,probs=c(.025,.25,.5,.75,.975))
  
  #default ordering of infected counties
  if(is.null(order)){
    order=order(q[3,],data)
  }

  #Alternate ordering
  if(all==T){
    #Ordering for including unifected counties
    inf<-q[,lateindex]
    uninf<-q[,-lateindex]
    q=cbind(inf[,order],uninf[,order(uninf[3,])])
    region<-c(region[lateindex][order],region[-lateindex][order(uninf[3,])])
  }
  else{
    #sort infected counties
    q=q[,order]
  }
 
  
  #change infinite values
  q[which(q==Inf)]=maxyear

  #creat plot
  plot(NA,xlim=c(0,d[1]),ylim=c(0,maxyear),xlab='',ylab='',main=main,xaxt='n',yaxt='n',bty='n')
  
  if(all==F){
    for(i in 1:d[1]){
      rect((i-1),q[1,i],i,q[5,i],border=bordercolor,col=boxcolors[1]) # 95% confidence interval
      rect((i-1),q[2,i],i,q[4,i],border=bordercolor,col=boxcolors[2]) # interquartile range
      rect((i-1),q[3,i],i,q[3,i],border=boxcolors[3],col="#000000") # median
    }
  }
  if(all==T){
    for(i in 1:d[1]){
      if(region[i] %in% c(13,5,8)){
        rect((i-1),q[1,i],i,q[5,i],border=bordercolor,col=boxcolors[1]) # 95% confidence interval
        rect((i-1),q[2,i],i,q[4,i],border=bordercolor,col=boxcolors[2]) # interquartile range
        rect((i-1),q[3,i],i,q[3,i],border=boxcolors[3],col="#000000") # median
      }
      else{
        rect((i-1),q[1,i],i,q[5,i],border=bordercolor,col=boxcolors[4]) # 95% confidence interval
        rect((i-1),q[2,i],i,q[4,i],border=bordercolor,col=boxcolors[5]) # interquartile range
        rect((i-1),q[3,i],i,q[3,i],border=boxcolors[3],col="#000000") # median
      }
    }
  }
  # add actual infection dates
  if(length(colors) == 1){
    colors <- rep(colors, length(data))
  }
    lines((1:length(order))-.5,data[order],col='#ffffff',bg=colors[order],type='p',pch=21,cex=psize)
  if(!is.null(cutoff)){
    abline(v=cutoff,lty=2, col=cutcolor,lwd=0.5)
    if (all==T) abline(v=55,lty=2, col=cutcolor, lwd=0.5)
  }
  if(all==F) return(c(correlation=cor(q[3,1:length(data)], data[order]), coverage=sum((data[order] >= q[1,]) & (data[order] <= q[5,]))/length(data)))
  if(all==T) return(c(infect.late[order],seq(1:290)[-infect.late][order(uninf[3,])]))
}

findmax <- function(gofsims, roundup=50){
  gofsims <- gofsims[,-1]
  q <- apply(gofsims,1,quantile,probs=.975)
  est.max<-max(q[which(q != Inf)])
  m <- est.max
  max <- m - (m %% roundup) + roundup
  return(max)
}

corcov <- NULL

# Create single panel plot for best fit model
boxcolors.ggb <- c('#1874cd40', '#104c8b40', '#ffffff')
gofsims.ggb.wa <- readRDS('newsims/sims.init.gravgroupborder.wa.RDS') 
pdf(width=7.5, height=4, file="gof.ggb.wa-colorswap2.pdf")
#png(width=7.5, height=4, file="gof.ggb.wa.png", units="in", res=300)
par(mar=c(6,4.1,1,0))
maxyear.ggb <- findmax(gofsims.ggb.wa[infect.late,]) # make single panel plot subset by infected counties
corcov <- rbind(corcov, gofPlot(sims=gofsims.ggb.wa[infect.late,], data=late.second[infect.late], order=order, maxyear=maxyear.ggb, colors=colors, boxcolors=boxcolors.ggb, bordercolor='#ffffff', cutcolor="#000000", cutoff=46, all=F))
axis(1,at=(0:54)+.5,labels=xlabels,las=2,cex.axis=.7)
axis(2,at=c(0,200,400,600,800),labels=ylabels2[c(0,200,400,600,800)+1],las=1,cex.axis=.7)
legend(x=3,y=900,legend=spellout[sort(unique(region[infect.late]))], pt.bg=colormod[sort(unique(region[infect.late]))], border='#ffffff', bty='n',pch=21,col='#ffffff',pt.cex=1.5)
text(x=c(40,52),y=c(888,888),labels=c("data used to fit model","subsequent infections"),cex=.7)
dev.off()

#Load data for the multipanel plot
boxcolors=c("#BFBFBF", "#404040", "#F2F2F2")
gofsims.const.wa <- readRDS('newsims/gofsims.const.wa.RDS')
gofsims.baseline.wa <- readRDS('newsims/sims.init.baseline.wa.RDS')
gofsims.exp.wa <- readRDS('newsims/sims.init.exp.wa.RDS')
gofsims.grav.wa <- readRDS('newsims/sims.init.grav.wa.RDS')
gofsims.gravpow.wa <- readRDS('newsims/sims.init.gravpow.wa.RDS')
gofsims.gravborder.wa <- readRDS('newsims/sims.init.gravborder.wa.RDS')
gofsims.rad.wa <- readRDS('newsims/sims.init.rad.wa.RDS')
gofsims.mobsen.wa <- readRDS('newsims/sims.init.mobsen.wa.RDS')
gofsims.pow.wa <- readRDS('newsims/sims.init.pow.wa.RDS')
gofsims.distmobsen.wa <- readRDS('newsims/sims.init.distmobsen.wa.RDS')
gofsims.expborder.wa <- readRDS('newsims/sims.init.expborder.wa.RDS')
gofsims.expgroupborder.wa <- readRDS('newsims/sims.init.expgroupborder.wa.RDS')
gofsims.pwo.wa <- readRDS('newsims/sims.init.pwo.wa.RDS')
gofsims.foi1.wa <- readRDS('newsims/sims.init.foi1.wa.RDS')
gofsims.foi2.wa <- readRDS('newsims/sims.init.foi2.wa.RDS')

# Multipanel plot
pdf(width=7.2,height=9.7,'gofs.wa2.pdf')
par(mfrow=c(4,2))

par(mar=c(1,3.5,3.5,.5), oma=c(4,1.5,0,0))
maxyear.const <- findmax(gofsims.const.wa[infect.late,], 1000)
corcov <- rbind(corcov, gofPlot(gofsims.const.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.const, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Constant"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
axis(2,at=c(0,1000,2000,3000,4000),labels=ylabels2[c(0,1000,2000,3000,4000)+1],las=1,cex.axis=.8)

#Use same maxyear for the remaining plots, that is maxyear.pwo 
maxyear.common <- findmax(gofsims.pwo.wa[infect.late,])
maxyear.baseline <- findmax(gofsims.baseline.wa[infect.late,]) ##Can be used to scale each plot independently
corcov <- rbind(corcov, gofPlot(gofsims.baseline.wa[infect.late,],late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Well-mixed"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
#axis(2,at=c(0,175,350,525,700),labels=ylabels[c(0,175,350,525,700)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.exp <- findmax(gofsims.exp.wa[infect.late,])
corcov <- rbind(corcov, gofPlot(gofsims.exp.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Diffusion"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
#axis(2,at=c(0,340,680,1020,1360),labels=ylabels[c(0,340,680,1020,1360)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.grav <- findmax(gofsims.grav.wa[infect.late,],100)
corcov <- rbind(corcov, gofPlot(gofsims.grav.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Gravity"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
#axis(2,at=c(0,325,650,975,1300),labels=ylabels[c(0,325,650,975,1300)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.gravpow <- findmax(gofsims.gravpow.wa[infect.late,])
corcov <- rbind(corcov, gofPlot(gofsims.gravpow.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Gravity + long distance"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
#axis(2,at=c(0,225,450,675,900),labels=ylabels[c(0,225,450,675,900)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.gravborder <- findmax(gofsims.gravborder.wa[infect.late,])
corcov <- rbind(corcov, gofPlot(gofsims.gravpow.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Gravity + country  borders"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
#axis(2,at=c(0,200,400,600,800),labels=ylabels[c(0,200,400,600,800)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.rad <- findmax(gofsims.rad.wa[infect.late,])
corcov <- rbind(corcov, gofPlot(gofsims.rad.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Radiation"))
axis(1,at=(0:54)+.5,labels=xlabels,las=2,cex.axis=.55)
#axis(2,at=c(0,350,700,1050,1400),labels=ylabels[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.mobsen <- findmax(gofsims.mobsen.wa[infect.late,])
corcov <- rbind(corcov, gofPlot(gofsims.mobsen.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Mobility (Senegal)"))
axis(1,at=(0:54)+.5,labels=xlabels,las=2,cex.axis=.55)
#axis(2,at=c(0,175,350,525,700),labels=ylabels[c(0,175,350,525,700)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.pow <- findmax(gofsims.pow.wa[infect.late,])
corcov <- rbind(corcov, gofPlot(gofsims.pow.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Diffusion + long distance"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
#axis(2,at=c(0,300,600,900,1200),labels=ylabels[c(0,300,600,900,1200)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.distmobsen <- findmax(gofsims.distmobsen.wa[infect.late,])
corcov <- rbind(corcov, gofPlot(gofsims.distmobsen.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Mobility (Senegal) + diffusion"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
#axis(2,at=c(0,325,650,975,1300),labels=ylabels[c(0,325,650,975,1300)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.expborder <- findmax(gofsims.expborder.wa[infect.late,])
corcov <- rbind(corcov, gofPlot(gofsims.expborder.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Diffusion + country borders"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
#axis(2,at=c(0,275,550,825,1100),labels=ylabels[c(0,275,550,825,1100)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.expgroupborder <- findmax(gofsims.expgroupborder.wa[infect.late,])
corcov <- rbind(corcov, gofPlot(gofsims.expborder.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Diffusion + core borders"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
#axis(2,at=c(0,275,550,825,1100),labels=ylabels[c(0,275,550,825,1100)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

maxyear.pwo <- findmax(gofsims.pwo.wa[infect.late,])
corcov <- rbind(corcov, gofPlot(gofsims.pwo.wa[infect.late,], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Population weighted opportunity"))
axis(1,at=(0:54)+.5,labels=FALSE,las=2,cex.axis=.55)
#axis(2,at=c(0,350,700,1050,1400),labels=ylabels[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

complete.foi1 <- which(apply(gofsims.foi1.wa!=Inf,2,all))
maxyear.foi1 <- findmax(gofsims.foi1.wa[infect.late, complete.foi1])
corcov <- rbind(corcov, gofPlot(gofsims.foi1.wa[infect.late, complete.foi1], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Force of infection"))
axis(1,at=(0:54)+.5,labels=xlabels,las=2,cex.axis=.55)
#axis(2,at=c(0,1000,2000,3000,4000),labels=ylabels[c(0,1000,2000,3000,4000)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

complete.foi2 <- which(apply(gofsims.foi2.wa!=Inf,2,all))
maxyear.foi2 <- findmax(gofsims.foi2.wa[infect.late, complete.foi2])
corcov <- rbind(corcov, gofPlot(gofsims.foi2.wa[infect.late, complete.foi2], late.second[infect.late], order=order, maxyear=maxyear.common, colors='#000000', boxcolors=boxcolors, psize=.8, cutoff=46, main="Force of infection (normalized)"))
axis(1,at=(0:54)+.5,labels=xlabels,las=2,cex.axis=.55)
#axis(2,at=c(0,1000,2000,3000,4000),labels=ylabels[c(0,1000,2000,3000,4000)+1],las=1,cex.axis=.8)
axis(2,at=c(0,350,700,1050,1400),labels=ylabels2[c(0,350,700,1050,1400)+1],las=1,cex.axis=.8)

dev.off()

#Create plot of correlation coverage tradeoff for a representative subset of the models.
pdf(height=4, width=5, file='corcov.pdf')
par(mar=c(5.1,4.1,4.1,8), xpd=NA, las=1)
plot(corcov[1:9,], xlim=c(0,1), ylim=c(0,1), pch=0:8)
legend(x=1.1,y=1, legend=c('Gravity + core borders','Constant','Well-mixed','Diffusion', 'Gravity', 'Gravity + long distance', 'Gravity + country  borders', 'Radiation', 'Mobility (Senegal)'), pch=0:8, bty='n', cex=.5, pt.cex=1, y.intersp=2)
dev.off()
saveRDS(corcov,"Correlation-Coverage.rds")

# Create single panel plot for best fit model that includes uninfected counties
boxcolors.ggb <- c('#1874cd40', '#104c8b40', '#ffffff',"dodgerblue","dodgerblue3")
#gofsims.ggb.wa <- readRDS('gofsims.gravgroupborder.wa.RDS')
pdf(width=7.5, height=4, file="gof.ggb.wa2.pdf")
par(mar=c(6,4,1,0))
maxyear.ggb <- findmax(gofsims.ggb.wa) # make single panel plot
order.counties<-gofPlot(sims=gofsims.ggb.wa, data=late.second[infect.late], lateindex=infect.late, order=order, maxyear=maxyear.ggb, region=region, colors=colors, boxcolors=boxcolors.ggb, bordercolor='#ffffff', 
        cutcolor="#000000", cutoff=46,psize=0.3,all=T)
#axis(1,at=c(0,50,100,150,200,250)+.5,labels=c(0,50,100,150,200,250),las=2,cex.axis=.7)
#mtext(side=1,line=2,"Counties in West Africa")
axis(1,at=(0:289)+.5,labels=as.character(latebreak$county_names)[order.counties],las=2,cex.axis=.19)
axis(2,at=c(0,300,600,900,1200,1500,1800),labels=ylabels2[c(0,300,600,900,1200,1500,1800)+1],las=1,cex.axis=.7)
#legend(x=3,y=900,legend=spellout[sort(unique(region[infect.late]))], pt.bg=colormod[sort(unique(region[infect.late]))], border='#ffffff', bty='n',pch=21,col='#ffffff',pt.cex=1.5)
text(x=c(40,50),y=c(1300,1300),labels=c("data used to fit model","subsequent observed infections"),cex=.7,srt=90)
dev.off()


#Diffusion + long distance
#Mobility (Senegal) + diffusion
#Diffusion + country borders
#Diffusion + core borders
#Population weighted opportunity
#Force of infection
#Force of infection (normalized)


