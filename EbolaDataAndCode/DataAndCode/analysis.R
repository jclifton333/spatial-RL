##Conducts correlation analysis on the network models to see if links are similar


net.corr <- function(net1, net2, lg=TRUE, method=c("pearson", "kendall", "spearman")){
  net1 <- as.matrix(net1)
  net2 <- as.matrix(net2)
  diag(net1) <- NA
  diag(net2) <- NA
  net1 <- as.numeric(net1)
  net2 <- as.numeric(net2)
  if(lg){
    net1 <- log(net1)
    net2 <- log(net2)
  }
  corr <- cor(net1, net2, use="complete.obs", method=method)
  return(corr)
}

corrnamefun <- function(name, net1, tack, method=c("pearson", "kendall", "spearman")){
  net2 <- get(paste("net", name, tack, sep='.'))
  ret <- net.corr(net1,net2,method=method)
  return(ret)
}

names=c('gravgroupborder', 'gravpow', 'grav', 'gravborder', 'exp', 'rad', 'mobsen')


for(i in 1:length(names)){
  assign(paste("net",names[i],"wa",sep='.'), readRDS(paste("net",names[i],"wa.RDS",sep='.')))
}

pearson.table.wa <- rbind(
  sapply(names,corrnamefun,net.gravgroupborder.wa, tack="wa", method="pearson"),
  sapply(names,corrnamefun,net.gravpow.wa, tack="wa", method="pearson"),
  sapply(names,corrnamefun,net.grav.wa, tack="wa", method="pearson"),
  sapply(names,corrnamefun,net.gravborder.wa, tack="wa", method="pearson"),
  sapply(names,corrnamefun,net.exp.wa, tack="wa", method="pearson"),
  sapply(names,corrnamefun,net.rad.wa, tack="wa", method="pearson"),
  sapply(names,corrnamefun,net.mobsen.wa, tack="wa", method="pearson")
)

spearman.table.wa <- rbind(
  sapply(names,corrnamefun,net.gravgroupborder.wa, tack="wa", method="spearman"),
  sapply(names,corrnamefun,net.gravpow.wa, tack="wa", method="spearman"),
  sapply(names,corrnamefun,net.grav.wa, tack="wa", method="spearman"),
  sapply(names,corrnamefun,net.gravborder.wa, tack="wa", method="spearman"),
  sapply(names,corrnamefun,net.exp.wa, tack="wa", method="spearman"),
  sapply(names,corrnamefun,net.rad.wa, tack="wa", method="spearman"),
  sapply(names,corrnamefun,net.mobsen.wa, tack="wa", method="spearman")
)

