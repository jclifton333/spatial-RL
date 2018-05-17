#Load Dependencies
source('collect.R')
source('ebola-pwo.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Calculate sij from Simini et al.
cm <- makecirclemass(dist, data$pop)

#Calculate Population Weighted Opportunity matrix
pwomat <- makepwomat(cm, data$pop)

#Estimate model parameters
beta <- getbeta(dist, data$pop, data$infectionDate, pwomat, c(5.61, 1.04, 5.51e-7))

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, data$pop, pwomat, dist, c(5.6061, 1.0406, 5.5145e-7), first=0, last=2000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, data$pop, pwomat, dist, c(5.6061, 1.0406, 5.5145e-7), first=157, last=2000)
