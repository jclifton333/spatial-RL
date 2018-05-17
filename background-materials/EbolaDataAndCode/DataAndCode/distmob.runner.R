#Load Dependencies
source('collect.R')
source('ebola-MobilityDist.R')

#Read in data
data <- collect.wa()
mobsen <- get.mobility.wa('sen')

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Estimate model parameters
beta <- getbeta(mobsen, dist, data$infectionDate, c(5.59, 3.33e-6, 1.05))

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, mobsen, dist, c(5.5937, 3.3346e-6, 1.0456), first=0, last=2000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, mobsen, dist, c(5.5937, 3.3346e-6, 1.0456), first=157, last=2000)
