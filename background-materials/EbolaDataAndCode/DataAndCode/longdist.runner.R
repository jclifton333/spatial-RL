#Load Dependencies
source('collect.R')
source('ebola-LongDist.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Estimate model parameters
beta <- getbeta(dist, data$infectionDate, c(.876, 5.75, .341))

#Estimate parameter ranges
delta <- getdelta(c(.87610, 5.7477, .34112), dist, data$infectionDate)

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, dist, c(.87610, 5.7477, .34112), first=0, last=2000)

#Simulate foreward from 1 October 2014
sims.initial <- makesims(data$infectionDate, data$PID, dist, c(.87610, 5.7477, .34112), first=157, last=2000)
