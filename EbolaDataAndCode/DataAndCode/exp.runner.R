#Load Dependencies
source('collect.R')
source('ebola-ExpNet.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Estimate model parameters
beta <- getbeta(dist, data$infectionDate, c(5.64, 1.03))

#Estimate parameter ranges
delta <- getdelta(c(5.6360, 1.0322), dist, data$infectionDate)

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, dist, c(5.6360, 1.0322), first=0, last=2000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, dist, c(5.6360, 1.0322), first=157, last=2000)
