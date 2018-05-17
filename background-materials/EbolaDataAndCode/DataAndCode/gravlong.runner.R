#Load Dependencies
source('collect.R')
source('ebola-GravLong.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Estimate model parameters
beta <- getbeta(dist, data$pop, data$infectionDate, c(3.66, 22.6, .593, .0797))

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, dist, data$pop, c(3.6586, 22.627, 0.59254, 0.079654), first=0, last=2000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, dist, data$pop, c(3.6586, 22.627, 0.59254, 0.079654), firrst=157, last=2000)
