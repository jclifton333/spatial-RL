#Load Dependencies
source('collect.R')
source('ebola-GravNet.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Estimate model parameters
beta <- getbeta(dist, data$pop, data$infectionDate, c(5.25, 156, .186))

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, data$pop, dist, c(5.2457, 155.76, .18569), firrst=0, last=2000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, data$pop, dist, c(5.2457, 155.76, .18569), firrst=157, last=2000)
