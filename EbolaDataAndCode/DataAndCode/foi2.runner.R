#Load Dependencies
source('collect.R')
source('ebola-alt2-foi.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Estimate model parameters
beta <- getbeta(dist, data$pop, data$infectionDate, c(-.113, .395, -1.08, 2.62, .404))

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, data$pop, dist, c(-0.11335, 0.39489, -1.0751, 2.6205, 0.40433), first=0, last=6000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, data$pop, dist, c(-0.11335, 0.39489, -1.0751, 2.6205, 0.40433), first=157, last=2000)
