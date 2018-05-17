#Load Dependencies
source('collect.R')
source('ebola-alt1-foi.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Estimate model parameters
beta <- getbeta(dist, data$pop, data$infectionDate, c(-12.1, .544, -1.27, 2.40))

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, data$pop, dist, c(-12.142, 0.54377, -1.2672, 2.3989), first=0, last=6000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, data$pop, dist, c(-12.142, 0.54377, -1.2672, 2.3989), first=157, last=2000)
