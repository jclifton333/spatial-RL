#Load Dependencies
source('collect.R')
source('ebola-radiation.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Calculate sij from Simini et al.
cm <- makecirclemass(dist, data$pop)

#Estimate model parameters
beta <- getbeta(dist ,data$pop, data$infectionDate, cm, c(5.59, 1.05, 1.61e-6))

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, data$pop, cm, dist, c(5.5882, 1.0462, 1.6126e-6), first=0, last=2000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, data$pop, cm, dist, c(5.5882, 1.0462, 1.6126e-6), first=157, last=2000)
