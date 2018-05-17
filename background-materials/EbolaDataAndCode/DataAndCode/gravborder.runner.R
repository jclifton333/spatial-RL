#Load Dependencies
source('collect.R')
source('ebola-GravNet-border.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Determine which links cross borders
cross <- makecross(data$region)

#Estimate model parameters
beta <- getbeta(dist, data$pop, cross, data$infectionDate, c(5.17, 157, .189, .507))

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, data$pop, cross, dist, c(5.1661, 157.14, 0.18945, 0.50694), first=0, last=2000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, data$pop, cross, dist, c(5.1661, 157.14, 0.18945, 0.50694), first=157, last=2000)
