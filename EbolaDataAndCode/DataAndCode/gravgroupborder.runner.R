#Load Dependencies
source('collect.R')
source('ebola-GravNet-border.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Determine which links cross borders
cross <- makecross.group(data$region)

#Estimate model parameters
beta <- getbeta(dist, data$pop, cross, data$infectionDate, c(5.79, 106, .186, .150))

#Simulate forward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, data$pop, cross, dist, c(5.7924, 105.67, 0.18597, 0.15021), first=0, last=2000)

#Simulate forward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, data$pop, cross, dist, c(5.7924, 105.67, 0.18597, 0.15021), first=157, last=2000)
