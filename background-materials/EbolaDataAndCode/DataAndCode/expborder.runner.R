#Load Dependencies
source('collect.R')
source('ebola-ExpNet-border.R')

#Read in data
data <- collect.wa()

#Calculate distance matrix
dist <- makedist(rbind(data$x,data$y))

#Determine which links cross borders
cross <- makecross(data$region)

#Estimate model parameters
beta <- getbeta(dist, cross, data$infectionDate, c(5.56, .806, .257))

#Estimate parameter ranges
delta <- getdelta(c(5.5610, .80598, .25677), dist, cross, data$infectionDate)

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, cross, dist, c(5.5610, .80598, .25677), first=0, last=2000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, cross, dist, c(5.5610, .80598, .25677), first=157, last=2000)
