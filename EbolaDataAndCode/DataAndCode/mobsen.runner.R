#Load Dependencies
source('collect.R')
source('ebola-ExpNet.R')

#Read in data
data <- collect.wa()
mobsen <- get.mobility.wa('sen')

#Estimate model parameters
beta <- getbeta(mobsen, data$infectionDate, c(9.89, -6.62e-6))

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, mobsen, c(9.8857, -6.6162e-6), first=0, last=2000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, mobsen, c(9.8857, -6.6162e-6), first=157, last=2000)
