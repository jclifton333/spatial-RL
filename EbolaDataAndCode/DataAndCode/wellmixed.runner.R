#Load Dependencies
source('collect.R')
source('ebola-baseline.R')

#Read in data
data <- collect.wa()

#Estimate model parameters
beta <- getbeta(data$infectionDate, 5.11e-5)

#Estimate parameter ranges
delta <- getdelta(5.1117e-5, data$infectionDate)

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, 5.1117e-5, first=0, last=2000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, 5.1117e-5, first=157, last=2000)
