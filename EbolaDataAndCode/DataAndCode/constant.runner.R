#Load Dependencies
source('collect.R')
source('ebola-constant.R')

#Read in data
data <- collect.wa()

#Estimate model parameters
beta <- getbeta(data$infectionDate, 1.04e-3)

#Estimate parameter ranges
delta <- getdelta(1.0427e-3, data$infectionDate)

#Simulate foreward from 26 April 2014
sims.initial <- makesims(data$infectionDate, data$PID, 1.0427e-3, first=0, last=5000)

#Simulate foreward from 1 October 2014
sims.after <- makesims(data$infectionDate, data$PID, 1.0427e-3, first=157, last=2000)
