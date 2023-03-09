# install.packages("deSolve")
# install.packages("FME")
#install.packages("pspline")
# loading library
library("deSolve")
library("rootSolve")
library("coda")
library("FME")
library('pspline')
#set current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
############################test for Logistic model#############################
#load data
data = read.table("data/Emerald1983g048.txt")
# fit <- smooth.Pspline(data$das, data$biomass)
# lines(fit, col = "blue")
# plot(data$das, data$biomass)

#define function
logistic <- function(t, y, parms) {
  with(as.list(parms,y), {
    dMdt <- r * y * (1 - y / Mmax)
    return(list(dMdt))
  })
}

#x and y
M <-as.numeric(data$biomass)
t <- as.numeric(data$das)

y0 <- c(M = M[1])
y0
parms <- c(r = 0.1, Mmax = max(M))
# #use desolve to fit the data
# fit <- ode(y = y0, times = t, func = logistic, parms = parms)
# parms
# 
plot(t, M, pch = 16, xlab = "Time", ylab = "M")
# lines(fit, col = "red")

#the first column of contains the name of the observed variable, if we only have biomass it is 'M' here
biomass_time_df <-data.frame(name=rep("M",length(t)),time=unlist(t),M=unlist(M))

#the function to minimize
ModelCost <- function(parms) {
  modelout <- as.data.frame(ode(y = y0, times = t, func = logistic, parms = parms))
  modelout
  modCost(model=modelout,obs=biomass_time_df,y="M")  # object of class modCost
}

Fit <- modFit(f = ModelCost, p = parms, method = "Marq") #fit the curve which minimize the ModelCost
summary(Fit)
out <- ode(y = y0, func = logistic, parms = Fit$par,
           times = t)
Fit$par
lines(out, col = "blue")
########################seems the code works for logistic ODE for one genotype and one environment###############
