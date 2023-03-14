### Completely clear the working space; It does NOT change your working directory
ls(all = TRUE)
rm(list = ls(all = TRUE)) 
ls()
ls(all = TRUE)
### Set the working directory, and check
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
#used for getting or setting the library trees that R knows about
.libPaths("library")
#install.packages("sde") 
library(sde) ### https://rdrr.io/cran/sde/man/sde.sim.html

### Data
b.f <- 1000
filename <- "data/Emerald1985g009.txt"
data.apsim <- read.table(filename, header = TRUE)
das <- data.apsim[5][,1] # same as: das <- data.apsim$das
biomass <- data.apsim[7][,1]
#biomass <- biomass/b.f #divided biomass by 1000 to get more 
end.time <- length(das)
end.biomass <- biomass[end.time]
time.vec <- 0:(end.time-1)

### Stochastic logistic equation
set.seed(123)
#r <- 0.5
#runif() generates random deviates of the uniform distribution
r <- 0.08 # runif(1, 0.5, 1.5) # 0.0887
#Mx <- 5000/b.f
Mx=max(biomass)
#Mx <- 5972/b.f
y0 <- biomass[1]
#y0 <- 25.66/b.f
d <- expression(r * x * (1 - x/Mx) )
s <- expression(0.05 * sqrt(x))
#delta: time step of the simulation,the fixed amount of time by which the simulation advances.
#N:number of simulation steps.
sde.sim(X0=y0, delta=1/100, N=(end.time*100), drift=d, sigma=s, M = 10) -> F
#png("sdelogistnp-biomass-noise.png", width = 4, height = 4, units = 'in', res = 300)
#plot(time.vec, (biomass*b.f), main = "Logistic SDE adapted", xlab = "time (days)", ylab = "biomass", ylim = c(0,7000), type = "l", lwd = 2)
#lines(F*b.f, col=3)
plot(F)
#dev.off() 
