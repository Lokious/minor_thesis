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
das <- data.apsim$das # time steps
biomass <- data.apsim[7][,1]
biomass <- biomass/b.f #divided biomass by 1000 
end.time <- length(das) # 118
end.biomass <- biomass[end.time] #5.8696
time.vec <- 0:(end.time-1) # define a time vector from 0 to 117

### Stochastic logistic equation
set.seed(123)
#r <- 0.5

df <- data.frame(matrix(nrow = end.time, ncol = 0))
r_list = list()
s_list = list()
for (i in c(0:100)){
  #runif() generates random deviates of the uniform distribution
  r <- runif(1, 0.1, 1.5)
  r
  Mmax=max(biomass)
  y0 <- biomass[1]
  d <- expression(r * x * (1 - x/Mmax) ) 
  s <- expression(0.05 * sqrt(x))# NOISE related to time
  
  #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
  #N:number of simulation steps.
  # diffusion coefficient: an expression of two variables t and x
  #M: number of trajectories.
  sde.sim(X0=y0, delta=1/100, N=(end.time-1), drift=d, sigma=s, M = 1) -> simulated_logistics_data
  class(simulated_logistics_data) #check datatype
  plot(simulated_logistics_data)
  df[ , ncol(df) + 1] = simulated_logistics_data
  r_list <- append(r_list,r)

  #dev.off() 
}
write.csv(df,"simulated_X_data_1.csv")
df_Y = data.frame(r_list)
colnames(df_Y) <- c(1:101)
write.csv(df_Y,"simulated_Y_data_1.csv")
