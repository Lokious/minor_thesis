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
library('pspline')
# ### Data
# b.f <- 1000
# filename <- "data/Emerald1985g009.txt"
# data.apsim <- read.table(filename, header = TRUE)
# das <- data.apsim$das # time steps
# biomass <- data.apsim[7][,1]
# biomass <- biomass/b.f #divided biomass by 1000 
# end.time <- length(das) # 118
# end.biomass <- biomass[end.time] # 5.8696
# time.vec <- 0:(end.time-1) # define a time vector from 0 to 117

# read parameters range
parameters_df <- read.csv("logistics_fit_parameters.csv")
# drop negetive r
parameters_df<-subset(parameters_df, r>0)
r_range <- c(min(parameters_df$r),max(parameters_df$r))
y0_range <- c(min(parameters_df$y0),max(parameters_df$y0))
biomass_max_range <- c(min(parameters_df$Mmax),max(parameters_df$Mmax))
end.time <-120
time.vec <- 0:119
### Stochastic logistic equation
set.seed(123)
#r <- 0.5

df <- data.frame(matrix(nrow = end.time, ncol = 0))
r_list = list()
s_list = list()
#label_list = list()
for (i in c(0:100)){
  #runif() generates random deviates of the uniform distribution
  r <- runif(1, 0.1, 1.5)
  # r_range[1]
  # r <-runif(1,r_range[1],r_range[2])
  Mmax<-runif(1,biomass_max_range[1],biomass_max_range[2])
  #Mmax <- runif(1,5000,6000)
  y0 <-runif(1,y0_range[1],y0_range[2])
  d <- expression(r * x * (1 - x/Mmax) ) 
  #noise <- runif(1,0.1,1)
  s <- expression(0.2*(x))# NOISE related to biomass
  #s<- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax)))
  #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
  #N:number of simulation steps.
  # diffusion coefficient: an expression of two variables t and x
  #M: number of trajectories.
  sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s, M = 1) -> simulated_logistics_data
  class(simulated_logistics_data) #check datatype
  plot(simulated_logistics_data)
  df[ , ncol(df) + 1] = simulated_logistics_data
  r_list <- append(r_list,r)
  #label_list <- append(label_list,"0")
  #dev.off()
  fit <- sm.spline(time.vec,simulated_logistics_data)
  derivative <- predict(fit,time.vec,nderiv=2)
  plot(derivative)
  plot(time.vec,simulated_logistics_data)
  lines(fit, col = "blue")
}
write.csv(df,"simulated_X_data_3.csv")

df_Y = data.frame(r_list)
colnames(df_Y) <- c(1:101)
write.csv(df_Y,"simulated_Y_data_3.csv")
#df_Y = data.frame(label_list)
#write.csv(df_Y,"simulated_label_data_3.csv")



fit <- smooth.Pspline(as.numeric(rownames(df)), df$V3)
plot(as.numeric(rownames(df)), df$V3)
lines(fit, col = "blue")
spline <-splinefun(as.numeric(rownames(df)), df$V3)
plot(spline(df$V3, deriv = 1), type="l")
# read ml predict biomass
predict <- read.csv("predict.csv")
orginal <-read.csv("original.csv")
fit <- smooth.Pspline(as.numeric(rownames(predict)), predict$X0)
plot(as.numeric(rownames(predict)), predict$X0)
lines(fit, col = "red")
spline <-splinefun(as.numeric(rownames(predict)), predict$X0)
plot(spline(predict$X0, deriv = 1), type="l", col = "red")

