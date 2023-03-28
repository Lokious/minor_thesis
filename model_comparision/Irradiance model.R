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


end.time = 120
# end.biomass <- biomass[end.time] #5.8696
time.vec <- 1:(end.time) # define a time vector from 0 to 117

### Stochastic logistic equation
set.seed(123)
#r <- 0.5
df <- data.frame(matrix(nrow = end.time, ncol = 0))
r_list = list()
s_list = list()
label_list = list()
# read parameters range
parameters_df <- read.csv("logistics_fit_parameters.csv")
# drop negetive r
parameters_df<-subset(parameters_df, r>0)
r_range <- c(min(parameters_df$r),max(parameters_df$r))
y0_range <- c(min(parameters_df$y0),max(parameters_df$y0))
biomass_max_range <- c(min(parameters_df$Mmax),max(parameters_df$Mmax))
### Stochastic logistic equation
set.seed(123)
#r <- 0.5

df <- data.frame(matrix(nrow = end.time, ncol = 0))
r_list = list()
s_list = list()
label_list = list()
for (i in c(1:301)){
  #runif() generates random deviates of the uniform distribution
  #r <- runif(1, 0.1, 0.2)
  r_range[1]
  r <-runif(1,r_range[1],r_range[2])
  Mmax<-runif(1,biomass_max_range[1],biomass_max_range[2])/1000
  #Mmax <- runif(1,5000,6000)
  y0 <-runif(1,y0_range[1],y0_range[2])/1000
  d <- expression(r * x * (1 - x/Mmax) ) 
  noise <- rnorm(1,0,0.25) 

  
  s <- expression(noise)# idependent NOISE 
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
  label_list <- append(label_list,"0")
  #dev.off()

}
plot(simulated_logistics_data)
write.csv(df,"simulated_X_data_logistic_time_independent_noise_0.25.csv")
df_Y = data.frame(label_list)
colnames(df_Y) <- c(1:301)
write.csv(df_Y,"simulated_label_data_logistic_time_independent_noise_0.25.csv")

###generate data from Irradiance model###
df <- data.frame(matrix(nrow = end.time, ncol = 0))
r_list = list()
a_list = list()
fi_list = list()
s_list = list()
label_list = list()
for (i in c(1:301)){
  #runif() generates random deviates of the uniform distribution
  r <-runif(1,r_range[1],r_range[2])
  #r<-0.1
  Mmax<-runif(1,biomass_max_range[1],biomass_max_range[2])/1000
  #Mmax <- runif(1,5000,6000)
  y0 <-runif(1,y0_range[1],y0_range[2])/1000
  #r <- runif(1, 0.1, 0.5)
 
  a <- runif(1, -0.2, 0.2)
  fi <- runif(1,1/365,182/365)
  while (all((r+a*sin((2*pi/365)*time.vec+fi))>0)==FALSE & all((r+a*sin((2*pi/365)*time.vec+fi))<r_range[2])==FALSE) {
    r <-runif(1,r_range[1],r_range[2])
    a <- runif(1, -0.2, 0.2)
    fi <- runif(1,-.2,0.2)
  }
    
  d_irradiance <- expression((r+a*sin((2*pi/365)*t+fi)) * x * (1 - x/Mmax))
  #s_irradiance <- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax)))# NOISE related to biomass
  noise <- rnorm(1,0,0.25) 
  s_irradiance <- expression(noise)# idependent NOISE 
  #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
  #N:number of simulation steps.
  # diffusion coefficient: an expression of two variables t and x
  #M: number of trajectories.
  sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_irradiance, sigma=s_irradiance, M = 1) -> simulated_irradiance_logistics_data
  class(simulated_irradiance_logistics_data) #check datatype
  plot(simulated_irradiance_logistics_data)
  df[ , ncol(df) + 1] = simulated_irradiance_logistics_data
  r_list <- append(r_list,r)
  a_list <- append(a_list,a)
  fi_list <- append(fi_list,fi)
  label_list <- append(label_list,'1')
  # if (sum(is.na(simulated_irradiance_logistics_data))==0){
  # fit <- sm.spline(time.vec,simulated_irradiance_logistics_data)
  # derivative <- predict(fit,time.vec,nderiv=1)
  # plot(derivative)
  # plot(time.vec,simulated_irradiance_logistics_data)
  # lines(fit, col = "red")}
  # #dev.off() 
}
plot(simulated_irradiance_logistics_data)
write.csv(df,"simulated_X_data_irradiance_time_independent_noise_0.25.csv")
df_Y = data.frame(label_list)
colnames(df_Y) <- c(1:301)
write.csv(df_Y,"simulated_label_data_irradiance_time_independent_noise_0.25.csv")

# ################temperature model#############
# df <- data.frame(matrix(nrow = end.time, ncol = 0))
# r_list = list()
# a_list = list()
# fi_list = list()
# s_list = list()
# label_list = list()
# for (i in c(0:615)){
#   #runif() generates random deviates of the uniform distribution
#   r <- runif(1, 0.1, 0.5)
# 
#   Mmax <- runif(1,5000,6000)
#   y0 <- biomass[1]
#   TAL= 20000
#   TL = 292
#   TAH = 60000
#   TH = 303
#   tempreture <- expression(273.15 +4.2*sin((t+1)*pi/180) +13.7) #use a function of day to simulated daily temperature.
#   d_temperature <- expression((r*(1+exp(TAL/(273.15 +4.2*sin((t+1)*pi/180) +13.7) -TAL/TL) + exp(TAH/TH -TAH/(273.15 +4.2*sin((t+1)*pi/180) +13.7)))^-1) * x * (1 - x/Mmax))
#   s_temperature <- expression(sqrt(x))# NOISE related to time(biomass) #0.2~0.3
#   
#   #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
#   #N:number of simulation steps.
#   # diffusion coefficient: an expression of two variables t and x
#   #M: number of trajectories.
#   sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_temperature, sigma=s_temperature, M = 1) -> simulated_temperature_logistics_data
#   class(simulated_temperature_logistics_data) #check datatype
#   plot(simulated_temperature_logistics_data)
#   df[ , ncol(df) + 1] = simulated_temperature_logistics_data
#   r_list <- append(r_list,r)
#   a_list <- append(a_list,a)
#   fi_list <- append(fi_list,fi)
#   label_list <- append(label_list,'2')
#   #dev.off() 
# }
# plot(simulated_temperature_logistics_data)
# write.csv(df,"simulated_X_data_temperature.csv")
# df_Y = data.frame(label_list)
# colnames(df_Y) <- c(1:616)
# write.csv(df_Y,"simulated_label_data_temperature.csv")
# 
# 
# ############water model#############
# df <- data.frame(matrix(nrow = end.time, ncol = 0))
# r_list = list()
# a_list = list()
# fi_list = list()
# s_list = list()
# label_list = list()
# for (i in c(0:615)){
#   #runif() generates random deviates of the uniform distribution
#   r <- runif(1, 0.1, 0.5)
#   
#   Mmax <- runif(1,5000,6000)
#   y0 <- biomass[1]
#   TAL= 20000
#   TL = 292
#   TAH = 60000
#   TH = 303
#   tempreture <- expression(273.15 +4.2*sin((t+1)*pi/180) +13.7) #use a function of day to simulated daily temperature.
#   d_water <- expression((r*(1+exp(TAL/(273.15 +4.2*sin((t+1)*pi/180) +13.7) -TAL/TL) + exp(TAH/TH -TAH/(273.15 +4.2*sin((t+1)*pi/180) +13.7)))^-1) * x * (1 - x/Mmax))
#   s_water <- expression(sqrt(x))# NOISE related to time(biomass)
#   
#   #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
#   #N:number of simulation steps.
#   # diffusion coefficient: an expression of two variables t and x
#   #M: number of trajectories.
#   sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_water, sigma=s_water, M = 1) -> simulated_water_logistics_data
#   class(simulated_water_logistics_data) #check datatype
#   plot(simulated_water_logistics_data)
#   df[ , ncol(df) + 1] = simulated_water_logistics_data
#   r_list <- append(r_list,r)
#   a_list <- append(a_list,a)
#   fi_list <- append(fi_list,fi)
#   label_list <- append(label_list,'1')
#   #dev.off() 
# }
# plot(simulated_water_logistics_data)
# write.csv(df,"simulated_X_data_water.csv")
# df_Y = data.frame(label_list)
# colnames(df_Y) <- c(1:616)
# write.csv(df_Y,"simulated_label_data_water.csv")
