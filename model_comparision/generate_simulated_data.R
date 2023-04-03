## this script is to generate the simulated dataset with SDE and it's derivative
### Completely clear the working space
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
library(naniar)
library(sde) ### https://rdrr.io/cran/sde/man/sde.sim.html
library('pspline')
library('sfsmisc') # get the derivative
# time steps is 120 days
end.time = 120
# end.biomass <- biomass[end.time] #5.8696
time.vec <- 1:(end.time) # define a time vector from 0 to 117

### Stochastic logistic equation model ###
set.seed(123)

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

### Stochastic logistic equation

df <- data.frame(matrix(nrow = end.time, ncol = 0))
derivative_df <- data.frame(matrix(nrow = end.time, ncol = 0))
r_list = list()
s_list = list()
label_list = list()
plot(x=0,y=0,xlim = c(0, 7),ylim = c(0.0,0.5))
for (i in c(1:301)){
  print(i)
  simulated_logistics_data <- NA
  while(sum(is.na(simulated_logistics_data)) !=0){
  #only keep the simulated data which doesn't have NA Inf or -Inf
  
  r <-runif(1,r_range[1],r_range[2])
  Mmax<-runif(1,5900,6100)/1000
  y0 <- 4.6/1000
  d <- expression(r * x * (1 - x/Mmax) ) 
  
  # noise <- rnorm(1,0,0.25) # idependent NOISE
  # s <- expression(noise)# idependent NOISE
  #s<- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax))) # biomass_dependent_noise_0.2
  #s <- expression(0.2*((2*(end.time-t)/end.time)*(1-(end.time-t)/end.time)))# time_dependent_noise_0.2
  s <- expression(0*x)
  #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
  #N:number of simulation steps.
  # diffusion coefficient: an expression of two variables t and x
  #M: number of trajectories.
  sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s, M = 1) -> simulated_logistics_data

  #plot(simulated_logistics_data)

  simulated_logistics_data <- replace(simulated_logistics_data, simulated_logistics_data==-Inf, NA)
  simulated_logistics_data <- replace(simulated_logistics_data, simulated_logistics_data==Inf, NA)
  }
  #plot(simulated_logistics_data)
  # save simulated data and parameters
  df[ , ncol(df) + 1] = simulated_logistics_data

  r_list <- append(r_list,r)
  label_list <- append(label_list,"0")

  fit <- sm.spline(x=time.vec,y=simulated_logistics_data)
  derivative <-D1tr(y=simulated_logistics_data, x = time.vec)
  derivative_df[ , ncol(derivative_df) + 1] = derivative
  lines(y=c(derivative),x=c(simulated_logistics_data),type='l', col = "green",xlim = c(0, 7),ylim = c(0.0,1.0))
  # get the max on y, and matching x value
  derivateMax <- max(derivative)
  x_index <-which.max(derivative)
  # mark the max point on the plot
  points(y=derivateMax, x = simulated_logistics_data[x_index], col = "green", pch = 19)
  # add vertical line
  abline(v = simulated_logistics_data[x_index], col = "green", lty = "dashed")

}
####save dataframe to files###
# write biomass at 120 time steps to csv
write.csv(df,"data/simulated_data/fixed_Max_range_of_parameters/simulated_X_data_logistic_time_without_noise.csv")
# rename label dataframe
df_Y = data.frame(label_list)
colnames(df_Y) <- c(1:301)
# write label dataframe
write.csv(df_Y,"data/simulated_data/fixed_Max_range_of_parameters/simulated_label_data_logistic_without_noise.csv")
# write derivative dataframe to csv
write.csv(derivative_df,"data/simulated_data/fixed_Max_range_of_parameters/simulated_derivative_data_logistic_without_noise.csv")


######generate data from Irradiance model######
df <- data.frame(matrix(nrow = end.time, ncol = 0))
derivative_df <- data.frame(matrix(nrow = end.time, ncol = 0))
r_list = list()
a_list = list()
fi_list = list()
s_list = list()
label_list = list()
for (i in c(1:301)){
  print(i)
  simulated_irradiance_logistics_data <- c(NA)
  Mmax<-runif(1,5900,6100)/1000
  while((sum(is.na(simulated_irradiance_logistics_data)) !=0) || (tail(simulated_irradiance_logistics_data, n=1)<(Mmax-1))){
    #only keep the simulated data which doesn't have NA Inf or -Inf
    #runif() generates random deviates of the uniform distribution
    r <-runif(1,r_range[1],r_range[2])
    y0 <-4.6/1000
    a <- runif(1, -0.5, 0.5)
    fi <- runif(1,1/365,182/365)
    
    d_irradiance <- expression((r+a*sin((2*pi/365)*t+fi)) * x * (1 - x/Mmax))
    # noise <- rnorm(1,0,0.25)# idependent NOISE
    # s_irradiance <- expression(noise)# idependent NOISE
    #s_irradiance <- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax)))# biomass_dependent_noise_0.2
    #s_irradiance <- expression(0.2*((2*(end.time-t)/end.time)*(1-(end.time-t)/end.time)))# time_dependent_noise_0.2
    s_irradiance <- expression(0*x)
    #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
    #N:number of simulation steps.
    # diffusion coefficient: an expression of two variables t and x
    #M: number of trajectories.
    sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_irradiance, sigma=s_irradiance, M = 1) -> simulated_irradiance_logistics_data
    #plot(simulated_irradiance_logistics_data)
    simulated_irradiance_logistics_data <- replace(simulated_irradiance_logistics_data, simulated_irradiance_logistics_data==-Inf, NA)
    simulated_irradiance_logistics_data <- replace(simulated_irradiance_logistics_data, simulated_irradiance_logistics_data==Inf, NA)

  }
  
  df[ , ncol(df) + 1] = simulated_irradiance_logistics_data
  r_list <- append(r_list,r)
  a_list <- append(a_list,a)
  fi_list <- append(fi_list,fi)
  label_list <- append(label_list,'1')
  if (sum(is.na(simulated_irradiance_logistics_data))==0){
    fit <- sm.spline(x=time.vec,y=simulated_irradiance_logistics_data)
    derivative <-D1tr(y=simulated_irradiance_logistics_data, x = time.vec)
    derivative_df[ , ncol(derivative_df) + 1] = derivative
    lines(y=c(derivative),x=c(simulated_irradiance_logistics_data),type='l', col = "red")
    # get the max on y, and matching x value
    derivateMax <- max(derivative)
    x_index <-which.max(derivative)
    # mark the max point on the plot
    points(y=derivateMax, x = simulated_irradiance_logistics_data[x_index], col = "red", pch = 19)
    # add vertical line
    abline(v = simulated_irradiance_logistics_data[x_index], col = "red", lty = "dashed")
  }
}

plot(simulated_irradiance_logistics_data)
write.csv(df,"data/simulated_data/fixed_Max_range_of_parameters/simulated_X_data_irradiance_without_noise.csv")
df_Y = data.frame(label_list)
colnames(df_Y) <- c(1:301)
write.csv(df_Y,"data/simulated_data/fixed_Max_range_of_parameters/simulated_label_data_irradiance_without_noise.csv")
# write derivative dataframe to csv
write.csv(derivative_df,"data/simulated_data/fixed_Max_range_of_parameters/simulated_derivative_data_irradiance_without_noise.csv")

### Allee model ###
### Stochastic logistic equation
set.seed(123)

df <- data.frame(matrix(nrow = end.time, ncol = 0))
r_list = list()
s_list = list()
label_list = list()

# drop negetive r
parameters_df<-subset(parameters_df, r>0)
r_range <- c(min(parameters_df$r),max(parameters_df$r))
y0_range <- c(min(parameters_df$y0),max(parameters_df$y0))

df <- data.frame(matrix(nrow = end.time, ncol = 0))
derivative_df <- data.frame(matrix(nrow = end.time, ncol = 0))
r_list = list()
s_list = list()
label_list = list()

for (i in c(1:301)){
  print(i)
  simulated_allee_data <- NA
  while(sum(is.na(simulated_allee_data)) !=0){
    #only keep the simulated data which doesn't have NA Inf or -Inf
    
    r <-runif(1,r_range[1],r_range[2])
    Mmax<-runif(1,5900,6100)/1000
    y0 <- 4.6/1000
    Ma <- runif(1,0,y0) #yo>Ma
    d <- expression(r*x*(1 - x/Mmax)*(x/(x + Ma))) 
    
    # noise <- rnorm(1,0,0.25) # idependent NOISE
    # s <- expression(noise)# idependent NOISE
    # s<- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax))) #biomass_dependent_noise_0.2
    # s <- expression(0.2*((2*(end.time-t)/end.time)*(1-(end.time-t)/end.time)))# time_dependent_noise_0.2
    s <- expression(0*x) #without_noise
    #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
    #N:number of simulation steps.
    # diffusion coefficient: an expression of two variables t and x
    #M: number of trajectories.
    sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s, M = 1) -> simulated_allee_data
    
    #plot(simulated_allee_data)
    
    simulated_allee_data <- replace(simulated_allee_data, simulated_allee_data==-Inf, NA)
    simulated_allee_data <- replace(simulated_allee_data, simulated_allee_data==Inf, NA)
  }
  #plot(simulated_allee_data)
  # save simulated data and parameters
  df[ , ncol(df) + 1] = simulated_allee_data
  r_list <- append(r_list,r)
  label_list <- append(label_list,"2")
  if (sum(is.na(simulated_allee_data))==0){
    fit <- sm.spline(x=time.vec,y=simulated_allee_data)
    derivative <-D1tr(y=simulated_allee_data, x = time.vec)
    derivative_df[ , ncol(derivative_df) + 1] = derivative
    lines(y=c(derivative),x=c(simulated_allee_data),type='l', col = "orange",xlim = c(0, 7),ylim = c(0.0,1.0))
    # get the max on y, and matching x value
    derivateMax <- max(derivative)
    x_index <-which.max(derivative)
    # mark the max point on the plot
    points(y=derivateMax, x = simulated_allee_data[x_index], col = "orange", pch = 19)
    # add vertical line
    abline(v = simulated_allee_data[x_index], col = "orange", lty = "dashed")
  }
}

plot(simulated_allee_data)
### without_noise; time_independent_noise_0.25; time_dependent_noise_0.2; biomass_dependent_noise_0.2
write.csv(df,"data/simulated_data/fixed_Max_range_of_parameters/simulated_X_data_Allee_without_noise.csv")
df_Y = data.frame(label_list)
colnames(df_Y) <- c(1:301)
write.csv(df_Y,"data/simulated_data/fixed_Max_range_of_parameters/simulated_label_data_Allee_without_noise.csv")
# write derivative dataframe to csv
write.csv(derivative_df,"data/simulated_data/fixed_Max_range_of_parameters/simulated_derivative_data_Allee_without_noise.csv")
