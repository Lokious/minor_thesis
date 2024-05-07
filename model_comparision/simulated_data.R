####This script generate figures shows in thesis report.####

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
#install.packages("plyr")
library(sde) ### https://rdrr.io/cran/sde/man/sde.sim.html
#library('pspline')
library('splines')
library('sfsmisc') # get the derivative
library("plyr")
library(ggplot2)
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

# set parameters range
r <- 0.15 #Aa
Mmax_range<-6000
Mmax <- 6000/1000
end.time <-120
time.vec <- 0:119
### Stochastic logistic equation



#runif() generates random deviates of the uniform distribution
# r <- rnorm(1,0.5,0.25)
# Mmax<-rnorm(1,6000,100)/1000
#Mmax <- runif(1,5000,6000)
y0 <- 4.6/1000
d <- expression(r * x * (1 - x/Mmax) ) 

s <- expression(0*x)#without_noise
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s, M = 1)*1000 -> simulated_logistics_data
derivative <-D1tr(y=simulated_logistics_data, x = time.vec)

plot(simulated_logistics_data,col = "black",lwd=2,xlab="",ylab="")
title(xlab="Time day",ylab=expression(Biomass~kg/m^2),line=2)

df_biomass <- data.frame(matrix(nrow = end.time, ncol = 0))
df_biomass_derivative <- data.frame(matrix(nrow = end.time, ncol = 0))
for (i in c(1:5)){
s_biomass <- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax)))# NOISE related to biomass
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s_biomass, M = 1)*1000 -> simulated_logistics_data_biomass
df_biomass[ , ncol(df_biomass) + 1] = simulated_logistics_data_biomass
derivative_biomass <-D1tr(y=simulated_logistics_data_biomass, x = time.vec)
df_biomass_derivative[ , ncol(df_biomass_derivative) + 1] = derivative_biomass
lines(simulated_logistics_data_biomass,col = "red",lwd=2,xlab="Time day",ylab=expression(Biomass~kg/m^2),ylim=c(0,6200))
}

plot(y=c(derivative),x=c(simulated_logistics_data), type="l",col = "black",lwd=2,ylim =c(0,500),xlab="",ylab="")
title(xlab=expression(Biomass~kg/m^2),ylab=expression(Derivatives~kg/(m^2~day)),line=2.5)
for (i in c(1:5)){
lines(y=c(unlist(df_biomass_derivative[i])),x=c(unlist(df_biomass[i])), type="l",col = "red",lwd=2,ylim =c(0,500),xlab=expression(Biomass~kg/m^2),ylab="Derivative")
}




plot(simulated_logistics_data,col = "black",lwd=2,xlab="",ylab="")
title(xlab="Time day",ylab=expression(Biomass~kg/m^2),line=2)
df_time_dependent <- data.frame(matrix(nrow = end.time, ncol = 0))
df_time_dependent_derivative <- data.frame(matrix(nrow = end.time, ncol = 0))
for (i in c(1:5)){
s_time <- expression(0.2*((2*(end.time-t)/end.time)*(1-(end.time-t)/end.time)))# time_dependent_noise_0.2
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s_time, M = 1)*1000 -> simulated_logistics_data_time_dependent
df_time_dependent[ , ncol(df_time_dependent) + 1] = simulated_logistics_data_time_dependent
lines(simulated_logistics_data_time_dependent,col = "green",lwd=2)
derivative_time_dependent <-D1tr(y=simulated_logistics_data_time_dependent, x = time.vec)
df_time_dependent_derivative[ , ncol(df_time_dependent_derivative) + 1] = derivative_time_dependent
#lines(y=c(derivative_time_dependent),x=c(simulated_logistics_data_time_dependent), type="l",col = "green",lwd=2,ylim =c(0,1000),xlab=expression(Biomass~kg/m^2),ylab="Derivative")
}
plot(y=c(derivative),x=c(simulated_logistics_data), type="l",col = "black",lwd=2,ylim =c(0,500),xlab="",ylab="")
title(xlab=expression(Biomass~kg/m^2),ylab=expression(Derivatives~kg/(m^2~day)),line=2.5)
for (i in c(1:5)){
  lines(y=c(unlist(df_time_dependent_derivative[i])),x=c(unlist(df_time_dependent[i])), type="l",col = "green",lwd=2,ylim =c(0,500),xlab=expression(Biomass~kg/m^2),ylab="Derivative")
}





plot(simulated_logistics_data,col = "black",lwd=2,xlab="",ylab="")
title(xlab="Time day",ylab=expression(Biomass~kg/m^2),line=2)
df_independent <- data.frame(matrix(nrow = end.time, ncol = 0))
df_independent_derivative <- data.frame(matrix(nrow = end.time, ncol = 0))

for (i in c(1:5)){
noise <- rnorm(1,0,0.25) # idependent NOISE
s_time_in <- expression(noise)# time_independent_noise_0.25
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s_time_in, M = 1)*1000 -> simulated_logistics_data_independent
df_independent[ , ncol(df_independent) + 1] = simulated_logistics_data_independent
derivative_independent <-D1tr(y=simulated_logistics_data_independent, x = time.vec)
df_independent_derivative[ , ncol(df_independent_derivative) + 1] = derivative_independent
lines(simulated_logistics_data_independent,col = "orange",lwd=2)
}
plot(y=c(derivative),x=c(simulated_logistics_data), type="l",col = "black",lwd=2,ylim =c(0,500),xlab="",ylab="")
title(xlab=expression(Biomass~kg/m^2),ylab=expression(Derivatives~kg/(m^2~day)),line=2.5)
for (i in c(1:5)){
  lines(y=c(unlist(df_independent_derivative[i])),x=c(unlist(df_independent[i])), type="l",col = "orange",lwd=2,ylim =c(0,500),xlab=expression(Biomass~kg/m^2),ylab="Derivative")
}




############plot four growth model on the same plot 
r <- 0.15 
Mmax<-6000/1000
end.time <-120
time.vec <- 0:119
### Stochastic logistic equation


y0 <- 4.6/1000
d <- expression(r * x * (1 - x/Mmax) ) 
s <- expression(0*x) #without_noise
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s, M = 1)*1000-> simulated_logistics_data
sp <-smooth.spline(y=simulated_logistics_data,x=time.vec,all.knots = TRUE)
derivative <-  predict(sp, x=time.vec, deriv = 1,)

#lines(y=c(derivative),x=time.vec,col = "red",lwd=2,xlab="Biomass",ylab="Derivative",type="l")
plot(y=c(derivative$y),x=c(simulated_logistics_data), type="l",col = "black",lwd=2,ylim =c(0,500),xlab="",ylab="")
title(xlab=expression(Biomass~kg/m^2),ylab=expression(Derivatives~kg/(m^2~day)),line=2.5)

# df[ , ncol(df) + 1] = simulated_logistics_data
# df[ , ncol(df) + 1] = derivative
r <- 0.05 #Aa
a <- 0.3
fi <- 0.01168281 #runif(1,1/365,182/365)
d_irradiance <- expression((r+(a*sin((2*pi/365)*t+fi))) * x * (1 - x/Mmax))
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_irradiance, sigma=s, M = 1)*1000 -> simulated_irradiance_data
sp <-smooth.spline(y=simulated_irradiance_data,x=time.vec,all.knots = TRUE)
derivative_irradiance <- predict(sp, x=time.vec, deriv = 1,)$y
#derivative_irradiance <-D1tr(y=simulated_irradiance_data, x = time.vec)
lines(y=c(derivative_irradiance),x=c(simulated_irradiance_data),col = "red",lwd=2)
#plot(derivative_irradiance)
r <- 0.15
Ma <- runif(1,0,y0) #yo>Ma
d_Allee <- expression(r*x*(1 - x/Mmax)*(x/(x + Ma))) #dM <- r*M*(1 - M/Mx)*(M/(M + Ma))

sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_Allee, sigma=s, M = 1)*1000 -> simulated_Allee_data
sp <-smooth.spline(y=simulated_Allee_data,x=time.vec,all.knots = TRUE)
derivative_Allee <-  predict(sp, x=time.vec, deriv = 1,)$y
#derivative_Allee <-D1tr(y=simulated_Allee_data, x = time.vec)
lines(y=c(derivative_Allee),x=c(simulated_Allee_data),col = "green",lwd=2)

r=0.3
TAL= 20000
TL = 292
TAH = 60000
TH = 303
weather_condition <- read.csv("data/simulated_data/netherland 2022-02-01 to 2022-05-31.csv")
temperature_list <- weather_condition$tempmin+20
fit_temperature<- smooth.spline(temperature_list,nknots=10)
temp_t <- fit_temperature$y
#if with temperature tolerance snp(dd), the parameter will be 0.5 so the is less affected by temperature
r.adapt <- (1 + ((exp(TAL/(temp_t + 273) - TAL/TL) + exp(TAH/TH - TAH/(temp_t + 273)))))^{-1}
d_temp <- expression(r.adapt[t]*r * x * (1 - x/Mmax)) 
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_temp, sigma=s, M = 1)*1000 -> simulated_temp_data
sp <-smooth.spline(y=simulated_temp_data,x=time.vec,all.knots = TRUE)
derivative_temp <-  predict(sp, x=time.vec, deriv = 1,)$y
#derivative_temp <-D1tr(y=simulated_temp_data, x = time.vec)
lines(y=c(derivative_temp),x=c(simulated_temp_data),col = "blue",lwd=2)

legend(x = "topright",     
       legend = c("Logistic", "Irradiane","Allee","Temperature"),
       lty = c(1, 1,1,1),           # Line types
       col = c("black","red","green","blue"),           # Line colors
       lwd = 1,
       cex=0.7)

plot(simulated_logistics_data,col = "black",lwd=2,ylab="",xlab="")
title(xlab="Time day",ylab=expression(Biomass~kg/m^2),line=2)
lines(simulated_irradiance_data,col = "red",lwd=2)
lines(simulated_Allee_data,col = "green",lwd=2)
lines(simulated_temp_data,col = "blue",lwd=2)
legend(x = "bottomright",     
       legend = c("Logistic", "Irradiane","Allee","Temperature"),
       lty = c(1, 1,1,1),           # Line types
       col = c("black","red","green","blue"),           # Line colors
       lwd = 2,
       cex=1)
# 

plot(temp_t,type="l",xlab="Time day",ylab="Temperature ℃",lwd = 2,col="blue")
r.adapt_function <- function(temp_t){
  TAL= 20000
  TL = 292
  TAH = 60000
  TH = 303
  (1 + ((exp(TAL/(temp_t + 273) - TAL/TL) + exp(TAH/TH - TAH/(temp_t + 273)))))^{-1}
  }
temperature = c(0:40)
factor = laply(temperature,r.adapt_function)
plot(x=temperature,factor,type="l",lwd = 2,col="blue",ylab = "r.adapt",xlab="Temperature ℃")
irradiance_effect <- function(t){
  a=0.3
  fi <- 60/365
  a*sin((2*pi/365)*t+fi)
}
t <- c(1:365)
Irradiance.effect = laply(t,irradiance_effect)
plot(y=Irradiance.effect,x=t,type="l",lwd = 2,col="red",xlab="Time day",xlim=c(1,365))
abline(v=120, col="black")

################################################################################################

# ############plot four growth model on different plot 
Mmax<-6000/1000
end.time <-120
time.vec <- 0:119
#create data frame
df_logistic <- data.frame(matrix(nrow = end.time, ncol = 0))
df_logistic_derivative <- data.frame(matrix(nrow = end.time, ncol = 0))
for (i in c(1:10)){
  simulated_logistics_data <- c(NA)
while(sum(is.na(simulated_logistics_data)) !=0||(tail(simulated_logistics_data, n=1)<(Mmax*1000-2000))){
r <- rnorm(1,0.25,0.25)
### Stochastic logistic equation
y0 <- 4.6/1000
d <- expression(r * x * (1 - x/Mmax) )
s <- expression(0*x) #without_noise
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s, M = 1)*1000-> simulated_logistics_data
sp <-smooth.spline(y=simulated_logistics_data,x=time.vec,all.knots = TRUE)
derivative <-  predict(sp, x=time.vec, deriv = 1)$y
}
  df_logistic[ , ncol(df_logistic) + 1] = simulated_logistics_data
  df_logistic_derivative[ , ncol(df_logistic_derivative) + 1] = derivative
  
  }
plot(y=0,x=0, type="l",col = "black",lwd=2,ylim =c(0,1000),xlim=c(0,6000),xlab="",ylab="")
title(xlab=expression(Biomass~kg/m^2),ylab=expression(Derivatives~kg/(m^2~day)),line=2.5)
plot(y=0,x=0, type="l",col = "black",lwd=2,ylim =c(0,1000),xlim=c(0,120),xlab="",ylab="")
title(xlab=expression(Time~day),ylab=expression(Derivatives~kg/(m^2~day)),line=2.5)
lines(y=c(unlist(df_logistic_derivative[i])),x=c(time.vec), type="l",col = "black",lwd=2)

for (i in c(1:10)){
  lines(y=c(unlist(df_logistic_derivative[i])),x=c(unlist(df_logistic[i])), type="l",col = "black",lwd=2)
}
plot(y=0,x=0,col = "black",lwd=2,ylab="",xlab="",ylim =c(0,6000),xlim=c(0,120))
title(xlab="Time day",ylab=expression(Biomass~kg/m^2),line=2)
for (i in c(1:10)){
  lines(c(unlist(df_logistic[i])), type="l",col = "black",lwd=2)
}


df_irradiance <- data.frame(matrix(nrow = end.time, ncol = 0))
df_irradiance_derivative <- data.frame(matrix(nrow = end.time, ncol = 0))

for (i in c(1:10)){
  simulated_irradiance_data <- c(NA)
  while(sum(is.na(simulated_irradiance_data)) !=0|| (tail(simulated_irradiance_data, n=1)<(Mmax*1000-2000))){
  r <- rnorm(1,0.25,0.25) 
  a <- runif(1,-0.5,0.5)
  fi <- runif(1,1/365,182/365)
  d_irradiance <- expression((r+(a*sin((2*pi/365)*t+fi))) * x * (1 - x/Mmax))
  sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_irradiance, sigma=s, M = 1)*1000 -> simulated_irradiance_data
  sp <-smooth.spline(y=simulated_irradiance_data,x=time.vec,all.knots = TRUE)
  derivative_irradiance <- predict(sp, x=time.vec, deriv = 1,)$y

  }
  
  df_irradiance[ , ncol(df_irradiance) + 1] = simulated_irradiance_data
  df_irradiance_derivative[ , ncol(df_irradiance_derivative) + 1] = derivative_irradiance
  
}
plot(y=0,x=0, type="l",col = "black",lwd=2,ylim =c(0,1000),xlim=c(0,6000),xlab="",ylab="")
title(xlab=expression(Biomass~kg/m^2),ylab=expression(Derivatives~kg/(m^2~day)),line=2.5)

for (i in c(1:10)){
  lines(y=c(unlist(df_irradiance_derivative[i])),x=c(unlist(df_irradiance[i])), type="l",col = "red",lwd=2,ylim =c(0,600),xlab=expression(Biomass~kg/m^2),ylab="Derivative")
}
plot(y=0,x=0,col = "black",lwd=2,ylab="",xlab="",ylim =c(0,6000),xlim=c(0,120))
title(xlab="Time day",ylab=expression(Biomass~kg/m^2),line=2)
for (i in c(1:10)){
  lines(c(unlist(df_irradiance[i])), type="l",col = "red",lwd=2,ylim =c(0,1000))
}


df_allee <- data.frame(matrix(nrow = end.time, ncol = 0))
df_allee_derivative <- data.frame(matrix(nrow = end.time, ncol = 0))

for (i in c(1:10)){
  simulated_Allee_data <- c(NA)
  while(sum(is.na(simulated_Allee_data)) !=0|| (tail(simulated_Allee_data, n=1)<(Mmax*1000-2000))){
r <- rnorm(1,0.25,0.25) 
Ma <- runif(1,0,y0) #yo>Ma
d_Allee <- expression(r*x*(1 - x/Mmax)*(x/(x + Ma))) #dM <- r*M*(1 - M/Mx)*(M/(M + Ma))

sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_Allee, sigma=s, M = 1)*1000 -> simulated_Allee_data
sp <-smooth.spline(y=simulated_Allee_data,x=time.vec,all.knots = TRUE)
derivative_Allee <-  predict(sp, x=time.vec, deriv = 1,)$y
}

df_allee[ , ncol(df_allee) + 1] = simulated_Allee_data
df_allee_derivative[ , ncol(df_allee_derivative) + 1] = derivative_Allee
}

plot(y=0,x=0, type="l",col = "black",lwd=2,ylim =c(0,1000),xlim=c(0,6000),xlab="",ylab="")
title(xlab=expression(Biomass~kg/m^2),ylab=expression(Derivatives~kg/(m^2~day)),line=2.5)

for (i in c(1:10)){
  lines(y=c(unlist(df_allee_derivative[i])),x=c(unlist(df_allee[i])), type="l",col = "green",lwd=2,ylim =c(0,600),xlab=expression(Biomass~kg/m^2),ylab="Derivative")
}
plot(y=0,x=0,col = "black",lwd=2,ylab="",xlab="",ylim =c(0,6000),xlim=c(0,120))
title(xlab="Time day",ylab=expression(Biomass~kg/m^2),line=2)
for (i in c(1:10)){
  lines(c(unlist(df_allee[i])), type="l",col = "green",lwd=2,ylim =c(0,1000))
}

df_temperature <- data.frame(matrix(nrow = end.time, ncol = 0))
df_temperature_derivative <- data.frame(matrix(nrow = end.time, ncol = 0))
for (i in c(1:10)){
  simulated_temp_data <- c(NA)
  while(sum(is.na(simulated_temp_data)) !=0|| (tail(simulated_temp_data, n=1)<(Mmax*1000-2000))){
  r <- rnorm(1,0.25,0.25) 
  TAL= 20000
  TL = 292
  TAH = 60000
  TH = 303
  weather_condition <- read.csv("data/simulated_data/netherland 2022-02-01 to 2022-05-31.csv")
  temperature_list <- weather_condition$tempmin+20
  fit_temperature<- smooth.spline(temperature_list,nknots=10)
  temp_t <- fit_temperature$y
  #if with temperature tolerance snp(dd), the parameter will be 0.5 so the is less affected by temperature
  r.adapt <- (1 + ((exp(TAL/(temp_t + 273) - TAL/TL) + exp(TAH/TH - TAH/(temp_t + 273)))))^{-1}
  d_temp <- expression(r.adapt[t]*r * x * (1 - x/Mmax)) 
  sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_temp, sigma=s, M = 1)*1000 -> simulated_temp_data
  sp <-smooth.spline(y=simulated_temp_data,x=time.vec,all.knots = TRUE)
  derivative_temp <-  predict(sp, x=time.vec, deriv = 1,)$y
}
#derivative_temp <-D1tr(y=simulated_temp_data, x = time.vec)
#lines(y=c(derivative_temp),x=c(simulated_temp_data),col = "blue",lwd=2)

df_temperature[ , ncol(df_temperature) + 1] = simulated_temp_data
df_temperature_derivative[ , ncol(df_temperature_derivative) + 1] = derivative_temp
}

plot(y=0,x=0, type="l",col = "black",lwd=2,ylim =c(0,1000),xlim=c(0,6000),xlab="",ylab="")
title(xlab=expression(Biomass~kg/m^2),ylab=expression(Derivatives~kg/(m^2~day)),line=2.5)

for (i in c(1:10)){
  lines(y=c(unlist(df_temperature_derivative[i])),x=c(unlist(df_temperature[i])), type="l",col = "blue",lwd=2,ylim =c(0,600),xlab=expression(Biomass~kg/m^2),ylab="Derivative")
}
plot(y=0,x=0,col = "black",lwd=2,ylab="",xlab="",ylim =c(0,6000),xlim=c(0,120))
title(xlab="Time day",ylab=expression(Biomass~kg/m^2),line=2)
for (i in c(1:10)){
  lines(c(unlist(df_temperature[i])), type="l",col = "blue",lwd=2,ylim =c(0,1000))
}








legend(x = "topright",     
       legend = c("Logistic", "Irradiane","Allee","Temperature"),
       lty = c(1, 1,1,1),           # Line types
       col = c("black","red","green","blue"),           # Line colors
       lwd = 1,
       cex=0.7)

plot(simulated_logistics_data,col = "black",lwd=2,ylab="",xlab="")
title(xlab="Time day",ylab=expression(Biomass~kg/m^2),line=2)
lines(simulated_irradiance_data,col = "red",lwd=2)
lines(simulated_Allee_data,col = "green",lwd=2)
lines(simulated_temp_data,col = "blue",lwd=2)
legend(x = "bottomright",     
       legend = c("Logistic", "Irradiane","Allee","Temperature"),
       lty = c(1, 1,1,1),           # Line types
       col = c("black","red","green","blue"),           # Line colors
       lwd = 2,
       cex=1)
# 

plot(temp_t,type="l",xlab="Time day",ylab="Temperature ℃",lwd = 2,col="blue")
r.adapt_function <- function(temp_t){
  TAL= 20000
  TL = 292
  TAH = 60000
  TH = 303
  (1 + ((exp(TAL/(temp_t + 273) - TAL/TL) + exp(TAH/TH - TAH/(temp_t + 273)))))^{-1}
}
temperature = c(0:40)
factor = laply(temperature,r.adapt_function)
plot(x=temperature,factor,type="l",lwd = 2,col="blue",ylab = "r.adapt",xlab="Temperature ℃")
irradiance_effect <- function(t){
  a=0.3
  fi <- 60/365
  a*sin((2*pi/365)*t+fi)
}
t <- c(1:365)
Irradiance.effect = laply(t,irradiance_effect)
plot(y=Irradiance.effect,x=t,type="l",lwd = 2,col="red",xlab="Time day",xlim=c(1,365))
abline(v=120, col="black")

