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
library('sfsmisc') # get the derivative
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
r <- 0.5 #Aa
Mmax_range<-6000
end.time <-120
time.vec <- 0:119
### Stochastic logistic equation
set.seed(123)


df <- data.frame(matrix(nrow = end.time, ncol = 0))
r_list = list()
s_list = list()
#label_list = list()
for (i in c(0:100)){
  #runif() generates random deviates of the uniform distribution
  # r <- rnorm(1,0.5,0.25)
  # Mmax<-rnorm(1,6000,100)/1000
  #Mmax <- runif(1,5000,6000)
  y0 <- 4.6/1000
  d <- expression(r * x * (1 - x/Mmax) ) 
  s <- expression(0*x) #without_noise
  noise <- rnorm(1,0,0.25) # idependent NOISE
  s <- expression(noise)# time_independent_noise_0.25
  s <- expression(0.2*((2*(end.time-t)/end.time)*(1-(end.time-t)/end.time)))# time_dependent_noise_0.2
  s_biomass <- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax)))# NOISE related to biomass
  sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s_biomass, M = 1) -> simulated_logistics_data
  df[ , ncol(df) + 1] = simulated_logistics_data
  plot(simulated_logistics_data,col = "red",lwd=2)
  s_time <- expression(0.2*((2*(end.time-t)/end.time)*(1-(end.time-t)/end.time)))# time_dependent_noise_0.2
  sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s_time, M = 1) -> simulated_logistics_data
  df[ , ncol(df) + 1] = simulated_logistics_data
  lines(simulated_logistics_data,col = "darkgreen",lwd=2)
  s_time_in <- expression(noise)# time_independent_noise_0.25
  sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s_time_in, M = 1) -> simulated_logistics_data
  df[ , ncol(df) + 1] = simulated_logistics_data
  lines(simulated_logistics_data,col = "darkorange",lwd=2)
  s <- expression(0*x)
  sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s, M = 1) -> simulated_logistics_data

  plot(simulated_logistics_data,col = "black",lwd=2)
  legend(x = "bottomright",
         legend = c("biomass dependent noise", "time dependent noise","time independent noise","without noise"),
         lty = c(1, 1,1,1),           # Line types
         col = c("red","darkgreen","darkorange","black"),           # Line colors
         lwd = 2)


  
  df[ , ncol(df) + 1] = simulated_logistics_data
  # r_list <- append(r_list,r)
  #label_list <- append(label_list,"0")
  #dev.off()
  derivative <-D1tr(y=simulated_logistics_data, x = time.vec)
  plot(derivative)
  derivative_fit_spline <- smooth.spline(y=c(derivative),x=c(simulated_logistics_data),nknots=36)
  plot(derivative_fit_spline, col = "darkgreen",type='l',xlim = c(0, 7),ylim = c(0.0,1.0),ylab="",xlab="")
  ## When the device is off, file writing is completed.
  dev.off()

  plot(y=c(derivative),x=c(simulated_logistics_data),type='l', col = "green",xlim = c(0, 7),ylim = c(0.0,1.0))
  
  plot(time.vec,simulated_logistics_data)
  lines(fit, col = "blue")
}
write.csv(df,"four_noise_types_example_use_for_plot.csv")

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


############plot four growth model on the same plot 
r <- 0.6 #Aa
Mmax_range<-6000
end.time <-120
time.vec <- 0:119
### Stochastic logistic equation
set.seed(123)

y0 <- 4.6/1000
d <- expression(r * x * (1 - x/Mmax) ) 
s <- expression(0*x) #without_noise
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s, M = 1) -> simulated_logistics_data
derivative <-D1tr(y=simulated_logistics_data, x = time.vec)
plot(y=c(derivative),x=c(simulated_logistics_data), type="l",col = "black",lwd=2,ylim =c(0,1),xlab="biomass",ylab="derivative")
df[ , ncol(df) + 1] = simulated_logistics_data
df[ , ncol(df) + 1] = derivative
a <- 0.5
fi <- runif(1,1/365,182/365)
d_irradiance <- expression((r+(a*sin((2*pi/365)*t+fi))) * x * (1 - x/Mmax))
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_irradiance, sigma=s, M = 1) -> simulated_irradiance_data
derivative_irradiance <-D1tr(y=simulated_irradiance_data, x = time.vec)
lines(y=c(derivative_irradiance),x=c(simulated_irradiance_data),col = "red",lwd=2)


Ma <- runif(1,0,y0) #yo>Ma
d_Allee <- expression(r*x*(1 - x/Mmax)*(x/(x + Ma))) 
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_Allee, sigma=s, M = 1) -> simulated_Allee_data
derivative_Allee <-D1tr(y=simulated_Allee_data, x = time.vec)
lines(y=c(derivative_Allee),x=c(simulated_Allee_data),col = "green",lwd=2)


TAL= 20000
TL = 292
TAH = 60000
TH = 303
temp_t = temperature_list
#if with temperature tolerance snp(dd), the parameter will be 0.5 so the is less affected by temperature
r.adapt <- (1 + ((exp(TAL/(temp_t + 273) - TAL/TL) + exp(TAH/TH - TAH/(temp_t + 273)))))^{-1}
d_temp <- expression(r.adapt*r * x * (1 - x/Mmax)) 
sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d_temp, sigma=s, M = 1) -> simulated_temp_data
derivative_temp <-D1tr(y=simulated_temp_data, x = time.vec)
lines(y=c(derivative_temp),x=c(simulated_temp_data),col = "blue",lwd=2)

legend(x = "topright",     
       legend = c("Logistic", "irradiane","Allee","temperature"),
       lty = c(1, 1,1,1),           # Line types
       col = c("black","red","green","blue"),           # Line colors
       lwd = 1,
       cex=0.7)

plot(simulated_logistics_data,col = "black",lwd=2,ylab="biomass")
lines(simulated_irradiance_data,col = "red",lwd=2)
lines(simulated_Allee_data,col = "green",lwd=2)
lines(simulated_temp_data,col = "blue",lwd=2)
legend(x = "bottomright",     
       legend = c("Logistic", "irradiane","Allee","temperature"),
       lty = c(1, 1,1,1),           # Line types
       col = c("black","red","green","blue"),           # Line colors
       lwd = 2,
       cex=1)
