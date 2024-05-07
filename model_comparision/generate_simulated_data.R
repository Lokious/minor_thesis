## this script is to generate the simulated dataset with SDE and it's derivative
#sm.spline() gives error while smooth.spline() does not
#https://stat.ethz.ch/pipermail/r-help/2011-May/279545.html
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
#install.packages("Cairo")
library(naniar)
library(sde) ### https://rdrr.io/cran/sde/man/sde.sim.html
library('pspline')
library('sfsmisc') # get the derivative
library('Cairo')
#temperature data is download from https://www.visualcrossing.com/weather/weather-data-services# netherland from Feb. to May and add 15
#because when tempreture is too low the tempreture curve grows too slow
weather_condition <- read.csv("data/simulated_data/netherland 2022-02-01 to 2022-05-31.csv")
temperature_list <- weather_condition$tempmin+20
fit_temperature<- smooth.spline(temperature_list,nknots=10)
temperature_list <- fit_temperature$y

set.seed(123)
# read weather condition from the ELOPE platform weather condition
#weather_condition <- read.csv("../elope-main/ELOPE_raw_data/data_platform/platform_envCovariates/WeatherConditions_Mean.csv")
# temperature_condition <- weather_condition[c('Day','tempmean')]
# # calculate daily mean
# df =aggregate(temperature_condition$tempmean,list(temperature_condition$Day),mean)
# temperature_list <- df$x



#simulated_snps which has effect on the Max_biomass(aa,Aa,AA):(Mmax<-runif(5800,5900),Mmax<-runif(5900,6000),Mmax<-runif(6000,6100)), growth rate:r(bb,Bb,BB) and high temperature change tolerance(cc), and irridance change tolerance(dd)
# combination of five snps, we assumne only cc and dd has the effect of the last two, and the first two is related to the number of A at the position.
# so we generate 3*3*2*2 =36 different genotypes based on that(2 for snps c and d, because there are no difference between Cc and CC)

# three different r range
r_range_1 <- 0.25 #aa
r_range_2 <- 0.5 #Aa
r_range_3 <- 0.75 #AA
# three different Mmax
Mmax_range_1<-5900#bb
Mmax_range_2<-6000 #Bb
Mmax_range_3<-6100 #BB
snp_1 = 0
for (r in list(r_range_1,r_range_2,r_range_3)){print(r)}

for (r_range in list(r_range_1,r_range_2,r_range_3)){
  snp_2 = 0

  for (Mmax_range in list(Mmax_range_1,Mmax_range_2,Mmax_range_3)){
    snp_3 = 0
    
    for (irradiance_tolarence in list(1,0.5)){
      snp_4 = 0
      # 1 means this genotype is sensitive to irradiance, while for 0.5 we use 0.5 multiply by irradiance effect
      for (temperature_tolerance in list(1,0.5)){
        #same as irradiance effect
        
        noise_type = c('time_independent_noise_0.25', 'time_dependent_noise_0.2', 'biomass_dependent_noise_0.2','without_noise')
        
        for (noise_name in noise_type){
          
          # time steps is 120 days
          end.time = 120
          # end.biomass <- biomass[end.time] #5.8696
          time.vec <- 1:(end.time) # define a time vector from 0 to 120
          
          ### Stochastic logistic equation model ###
          
          df <- data.frame(matrix(nrow = end.time, ncol = 0))


          ### Stochastic logistic equation
          
          df <- data.frame(matrix(nrow = end.time, ncol = 0))
          derivative_df <- data.frame(matrix(nrow = end.time, ncol = 0))
          smooth_derivative_df<- data.frame(matrix(nrow = length(seq(1, 6, 0.25)), ncol = 0)) #use for saving predicted derivative from spline per 0.25 biomass
          r_list = list()
          Mmax_list = list()
          label_list = list()
          snp_1_list = list()
          snp_2_list = list()
          snp_3_list = list()
          snp_4_list = list()
          #plot(x=0,y=0,xlim = c(0, 7),ylim = c(0.0,0.5),xlab="biomass",ylab="derivative")
          for (i in c(1:50)){
            print(i)
            simulated_logistics_data <- c(NA)
            while(sum(is.na(simulated_logistics_data)) !=0|| (tail(simulated_logistics_data, n=1)<(Mmax-2))){
              #only keep the simulated data which doesn't have NA Inf or -Inf
              
              r <-rnorm(1,r_range,0.25)
              Mmax<-rnorm(1,Mmax_range,100)/1000
              # Mmax <-6
              y0 <- 4.6/1000
              d <- expression(r * x * (1 - x/Mmax)) 
              if (noise_name=="without_noise"){
                s <- expression(0*x) #without_noise
              }
              if (noise_name=="biomass_dependent_noise_0.2"){
                s<- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax))) # biomass_dependent_noise_0.2
              }
              if (noise_name=="time_independent_noise_0.25"){
                noise <- rnorm(1,0,0.25) # idependent NOISE
                s <- expression(noise)# time_independent_noise_0.25
              }
              if (noise_name=="time_dependent_noise_0.2"){
                s <- expression(0.2*((2*(end.time-t)/end.time)*(1-(end.time-t)/end.time)))# time_dependent_noise_0.2
                
              }
              
              #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
              #N:number of simulation steps.
              # diffusion coefficient: an expression of two variables t and x
              #M: number of trajectories.
              sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s, M = 1) -> simulated_logistics_data
              
              simulated_logistics_data <- replace(simulated_logistics_data, simulated_logistics_data==-Inf, NA)
              simulated_logistics_data <- replace(simulated_logistics_data, simulated_logistics_data==Inf, NA)
            }
            #plot(simulated_logistics_data)
            # save simulated data and parameters
            df[ , ncol(df) + 1] = simulated_logistics_data
            #save growth curve plot
            Cairo(file=paste0("data/simulated_data/simulated_with_different_gene_type/plot/growth_curve/simulated_X_data_logistic_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,"_",i,"_.tiff",sep=""),
                  type="tiff",
                  units="px", 
                  width=256, 
                  height=256, 
                  pointsize=12, 
                  dpi="auto")
            
            plot(simulated_logistics_data, col = "darkgreen",type='l',xlim = c(0, 120),ylim = c(0.0,7.0),ylab="",xlab="")
            dev.off()
            r_list <- append(r_list,r)
            Mmax_list <- append(Mmax_list,Mmax)
            label_list <- append(label_list,"0")
            # snp_1_list <- append(snp_1_list,snp_1)
            # snp_2_list <- append(snp_2_list,snp_2)
            # snp_3_list <- append(snp_3_list,snp_3)
            # snp_4_list <- append(snp_4_list,snp_1)
            fit <- smooth.spline(x=time.vec,y=simulated_logistics_data)
            derivative <-D1tr(y=simulated_logistics_data, x = time.vec)
            Cairo(file=paste0("data/simulated_data/simulated_with_different_gene_type/plot/smooth_derivative/simulated_X_data_logistic_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,"_",i,"_.tiff",sep=""),
                  type="tiff",
                  units="px", 
                  width=256, 
                  height=256, 
                  pointsize=12, 
                  dpi="auto")
            derivative_fit_spline <- smooth.spline(y=c(derivative),x=c(simulated_logistics_data),nknots=36)
            plot(derivative_fit_spline, col = "darkgreen",type='l',xlim = c(0, 7),ylim = c(0.0,2.0),ylab="",xlab="")
            ## When the device is off, file writing is completed.
            dev.off()
            predicted_derivative <- predict(derivative_fit_spline,x=seq(1, 6, 0.25))
            smooth_derivative_df[ , ncol(derivative_df) + 1] = predicted_derivative$y
            derivative_df[ , ncol(derivative_df) + 1] = derivative
            plot(y=c(derivative),x=c(simulated_logistics_data),type='l', col = "green",xlim = c(0, 7),ylim = c(0.0,1.0))
            
            # get the max on y, and matching x value
            derivateMax <- max(derivative)
            x_index <-which.max(derivative)
            # mark the max point on the plot
            points(y=derivateMax, x = simulated_logistics_data[x_index], col = "green", pch = 19)
            # add vertical line
            abline(v = simulated_logistics_data[x_index], col = "green", lty = "dashed")
          }
          
          ####save dataframe to files###
          #write biomass at 120 time steps to csv
          write.csv(df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_X_data_logistic_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          # rename label dataframe
          df_Y = data.frame(label_list)
          colnames(df_Y) <- c(1:50)
          # write label dataframe
          write.csv(df_Y,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_label_data_logistic_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          # write derivative dataframe to csv
          write.csv(derivative_df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_derivative_data_logistic_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          # write smoothed derivative dataframe to csv
          write.csv(smooth_derivative_df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_smoothed_derivative_data_logistic_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          df_r <- data.frame(r_list)
          df_new = rbind(df_r,Mmax_list)
          rownames(df_new)  <-c("r","Mmax")
          colnames(df_new) <- c(1:50)
          write.csv(df_new,paste0("data/simulated_data/simulated_with_different_gene_type/parameters_list_simulated_data_logistic_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          
          ######generate data from Irradiance model######
          df <- data.frame(matrix(nrow = end.time, ncol = 0))
          derivative_df <- data.frame(matrix(nrow = end.time, ncol = 0))
          smooth_derivative_df<- data.frame(matrix(nrow = length(seq(1, 6, 0.25)), ncol = 0))
          #creat list for saving parameters
          r_list = list()
          a_list = list()
          phi_list = list()
          Mmax_list = list()
          label_list = list()
          for (i in c(1:50)){
            print(i)
            simulated_irradiance_logistics_data <- c(NA)
            Mmax<-rnorm(1,Mmax_range,100)/1000
 
            while((sum(is.na(simulated_irradiance_logistics_data)) !=0) || (tail(simulated_irradiance_logistics_data, n=1)<(Mmax-2))){
              #only keep the simulated data which doesn't have NA Inf or -Inf
              #runif() generates random deviates of the uniform distribution
              r <-rnorm(1,r_range,0.25)
              y0 <-4.6/1000
              a <- runif(1, -0.5, 0.5)
              fi <- runif(1,1/365,182/365)
              
              d_irradiance <- expression((r+irradiance_tolarence*(a*sin((2*pi/365)*t+fi))) * x * (1 - x/Mmax))
              if (noise_name=="without_noise"){
                s_irradiance <- expression(0*x) #without_noise
              }
              if (noise_name=="biomass_dependent_noise_0.2"){
                s_irradiance<- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax))) # biomass_dependent_noise_0.2
              }
              if (noise_name=="time_independent_noise_0.25"){
                noise <- rnorm(1,0,0.25) # idependent NOISE
                s_irradiance <- expression(noise)# time_independent_noise_0.25
              }
              if (noise_name=="time_dependent_noise_0.2"){
                s_irradiance <- expression(0.2*((2*(end.time-t)/end.time)*(1-(end.time-t)/end.time)))# time_dependent_noise_0.2
                
              }
              
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
            phi_list <- append(phi_list,fi)
            Mmax_list <- append(Mmax_list,Mmax)
            label_list <- append(label_list,'1')
            
            Cairo(file=paste0("data/simulated_data/simulated_with_different_gene_type/plot/growth_curve/simulated_X_data_irradiance_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,"_",i,"_.tiff",sep=""),
                  type="tiff",
                  units="px", 
                  width=256, 
                  height=256, 
                  pointsize=12, 
                  dpi="auto")
            
            plot(simulated_irradiance_logistics_data, col = "red",type='l',xlim = c(0, 120),ylim = c(0.0,7.0),ylab="",xlab="")
            dev.off()
            fit <- smooth.spline(x=time.vec,y=simulated_irradiance_logistics_data)
            derivative <-D1tr(y=simulated_irradiance_logistics_data, x = time.vec)
            derivative_fit_spline <- smooth.spline(y=c(derivative),x=c(simulated_irradiance_logistics_data),nknots=36)
            predicted_derivative <- predict(derivative_fit_spline,x=seq(1, 6, 0.25))
            smooth_derivative_df[ , ncol(derivative_df) + 1] = predicted_derivative$y
            derivative_df[ , ncol(derivative_df) + 1] = derivative
            plot(y=c(derivative),x=c(simulated_irradiance_logistics_data),type='l', col = "red",xlim = c(0, 7),ylim = c(0.0,2.0))
            Cairo(file=paste0("data/simulated_data/simulated_with_different_gene_type/plot/smooth_derivative/simulated_X_data_irradiance_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,"_",i,"_.tiff",sep=""),
                  type="tiff",
                  units="px", 
                  width=256, 
                  height=256, 
                  pointsize=12, 
                  dpi="auto")
            plot(derivative_fit_spline, col = "salmon",type='l',xlim = c(0, 7),ylim = c(0.0,2.0),xlab="",ylab="")
            dev.off()
            # get the max on y, and matching x value
            derivateMax <- max(derivative)
            x_index <-which.max(derivative)
            # mark the max point on the plot
            points(y=derivateMax, x = simulated_irradiance_logistics_data[x_index], col = "red", pch = 19)
            # add vertical line
            abline(v = simulated_irradiance_logistics_data[x_index], col = "red", lty = "dashed")
            
          }
          
          #plot(simulated_irradiance_logistics_data)
          write.csv(df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_X_data_irradiance_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          df_Y = data.frame(label_list)
          colnames(df_Y) <- c(1:50)
          write.csv(df_Y,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_label_data_irradiance_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          # write derivative dataframe to csv
          write.csv(derivative_df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_derivative_data_irradiance_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          # write smoothed derivative dataframe to csv
          write.csv(smooth_derivative_df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_smoothed_derivative_data_irradiance_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          df_r <- data.frame(r_list)
          df_new = rbind(df_r,Mmax_list)
          df_new = rbind(df_new,phi_list)
          df_new = rbind(df_new,a_list)
          rownames(df_new)  <-c("r","Mmax","Phi","a")
          colnames(df_new) <- c(1:50)
          write.csv(df_new,paste0("data/simulated_data/simulated_with_different_gene_type/parameters_list_simulated_data_irradiance_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          
          ### Allee model ###
          set.seed(123)
          
          parameters_df<-subset(parameters_df, r>0)
          r_range <- c(min(parameters_df$r),max(parameters_df$r))
          y0_range <- c(min(parameters_df$y0),max(parameters_df$y0))
          
          df <- data.frame(matrix(nrow = end.time, ncol = 0))
          derivative_df <- data.frame(matrix(nrow = end.time, ncol = 0))
          smooth_derivative_df<- data.frame(matrix(nrow = length(seq(1, 6, 0.25)), ncol = 0))
          #list for saving parameters
          r_list = list()
          Ma_list = list()
          Mmax_list = list()
          label_list = list()
          
          for (i in c(1:50)){
            print(i)
            simulated_allee_data <- NA
            while(sum(is.na(simulated_allee_data)) !=0 || (tail(simulated_allee_data, n=1)<(Mmax-2))){
              #only keep the simulated data which doesn't have NA Inf or -Inf
              
              r <-rnorm(1,r_range,0.25)
              Mmax<-rnorm(1,Mmax_range,100)/1000
              y0 <- 4.6/1000
              Ma <- runif(1,0,y0) #yo>Ma
              d <- expression(r*x*(1 - x/Mmax)*(x/(x + Ma))) 
              if (noise_name=="without_noise"){
                s_allee <- expression(0*x) #without_noise
              }
              if (noise_name=="biomass_dependent_noise_0.2"){
                s_allee<- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax))) # biomass_dependent_noise_0.2
              }
              if (noise_name=="time_independent_noise_0.25"){
                noise <- rnorm(1,0,0.25) # idependent NOISE
                s_allee <- expression(noise)# time_independent_noise_0.25
              }
              if (noise_name=="time_dependent_noise_0.2"){
                s_allee <- expression(0.2*((2*(end.time-t)/end.time)*(1-(end.time-t)/end.time)))# time_dependent_noise_0.2
                
              }
              #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
              #N:number of simulation steps.
              # diffusion coefficient: an expression of two variables t and x
              #M: number of trajectories.
              sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s_allee, M = 1) -> simulated_allee_data
              
              #plot(simulated_allee_data)
              
              simulated_allee_data <- replace(simulated_allee_data, simulated_allee_data==-Inf, NA)
              simulated_allee_data <- replace(simulated_allee_data, simulated_allee_data==Inf, NA)
            }
            #plot(simulated_allee_data)
            # save simulated data and parameters
            df[ , ncol(df) + 1] = simulated_allee_data
            r_list <- append(r_list,r)
            Ma_list <-append(Ma_list,Ma)
            Mmax_list <-append(Mmax_list,Mmax)
            label_list <- append(label_list,"2")
            Cairo(file=paste0("data/simulated_data/simulated_with_different_gene_type/plot/growth_curve/simulated_X_data_Allee_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,"_",i,"_.tiff",sep=""),
                  type="tiff",
                  units="px", 
                  width=256, 
                  height=256, 
                  pointsize=12, 
                  dpi="auto")
            
            plot(simulated_allee_data, col = "orange",type='l',xlim = c(0, 120),ylim = c(0.0,7.0),ylab="",xlab="")
            dev.off()
            fit <- smooth.spline(x=time.vec,y=simulated_allee_data)
            derivative <-D1tr(y=simulated_allee_data, x = time.vec)
            derivative_fit_spline <- smooth.spline(y=c(derivative),x=c(simulated_allee_data),nknots=36)
            predicted_derivative <- predict(derivative_fit_spline,x=seq(1, 6, 0.25))
            smooth_derivative_df[ , ncol(derivative_df) + 1] = predicted_derivative$y
            derivative_df[ , ncol(derivative_df) + 1] = derivative
            
            Cairo(file=paste0("data/simulated_data/simulated_with_different_gene_type/plot/smooth_derivative/simulated_X_data_Allee_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,"_",i,"_.tiff",sep=""),
                  type="tiff",
                  units="px", 
                  width=256, 
                  height=256, 
                  pointsize=12, 
                  dpi="auto")
            plot(derivative_fit_spline, col = "darkorange",type='l',xlim = c(0, 7),ylim = c(0.0,2.0),xlab="",ylab="")
            dev.off()
            plot(y=c(derivative),x=c(simulated_allee_data),type='l', col = "orange",xlim = c(0, 7),ylim = c(0.0,2.0))
            # get the max on y, and matching x value
            derivateMax <- max(derivative)
            x_index <-which.max(derivative)
            # mark the max point on the plot
            points(y=derivateMax, x = simulated_allee_data[x_index], col = "orange", pch = 19)
            # add vertical line
            abline(v = simulated_allee_data[x_index], col = "orange", lty = "dashed")
            
          }
          
          #plot(simulated_allee_data)
          ### without_noise; time_independent_noise_0.25; time_dependent_noise_0.2; biomass_dependent_noise_0.2
          write.csv(df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_X_data_Allee_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          df_Y = data.frame(label_list)
          colnames(df_Y) <- c(1:50)
          colnames(derivative_df) <- c(1:50)
          write.csv(df_Y,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_label_data_Allee_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          # write derivative dataframe to csv
          write.csv(derivative_df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_derivative_data_Allee_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          # write smoothed derivative dataframe to csv
          write.csv(smooth_derivative_df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_smoothed_derivative_data_Allee_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          df_r <- data.frame(r_list)
          df_new = rbind(df_r,Mmax_list)
          df_new = rbind(df_new,Ma_list)
          rownames(df_new)  <-c("r","Mmax","Ma")
          colnames(df_new) <- c(1:50)
          write.csv(df_new,paste0("data/simulated_data/simulated_with_different_gene_type/parameters_list_simulated_data_Allee_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          
          # ### temperature model ###
          set.seed(123)
          
          parameters_df<-subset(parameters_df, r>0)
          r_range <- c(min(parameters_df$r),max(parameters_df$r))
          y0_range <- c(min(parameters_df$y0),max(parameters_df$y0))
          
          df <- data.frame(matrix(nrow = end.time, ncol = 0))
          derivative_df <- data.frame(matrix(nrow = end.time, ncol = 0))
          smooth_derivative_df<- data.frame(matrix(nrow = length(seq(1, 6, 0.25)), ncol = 0))
          r_list = list()
          Mmax_list = list()
          label_list = list()
          
          for (i in c(1:50)){
            print(i)
            simulated_temperature_data <- NA
            while(sum(is.na(simulated_temperature_data)) !=0 || (tail(simulated_temperature_data, n=1)<(Mmax-2))){ 
              #only keep the simulated data which doesn't have NA Inf or -Inf
              
              r <-rnorm(1,r_range,0.25)
              Mmax<-rnorm(1,Mmax_range,100)/1000
              y0 <- 4.6/1000
              TAL= 20000
              TL = 292
              TAH = 60000
              TH = 303
              temp_t = temperature_list
              #if with temperature tolerance snp(dd), the parameter will be 0.5 so the is less affected by temperature
              r.adapt <- (1 + temperature_tolerance*((exp(TAL/(temp_t + 273) - TAL/TL) + exp(TAH/TH - TAH/(temp_t + 273)))))^{-1}
            
              d <- expression(r.adapt[t]*r * x * (1 - x/Mmax)) 
              
              if (noise_name=="without_noise"){
                s_temperature <- expression(0*x) #without_noise
              }
              if (noise_name=="biomass_dependent_noise_0.2"){
                s_temperature<- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax))) # biomass_dependent_noise_0.2
              }
              if (noise_name=="time_independent_noise_0.25"){
                noise <- rnorm(1,0,0.25) # idependent NOISE
                s_temperature <- expression(noise)# time_independent_noise_0.25
              }
              if (noise_name=="time_dependent_noise_0.2"){
                s_temperature <- expression(0.2*((2*(end.time-t)/end.time)*(1-(end.time-t)/end.time)))# time_dependent_noise_0.2
                
              }
              #delta: time step of the simulation,the fixed amount of time by which the simulation advances.
              #N:number of simulation steps.
              # diffusion coefficient: an expression of two variables t and x
              #M: number of trajectories.
              sde.sim(X0=y0, delta=1, N=(end.time-1), drift=d, sigma=s_temperature, M = 1) -> simulated_temperature_data
              
              plot(simulated_temperature_data)
              
              simulated_temperature_data <- replace(simulated_temperature_data, simulated_temperature_data==-Inf, NA)
              simulated_temperature_data <- replace(simulated_temperature_data, simulated_temperature_data==Inf, NA)
            }
            #plot(simulated_temperature_data)
            # save simulated data and parameters
            df[ , ncol(df) + 1] = simulated_temperature_data
            r_list <- append(r_list,r)
            Mmax_list <-append(Mmax_list,Mmax)
            label_list <- append(label_list,"3")
            Cairo(file=paste0("data/simulated_data/simulated_with_different_gene_type/plot/growth_curve/simulated_X_data_Temperature_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,"_",i,"_.tiff",sep=""),
                  type="tiff",
                  units="px",
                  width=256,
                  height=256,
                  pointsize=12,
                  dpi="auto")
            
            plot(simulated_temperature_data, col = "grey",type='l',xlim = c(0, 120),ylim = c(0.0,7.0),ylab="",xlab="")
            dev.off()
            fit <- smooth.spline(x=time.vec,y=simulated_temperature_data)
            derivative <-D1tr(y=simulated_temperature_data, x = time.vec)
            derivative_fit_spline <- smooth.spline(y=c(derivative),x=c(simulated_temperature_data),nknots=36)
            predicted_derivative <- predict(derivative_fit_spline,x=seq(1, 6, 0.25))
            smooth_derivative_df[ , ncol(derivative_df) + 1] = predicted_derivative$y
            derivative_df[ , ncol(derivative_df) + 1] = derivative
            plot(y=c(derivative),x=c(simulated_temperature_data),type='l', col = "grey",xlim = c(0, 7),ylim = c(0.0,2.0))
            Cairo(file=paste0("data/simulated_data/simulated_with_different_gene_type/plot/smooth_derivative/simulated_X_data_Temperature_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,"_",i,"_.tiff",sep=""),
                  type="tiff",
                  units="px",
                  width=256,
                  height=256,
                  pointsize=12,
                  dpi="auto")
            plot(derivative_fit_spline, col = "darkgrey",type='l',xlim = c(0, 7),ylim = c(0.0,2.0),xlab="",ylab="")
            dev.off()
            # get the max on y, and matching x value
            derivateMax <- max(derivative)
            x_index <-which.max(derivative)
            # mark the max point on the plot
            points(y=derivateMax, x = simulated_temperature_data[x_index], col = "grey", pch = 19)
            # add vertical line
            abline(v = simulated_allee_data[x_index], col = "grey", lty = "dashed")
            
          }
          
          plot(simulated_temperature_data)
          
          write.csv(df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_X_data_Temperature_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          df_Y = data.frame(label_list)
          colnames(df_Y) <- c(1:50)
          colnames(derivative_df) <- c(1:50)
          write.csv(df_Y,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_label_data_Temperature_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          # write derivative dataframe to csv
          write.csv(derivative_df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_derivative_data_Temperature_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          # write smoothed derivative dataframe to csv
          write.csv(smooth_derivative_df,paste0("data/simulated_data/simulated_with_different_gene_type/simulated_smoothed_derivative_data_Temperature_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
          df_r <- data.frame(r_list)
          df_new = rbind(df_r,Mmax_list)
          rownames(df_new)  <-c("r","Mmax")
          colnames(df_new) <- c(1:50)
          write.csv(df_new,paste0("data/simulated_data/simulated_with_different_gene_type/parameters_list_simulated_data_Temperature_",noise_name,"_",snp_1,"_",snp_2,"_",snp_3,"_",snp_4,".csv",sep=""))
        }
        snp_4 = snp_4+2
      }
      snp_3 = snp_3+2
    }
    snp_2 = snp_2+1
  }
  snp_1 = snp_1+1
  
}