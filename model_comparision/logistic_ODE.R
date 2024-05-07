#install.packages("deSolve")
#install.packages("FME")
#install.packages("pspline")
install.packages("segmented")

# loading library
library("deSolve")
library("rootSolve")
library("coda")
library("FME")
library('pspline')
library('segmented')
# loading library for EPOLE data
library("rstudioapi")
library("statgenHTP")
library('Cairo')
#set current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

# #load data and split based on location
# for (loc_name in c("Emerald","Merredin","Narrabri","Yanco")){
#   mergerddata <- read.csv("data/biomass.csv",row.names = 1,header= TRUE)
#   Emerald_data <- subset(mergerddata,mergerddata$loc==loc_name)
#   row.names(Emerald_data) <- NULL
#   file_name <- sprintf("data/%s_data.csv", loc_name)
#   write.csv(Emerald_data,file_name)
# }

# theFiles <- list.files("data/",pattern="*.txt",full.names = TRUE)
# # list the files
# theFiles
# theFiles <- sample(theFiles, 600, replace=TRUE)
# #theFiles <-c('data/Emerald2002g007.txt')
# ############################test for fitting Logistic model for silco data#############################
# logistic_Fit_parameters = data.frame()
# index <- 0
# for (textfile in theFiles){
#   data = read.table(textfile)
#   # fit <- smooth.Pspline(data$das, data$biomass)
#   #
#   # plot(data$das, data$biomass)
#   # lines(fit, col = "blue")
#   #define function
#   logistic <- function(t, y, parms) {
#     with(as.list(parms,y,t), {
#       dMdt <- r * y * (1 - y / Mmax)
#       return(list(dMdt))
#     })
#   }
# 
#   #x and y
#   M <-as.numeric(data$biomass)/1000
#   t <- as.numeric(data$das)
#   y0 <- c(M = M[1])
#   y0
#   parms <- c(r = 0.1, Mmax = max(M))
#   # #use desolve to fit the data
#   # fit <- ode(y = y0, times = t, func = logistic, parms = parms)
#   # parms
#   #
#   plot(t, M, pch = 16, xlab = "Time", ylab = "M")
#   # lines(fit, col = "red")
# 
#   #the first column of contains the name of the observed variable, if we only have biomass it is 'M' here
#   biomass_time_df <-data.frame(name=rep("M",length(t)),time=unlist(t),M=unlist(M))
# 
#   #the function to minimize
#   ModelCost <- function(parms) {
#     modelout <- as.data.frame(ode(y = y0, times = t, func = logistic, parms = parms))
#     modCost(model=modelout,obs=biomass_time_df,y="M")  # object of class modCost
#   }
# 
#   Fit <- modFit(f = ModelCost, p = parms, method = "Port") #fit the curve which minimize the ModelCost
#   summary(Fit)
#   out <- ode(y = y0, func = logistic, parms = Fit$par,
#              times = t)
# 
#   logistic_Fit_parameters <- rbind(logistic_Fit_parameters, c(Fit$par,y0))
#   lines(out, col = "blue")
# }
# Fit$par
# max(M)
# colnames(logistic_Fit_parameters) <- c('r','Mmax','y0')
# 
# #save fit parameters, use as the parameters for genrate the data
# write.csv(logistic_Fit_parameters,"logistics_fit_parameters.csv")

#########################Fit logistic model with EPLOE data################################

# remove out-lier from the biomass data
#set current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
#loading prepossessed data:
platformdata <- read.csv("../New Folder/data/image_DHline_data_after_average_based_on_day.csv",row.names = 1,header= TRUE)

# plotId: unique ID of the plant, create time Points
phenoTP <- createTimePoints(dat = platformdata,
                            experimentName = "DHline11",
                            genotype = "genotype_name",
                            timePoint = "DAS",
                            repId = "Rep",
                            plotId = "XY",
                            rowNum = "Line", colNum = "Position",
)
attr(phenoTP, 'plotLimObs') 
timepoints <- getTimePoints(phenoTP)
summary(phenoTP) 

# detect and remove outlier
Biomass_singleOut <- detectSingleOut(TP = phenoTP,
                                     trait = "Biomass_Estimated",
                                     plotIds = unique(platformdata$XY),
                                     confIntSize = 5,
                                     nnLocfit = 0.5)
phenoTP_remove_out_Biomass <- removeSingleOut(phenoTP, Biomass_singleOut)

#save to dataframe
after_reomve_outlier_df = data.frame()
for (time_step in phenoTP_remove_out_Biomass){
  print(time_step)
  after_reomve_outlier_df = rbind(after_reomve_outlier_df,time_step)
}
#save the dataframe after removing outliers as csv file
write.csv(after_reomve_outlier_df,"biomass_remove_single_outlier.csv")

### then run python code to filter and calculate gene average###
#reformat.read_biomass_and_filter()


#load data
library(dplyr)
library(tidyr)
#loading prepossessed data:
platformdata <- read.csv("../New Folder/script/biomass_average_based_on_genotype.csv",row.names = 1,header= TRUE)
groups <- split(platformdata,platformdata$genotype)


predicted_biomass_from_smooth = data.frame()
for (group in groups){
  # group$Biomass_Estimated_mean <- replace(group$Biomass_Estimated_mean, group$Biomass_Estimated_mean==-Inf, NA)
  # group$Biomass_Estimated_mean <- replace(group$Biomass_Estimated_mean, group$Biomass_Estimated_mean==Inf, NA)
  #sorted DAS
  group<- group[order(group$Day),]
  
  #plot(x=group$Day,y=group$Biomass_Estimated_mean,type="o",pch=19)
  #replace NA with smooth prediction
  group_droped_na = group[!is.na(group$Biomass_Estimated_mean),]
  if (length(group_droped_na$Biomass_Estimated_mean)<20){
    next
  }
  else{
    Cairo(file=paste0("growth_curve_fit/fill_na/",unique(group$genotype),"_.tiff",sep=""),
          type="tiff",
          units="px", 
          width=512, 
          height=512, 
          pointsize=12, 
          dpi="auto")
    plot(x=group$Day,y=group$Biomass_Estimated_mean,type="o",pch=19)
    
    smooth_line <-smooth.spline(group_droped_na$Day,group_droped_na$Biomass_Estimated_mean)
    predicted_biomass <- with(group,predict(smooth_line,group$Day[is.na(group$Biomass_Estimated_mean)]))
    points(predicted_biomass,pch=19,col="red")
    dev.off()
    group[is.na(group$Biomass_Estimated_mean),'Biomass_Estimated_mean'] <- predicted_biomass$y
   
    # genotype_name <- rep(unique(group$genotype),20)
    # DAS<- c(1:20)
    # df = data.frame(genotype_name,DAS,predicted_biomass$y) 
    predicted_biomass_from_smooth = rbind(predicted_biomass_from_smooth,as.data.frame(group))
  }
}

predicted_biomass_from_smooth_groups <- split(predicted_biomass_from_smooth,predicted_biomass_from_smooth$genotype)
logistic_Fit_parameters = data.frame()
index <- 0
predicted_biomass_from_smooth = data.frame()
fit_R_squared = data.frame()
for (group in predicted_biomass_from_smooth_groups){
  
  logistic <- function(t, y, parms) {
    with(as.list(parms,y,t), {
      dMdt <- r * y * (1 - y / Mmax)
      return(list(dMdt))
    })
  }
  
  #x and y
  M <-as.numeric(group$Biomass_Estimated_mean)/100
  t <- as.numeric(group$Day)

  #fit spline
  # fit <- smooth.spline(x=t,y=M)
  # M <- predict(fit,x=t)$y
  y0 <- c(M = M[1])
  y0
  parms <- c(r = 0.1, Mmax = max(M))
  #use desolve to fit the data
  fit <- ode(y = y0, times = t, func = logistic, parms = parms)
  Cairo(file=paste0("growth_curve_fit/ODE/",unique(group$genotype),"_.tiff",sep=""),
        type="tiff",
        units="px", 
        width=512, 
        height=512, 
        pointsize=12, 
        dpi="auto")
  
  plot(t, M, pch = 16, xlab = "Time", ylab = "M")

  #the first column of contains the name of the observed variable, if we only have biomass it is 'M' here
  biomass_time_df <-data.frame(name=rep("M",length(t)),time=unlist(t),M=unlist(M))

  #the function to minimize
  ModelCost <- function(parms) {
    modelout <- as.data.frame(ode(y = y0, times = t, func = logistic, parms = parms))
    modCost(model=modelout,obs=biomass_time_df,y="M")  # object of class modCost
  }

  Fit <- modFit(f = ModelCost, p = parms, method = "Nelder-Mead") #fit the curve which minimize the ModelCost
  summary(Fit)
  out <- ode(y = y0, func = logistic, parms = Fit$par,
             times = t)
  print(Fit$par)
  logistic_Fit_parameters <- rbind(logistic_Fit_parameters, c(Fit$par,y0))
  lines(out, col = "blue")
  log_M <- log(as.numeric(group$Biomass_Estimated_mean))
  dev.off()
  Cairo(file=paste0("growth_curve_fit/segment_linear_regression/",unique(group$genotype),"_.tiff",sep=""),
        type="tiff",
        units="px", 
        width=512, 
        height=512, 
        pointsize=12, 
        dpi="auto")
  plot(t, log_M, pch = 16, xlab = "Time", ylab = "log(Biomass)")
  # Fit the segmented linear regression model for log(Biomass)
  seg.lm <- segmented(lm(log_M ~ t), seg.Z = ~t ) #seg.Z the variaable need to be segmented
  # Show the summary of the model
  summary_seg <- summary(seg.lm)
  # Plot the data and the segmented linear regression model
  lines(t, predict(seg.lm), col = "red")
  linear_log_fit <- lm(log_M ~ t)
  summary_linear <- summary(linear_log_fit)
  lines(t, predict(linear_log_fit), col = "green")
  dev.off()
  fit_R_squared = rbind(fit_R_squared, c(summary_seg$adj.r.squared,summary_linear$adj.r.squared))
}
colnames(fit_R_squared) = c('seg_linear','linear')
t_test <- t.test(fit_R_squared$seg_linear, fit_R_squared$linear)

# print the results
print(t_test)
#save fit parameters, use as the parameters for generate the data
write.csv(logistic_Fit_parameters,"eploe_logistics_fit_parameters.csv")
write.csv(fit_R_squared,"eploe_logistics_fit_adjust_r_suqared.csv")
# predicted_biomass_from_smooth_groups <- split(predicted_biomass_from_smooth,predicted_biomass_from_smooth$genotype)
# 
# for (group in predicted_biomass_from_smooth_groups){
#   M <-log(as.numeric(group$Biomass_Estimated_mean))
#   t <- as.numeric(group$Day)
#   plot(t,M)
# 
# }
# 















########################seems the code works for logistic ODE for one genotype and one environment###############

############################test for Irradiance model#############################
# something wong when estimate best parameters, check this link similar problem as this
# https://stackoverflow.com/questions/67478946/in-r-fme-desolve-sir-fitting-time-varying-parameters
# #load data
# irradiance_Fit_parameters = data.frame()
# index <- 0
# for (textfile in theFiles){
#   
#   print(index)
#   index = index+1
#   data = read.table(textfile)
#   #inilize parameters and plot curve
#   M <-as.numeric(data$biomass)/1000
#   t <- as.numeric(data$das)
#   y0 <- c(M = M[1])
#   biomass_time_df <-data.frame(name=rep("M",length(t)),time=unlist(t),M=unlist(M))
#   plot(t, M, pch = 16, xlab = "Time", ylab = "M")
#   irradiance_parms <- c(r = 0.5, Mmax = 5,a=0.1,fi=0.18)
#   #define function
#   Irradiance_model <- function(t, y, irradiance_parms) {
#     with(as.list(irradiance_parms,y,t), {
#       dMdt <- (r+a*sin((2*pi/365)*t+fi))* y * (1 - y / Mmax)
#       return(list(dMdt))
#     })
#   }
#   
#   #the function to minimize
#   Irradiance_ModelCost <- function(parms) {
#     modelout <- as.data.frame(ode(y = y0, times = t, func = Irradiance_model, parms = irradiance_parms))
# 
#     modCost(model=modelout,obs=biomass_time_df,y="M")  # object of class modCost
#   }
#   
#   irradiance_Fit <- modFit(f = Irradiance_ModelCost, p = irradiance_parms)#, method = "Nelder-Mead") #fit the curve which minimize the ModelCost
#   #summary(irradiance_Fit) # for Port method: In summary.modFit(Fit) : Cannot estimate covariance; system is singular
#   irradiance_out <- ode(y = y0, func = Irradiance_model, parms = irradiance_Fit$par,
#              times = t)
#   irradiance_Fit$par
#   irradiance_Fit_parameters <- rbind(irradiance_Fit_parameters, irradiance_Fit$par)
#   lines(irradiance_out, col = "green")
# }
# ###############################################################################
# irradiance_Fit$par
# summary(irradiance_Fit)
# ############################test for temperature model#############################
# 
# tempreture_parms <- c(r = 0.1, Mmax = max(M))
# #define function
# tempreture_model <- function(t, tempreture,y, tempreture_parms) {
#   with(as.list(tempreture_parms,y,tempreture,t), {
#     TAL= 20000
#     TL = 292
#     TAH = 60000
#     TH = 303
#     Ft <- (1+exp(TAL/tempreture -TAL/TL) + exp(TAH/TH -TAH/tempreture))^-1
#     dMdt <- (r*Ft)* y * (1 - y / Mmax)
#     return(list(dMdt))
#   })
# }
# 
# #the function to minimize
# Tempreture_ModelCost <- function(tempreture_parms) {
#   modelout <- as.data.frame(ode(y = y0, times = t, func = tempreture_model, parms = tempreture_parms))
#   modelout
#   modCost(model=modelout,obs=biomass_time_df,y="M")  # object of class modCost
# }
# 
# tempreture_Fit <- modFit(f = Tempreture_ModelCost, p = tempreture_parms) #fit the curve which minimize the ModelCost
# summary(Fit) # for Port method: In summary.modFit(Fit) : Cannot estimate covariance; system is singular
# tempreture_out <- ode(y = y0, func = tempreture_model, parms = tempreture_Fit$par,
#                       times = t)
# 
# lines(tempreture_out, col = "orange")
# ###############################################################################
