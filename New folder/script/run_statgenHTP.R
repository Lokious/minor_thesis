# install.packages("statgenHTP")
# install.packages("rstudioapi")
# loading library
library("rstudioapi")
library("statgenHTP")
#set current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
#loading prepossessed data:
platformdata <- read.csv("../data/image_DHline_data_after_average_based_on_day.csv",row.names = 1,header= TRUE)

# plotId: unique ID of the plant
# multiple value in some day, need to take mean instead of using the raw data at each timepoint
phenoTP <- createTimePoints(dat = platformdata,
                            experimentName = "DHline11",
                            genotype = "genotype_name",
                            timePoint = "Day",
                            repId = "Rep",
                            plotId = "XY",
                            rowNum = "Line", colNum = "Position",
)
attr(phenoTP, 'plotLimObs') 
timepoints <- getTimePoints(phenoTP)
summary(phenoTP) 

# outline detection
LA_singleOut <- detectSingleOut(TP = phenoTP,
                                      trait = "LA_Estimated",
                                      plotIds = platformdata$XY,
                                      confIntSize = 5,
                                      nnLocfit = 0.5)

Height_singleOut <- detectSingleOut(TP = phenoTP,
                                          trait = "Height_Estimated",
                                          plotIds = platformdata$XY,
                                          confIntSize = 5,
                                          nnLocfit = 0.5)
#PLOT
plot(LA_singleOut, outOnly = FALSE, plotIds = platformdata$XY[1:3])
plot(Height_singleOut, outOnly = FALSE, plotIds = platformdata$XY[1:3])
# count outlier
sum(LA_singleOut$outlier)
sum(Height_singleOut$outlier)
# remove single outlier: Height, LA
phenoTP_remove_out <- removeSingleOut(phenoTP, LA_singleOut)
phenoTP_remove_out <- removeSingleOut(phenoTP_remove_out, Height_singleOut)

# fit model after remove single outlier, will cause error while include the last time point
LA_spline_model <- fitModels(TP = phenoTP_remove_out,
                             trait = "LA_Estimated",
                             timePoints = 1:44,
                             what = "fixed",
                             useRepId = TRUE)

Height_spline_model <- fitModels(TP = phenoTP_remove_out,
                                  trait = "Height_Estimated",
                                 timePoints = 1:44,
                                 what = "fixed",
                                 useRepId = TRUE)
summary(LA_spline_model)
summary(Height_spline_model)
