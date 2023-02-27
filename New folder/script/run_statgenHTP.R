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
# which is alredy done
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

# # outline detection use log transformed data
# LA_singleOut <- detectSingleOut(TP = phenoTP,
#                                       trait = "LA_Estimated_log_transformed",
#                                       plotIds = platformdata$XY,
#                                       confIntSize = 5,
#                                       nnLocfit = 0.5)
# 
# Height_singleOut <- detectSingleOut(TP = phenoTP,
#                                           trait = "Height_Estimated_log_transformed",
#                                           plotIds = platformdata$XY,
#                                           confIntSize = 5,
#                                           nnLocfit = 0.5)
# 
# Biomass_singleOut <- detectSingleOut(TP = phenoTP,
#                                     trait = "Biomass_Estimated_log_transformed",
#                                     plotIds = platformdata$XY,
#                                     confIntSize = 5,
#                                     nnLocfit = 0.5)

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

Biomass_singleOut <- detectSingleOut(TP = phenoTP,
                                    trait = "Biomass_Estimated",
                                    plotIds = platformdata$XY,
                                    confIntSize = 5,
                                    nnLocfit = 0.5)
#PLOT
plot(LA_singleOut, outOnly = FALSE, plotIds = platformdata$XY[1:3])
plot(Height_singleOut, outOnly = FALSE, plotIds = platformdata$XY[1:3])
plot(Biomass_singleOut, outOnly = FALSE, plotIds = platformdata$XY[1:3])
# count outlier: around 10% different from killian got, because i did not set 
sum(LA_singleOut$outlier)
sum(Height_singleOut$outlier)
sum(Biomass_singleOut$outlier)

# remove single outlier: Height, LA ->outlying point will be replace by NA
phenoTP_remove_out_LA <- removeSingleOut(phenoTP, LA_singleOut)
phenoTP_remove_out_LA_Height <- removeSingleOut(phenoTP_remove_out_LA, Height_singleOut)
phenoTP_remove_out_LA_Height_Biomass <- removeSingleOut(phenoTP_remove_out_LA_Height, Biomass_singleOut)

#It is only possible to use the combination of check and genotype as random.
#we use genotype as fix without check genotype

# fit model after remove single outlier, will cause error while include the last time point
LA_spline_model <- fitModels(TP = phenoTP_remove_out_LA_Height_Biomass,
                             trait = "LA_Estimated",
                             timePoints = 1:44,
                             what = "fixed",
                             useRepId = TRUE)

Height_spline_model <- fitModels(TP = phenoTP_remove_out_LA_Height_Biomass,
                                  trait = "Height_Estimated",
                                 timePoints = 1:44,
                                 what = "fixed",
                                 useRepId = TRUE)

#error with Biomass
# Biomass_spline_model <- fitModels(TP = phenoTP_remove_out_LA_Height_Biomass,
#                                  trait = "Biomass_Estimated",
#                                  timePoints = 1:44,
#                                  what = "fixed",
#                                  useRepId = TRUE)
summary(LA_spline_model)
summary(Height_spline_model)
#summary(Biomass_spline_model)

plot(LA_spline_model,
     timePoints = 44,
     plotType = "spatial",
     spaTrend = "raw")

plot(Height_spline_model,
     timePoints = 44,
     plotType = "spatial",
     spaTrend = "raw")

# Extracting the spatially corrected data which will use to fit the Spline
corrected_LA_spline <- getCorrected(LA_spline_model)
corrected_Height_spline <- getCorrected(Height_spline_model)

# Fitting splines: knot larger-> more smooth, (knot=30 Warning: No convergence after 250 iterations) 
LA_spline_lines <- fitSpline(inDat = corrected_LA_spline,
                             trait = "LA_Estimated_corr",
                             genotypes = unique(as.character(corrected_LA_spline$genotype)),
                             knots = 20,
                             minNoTP = 10)

Height_spline_lines <- fitSpline(inDat = corrected_Height_spline,
                                 trait = "Height_Estimated_corr",
                                 genotypes = unique(as.character(corrected_Height_spline$genotype)),
                                 knots = 20,
                                 minNoTP = 10)


plot(LA_spline_lines,genotypes = "DH_KE0006" )
plot(Height_spline_lines,genotypes = "DH_KE0006" )

# Extracting the predicted values and coefficients for outlier detection
LA_predDat <- LA_spline_lines$predDat
LA_coefDat <- LA_spline_lines$coefDat
Height_predDat <- Height_spline_lines$predDat
Height_coefDat <- Height_spline_lines$coefDat

LA_serieOut <- detectSerieOut(corrDat = corrected_LA_spline,
                                    predDat = LA_predDat,
                                    coefDat = LA_coefDat,
                                    trait = "LA_Estimated_corr",
                                    genotypes = unique(as.character(corrected_LA_spline$genotype)),
                                    thrCor = 0.60,
                                    thrPca = 90,
                                    thrSlope = 0.60)


Height_serieOut <- detectSerieOut(corrDat = corrected_Height_spline,
                                    predDat = Height_predDat,
                                    coefDat = Height_coefDat,
                                    trait = "Height_Estimated_corr",
                                    genotypes = unique(as.character(corrected_Height_spline$genotype)),
                                    thrCor = 0.60,
                                    thrPca = 90,
                                    thrSlope = 0.60)

plot(LA_serieOut, genotypes = "DH_KE0002")
# remove the outlier series
LA_remove_series_out <- removeSerieOut(dat = corrected_LA_spline,
               serieOut = LA_serieOut)
Height_remove_series_out <- removeSerieOut(dat = corrected_Height_spline,
                                       serieOut = Height_serieOut)

# Fitting splines: knot larger-> more smooth, (knot=30 Warning: No convergence after 250 iterations) 
# Warning: More than 5 plotIds have observations for less than the minimum number of time points, which is 10. 
# The  first 5 are printed, to see them all run attr(..., 'plotLimObs') on the output 01_13, 01_29, 01_40, 01_41, 01_44
LA_spline_lines_after_remove_outlier_series <- fitSpline(inDat = LA_remove_series_out,
                             trait = "LA_Estimated_corr",
                             genotypes = unique(as.character(LA_remove_series_out$genotype)),
                             knots = 20,
                             minNoTP = 10)

plot(LA_spline_lines,genotypes = "DH_KE0006" )

Height_spline_lines_after_remove_outlier_series <- fitSpline(inDat = Height_remove_series_out,
                                 trait = "Height_Estimated_corr",
                                 genotypes = unique(as.character(Height_remove_series_out$genotype)),
                                 knots = 20,
                                 minNoTP = 10)
# extract some features and predicted values use for following prediction.
Height_predict <- na.omit(Height_spline_lines_after_remove_outlier_series$predDat)
Height_coefient<- na.omit(Height_spline_lines_after_remove_outlier_series$coefDat)

LA_predict <- na.omit(LA_spline_lines_after_remove_outlier_series$predDat)
LA_coefient <- na.omit(LA_spline_lines_after_remove_outlier_series$coefDat)

# extract AUC based on days
LA.AUC <- estimateSplineParameters(x = LA_spline_lines_after_remove_outlier_series,
                                   estimate = "predictions",
                                   what = "AUC",
                                   AUCScale = "days"
                             )

Height.AUC <- estimateSplineParameters(x = Height_spline_lines_after_remove_outlier_series,
                                       estimate = "predictions",
                                       what = "AUC",
                                       AUCScale = "days"
                                       )

LA.maxDeriv <- estimateSplineParameters(x = LA_spline_lines_after_remove_outlier_series,
                                        estimate = "derivatives",
                                        what = "max",
                                      )

Height.maxDeriv <- estimateSplineParameters(x = Height_spline_lines_after_remove_outlier_series,
                                            estimate = "derivatives",
                                            what = "max",
                                      )

LA.maxPred <- estimateSplineParameters(x = LA_spline_lines_after_remove_outlier_series,
                                       estimate = "predictions",
                                       what = "max",
                                       )

Height.maxPred <- estimateSplineParameters(x = Height_spline_lines_after_remove_outlier_series,
                                           estimate = "predictions",
                                           what = "max",
                                          )

LA.meanDeriv <- estimateSplineParameters(x = LA_spline_lines_after_remove_outlier_series,
                                         estimate = "derivatives",
                                         what = "mean",
                                        )

Height.meanDeriv <- estimateSplineParameters(x = Height_spline_lines_after_remove_outlier_series,
                                             estimate = "derivatives",
                                             what = "mean",
                                         )

LA_AUC <- droplevels(na.omit(LA.AUC))
Height_AUC <- droplevels(na.omit(Height.AUC))

LA_maxDeriv <- droplevels(na.omit(LA.maxDeriv))
Height_maxDeriv <- droplevels(na.omit(Height.maxDeriv))

LA_maxPred <- droplevels(na.omit(LA.maxPred))
Height_maxPred <- droplevels(na.omit(Height.maxPred))

LA_meanDeriv <- droplevels(na.omit(LA.meanDeriv))
Height_meanDeriv <- droplevels(na.omit(Height.meanDeriv))

#merge data for LA and Height
library(tidyverse)
## put all data frames into list
LA_df_list <- list(LA_coefient, LA_predDat, LA_AUC,LA_maxDeriv,LA_maxPred,LA_meanDeriv)      
Height_df_list <- list(Height_coefient, Height_predDat, Height_AUC,Height_maxDeriv,Height_maxPred,Height_meanDeriv)      

#merge all data frames together
Reduce(function(x, y) merge(x, y, all=FALSE), list_df)
LA_df_list %>% reduce(full_join_LA, by='plotid')
Height_df_list %>% reduce(full_join_Height, by=c('plotid'))

# droplevels(): remove unused levels from a factor variable
write.csv(full_join_LA,"../data/LA_features_from_spline.csv")
write.csv(full_join_Height,"../data/Height_features_from_spline.csv")

