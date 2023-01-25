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


data("PhenovatorDat1")
# plotId: unique ID of the plant
# missing value in each day, need to take mean instead of using the raw data at each timepoint
phenoTP.lines <- createTimePoints(dat = platformdata,
                            experimentName = "DHline11",
                            genotype = "genotype_name",
                            timePoint = "Day",
                            repId = "Rep",
                            plotId = "XY",
                            rowNum = "Line", colNum = "Position",
)
attr(phenoTP.lines, 'plotLimObs') 
timepoints.lines <- getTimePoints(phenoTP.lines)
summary(phenoTP.lines) 

LA.singleOut.lines <- detectSingleOut(TP = phenoTP.lines,
                                      trait = "LA_Estimated",
                                      plotIds = platformdata$XY,
                                      confIntSize = 5,
                                      nnLocfit = 0.5)

Height.singleOut.lines <- detectSingleOut(TP = phenoTP.lines,
                                          trait = "Height_Estimated",
                                          plotIds = platformdata$XY,
                                          confIntSize = 5,
                                          nnLocfit = 0.5)

# Plots look good:
plot(LA.singleOut.lines, outOnly = FALSE, plotIds = platformdata$XY[1:3])
plot(Height.singleOut.lines, outOnly = FALSE, plotIds = platformdata$XY[1:3])
