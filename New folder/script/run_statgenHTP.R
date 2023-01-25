install.packages("statgenHTP")
install.packages("rstudioapi")
# loading library
library("rstudioapi")
library("statgenHTP")
#set current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
#loading prepossessed data:
platformdata <- read.csv("../data/image_DHline_data.csv")


data("PhenovatorDat1")
# plotId: unique ID of the plant
# missing value in each day, need to take mean instead of using the raw data at each timepoint
phenoTP.lines <- createTimePoints(dat = platformdata,
                            experimentName = "DHline11",
                            genotype = "genotype_name",
                            timePoint = "datetime",
                            repId = "Rep",
                            plotId = "XY",
                            rowNum = "Line", colNum = "Position",
)
attr(phenoTP.lines, 'plotLimObs') 
timepoints.lines <- getTimePoints(phenoTP.lines)
