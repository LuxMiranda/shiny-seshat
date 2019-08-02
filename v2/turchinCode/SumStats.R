############  Summary statistics on CCs. Uses SCdat.csv
load("PolsVars.Rdata")
Section <- "Social Complexity variables"
PropCoded_threshold <- 30  #### Only use polities with better than 30% coded data
Vars <- variables[variables[,6]==Section,]
NGAs <- unique(polities$NGA)
source("ConstrAvg.R")
ConstrDat <- OutPolity
source("fAggr.R")

#### Plot frequency distributions for all CCs
PlotNames <- c("PolPop","PolTerr","CapPop","Hier","Gov","Infra","Info","Money")
layout(matrix(c(1:8),4,2)) # 4 x 2 graphs/page
for(i in 1:8){
  dat <- AggrDat[,4+i]
  dat <- dat[is.na(dat)==FALSE]
  PlotLabel <- paste0(PlotNames[i], " (n = ",length(dat),")")
  hist(dat, main = PlotLabel, xlab = "")
  }

# Number of complete rows
CompletDat <- data.frame()
for(i in 1:nrow(AggrDat)){
  if(all(is.na(AggrDat[i,]) == FALSE) ){  CompletDat <- rbind(CompletDat,AggrDat[i,]) }
}




