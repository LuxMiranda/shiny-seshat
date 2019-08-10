############   !MI_SC  generates a multiple imputation dataset with seven CCs (sans the response CC), nrep = 20
# 
########### The commented out scripts are not included in the release
#    Run update_PolsVars.R after changing !PolsVars spreadsheet
#load("PolsVars.Rdata")
#Section <- "Social Complexity variables"
#PropCoded_threshold <- 30  #### Only use polities with better than 30% coded data
#source("precheck.R")
#########    generates SCDat.csv (prechecked, selected SC data), writes SCDat.csv
#rm(errors)
#==========================================   end of data checking and prep section

##########################   Uses SCDat.csv. Construct ImpDatRepl
load("PolsVars.Rdata")
Section <- "Social Complexity variables"
PropCoded_threshold <- 30  #### Only use polities with better than 30% coded dataload("PolsVars.Rdata")
Vars <- variables[variables[,6]==Section,]
NGAs <- unique(polities$NGA)
iResp <- 7   ##### Response Variable = Info   ALSO SET IN LINE 65

nrep <- 20
ImpDatRepl <- matrix(NA, nrow=0, ncol=14)
for(irep in 1:nrep){
  print(irep)
  source("ConstrMI.R")
  source("fAggr.R")
  source("ImputeMI.R")
  ones <- matrix(data=1,nrow=length(AggrDat[,1]),ncol=1)
  colnames(ones) <- "irep"
  ImpDat <- cbind(AggrDat[,1:4],ImpDat,(ones*irep))
  ImpDatRepl <- rbind(ImpDatRepl,ImpDat)
}

####### Remove polity-dates that didn't yield nrep repl
load("PolsVars.Rdata")
dat_temp <- ImpDatRepl
for(i in 1:nrow(polities)){
  dat <- ImpDatRepl[as.character(ImpDatRepl[,2])==as.character(polities[i,2]),]
  if(nrow(dat)!=0){
    Time <- unique(dat$Time)
    for(j in 1:length(Time)){
      dt <- dat[dat$Time==Time[j],]
      if(nrow(dt) != nrep){
        print(nrow(dt))
        print(dt[1,1:3])
        dat_temp[as.character(dat_temp$PolID)==as.character(dat$PolID[1]) & dat_temp$Time==Time[j],length(dat_temp)] <- -99999
      }
    }
  }
}
ImpDatRepl <- dat_temp[dat_temp$irep!=-99999,]
write.csv(ImpDatRepl, file="ImpDatRepl.csv",  row.names=FALSE)
#============================================================== Multiple Imputation is done

######################################################  Switching to averaged data
#####     Uses the average of nrep CCs
load("PolsVars.Rdata")
Section <- "Social Complexity variables"
PropCoded_threshold <- 30  #### Only use polities with better than 30% coded data
Vars <- variables[variables[,6]==Section,]
NGAs <- unique(polities$NGA)
source("ConstrAvg.R")
ConstrDat <- OutPolity
source("fAggr.R")

iResp <- 7     ##### Response Variable = Info   ALSO SET IN LINE 19
load("SC.Rdata")

#### Read CCs from file
ImpCCrepl <- read.table('ImpDatRepl.csv', sep=",", header=TRUE)
nrep <- ImpCCrepl$irep[nrow(ImpCCrepl)]
for(i in 1:2){ImpCCrepl[,i] <- as.character(ImpCCrepl[,i])}
n <- nrow(ImpCCrepl)/nrep   #### Number of data points
nCC <- length(ImpCCrepl) - 5

######## Calculate average CCs
CC <- matrix(0,n,nCC)
for(irep in 1:nrep){
  dat <- ImpCCrepl[ImpCCrepl$irep==irep,5:(4+nCC)]
  CC <- CC + dat
}
CC <- CC/nrep
Predictors <- cbind(ImpCCrepl[ImpCCrepl$irep==1,1:3],CC)
RespVar <- AggrDat[,c(1:3, 4+iResp)]
RespVar <- RespVar[is.na(RespVar[,4]) == FALSE,]

###   Merge Responses with Predictors
RespPred <- data.frame()
for(i in 1:nrow(RespVar)){
  record <- Predictors[RespVar[i,1] == Predictors[,1] & RespVar[i,2] == Predictors[,2] & RespVar[i,3] == Predictors[,3],]
  record <- cbind(RespVar[i,], record[1,4:ncol(record)])
  RespPred <- rbind(RespPred,record)
}

#####  Construct InterpDat: data with interpolated steps=100 y
####### Interpolate centuries
load("PolsVars.Rdata")
polities <- polities[polities$Dupl == "n",]
InterpDat <- data.frame()
for(iNGA in 1:length(NGAs)){
  dat <- RespPred[RespPred$NGA==NGAs[iNGA],]
  dt <- data.frame()
  if(nrow(dat) > 1){
    for(j in 2:nrow(dat)){
      Tstart <- dat$Time[j-1]
      PolID <- dat$PolID[j-1]
      Tend <- dat$Time[j] - 100
      for(t in seq(Tstart,Tend,100)){
        dt <- rbind(dt,dat[j-1,])
        dt$Time[nrow(dt)] <- t
      }
    }
    dt <- rbind(dt,dat[j,])
  }
  InterpDat <- rbind(InterpDat,dt)
}
##### Take out centuries that are outside polity temporal bounds
for(i in 1:nrow(InterpDat)){
  PolID <- InterpDat$PolID[i]
  Time <- InterpDat$Time[i]
  Start <- as.numeric(polities$Start[polities$PolID==PolID])
  End <- as.numeric(polities$End[polities$PolID==PolID])
  if(length(End)>1){print(i)}
  if(Time < Start){InterpDat[i,3] <- -99999}
  if(Time > End){InterpDat[i,3] <- -99999}
}
InterpDat <- InterpDat[InterpDat$Time!=-99999,]

#dat <- InterpDat
#for(i in 1:nrow(InterpDat)){if(is.nan(dat$Hier[i]) == TRUE){dat$Hier[i] <- NA}}  # Replace Nan with NA

#### Save (or resave) everything in SC.Rdata
PolityDat <- RespPred
save(polities, coords, NGAs, PolityDat,InterpDat, DistMatrix, file="SC.Rdata")
rm(dat,dt,ImpCCrepl,Predictors,End,i,iNGA,irep,j,PolID,Start,t,Tend,Time,Tstart,RespPred)
rm(n,nrep,nCC,NGAs,coords,DistMatrix,polities,record,CC, Section, PropCoded_threshold,variables)

#######################################################################
#######################################################################
######### end of the new scrape and imputation 

########################################################
##### Distribution of distances
#dist <- vector()
#mindist <- dist
#for(i in 1:(nrow(DistMatrix)-1)){
#  row <- DistMatrix[i,]
#  row <- row[row != 0]
#  mindist <- c(mindist,min(row))
#  for(j in (i+1):ncol(DistMatrix)){dist <- c(dist,DistMatrix[i,j])}
#}
#hist(mindist, breaks=seq(0,6000,by=500))



# cor(as.numeric(output[,8]),as.numeric(output[,9]), use="complete.obs")
# cor(as.numeric(output[,8]),as.numeric(output[,11]), use="complete.obs")
# plot(as.numeric(output[,8]),as.numeric(output[,11]))

#plot(CC$PolTerr,CC$government)
#res <- loess(CC$government ~ CC$PolTerr, span=0.5)
#points(predict(res), x=CC$PolTerr, col="red")

#plot(CC$PolPop,CC$government)
#res <- loess(CC$government ~ CC$PolPop, span=0.5)
#points(predict(res), x=CC$PolPop, col="red")

#plot(CC$CapPop,CC$government)
#res <- loess(CC$government ~ CC$CapPop, span=0.5)
#points(predict(res), x=CC$CapPop, col="red")

#plot(CC$levels,CC$government)
#res <- loess(CC$government ~ CC$levels, span=0.5)
#points(predict(res), x=CC$levels, col="red")


