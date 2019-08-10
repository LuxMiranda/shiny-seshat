#####     Runs dynamic regression analysis separately for each of nrep CCs
outAll <- data.frame()

for(irep in 1:nrep){

load("PolsVars.Rdata")
Section <- "Social Complexity variables"
PropCoded_threshold <- 30  #### Only use polities with better than 30% coded data
RemPols <- c("EsHabsb","InGaroL","CnHChin","PgOrokL","FmTrukL","InGupta", "IsCommw","NorKing","GhAshnE","GhAshnL") 
dpar <- 1000
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

CC <- ImpCCrepl[ImpCCrepl$irep==irep,5:(4+nCC)]
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

source("fRegrDat.R")
for(i in 1:length(RemPols)){NGARegrDat <- NGARegrDat[NGARegrDat$PolID != RemPols[i],] }
#### Rename variables
colnames(NGARegrDat)[4:5] <- c("Info","Lag1")
colnames(NGARegrDat)[9:12] <- c("Hier","Gov","Infra","Money")
RegrDat <- NGARegrDat[,4:ncol(NGARegrDat)]
RegrDat <- RegrDat[,c(1:9,11:12)]   ####  Drop Space as not significant

####  Exhaustive regressions with linear terms
print(paste("Response variable =",colnames(RegrDat)[1],"        irep =",irep))
Predictors <- 3:ncol(RegrDat)
output <- data.frame()
for (nPred in 1:length(Predictors)){ print(nPred)
  Preds<- combn(Predictors, nPred)
  for(j in 1:length(Preds[1,])){
    fit <- lm(RegrDat[, c(1:2, Preds[,j])])
    Pval <- summary(fit)$coefficients[,4]
    tval <- summary(fit)$coefficients[,3]
    out <- vector("numeric",length = length(RegrDat))
    out[c(1:2,Preds[,j])] <- tval
    out <- c(out,summary(fit)$r.sq)
    fit <- glm(RegrDat[, c(1:2, Preds[,j])])
    out <- c(out,summary(fit)$aic)
    output <- rbind(output,out)
  }
}
colnames(output) <- c(colnames(RegrDat),"R-sq","delAIC")
output <- output[order(output$delAIC),]
output$delAIC <- output$delAIC - min(output$delAIC)
outAll <- rbind(outAll,output[1:10,])
}

write.csv(outAll, file="output.csv",  row.names=FALSE)
####

