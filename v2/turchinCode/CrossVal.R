############  Averages uncertainty and disagreement
#    Run update_PolsVars.R after changing !PolsVars spreadsheet
#    source("precheck.R")   #### To run from !MI_SC.R after a new scrape. 
load("PolsVars.Rdata")
Section <- "Social Complexity variables"
PropCoded_threshold <- 30  #### Only use polities with better than 30% coded data

Vars <- variables[variables[,6]==Section,]
NGAs <- unique(polities$NGA)
source("ConstrAvg.R")
ConstrDat <- OutPolity
source("fAggr.R")
load("PolsVars.Rdata")
for(i in 1:nrow(AggrDat)){
  region <- polities$World.Region[polities$NGA == AggrDat$NGA[i]]
  AggrDat[i,4] <- region[1]
}
colnames(AggrDat)[4] <- "Region"

############################    Cross-Validation
iResp <- 7 #### Response variable: 1=PolPop, 2=PolTerr, ... 8=money
ObsPred <- AggrDat[,c(1:4, (iResp+4), 5)]
colnames(ObsPred)[6] <- "Pred"
index <- 1:(ncol(AggrDat)-4)
index <- index[index != iResp]
data <- AggrDat[,4+c(iResp,index)]
regions <- AggrDat$Region
####   Omit NAs in the response variable
regions <- regions[is.na(data[,1])==FALSE]
ObsPred <- ObsPred[is.na(data[,1])==FALSE,]
data <- data[is.na(data[,1])==FALSE,]

##### Run cross-validation
for(i in 1:nrow(data)){print(i)
    index <- c(1:ncol(data))[is.na(data[i,]) == FALSE]
  dat <- data[regions[i] != regions,index]                   # Predict outside region
  for(j in 1:ncol(dat)){dat <- dat[is.na(dat[,j])==FALSE,]}  # Omit rows with NAs
  Predictors <- 2:ncol(dat)                                  # Exhaustive regressions
  output <- data.frame()
  for(nPred in 1:length(Predictors)){
    Preds<- combn(Predictors, nPred)
    for(j in 1:ncol(Preds)){
      fit <- glm(dat[, c(1, Preds[,j])])
      out <- head(c(Preds[,j], 0,0,0,0,0,0,0,0,0),length(Predictors))
      out <- c(out,summary(fit)$aic)
      output <- rbind(output,out)
    }
  }
  output <- output[order(output[,ncol(output)]),]
  Preds <- output[1,1:(ncol(output)-1)]
  Preds <- Preds[Preds != 0]
  fit <- glm(dat[, c(1, Preds)])
  PredVar <- data[i,index[c(1,Preds)]]
  PredVar[1] <- 1
  ObsPred[i,6] <- sum(coefficients(fit)*PredVar)
}

Rsq_out <- data.frame(NA, 10, 3)
Regions <- unique(ObsPred$Region)
for(i in 1:length(Regions)){
  X <- ObsPred$info[ObsPred$Region == Regions[i]]
  Y <- ObsPred$Pred[ObsPred$Region == Regions[i]]
  rsq <- 1 - sum((X-Y)^2)/sum((X-mean(X))^2)
  Rsq_out[i,] <- c(Regions[i],rsq,length(X))
}
colnames(Rsq_out) <- c("Region", "R-sq", "n")
write.csv(Rsq_out, file="output.csv",  row.names=FALSE)
####
#############################################   Plot predicted - observed by region
summary(lm(ObsPred[,5:6]))
X <- ObsPred$info
Y <- ObsPred$Pred
rsq <- 1 - sum((X-Y)^2)/sum((X-mean(X))^2)
rsq <- round(1000*rsq)/1000

colors <- c("red","blue","darkgreen","purple","brown","tan","darkgrey","orange","cyan","black")
pointshapes <-c(15:19,15:19)
xcoord <-c(0,1)
ycoord <-c(0,1)
plot(xcoord,ycoord,"n",xaxp=c(0,1,10),yaxp=c(0,1,10),xlab="Predicted",ylab="Observed",
     main=paste("Info: observed vs. predicted.    Prediction R-sq =",rsq))
lines(x=c(0,1),y=c(0,1), lty = 2, lwd=2)
xcoord <- 0
ycoord <- 1  ### To put labels in the upper left corner
for(i in 1:length(Regions)){
  gdat <- ObsPred[ObsPred$Region == Regions[i],6:5]
  points(gdat, col=colors[i], pch=pointshapes[i])
  text(x=xcoord, y=(ycoord-0.05*(i-1)), Regions[i], col=colors[i], pos=4)
  points(x=xcoord, y=(ycoord-0.05*(i-1)), col=colors[i], pch=pointshapes[i])
}

