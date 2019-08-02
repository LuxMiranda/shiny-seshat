#####  Regressions on SC data
### Run !MI_SC.R after a new scrape. ImpDatRepl.csv has nrep imputations
load("SC.Rdata")             ###  InterpDat is the average of nrep imputations
RemPols <- c("EsHabsb","InGaroL","CnHChin","PgOrokL","FmTrukL","InGupta")   ### Polities to remove from analysis
# RemPols <- c("EsHabsb","InGaroL","CnHChin","PgOrokL","FmTrukL","InGupta", "IsCommw","NorKing","GhAshnE","GhAshnL") ### eliminate spurious Infra

dpar <- 1000   ### d parameter that determines how rapidly geographic influence declines with distance

#### Construct RegrDat from InterpDat 
source("fRegrDat.R")
 for(i in 1:length(RemPols)){NGARegrDat <- NGARegrDat[NGARegrDat$PolID != RemPols[i],] }
#### Rename variables
colnames(NGARegrDat)[4:5] <- c("Info","Lag1")
colnames(NGARegrDat)[9:12] <- c("Hier","Gov","Infra","Money")
RegrDat <- NGARegrDat[,4:ncol(NGARegrDat)]
rm(coords,DelDat,DistMatrix, polities,dpar,i, NGAs,RemPols, InterpDat, PolityDat)
#     write.csv(NGARegrDat, file="NGARegrDat.csv",  row.names=FALSE)
RegrDat <- RegrDat[,c(1:9,11:12)]   ####  Drop Space as not significant

####  Exhaustive regressions with linear terms
print(paste("Response variable =",colnames(RegrDat)[1]))
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
write.csv(output, file="output.csv",  row.names=FALSE)
####

#### Best model for Info by AIC, standardized coefficients  
#RegrDat <- RegrDat[,1:(ncol(RegrDat)-1)]          ###  Omit Lag2 if not significant, 
RegrDat <- RegrDat[is.na(RegrDat[,ncol(RegrDat)]) == FALSE,]  ###  or omit missing values in Lag2 if significant
for(i in 1:ncol(RegrDat)){   RegrDat[,i] <- (RegrDat[,i] - mean(RegrDat[,i]))/sd(RegrDat[,i]) } # For standardized coefficients
#options(scipen=999, digits = 5)
summary(fit <- glm(RegrDat[, c(1,2,3,7,8,9,10,11) ]))

###  Diagnostics
layout(matrix(c(1,2,3,4),2,2)) # 4 graphs/page
plot(fit)

####################################################################################################
#### Fixed-effects regression: NGAs
dt <- NGARegrDat[,c(4,5,15,6,10,11,12,14,1)]
dt <- dt[is.na(dt$Lag2) == FALSE,]  ###  or omit missing values in Lag2 if significant
for(i in 1:(ncol(dt)-1)){   dt[,i] <- (dt[,i] - mean(dt[,i]))/sd(dt[,i]) } # For standardized coefficients
summary(fit <- glm(dt))
#summary(fit <- lm(dt$Info ~ dt$Lag1 + dt$Lag1.sq + dt$Lag2 +dt$PolPop +dt$Infra + dt$Money + dt$Phylogeny + factor(dt$NGA), data=dt))

#### Take out Ghana and Iceland
dt <- dt[(dt$NGA != "Iceland") & (dt$NGA != "Ghanaian Coast"),]
summary(fit <- glm(dt))

#### Best Linear Model without Ghana and Iceland
summary(fit <- glm(dt[,c(1:5,7:8)]))

#### Absolute time as a covariate
dt <- NGARegrDat[(NGARegrDat$NGA != "Iceland") & (NGARegrDat$NGA != "Ghanaian Coast"),c(4,5,15,6,10,12,14,3)]
dt <- dt[is.na(dt$Lag2) == FALSE,]  ###  omit missing values in Lag2
for(i in 1:(ncol(dt))){   dt[,i] <- (dt[,i] - mean(dt[,i]))/sd(dt[,i]) } # For standardized coefficients
summary(fit <- glm(dt))


#### Tests for nonlin effects
iVar <- 2  ### adding Lag1 squared
NLDat <- dt 
NLDat <- cbind(NLDat, NLDat[,iVar]^2)
colnames(NLDat)[9] <- paste0(colnames(NLDat)[iVar],".sq")
NLDat <- NLDat[,c(1,2,9,3:8)]
summary(fit <- glm(NLDat))
####

iVar <- 7  ### 4: Lag2, 5:PolPop, 6:Gov, 7:Money
NLDat1 <- cbind(NLDat, NLDat[,iVar]^2)
colnames(NLDat1)[10] <- paste0(colnames(NLDat)[iVar],".sq")
for(i in 1:(ncol(NLDat1))){ NLDat1[,i] <- (NLDat1[,i] - mean(NLDat1[,i]))/sd(NLDat1[,i]) }
summary(fit <- glm(NLDat1))
####

#### The best overall model for Info
summary(fit <- glm(NLDat1[,c(1:4,6:7,10,8:9)]))
###


################################################################################################
# Normality of Residuals
library(MASS)
sresid <- studres(fit)
hist(sresid, freq=FALSE, main="Distribution of Studentized Residuals")
xfit<-seq(min(sresid),max(sresid),length=40)
yfit<-dnorm(xfit)
lines(xfit, yfit) 
rm(sresid,xfit,yfit)
####


############################################################################
####     Estimate d parameter
#load("SC.Rdata")             ###  InterpDat is the average of nrep imputations
#out <- data.frame()
#for(dpar in seq(100,2000, by=100)){
#  source("fRegrDat.R")
#  fit <- lm(RegrDat)
#  print(summary(fit))
#  out <- rbind(out,c(dpar,summary(fit)$r.sq))
#}

