# Single Imputation function using stochastic regression 

dat <- AggrDat[,5:length(AggrDat)]

#### Omit the Response Variable
index <- 1:ncol(dat)
index <- index[index != iResp]
dat <- dat[,index]
ImpDat <- dat

for(j in 1:length(dat[1,])){
   for(i in 1:length(dat[,1])){
      if(is.na(dat[i,j])==TRUE){
         index <- c(1:length(dat[1,]))
         index <- index[is.na(dat[i,])==FALSE]
         RegrDat <- dat[,c(j,index)]
         RegrDat <- RegrDat[is.na(RegrDat[,1])==FALSE,]
         fit <- lm(RegrDat)
         Pval <- summary(fit)$coefficients[,4]
         Pval <- Pval[-1]
         if (all((Pval < 0.05)==FALSE)){Pval[Pval==min(Pval)] <- 0.01}
         index <- index[Pval < 0.05] 
         RegrDat <- dat[,c(j,index)]
         RegrDat <- RegrDat[is.na(RegrDat[,1])==FALSE,]
         fit <- lm(RegrDat)         
         predictors <- dat[i,c(j,index)]
         predictors[1] <- 1
         coeff <- coefficients(fit)
         ImpDat[i,j] <- sum(coeff*predictors) + sample(fit$residuals,1)
      }
   }
}

rm(dat,predictors,RegrDat,coeff,fit,i,index,j,Pval)
