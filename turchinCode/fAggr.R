# fAggr.R  -- universal for averaged and MI data
# Aggregate data into CCs: scale, hierarchy, government, infrastr, information, money
data <- ConstrDat[ConstrDat$PropCoded > PropCoded_threshold,]  ### From ConstrAvg.R, or ConstrMI.R; Omit sparsely coded polities
dat <- data[,5:ncol(data)] 
AggrDat <- matrix(NA, 0, 0)
for(i in 1:nrow(dat)){
   row <- log10(dat[i,1:3])                                         # scale variables, log-transformed based 10
   dt <- dat[i,c(4:7)]
     dt <- dt[is.na(dt)==FALSE]
     row <- cbind(row,NA)
     if(length(dt)!=0){row[length(row)] <- mean(dt) }               # hierarchy, averaging over non-missing values
   dt <- dat[i,8:18]
   if(is.na(dt$ExamSyst)){dt$ExamSyst <- 0}                         # Missing => absent (only secure presence counts)
   if(is.na(dt$MeritProm)){dt$MeritProm <- 0}
   row <- cbind(row,mean(dt[is.na(dt)==FALSE]))                     # government
   dt <- dat[i,19:30]
   row <- cbind(row,mean(dt[is.na(dt)==FALSE]))                     # infrastr
   writing <- dat[i,c(31,32,34,33)]
   writing[is.na(writing)] <- 0                                     # Missing => absent
   writing[writing < 0.5] <- 0                                      # Inferred absent => absent
   writing[writing > 0.5] <- 1                                      # Inferred present => present
   writing <- max(writing * 1:4)                                    # Code writing on scale of 0 to 4
   texts <- dat[i, 37:45]
   texts[is.na(texts)] <- 0                                         # Count only securely "present"
   texts[texts < 1] <- 0
   texts <- (sum(texts))
   row <- cbind(row,(writing+texts)/13)                             # info = writing + texts, scaled between 0 and 1
   dt <- dat[i,46:51]
   money <- dt*1:6                                                  # money = coded on scale of 1 to 6
      money <- money[is.na(money)==FALSE]
      row <- cbind(row,NA)
      if(length(money)!=0){row[length(row)] <- max(money) }
   AggrDat <- rbind(AggrDat, row)
} 
AggrDat <- cbind( data[,1:4] ,AggrDat )
colnames(AggrDat) <- c("NGA", "PolID","Time", "PropCoded", "PolPop","PolTerr","CapPop","hierarchy", "government", "infrastr", "info", "money")
row.names(AggrDat) <- NULL

rm(data,dat,dt,row,i,money,writing, texts,polities)

