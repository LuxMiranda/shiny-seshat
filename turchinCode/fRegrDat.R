#### fRegrDat constructs RegrDat, using InterpDat from SC.Rdata
##### Reconfigure InterpDat with response variable and lagged predictors
RegrDat <- data.frame()
for(iNGA in 1:length(NGAs)){
  dat <- InterpDat[InterpDat$NGA == NGAs[iNGA],]
  n <- nrow(dat)
  if(n != 0){rdat <- cbind(dat[2:n,1:4],dat[1:(n-1),4:length(dat)])
  RegrDat <- rbind(RegrDat,rdat)
  }
}
RegrDat$Time <- RegrDat$Time - 100 ### Set RegrDat$Time to t, not t+1
nm <- colnames(RegrDat)
nm[4] <- paste0(nm[4],"(t+1)")
colnames(RegrDat) <- nm

##### Calculate Space using estimated dpar (set in !SC_analyz)
Space <- RegrDat[,1:4]
Space[,4] <- 0
colnames(Space) <- c("NGA","PolID","Time","Space")

colMat <- colnames(DistMatrix)
rowMat <- rownames(DistMatrix)
for(i in 1:nrow(RegrDat)){
  t1 <- RegrDat$Time[i]
  dat <- RegrDat[RegrDat$Time==t1,c(1:3,5)]
  if(nrow(dat) > 1){
    delta <- vector(length=nrow(dat))
    for(j in 1:nrow(dat)){
      dt <- DistMatrix[colMat==dat$NGA[j],]
      delta[j] <- dt[rowMat==RegrDat$NGA[i]]
    }
    s <- exp(-delta/dpar)*dat[,4]
    s <- s[delta != 0]  ### Exclude i=j
    Space$Space[i] <- mean(s)
  }
}
RegrDat <- cbind(RegrDat,Space$Space)
nm <- colnames(RegrDat)
nm[length(nm)] <- "Space"
colnames(RegrDat) <- nm


#### Calculate Language = matrix of linguistic distances
Phylogeny <- RegrDat[,1:4]
Phylogeny[,4] <- 0
colnames(Phylogeny) <- c("NGA","PolID","Time","Phylogeny")

for(i in 1:nrow(RegrDat)){
  t1 <- RegrDat$Time[i]
  dat <- RegrDat[RegrDat$Time==t1,c(1:3,5)]
  dat <- dat[dat$NGA != RegrDat$NGA[i],]   ### Exclude i = j
  PolID <- RegrDat$PolID[i]
  PolLang <- polities[polities$PolID==PolID,9:11]
  if(nrow(dat) > 1){
    weight <- vector(length=nrow(dat)) * 0
    for(j in 1:nrow(dat)){
      dt <- dat[j,]
      PolLang2 <- polities[polities$PolID==dt$PolID,9:11]
      if(PolLang[1,3]==PolLang2[1,3]){weight[j] <- 0.25}
      if(PolLang[1,2]==PolLang2[1,2]){weight[j] <- 0.5}
      if(PolLang[1,1]==PolLang2[1,1]){weight[j] <- 1}
    }
    s <- weight*dat[,4]
    Phylogeny$Phylogeny[i] <- mean(s)
  }
}
RegrDat <- cbind(RegrDat,Phylogeny$Phylogeny)
nm <- colnames(RegrDat)
nm[length(nm)] <- "Phylogeny"
colnames(RegrDat) <- nm

#### Add time lag
Lag2 <- RegrDat[,1:4]
Lag2[,4] <- NA
for(i in 1:nrow(RegrDat)){
  t2 <- RegrDat$Time[i] - 100
  NGA <- RegrDat$NGA[i]
  dat <- RegrDat[RegrDat$Time==t2 & RegrDat$NGA == NGA,]
  if(nrow(dat)>1){print("Error: more than one Lag2")}
  if(nrow(dat)==1){Lag2[i,4] <- dat[1,5]}
}
RegrDat <- cbind(RegrDat,Lag2[,4])

nm <- colnames(RegrDat)
nm[length(nm)] <- "Lag2"
colnames(RegrDat) <- nm
NGARegrDat <- RegrDat  ##### keep the NGA, PolID, Time
RegrDat <- RegrDat[,4:length(RegrDat)]  #### Drop NGA, PolID, and Time

#### Differenced response variable
DelDat <- RegrDat
DelDat[,1] <- DelDat[,1] - DelDat[,2]
nm <- colnames(DelDat)
nm[1] <- paste("del",nm[1])
colnames(DelDat) <- nm


#### Save (or resave) everything in HS.Rdata
#save(polities, coords, NGAs, RegrDat, DistMatrix, file="HS.Rdata")
rm(dat,dt,Lag2,Phylogeny,PolLang,PolLang2,rdat,Space)
rm(colMat,delta,i,iNGA,j,n,NGA,nm,PolID,rowMat,s,t1,t2,weight)






