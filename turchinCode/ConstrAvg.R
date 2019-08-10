# Contructs Soc Complx data from SCdat.csv for PCA and regressions
# Uses averages for ranges and disagreements

output <- matrix(nrow = 0, ncol = (5+nrow(Vars)))
for(iNGA in 1:length(NGAs)){
   NGA <- NGAs[iNGA]
   dat <- read.table('SCdat.csv', sep=",", header=TRUE, colClasses = "character")
   load("PolsVars.Rdata")
   polities <- polities[polities[,1]==NGA,]      # Use only one NGA at a time
   polities <-polities[polities[,8]=="n",]       # Exclude duplicates
   row.names(polities) <- NULL
   dat <- dat[dat$NGA==NGA,]                     # Select data for the NGA
   row.names(dat) <- NULL
   dat_temp <- matrix(nrow = 0, ncol = 9)        # Make sure all data are for polities in PolVars
   for(i in 1:nrow(polities)){
      dat_temp <- rbind(dat_temp,dat[dat$Polity == polities[i,2],])
      }
   dat <-dat_temp

   # If dat is empty, skip the rest and go to the next iNGA
   if(length(dat[,1]!=0)){

   # Reduce ranges to averages
   for(i in 1:nrow(dat)){
      if(dat[i,5]!="") {
         dat[i,4] <- mean(as.numeric(dat[i,4:5]))
         dat$Value.Note[i] <- "replaced"
      }}

# Substitute disputed and uncertain with averages, eliminate extra rows
   dat_temp<-dat
   for(i in 1:(nrow(dat)-1)){
   if(dat[i,9]=="disputed" | dat[i,9]=="uncertain"){value <- as.numeric(dat[i,4])
      for(j in (i+1):nrow(dat)){
         if(dat[i,1] == dat[j,1] & dat[i,3] == dat[j,3] & dat[i,6] == dat[j,6] & dat[i,7] == dat[j,7] & dat[i,9] == dat[j,9]){
            value <- c(value,as.numeric(dat[j,4]))
            dat[j,9] <- "delete"}}
            dat[i,4] <- mean(value)         
   }  }   

   datSC <- dat
   datSC <- datSC[datSC[,9] != "delete",]
   row.names(datSC) <- NULL
   datSC <- datSC[,c(2,3,4,6,7)]
   colnames(datSC) <- c("PolName","Variable","Value", "Date", "DateTo")

# Construct output
   tmin <- ceiling(0.01*min(polities$Start[polities$NGA==NGA]))
   tmax <- floor(0.01*max(polities$End[polities$NGA==NGA]))
   out <- matrix(nrow=c(length(100*tmin:tmax)),ncol=(4+nrow(Vars)))
   colnames(out) <- c("NGA","PolName", "PolID", "Date", Vars[,3])  # Use short names for variables
   out[,1] <- as.character(NGA)
   out[,4] <- 100*tmin:tmax

for(i in 1:nrow(out)){ 
   for(j in 1:nrow(polities)){
      if( (as.numeric(out[i,4]) <= as.numeric(polities[j,5])) & 
             (as.numeric(out[i,4]) >= as.numeric(polities[j,4])) ){ 
         out[i,2] <- as.character(polities[j,2]) 
         out[i,3] <- as.character(polities[j,3])
      }}}
out <- out[is.na(out[,2])==FALSE,]   # Eliminate centuries for which a polity is lacking

# First populate 'out' with data tied to polities, not dates
for(ivar in 1:nrow(Vars)){
   datV <- datSC[(datSC[,2]==Vars[ivar,1]) & (datSC[,4]==""),]
   if(is.null(nrow(datV))){datV <- array(datV,c(1,4))}
   for(i in 1:nrow(datV)){
      for(j in 1:nrow(out)){
         if(nrow(datV) != 0){
            if(out[j,2] == datV[i,1]){out[j,ivar+4] <- datV[i,3]
            }}}}}
# Next populate 'out' with data tied to a single date
for(ivar in 1:nrow(Vars)){
   datV <- datSC[((datSC[,2]==Vars[ivar,1]) & (datSC[,4]!="") & (datSC[,5]=="")),]
   if(is.null(nrow(datV))){datV <- array(datV,c(1,5))}
   for(i in 1:nrow(datV)){
      for(j in 1:nrow(out)){
         if(nrow(datV) != 0){
            century <- 100*round(as.numeric(datV[i,4])/100)
            if(out[j,4] == as.character(century)){out[j,ivar+4] <- datV[i,3]
            }}}}}
# Finally populate 'out' with data tied to a range of dates
for(ivar in 1:nrow(Vars)){
   datV <- datSC[((datSC[,2]==Vars[ivar,1]) & (datSC[,4]!="") & (datSC[,5]!="")),]
   if(is.null(nrow(datV))){datV <- array(datV,c(1,5))}
   for(i in 1:nrow(datV)){
      for(j in 1:nrow(out)){
         if(nrow(datV) != 0){
            century <- as.numeric(out[j,4])
            tmin <- as.numeric(datV[i,4])
            tmax <- as.numeric(datV[i,5])
            if(century >= tmin & century <= tmax){out[j,ivar+4] <- datV[i,3]
            }}}}}

# Calculate the proportion of data coded by century
PropCoded <- array(0,c(nrow(out),2))
PropCoded[,1] <- out[,4]
colnames(PropCoded) <- c("Date1","PropCoded")
for(i in 1:nrow(out)){j <- 0  
                      for(ivar in 1:nrow(Vars)){
                         if(is.na(out[i,ivar+4])){j <- j+1} }
                      PropCoded[i,2] <- 0.1*round((nrow(Vars) - j)/nrow(Vars)*1000) # Keep 3 sign digits
}
out <- cbind(out[,1:4],PropCoded[,2],out[,5:(nrow(Vars)+4)])
colnames(out) <- c("NGA","PolName", "PolID", "Date", "PropCoded", Vars[,3])  

output <- rbind(output,out)
}  # Closing the if loop testing for empty dat
   else{print(c("No data for ", NGA))}
}  # Closing the iNGA loop

output <- output[output[,5] != 0,]  # Remove rows of missing values
output <- output[,c(1,3:ncol(output))]   #### Remove redundant PolName 
output <- as.data.frame(output, stringsAsFactors = FALSE)
for (i in 3:ncol(output)){output[,i] <- as.numeric(output[,i])}
OutSeries <- output

outNA <- output
outNA[is.na(outNA)] <- -1
outNA[,3] <- 0
OutPolity <- data.frame()
OutPolity <- rbind(OutPolity, output[1,])
for(i in 2:nrow(output)) {
   if(all(outNA[i,] == outNA[i-1,]) == FALSE) OutPolity <- rbind(OutPolity,output[i,])
}

#write.csv(output, file="output.csv",  row.names=FALSE)
#write.csv(OutPolity, file="OutPolity.csv",  row.names=FALSE)
rm(i,j,tmin,tmax,datV,century,ivar,iNGA,NGA,PropCoded,out,value,dat,dat_temp,datSC, outNA, output)   # clean up the workspace

