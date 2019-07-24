#library(calibrator)
library(mvtnorm)

library(DiceKriging) # only for method requiring a surrogate
#library(BayProjected);library(lhs) # only for KO full MCMC, L2 and Projected
library(MASS)
#library(RobustCalibration);library(nloptr)
#library(extrafont);#font_import()
#loadfonts()
set.seed(0)


#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)



results_dir = args[1]
output_dir = args[2]
dataset_dir = args[3]


mean_VIcalib  <- as.numeric(c(args[4],args[5],args[6]))
sd_VIcalib  <- as.numeric(c(args[7],args[8], args[9]))


mean_VIcalib_NA  <- as.numeric(c(args[10],args[11],args[12]))
sd_VIcalib_NA  <- as.numeric(c(args[13],args[14], args[15]))

# load data
T     <- as.matrix(read.csv(paste(dataset_dir,"T.csv"    ,sep=""),sep=";"))
XStar <- as.matrix(read.csv(paste(dataset_dir,"XStar.csv",sep=""),sep=";"))
X     <- as.matrix(read.table(paste(dataset_dir,"X.csv",sep=""), quote="\"", comment.char="",sep=";"))
Y     <- as.matrix(read.table(paste(dataset_dir,"Y.csv",sep=""), quote="\"", comment.char="",sep=";"))
Z     <- as.matrix(read.csv(paste(dataset_dir,"Z.csv"    ,sep=""),sep=";"))


bornInfX <- min(X,XStar)
bornSupX <- max(X,XStar)
Xnorm <- (X-bornInfX)/(bornSupX-bornInfX)
XStarNorm <- (XStar-bornInfX)/(bornSupX-bornInfX)

d1 <- ncol(X)
d2 <- ncol(T)
N <- nrow(Z)
n <- nrow(Y)


# create surrogate computer model
#kmObj <- km(~1,design = cbind(XStarNorm,T),response = c(Z))
#save("kmObj",file=paste(fileLocation,"/data/kmObj.RData",sep=""))
load(paste(results_dir,"/kmObj.RData",sep=""))
computerModel <- function(x,theta) {
  if (is.matrix(x) && is.vector(theta)) {
    newData <- cbind(x,t(matrix(theta,length(theta),nrow(x))))
  }
  return(predict.km(kmObj,newdata=newData,type="UK",se.compute=FALSE,light.return=TRUE,checkNames=FALSE)$mean)
}


# L2 calibration

#timeL2 <- system.time(source(paste(fileLocation,"/L2.R",sep="")))


# Bayesian Projected Calibration
#timeProj <- system.time(source(paste(fileLocation,"/proj.R",sep="")))

# KOH
#timeKOH <- system.time(source(paste(fileLocation,"/koh.R",sep="")))

# robust calibration
#timeRobust <- system.time(source(paste(fileLocation,"/RobustCalibration.R",sep="")))
#robust_calib@post_sample <- robust_calib@post_sample*10

# checkpoint 1
#save.image("~/Collaboration avec Maurizio/GPcalibration/code/Comparisons/test case 2/checkpoint1.RData")
load(paste(results_dir,"checkpoint1.RData",sep=""))
computerModel <- function(x,theta) {
  if (is.matrix(x) && is.vector(theta)) {
    newData <- cbind(x,t(matrix(theta,length(theta),nrow(x))))
  }
  return(predict.km(kmObj,newdata=newData,type="UK",se.compute=FALSE,light.return=TRUE,checkNames=FALSE)$mean)
}


## PreProc
x = sort(proj_calib$x, decreasing = FALSE, index.return = TRUE)
x_seq = x$x
nPost <- 100
givePosteriorCM <- function(x,thetaPost,computerModel) {
  result <- matrix(NA,nrow(thetaPost),length(x))
  for (i in 1:nrow(thetaPost))
    result[i,] <- computerModel(matrix(x,ncol=1),thetaPost[i,])
  return(result)
}
# Proj
theta_star_mc = proj_calib$theta_star_mc
indFi1 <- apply(theta_star_mc>2,1,prod)*apply(theta_star_mc<6,1,prod)
indFilt <- rep(0,sum(indFi1))
indFilt[sample(1:sum(indFi1),size = nPost)] <- 1
theta_star_mc_f <- theta_star_mc[indFi1==1,][indFilt==1,]
post_proj <- givePosteriorCM(x_seq,theta_star_mc_f,computerModel)
post_proj_mean <- apply(post_proj,2,median)
# VI 
resMean <- mean_VIcalib*10
resSd <- sd_VIcalib*10
thetaRF <- t(t(matrix(rnorm(nPost*length(resMean)),nrow=nPost))*resSd+resMean)
post_RF <- givePosteriorCM(x_seq,thetaRF,computerModel)
post_RF_mean <- apply(post_RF,2,median)
# VINA
resMeanNA <- mean_VIcalib_NA*10
resSdNA <- sd_VIcalib_NA*10
thetaRFNA <- t(t(matrix(rnorm(nPost*length(resMeanNA)),nrow=nPost))*resSdNA+resMeanNA)
post_RFNA <- givePosteriorCM(x_seq,thetaRFNA,computerModel)
post_RF_meanNA <- apply(post_RFNA,2,median)
#KOH
indFi1 <- apply(KO_calib>3.2,1,prod)*apply(KO_calib<6,1,prod)
indFilt <- rep(0,sum(indFi1))
indFilt[sample(1:sum(indFi1),size = nPost)] <- 1
KO_calib_f <- KO_calib[indFi1==1,][indFilt==1,]
post_KO <- givePosteriorCM(x_seq,KO_calib_f,computerModel)
post_KO_mean <- apply(post_KO,2,median)
# robust
indFi1 <- rep(1,nrow(robust_calib@post_sample))#apply(robust_calib@post_sample[,1:3]>2,1,prod)*apply(robust_calib@post_sample[,1:3]<6,1,prod)
indFilt <- rep(0,sum(indFi1))
indFilt[sample(1:sum(indFi1),size = nPost)] <- 1
robust_calib_f <- robust_calib@post_sample[,1:3][indFi1==1,][indFilt==1,]
post_robust <- givePosteriorCM(x_seq,robust_calib_f,computerModel)
post_robust_mean <- apply(post_robust,2,median)

##plots
xlim <- c(0,1); ylim <- c(0,max(Y,post_RF,post_proj))
pdf(paste(output_dir,"calibrated.pdf",sep=""),width=5, height=5*5/8)
# L2
par(mfrow=c(2,3),mar=c(0.1,0.1,3,0.1),oma=c(3,3,0.1,0))
plot(NaN,xlim=xlim,ylim = ylim,xaxt="n")
axis(1,at=seq(0,1,.2),label=rep("",6))
points(XStarNorm,Z,pch=20,col="lightgrey")
lines(x_seq,computerModel(matrix(x_seq,ncol=1),L2_calib$theta_L2),col="black")
points(Xnorm,Y,pch=19,col="black",cex=1.2)
points(Xnorm,Y,pch=19,col="white",cex=.7)
mtext(expression(L[2]))
mtext(expression(italic(y)),side = 2,line = 2,cex=.8)
# Proj
plot(NaN,xlim=xlim,ylim = ylim,yaxt="n",xaxt="n")
points(XStarNorm,Z,pch=20,col="lightgrey")
axis(1,at=seq(0,1,.2),label=rep("",6))
matlines(x_seq,t(post_proj),type="l",col="orange",lty=1)
lines(x_seq,post_proj_mean,type="l",lty=1)
points(Xnorm,Y,pch=19,col="black",cex=1.2)
points(Xnorm,Y,pch=19,col="white",cex=.7)
mtext(expression("Projected"))
# VI
plot(NaN,xlim=xlim,ylim = ylim,yaxt="n",xaxt="n")
points(XStarNorm,Z,pch=20,col="lightgrey")
axis(1,at=seq(0,1,.2),label=rep("",6))
matlines(x_seq,t(post_RF),type="l",col="orange",lty=1)
lines(x_seq,post_RF_mean,type="l",lty=1)
mtext(expression("V-cal"*" "[" "]))
points(Xnorm,Y,pch=19,col="black",cex=1.2)
points(Xnorm,Y,pch=19,col="white",cex=.7)
# VINA
plot(NaN,xlim=xlim,ylim = ylim,xaxt="n")
points(XStarNorm,Z,pch=20,col="lightgrey")
axis(1,at=seq(0,1,.2),label=rep("",6))
matlines(x_seq,t(post_RFNA),type="l",col="orange",lty=1)
lines(x_seq,post_RF_meanNA,type="l",lty=1)
points(Xnorm,Y,pch=19,col="black",cex=1.2)
points(Xnorm,Y,pch=19,col="white",cex=.7)
mtext(expression("V-cal non-add."*" "[" "]))
mtext(expression(italic(x)),side = 1,line = 2,cex=.8)
axis(1,at = c(0,1))
axis(1,labels = FALSE)
mtext(expression(italic(y)),side = 2,line = 2,cex=.8)
# KO
plot(NaN,xlim=xlim,ylim = ylim,xaxt="n",yaxt="n")
points(XStarNorm,Z,pch=20,col="lightgrey")
matlines(x_seq,t(post_KO),type="l",col="orange",lty=1)
axis(1,at = c(0,1))
axis(1,labels = FALSE)
lines(x_seq,post_KO_mean,type="l",lty=1)
points(Xnorm,Y,pch=19,col="black",cex=1.2)
points(Xnorm,Y,pch=19,col="white",cex=.7)
mtext(expression("KOH"))
mtext(expression(italic(x)),side = 1,line = 2,cex=.8)
# robust
plot(NaN,xlim=xlim,ylim = ylim,yaxt="n",xaxt="n")
points(XStarNorm,Z,pch=20,col="lightgrey")
axis(1,at = c(0,1))
axis(1,labels = FALSE)
matlines(x_seq,t(post_robust),type="l",col="orange",lty=1)
lines(x_seq,post_robust_mean,type="l",lty=1)
points(Xnorm,Y,pch=19,col="black",cex=1.2)
points(Xnorm,Y,pch=19,col="white",cex=.7)
mtext(expression("Robust"))
mtext(expression(italic(x)),side = 1,line = 2,cex=.8)
dev.off()
#})
#}




# MSE
Pred_L2 <- computerModel(Xnorm,L2_calib$theta_L2)
sum((c(Y)-Xnorm)^2)
Pred_VI  <- Pred_VINA <- Pred_proj <- Pred_KO <- Pred_robust <- matrix(NaN,nPost,nrow(X))
for (i in 1:nPost) {
  Pred_VI[i,] <- computerModel(Xnorm,thetaRF[i,])
  Pred_VINA[i,] <- computerModel(Xnorm,thetaRFNA[i,])
  Pred_proj[i,] <- computerModel(Xnorm,theta_star_mc_f[i,])
  Pred_KO[i,] <- computerModel(Xnorm,KO_calib_f[i,])
  Pred_robust[i,] <- computerModel(Xnorm,robust_calib_f[i,])
}
timeVal <- (c(timeL2[1]+timeL2[2],timeProj[1]+timeProj[2],79,timeKOH[1]+timeKOH[2],timeRobust[1]+timeRobust[2]))
MSE <-c(L2=sum((Y-Pred_L2)^2),
proj=mean(apply((t((c(Y)-t(Pred_proj)))^2),1,sum)),
vi=mean(apply((t((c(Y)-t(Pred_VI)))^2),1,sum)),
vina=mean(apply((t((c(Y)-t(Pred_VINA)))^2),1,sum)),
ko=mean(apply((t((c(Y)-t(Pred_KO)))^2),1,sum)),
robust=mean(apply((t((c(Y)-t(Pred_robust)))^2),1,sum))
)
#print(round(timeVal))
print("########## case2 MSE tab")
print(round(MSE*1000,2))


# Contour
indices <- c(2,3)
lims <- c(2,8,1.2,8)
density_proj <-  kde2d(theta_star_mc[,indices[1]],theta_star_mc[,indices[2]],n = 60,lims = c(2,8,1.5,8))
density_VI <- matrix(dmvnorm(as.matrix(expand.grid(density_proj$x,density_proj$y)),mean = c(resMean[indices[1]],resMean[indices[2]]),sigma = diag((c(resSd[indices[1]],resSd[indices[2]])))),ncol = ncol(density_proj$z))
density_VINA <- matrix(dmvnorm(as.matrix(expand.grid(density_proj$x,density_proj$y)),mean = c(resMeanNA[indices[1]],resMeanNA[indices[2]]),sigma = diag((c(resSdNA[indices[1]],resSdNA[indices[2]])))),ncol = ncol(density_proj$z))
density_KO <-  kde2d(KO_calib[450:nrow(KO_calib),indices[1]],KO_calib[450:nrow(KO_calib),indices[2]],n=60,lims=lims)
density_robust <-  kde2d(robust_calib@post_sample[1:nrow(robust_calib@post_sample[,1:3]),indices[1]],robust_calib@post_sample[1:nrow(robust_calib@post_sample[,1:3]),indices[2]],n=60,lims=lims,h=0.5)




ratio = 4/5
wi = 3.5
pdf(paste(output_dir,"posteriorDist.pdf",sep=""),width=wi, height=ratio*wi)
par(mfrow=c(2,2),mar=c(.1,0.1,1.8,0.1),oma=c(3,3.1,0,.1))
contour(density_proj,cex.axis=1,asp=1,xaxt="n")
points(L2_calib$theta_L2[indices[1]],L2_calib$theta_L2[indices[2]],col="red",pch=19)
mtext(expression("Projected"))
mtext(expression(italic(theta)[3]),side = 2,line = 2)
contour(density_proj$x,density_proj$y,density_VI,yaxt="n",cex.axis=1,asp=1,xaxt="n",nlevels = 4,col="green")
contour(density_proj$x,density_proj$y,density_VINA,yaxt="n",cex.axis=1,asp=1,xaxt="n",add=TRUE,col="blue",nlevels = 4)
points(L2_calib$theta_L2[indices[1]],L2_calib$theta_L2[indices[2]],col="red",pch=19)
mtext(expression("V-cal"*" "[" "]))
legend("topleft",col=c("green","blue"),lwd=1,c("additive","general"),bty="n")
contour(density_proj$x,density_proj$y,density_KO$z,cex.axis=1,asp=1)
points(L2_calib$theta_L2[indices[1]],L2_calib$theta_L2[indices[2]],col="red",pch=19)
mtext(expression(italic(theta)[3]),side = 2,line = 2)
points(L2_calib$theta_L2[indices[1]],L2_calib$theta_L2[indices[2]],col="red",pch=19)
mtext(expression("KOH"))
mtext(expression(italic(theta)[2]),side = 1,line = 2)
contour(density_proj$x,density_proj$y,density_robust$z,cex.axis=1,yaxt="n",nlevel=5,asp=1)
points(L2_calib$theta_L2[indices[1]],L2_calib$theta_L2[indices[2]],col="red",pch=19)
mtext(expression("Robust"))
mtext(expression(italic(theta)[2]),side = 1,line = 2)
dev.off()


