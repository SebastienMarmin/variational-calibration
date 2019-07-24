#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

set.seed(0)


results_dir = args[1]
output_dir = args[2]
dataset_dir = args[3]


trueValue <- c(0.0894107602071017,0.307715772418305,0.372464679181576)

mean_VIcalib  <- as.numeric(c(args[4],args[5],args[6]))
sd_VIcalib  <- as.numeric(c(args[7],args[8], args[9]))




axialPre <- 150
x <- seq(0,1,length.out =  axialPre)
p_VIcalib1 <- dnorm(x,mean = mean_VIcalib[1],sd=sd_VIcalib[1])
p_VIcalib2 <- dnorm(x,mean = mean_VIcalib[2],sd=sd_VIcalib[2])
p_VIcalib3 <- dnorm(x,mean = mean_VIcalib[3],sd=sd_VIcalib[3])

laGP_calib <- rep(NA,3)#runif(3)

thetas_tree <- read.table(paste(results_dir,"tree.txt",sep=""))
p_trCalib1Dens1 <- density(thetas_tree[,1])
p_trCalib1Dens2 <- density(thetas_tree[,2])
p_trCalib1Dens3 <- density(thetas_tree[,3])
p_trCalib1Y1 <- c(0,0,p_trCalib1Dens1$y,0,0)
p_trCalib1Y2 <- c(0,0,p_trCalib1Dens2$y,0,0)
p_trCalib1Y3 <- c(0,0,p_trCalib1Dens3$y,0,0)
p_trCalib1X1 <- c(0,min(p_trCalib1Dens1$x),p_trCalib1Dens1$x,max(p_trCalib1Dens1$x),1)
p_trCalib1X2 <- c(0,min(p_trCalib1Dens2$x),p_trCalib1Dens2$x,max(p_trCalib1Dens2$x),1)
p_trCalib1X3 <- c(0,min(p_trCalib1Dens3$x),p_trCalib1Dens3$x,max(p_trCalib1Dens3$x),1)


ylim <- c(0,6)

ordre <-    c(4,     1,     2,      3, 5)
#           VI     True   laGP    Tree
colors <- c("blue","black","red","darkgreen","orange")
methNames <- c("V-cal","True","Mod. LAGP","Sum-of-trees","None")

abscissa <- list(list(x,
                 c(trueValue[1],trueValue[1]),
                 c(laGP_calib[1],laGP_calib[1]),
                 p_trCalib1X1,
                 x),
                 list(x,
                      c(trueValue[2],trueValue[2]),
                      c(laGP_calib[2],laGP_calib[2]),
                      p_trCalib1X2,
                      x),
                 list(x,
                      c(trueValue[3],trueValue[3]),
                      c(laGP_calib[3],laGP_calib[3]),
                      p_trCalib1X3,
                      x))
ordinate <- list(list(p_VIcalib1,
                      c(0,2*ylim[2]),
                      c(0,2*ylim[2]),
                      p_trCalib1Y1,
                      rep(NA,length(x))),
                 list(p_VIcalib2,
                      c(0,2*ylim[2]),
                      c(0,2*ylim[2]),
                      p_trCalib1Y2,
                      rep(NA,length(x))),
                 list(p_VIcalib3,
                      c(0,2*ylim[2]),
                      c(0,2*ylim[2]),
                      p_trCalib1Y3,
                      rep(NA,length(x))))

pdf(paste(output_dir,"boreholeCalibrated.pdf",sep=""),width = 3.5,height = 1.2,onefile = FALSE)
lwd=2
par(mfrow=c(1,3),mar=c(2.5,.3,.1,.1),oma=c(0,3.7,0,0), xpd=NA)
plot(NA,xlim=c(0,1),ylim=ylim,xaxt="n",xlab="",ylab="Density")
axis(side=1,at=seq(0,1,.2),labels = c(0,"","","","",1))
mtext(expression(italic(theta)[1]),side = 1,line=1.5,cex=.8)
for (i in 1:length(ordre)){
  lines(abscissa[[1]][[ordre[i]]],ordinate[[1]][[ordre[i]]],col=colors[ordre[i]],lwd=lwd)
}
plot(NA,xlim=c(0,1),ylim=ylim,xaxt="n",yaxt="n",xlab="",ylab="")
axis(side=1,at=seq(0,1,.2),labels = c(0,"","","","",1))
mtext(expression(italic(theta)[2]),side = 1,line=1.5,cex=.8)
for (i in 1:length(ordre)){
  lines(abscissa[[2]][[ordre[i]]],ordinate[[2]][[ordre[i]]],col=colors[ordre[i]],lwd=lwd)
}
plot(NA,xlim=c(0,1),ylim=ylim,xaxt="n",yaxt="n",xlab="",ylab="")
mtext(expression(italic(theta)[3]),side = 1,line=1.5,cex=.8)
axis(side=1,at=seq(0,1,.2),labels = c(0,"","","","",1))
for (i in 1:length(ordre)){
  lines(abscissa[[3]][[ordre[i]]],ordinate[[3]][[ordre[i]]],col=colors[ordre[i]],lwd=lwd)
}
par(mfrow=c(1,1))
#legend("topright",lwd=lwd,col=colors[ordre],legend=methNames[ordre],bty="n",cex=.7,inset=c(-.43,0))
dev.off()












#### MSE TAB


library(lhs)
set.seed(0)
caldim <- c(1,2,5)
vardim <- (1:8)[-caldim]
borehole <- function(inputx){
  x <- rep(NA,8)#inputx#
  x[caldim] <- tail(inputx,length(caldim))
  x[vardim] <- head(inputx,length(vardim))
  rw <- x[1] * (0.15 - 0.05) + 0.05
  r <-  x[2] * (50000 - 100) + 100
  Tu <- x[3] * (115600 - 63070) + 63070
  Hu <- x[4] * (1110 - 990) + 990
  Tl <- x[5] * (116 - 63.1) + 63.1
  Hl <- x[6] * (820 - 700) + 700
  L <-  x[7] * (1680 - 1120) + 1120
  Kw <- x[8] * (12045 - 9855) + 9855
  m1 <- 2 * pi * Tu * (Hu - Hl)
  m2 <- log(r / rw)
  m3 <- 1 + 2 * L * Tu / (m2 * rw^2 * Kw) + Tu / Tl
  return(m1/m2/m3/150-0.5)
}

bias <- function(x) 
{
  x <- as.matrix(x)   
  out <- 2 * (10 * x[ ,1]^2 + 4 * x[ ,2]^2) / (50 * x[ ,3] * x[ ,4] + 10)
  return(0.0051*out-0.01)
}

repe <- 6#60#100
D1 <- length(vardim)
D2 <- length(caldim)

nIntegr <- 1000#10000#100000
Xintegr <- randomLHS(nIntegr,D1)

XTintegr <- cbind(Xintegr,matrix(rep(trueValue,nrow(Xintegr)),byrow = TRUE,ncol=D2))
Zintegr <- apply(XTintegr,1,borehole)
Yintegr <- Zintegr + bias(Xintegr)

computeMSE <- function(theta,Xintegr,borehole,Yintegr) {
  Xtheta_rep <- cbind(Xintegr,matrix(rep(theta,nrow(Xintegr)),byrow = TRUE,ncol=length(theta)))
  Ztheta_rep <- apply(Xtheta_rep,1,borehole)
  return(sqrt(mean((Ztheta_rep-Yintegr)^2)))
}

randomTheta <- matrix(runif(repe*D2),ncol=D2)

#mean_VIcalib  <- c( 7.9640e-02, 3.8901e-01, 4.1594e-01)
#sd_VIcalib  <- c(0.0697, 0.1155, 0.1034)
VItheta <- matrix(rnorm(repe*D2)*sd_VIcalib+ mean_VIcalib,byrow = TRUE,ncol=D2)

treeTheta <- as.matrix(read.table(paste(results_dir,"tree.txt",sep="")))

mseVI <- apply(VItheta, 1,  computeMSE,Xintegr=Xintegr,borehole=borehole,Yintegr=Yintegr)
#plot(density(mseVI))
mseRandom <- apply(randomTheta, 1,  computeMSE,Xintegr=Xintegr,borehole=borehole,Yintegr=Yintegr)
#plot(density(mseRandom))
mseTree <- apply(treeTheta, 1,  computeMSE,Xintegr=Xintegr,borehole=borehole,Yintegr=Yintegr)
#plot(density(mseTree))

print("########## borehole MSE tab")
print(paste("None         |",mean(mseRandom)))
print(paste("v-cal        |",mean(mseVI)))
print(paste("sum-of-trees |",mean(mseTree)))





