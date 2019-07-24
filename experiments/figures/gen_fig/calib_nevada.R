#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
set.seed(0)

results_dir = args[1]
output_dir = args[2]
dataset_dir = args[3]



#library(extrafont)
#font_install('fontcm')
#font_import()
#loadfonts()
#fonts()

#script.dir <- dirname(sys.frame(1)$ofile)
#setwd(script.dir)
XStar <- as.matrix(read.csv(paste(dataset_dir,"XStar.csv",sep=""), header=FALSE, sep=";"))
Z <- as.numeric(as.matrix(read.csv(paste(dataset_dir,"Z.csv",sep=""), header=FALSE, sep=";")))
X <- as.matrix(read.csv(paste(dataset_dir,"X.csv",sep=""), header=FALSE, sep=";"))
Y <- as.numeric(as.matrix(read.csv(paste(dataset_dir,"Y.csv",sep=""), header=FALSE, sep=";")))
posteriorMeanY_tree <- rep(NA,length(Y))

data_laGP <- as.matrix(read.table(paste(results_dir,"postMeanLaGP.csv",sep=""), quote="\"", comment.char=""))
posteriorMeanY_laGP <- data_laGP[,2]
abscissa_laGP <- data_laGP[,1]
data_mulLayer <- as.matrix(read.table(paste(results_dir,"postMeanDeep.csv",sep=""), quote="\"", comment.char=""))
posteriorMeanY_mulLayer <- data_mulLayer[,1]

abscissa_mulLayer <- seq(0,1,length.out=length(posteriorMeanY_mulLayer))#data_mulLayer[,1]
data_zeroLayer <- as.matrix(read.table(paste(results_dir,"postMeanShallow.csv",sep=""), quote="\"", comment.char=""))
posteriorMeanY_zeroLayer <- data_zeroLayer[,1] #data_zeroLayer[,2]
abscissa_zeroLayer <- seq(0,1,length.out=length(posteriorMeanY_zeroLayer))#data_zeroLayer[,1]
data_tree <- read.table(paste(results_dir,"postMeanTree.csv",sep=""), quote="\"", comment.char="")
posteriorMeanY_tree <- data_tree[,2]
abscissa_tree <- data_tree[,1]

print("obs")
print(str(posteriorMeanY_mulLayer))

methodNames <- c("Modul. LaGP","Shallow V-cal","Deep V-cal","Sum-of-trees")
colNames <- c("red","blue","darkblue","darkgreen")
lty <- rep(1,length(colNames))
lwd=3

maskStar <- XStar[,1]==1
mask     <- X[,1]==1
ratiIm <- 3.5/8
wi <- 5
pdf(file = paste(output_dir,"plotNevadaCM.pdf",sep=""),width = wi,height = ratiIm*wi,onefile = FALSE)#,family = "CM Roman")
par(mar=c(2.5,3.1,.1,.1))
plot(XStar[maskStar,2],Z[maskStar],pch=20,col="lightgrey",xaxt="n",yaxt="n",xlab="",ylab="")
axis(1,at = seq(0,1,.2),labels = c("0","0.2","","","0.8","1"))
axis(2,at = seq(-1.5,1.5,.5),labels = c("-1.5","","","0","","1","1.5"))
mtext(expression(italic(x)[2]),side = 1,line=1.5)
mtext(expression(italic(y)),side = 2,line=2.2)
lines(abscissa_laGP,posteriorMeanY_laGP,col=colNames[1],lwd=lwd,lty=lty[1])
lines(abscissa_zeroLayer,posteriorMeanY_zeroLayer,col=colNames[2],lwd=lwd,lty=lty[2])
lines(abscissa_zeroLayer,posteriorMeanY_mulLayer,col=colNames[3],lwd=lwd,lty=lty[3])
lines(abscissa_tree,posteriorMeanY_tree,col=colNames[4],lwd=lwd,lty=lty[4])
legend(-.06,-2,methodNames[1:2],col=colNames[1:2],lwd=lwd,bty="n",lty=lty[1:2],xjust = 0,yjust = 0)
legend(0.42,-2,methodNames[3:4],col=colNames[3:4],lwd=lwd,bty="n",lty=lty[3:4],xjust = 0,yjust = 0)
points(X[mask,2],Y[mask],pch=19,col="black",cex=1.2)
points(X[mask,2],Y[mask],pch=19,col="white",cex=.7)
dev.off()
#embed_fonts("plotNevadaCM.pdf", outfile="plotNevadaCM_emb.pdf")

