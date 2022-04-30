# Description: Shows how to use the Heston neural network implementation.

# Remarks: 
# o Although still fast, the code has not been optimised (the Matlab version has).
# o The underlying models are not implemented in R. You must therefore 
#	  consult the Matlab code to check the neural networks against the 
#	  actual pricing models.

# Load libraries:
#install.packages("jsonlite")
#install.packages("ggplot2")
#install.packages("gridExtra")
library(jsonlite)
library(ggplot2)
library(gridExtra)

# Set paths to source code, weights and more:
# Remark: Works in Rstudio, otherwise you can set the folders manually.
project_folder <- dirname(dirname(dirname(rstudioapi::getSourceEditorContext()$path)))
code_folder <- gsub("/","//",paste(project_folder,"/code/r_code",sep=""))
contract_grid_folder <- gsub("/","//",paste(project_folder,"/code/neural_networks/data",sep=""))
weights_folder <- gsub("/","//",paste(project_folder,"/code/neural_networks/data/neural_network_weights/heston",sep=""))
example_contracts_folder <- gsub("/","//",paste(project_folder,"/get_started",sep=""))
source(gsub("/","//",paste(code_folder,"/NeuralNetworkPricing.r",sep="")))

# Load model:
model <- LoadModel(contract_grid_folder,weights_folder,"heston")

# Plot contract grid:
data_tmp <- data.frame(model$T,model$k)
ggplot(data_tmp, aes(x=model.T, y=model.k)) + geom_point(color="darkblue") + 
	   ggtitle("Contract grid") + xlab("Expiries") + ylab("Log-moneyness") +
	   theme(plot.title = element_text(hjust = 0.5))

# Set parameters:
# Remark: The order is (kappa,vbar,eta,rho,v0)
par <- c(4,0.3^2,2,-0.65,0.15^2)

# Evaluate network in grid:
iv <- GetImpliedVolatility(model,par,model$k,model$T)

# Plot a few expiries:
uniqT <- unique(model$T)
idx_plot <- c(1,10,20,50)
data <- data.frame(model$k,model$T,iv)
p <- list()
for (i in 1:length(idx_plot)){
  p[[i]] <- ggplot(data[uniqT[idx_plot[i]] == data$model.T,], aes(x=model.k, y=iv)) +
            geom_line(color="darkblue") + geom_point(color="darkblue") + 
            ggtitle(paste("T = ",uniqT[idx_plot[i]],sep="")) +
            xlab("Log-moneyness") + ylab("Implied volatility") +
            theme(plot.title = element_text(hjust = 0.5))
}
grid.arrange(p[[1]],p[[2]],p[[3]],p[[4]],ncol=2)

# Load example contracts:
tmp <- read.delim(paste(example_contracts_folder,"//example_contracts.txt",sep=""), header = FALSE)
k_orig <- tmp$V1
T_orig <- tmp$V2

# Remove observed contracts outside the neural network domain:
idxKeep <- AreContractsInDomain(model,k_orig,T_orig)
k_obs <- k_orig[idxKeep]
T_obs <- T_orig[idxKeep]

# Store in a data.frame:
data1 <- data.frame(k_orig,T_orig,rep(FALSE,length(k_orig),1))
names(data1) <- c("k","T","Filtered")
data2 <- data.frame(k_obs,T_obs,rep(TRUE,length(k_obs),1))
names(data2) <- c("k","T","Filtered")
data <- rbind(data1,data2)

# Plot observed contracts (before and after filtering):
ggplot(data, aes(x = T, y = k)) + geom_point(aes(color = Filtered)) + 
       ggtitle("Contracts") + xlab("Expiries") + ylab("Log-moneyness") +
       theme(plot.title = element_text(hjust = 0.5)) + ylim(c(-2,0.5))

# The number of contracts is:
length(k_obs)

# The number of (unique) expiries is:
uniqT <- unique(T_obs)
length(uniqT)

# Generate synthetic prices:
parTrue <- c(4,0.3^2,2,-0.65,0.15^2)
iv_obs <- GetImpliedVolatility(model,parTrue,k_obs,T_obs)
iv_obs

# Calibrate the model to see if we can recover the parameters:
err_fun <- function(par){sum((GetImpliedVolatility(model,par,k_obs,T_obs)-iv_obs)^2)}
par0 <- c(2,0.2^2,1,-0.4,0.25^2)
calib <- optim(par0,err_fun,method="L-BFGS-B",
               lower=model$lb,
               upper=model$ub)

# Compare calibrated parameters to the true ones:
calib$par
parTrue

# Plot fit (blue = observed, red = fit):
idx_plot <- c(1,7,10,15)
T_plot <- uniqT[idx_plot]
iv_fit <- GetImpliedVolatility(model,calib$par,k_obs,T_obs)
data1 <- setNames(data.frame(k_obs,T_obs,iv_obs,rep(TRUE,length(k_obs),1)),c("k_obs","T_obs","iv","obs"))
data2 <- setNames(data.frame(k_obs,T_obs,iv_fit,rep(FALSE,length(k_obs),1)),c("k_obs","T_obs","iv","obs"))
data <- rbind(data1,data2)
p <- list()
for (i in 1:length(idx_plot)){
  dataTmp <- data[T_plot[i] == data$T_obs,]
  p[[i]] <- ggplot(dataTmp, aes(x=k_obs, y=iv)) + geom_point(data=dataTmp[dataTmp$obs==TRUE,],color="blue") + 
            geom_line(data=dataTmp[dataTmp$obs==FALSE,],color="red")  + ggtitle(paste("T = ",round(uniqT[idx_plot[i]],4),sep="")) +
            xlab("Log-moneyness") + ylab("Implied volatility") + theme(plot.title = element_text(hjust = 0.5)) +
            scale_colour_manual("",values=c("blue","red"))
}
grid.arrange(p[[1]],p[[2]],p[[3]],p[[4]],ncol=2)

