###############
#### Lab 4 ####
###############

### Libraries
library("kernlab")
library(AtmRay)

### Functions

confusion_matrix = function(y, y.pred) {
  return(table(y, y.pred, dnn=c("TRUE", "PRED")))
}

accuracy = function(y, y.pred) {
  return(1-mean(abs(y-y.pred)))
}

Visualize = function(X, y, XStar, res) {
  mean = res$mean
  sd = sqrt(res$variance)
  plot(XStar, mean, type="l", ylim=c(-3, 4),
       xlab="x", ylab="y", main="Posterior Gaussian Process")
  lines(XStar, mean - 1.96 * sd, type="l", col="blue")
  lines(XStar, mean + 1.96 * sd, type="l", col="blue")
  points(X, y, pch=19)
  legend("topright", 
         legend=c("posterior mean", "probability band", "observations"), 
         col=c("black", "blue", "black"), lty=c(1,1,0), pch=c(-1,-1,19))
}

############
## Task 1 ##
############

SquaredExpKernel = function(x1, x2, sigmaF, l) {
  n1 = length(x1)
  n2 = length(x2)
  K = matrix(NA, n1, n2)
  for(i in 1:n2) {
    for(j in 1:n1) {
      K[j,i] = sigmaF^2 * exp(-0.5 * ( (x2[i] - x1[j])/l)^2 )
    }
  }
  return(K)
}

posteriorGP = function(X, y, XStar, sigmaNoise, K, ...) {
  n = length(X)
  I = diag(n)
  L = t(chol(K(X,X, ...) + sigmaNoise^2 * I))
  KStar = K(X, XStar, ...)
  
  # Predictive Mean
  alpha = solve(t(L), solve(L, y))
  fStarBar = t(KStar) %*% alpha
  
  # Predictive Variance
  v = solve(L, KStar)
  V = K(XStar, XStar, ...) - t(v) %*% v
  
  return(data.frame(mean = fStarBar, variance = diag(V)))
}
# (1)
XStar = seq(-1, 1, 0.05)

# (2)
X2 = c(0.4)
y2 = c(0.719)
res = posteriorGP(X2, y2, XStar, sigmaNoise = 0.1, K = SquaredExpKernel, sigmaF = 1, l = 0.3)
Visualize(X2, y2, XStar, res)

# (3)
X3 = c(0.4, -0.6)
y3 = c(0.719, -0.044)
res = posteriorGP(X3, y3, XStar, sigmaNoise = 0.1, K = SquaredExpKernel, sigmaF = 1, l = 0.3)
Visualize(X3, y3, XStar, res)

# (4)
X4 = c(-1.0, -0.6, -0.2, 0.4, 0.8)
y4 = c(0.768, -0.044, -0.940, 0.719, -0.644)
res = posteriorGP(X4, y4, XStar, sigmaNoise = 0.1, K = SquaredExpKernel, sigmaF = 1, l = 0.3)
Visualize(X4, y4, XStar, res)

# (5)
res = posteriorGP(X4, y4, XStar, sigmaNoise = 0.1, K = SquaredExpKernel, sigmaF = 1, l = 1)
Visualize(X4, y4, XStar, res)
# Comment: Higher l -> More smoothness, fit does look worse

#############
### Task 2 ##
#############
# Download data
data.temp = read.csv("https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/Code/TempTullinge.csv", header=TRUE, sep=";")
time = seq(1, 2190, 1)
day = rep(seq(1, 365, 1), 6)
data.temp["time"] = time
data.temp["day"] = day

# Subsample data
n = dim(data.temp)[1]
data.temp = data.temp[seq(1, n, 5),]


# (1)
# Squared Exponential Kernel function 
SEkernel = function(sigmaF = 1, ell = 1)
{
  SquaredExpKernel <- function(x, y = NULL) {
    n1 = length(x)
    n2 = length(y)
    K = matrix(NA, n1, n2)
    for(i in 1:n2) {
      for(j in 1:n1) {
        K[j,i] = sigmaF^2 * exp(-0.5 * ( (x[i] - y[j]) / ell)^2 )
      }
    }
    return(K)
  }
  class(SquaredExpKernel) <- "kernel"
  return(SquaredExpKernel)
}
k = SEkernel()

## Evaluate kernel
# x = 1, x' = 2
k(1,2)
# 0.6065307

# X = (1, 3, 4), X' = (2, 3, 4)
kernelMatrix(k, c(1,3,4), c(2,3,4))
# 0.6065307 0.6065307 0.1353353
# 0.1353353 1.0000000 0.6065307
# 0.0111090 0.6065307 1.0000000

# (2)
# Fit lm to estimate residual variance
lm = lm(temp ~ time + I(time^2), data = data.temp)
sigmaN = sd(lm$residuals) # Estimate sigmaN using the standard deviation of the residuals

# Estimate Gaussian Process
k = SEkernel(sigmaF = 20, ell = 0.2)
GPTime = gausspr(temp ~ time, data = data.temp, kernel = k, var = sigmaN^2)
# GPTime = gausspr(x = data.temp$time, y = data.temp$temp, kernel = SEkernel, 
#                  kpar = list(sigmaF = 20, ell = 0.2), var = sigmaN^2)
postMeanTime = predict(GPTime, data.temp)

# Plot data and posterior mean
plot(data.temp$time, data.temp$temp, ylim=c(-35, 35), pch=19,
     main="Gaussian Process", xlab="Time", ylab="Temp")
lines(data.temp$time, postMeanTime, col="red", lwd=2)
legend("bottomright", legend=c("predictive mean", "observations"),
       col=c("red", "black"), lty=c(1,0), pch=c(-1,19), lwd=c(2,0))

# (3)
X = scale(data.temp$time)
XStar = scale(X)
n = length(X)

KStarStar = kernelMatrix(kernel = k, x = XStar, y = XStar)
KStar = kernelMatrix(kernel = k, x = X, y = XStar)
K = kernelMatrix(kernel = k, x = X, y = XStar)
V = diag(KStarStar - t(KStar) %*% solve(K + sigmaN^2 * diag(n), KStar))
# V = posteriorGP(X, data.temp$temp, XStar, sigmaN, k)$variance # same result

# Superimpose the predictive interval
lines(data.temp$time, postMeanTime - 1.96 * sqrt(V), col="blue", lwd=2)
lines(data.temp$time, postMeanTime + 1.96 * sqrt(V), col="blue", lwd=2)
legend("bottomright", legend=c("predictive mean", "predictive interval", "observations"), 
       col=c("red", "blue", "black"), lty=c(1,1,0), pch=c(-1,-1,19), lwd=c(2,2,0))
# (4)
k = SEkernel(sigmaF = 20, ell = 0.2)
GPDay = gausspr(temp ~ day, data = data.temp, kernel = k, var = sigmaN^2)
postMeanDay = predict(GPDay, data.temp)
plot(data.temp$time, data.temp$temp, ylim=c(-35, 20), pch=19,
     main="Gaussian Process", xlab="Time", ylab="Temp")
lines(data.temp$time, postMeanDay, lwd=2, col="blue")
lines(data.temp$time, postMeanTime, lwd=2, col="red")
legend("bottomright", legend=c("predictive mean (day)", "predictive mean (time)", "observations"),
       col=c("blue", "red", "black"), lty=c(1,1,0), pch=c(-1,-1,19), lwd=c(2,2,0))
# Comment: Using day is less sensitive to years with more elastic temperature. Similar predictions each year.

# (5)


# Period Kernel function
Pkernel = function(sigmaF = 1, l1 = 1, l2 = 1, d)
{
  PeriodicKernel <- function(x, y) {
    K = sigmaF^2 * 
      exp(-2 * sin(pi * abs(y - x) / d)^2 / l1^2 ) * 
      exp(-0.5 * (y - x)^2 / l2^2 )
    return(K)
  }
  class(PeriodicKernel) <- "kernel"
  return(PeriodicKernel)
}
d = 365 / sd(time)
k = Pkernel(sigmaF = 20, l1 = 1, l2 = 10, d)
GPPeriodic = gausspr(temp ~ time, data = data.temp, kernel = k, var = sigmaN^2)
postMeanPeriodic = predict(GPPeriodic, data.temp)
plot(data.temp$time, data.temp$temp, ylim=c(-35, 20), pch=19,
     main="Gaussian Process", xlab="Time", ylab="Temp")
lines(data.temp$time, postMeanDay, lwd=2, col="blue")
lines(data.temp$time, postMeanTime, lwd=2, col="red")
lines(data.temp$time, postMeanPeriodic, lwd=2, col="green")
legend("bottomright", 
       legend=c("predictive mean (period)", "predictive mean (day)", "predictive mean (time)", "observations"),
       col=c("green", "blue", "red", "black"), lty=c(1,1,1,0), 
       pch=c(-1,-1,-1,19), lwd=c(2,2,2,0))

# Comment: Performs worse than previous kernal/covariates, may be because of hyperparamter tuning.

##############
### Task 3 ###
##############
# Download data
data.fraud <- read.csv("https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/Code/banknoteFraud.csv", header=FALSE, sep=",")
names(data.fraud) <- c("varWave","skewWave","kurtWave","entropyWave","fraud")
data.fraud[,5] <- as.factor(data.fraud[,5])
# (1)
# Extract train & test data
set.seed(111)
SelectTraining <- sample(1:dim(data.fraud)[1], size = 1000, replace = FALSE)
train = data.fraud[SelectTraining,]
test = data.fraud[-SelectTraining,]

# Fit GP with varWave & skewWave as covariates
set.seed(123)
fit = gausspr(fraud ~ varWave + skewWave, data = train)

# Class predictions on train data
pred = predict(fit, train)
probs = predict(fit, train, type="probabilities")
preds = ifelse(probs[,2] > 0.5, 1, 0)
CM = confusion_matrix(train$fraud, preds)
#     PRED
# TRUE   0   1
# 0    512  44
# 1     24 420
accuracy(as.integer(train$fraud)-1, preds)
# 0.932

# Plot Contour
plot_contour = function(fit, train) {
  # Grid setup
  x1 = seq(min(train$varWave), max(train$varWave), length=100)
  x2 = seq(min(train$skewWave), max(train$skewWave), length=100)
  grid = meshgrid(x1, x2)
  grid <- data.frame(cbind(c(grid$x), c(grid$y)))
  names(grid) = c("varWave", "skewWave")
  probsGrid = predict(fit, grid, type="probabilities")
  
  # Plot grid
  contour(x1, x2, matrix(probsGrid[,2], 100, byrow=TRUE), 20,
          xlab = "varWave", ylab = "skewWave", main = "Prob(fraud)")
  points(grid, col=rgb(1-probsGrid[,2], 0, probsGrid[,2], 0.025), pch=15, cex=3)
  points(train$varWave, train$skewWave, 
         col=rgb(2-as.integer(train$fraud), 0, as.integer(train$fraud)-1), pch=19)
}
plot_contour(fit, train)

# (2)
# Class predictions on test data
probs = predict(fit, test, type="probabilities")
preds = ifelse(probs[,2] > 0.5, 1, 0)
CM = confusion_matrix(test$fraud, preds)
#     PRED
# TRUE   0   1
#    0 191  15
#    1   9 157
accuracy(as.integer(test$fraud)-1, preds)
# 0.9354839
# Comment: High accuracy for the test data, more or less the same as the accuracy for the training data.

# (3)
fit = gausspr(fraud ~ ., data = train)
# Class predictions on test data
probs = predict(fit, test, type="probabilities")
preds = ifelse(probs[,2] > 0.5, 1, 0)
CM = confusion_matrix(test$fraud, preds)
#     PRED
# TRUE   0   1
#    0 205   1
#    1   0 166
accuracy(as.integer(test$fraud)-1, preds)
# 0.9973118
# Comment: Extremly high accuracy, performs better than the model fitted using only two covariates,
# Overfitting does not seem to be an issue
