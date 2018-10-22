#### Libraries
library("HMM")
library("entropy")

### Setup
set.seed(123)
par(mfrow=c(1,1))

#### Functions

get_mcr = function(pred) {
  return(1 - sum(pred) / length(pred))
}

#### Implementation

### Task 1
states = c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
symbols = c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
startProbs = rep(0.1, 10)
transProbs = t(matrix(
  c(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5,
    0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5
    ),
  nrow = 10, ncol = 10
))
emissionProbs = t(matrix(
  c(0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0.2, 0.2,
    0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0.2,
    0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0,
    0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0,
    0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0,
    0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0,
    0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0,
    0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2,
    0.2, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2,
    0.2, 0.2, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2
  ),
  nrow = 10, ncol = 10
))
hmm = initHMM(states, symbols, startProbs, transProbs, emissionProbs)
best = c(0, 0, 0)
for (i in 1:10) {
  
  ### Task 2
  T = 100
  sim = simHMM(hmm, T)
  
  ### Task 3
  # Calculate alpha & beta from observations
  obs = sim$observation
  alpha = exp(forward(hmm, obs))
  beta = exp(backward(hmm, obs))
  
  # Setup calculation of filtering & smoothing
  filtering = matrix(0, 10, T)
  smoothing = matrix(0, 10, T)
  
  # Calculate filtering & smoothing
  alphaSum = apply(alpha, 2, sum)
  alphaBetaSum = apply(alpha * beta, 2, sum)
  for (t in 1:T) {
    filtering[,t] = alpha[,t] / alphaSum[t]
    smoothing[,t] = (alpha[,t] * beta[,t]) / alphaBetaSum[t]
  }
  
  # Predict most probable path
  y.path = viterbi(hmm, obs)
  
  ### Task 4
  # Extract predicted states of filtering & smoothing
  y.filtering = c()
  y.smoothing = c()
  for (t in 1:T) {
    y.filtering[t] = states[which.max(filtering[,t])]
    y.smoothing[t] = states[which.max(smoothing[,t])]
  }
  
  # Determine correct prediction states of filtering, smoothing and viterbi in comparison to the true path
  y = sim$states
  pred.filtering = y.filtering == y
  pred.smoothing = y.smoothing == y
  pred.path = y.path == y
  
  # Provide missclassification rates of the different methods of prediction
  mcr.filtering = get_mcr(pred.filtering)
  # 0.47
  mcr.smoothing = get_mcr(pred.smoothing)
  # 0.36
  mcr.path = get_mcr(pred.path)
  # 0.64 
  
  ### Task 5
  which.best = which.min(c(mcr.filtering, mcr.smoothing, mcr.path))
  best[which.best] = best[which.best] + 1
  
  # Results
  # Filtering: 0 
  # Smoothing: 10 
  # Viterbi: 0
  # Smoothing outperforms the other two each simulation.
  # Smoothin outperforms filtering since it takes the future t+1 into account (via beta)
  # Smoothing outperforms most probable path since it focuses on making the best prediction on current t
  # compared to the entire path of most probable path
}

### Task 6
entropy = c()
for (t in 1:T) {
  entropy[t] = entropy.empirical(filtering[,t])
}
plot(entropy, type="l")
# Comment:
# Entropi = hur rörig din information är, hur mycket osäkerhet i din fördelning, likformig = max entropi, 
# Går nedåt -> Vet bättre, om det går uppåt -> Vet mindre

### Task 7
p.100 = filtering[, 100] # same as smoothing, since its based on the last observation (beta = 1)
p.101 = t(transProbs) %*% p.100 # transformation matrix
