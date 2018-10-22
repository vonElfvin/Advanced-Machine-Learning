#### Lab 1 - Graphical Models
### Libraries
library("bnlearn")
library("gRain")

### Setup
data("asia")
data.asia = asia

### Functions
get_restarts = function(i) {
  return(10 ^ ((i - 1) %% 3))
}

## Assignment 1
network.scores = c("loglik", "bde", "bic")
range = 1:9

# Learn structure with different configurations (network scores and number of restarts)
BN.structures = list()
for (i in range) {
  network.score = network.scores[ceiling(i/3)]
  
  # construct two BNs with the same configuration but different seeds.
  set.seed(i)
  BN1 = hc(asia, restart = get_restarts(i), score = network.score)
  set.seed(i+10)
  BN2 = hc(asia, restart = get_restarts(i), score = network.score)
  CP1 = cpdag(BN1)
  CP2 = cpdag(BN2)
  
  print(paste("score:", network.score, "| restarts:", get_restarts(i)  ,"| equal:", all.equal(CP1, CP2)))
  
  BN.structures[[i]] = BN1 # store one for visualization and comparision between configurations
}

# Compare resulting structures to see if they are equivalent
BN.scores = numeric()
par(mfrow=c(3,3))
for (i in range) {
  BN.scores[i] = score(BN.structures[[i]], data.asia, type = "bic")
  plot(BN.structures[[i]], main = paste("Scoring:", network.scores[ceiling(i/3)], ", Restarts:", get_restarts(i)))
}
par(mfrow=c(1,1))

# Plot scores to see difference among the structures
hist(BN.scores)

## Assignment 2
# Setup train/test data
n = dim(data.asia)[1]
train = data.asia[1:4000,]
test = subset(data.asia[4001:n, ], select = -c(S))
y.test = data.asia[4001:n, ]$S
nodes = colnames(test)

get_confusion_matrix = function(y, y.pred) {
  return(table(y, y.pred, dnn=c("TRUE", "PRED")))
}

get_missclassifcation_rate = function(y, y.pred) {
  n = length(y)
  k = sum(abs(ifelse(y == "yes", 1, 0) - ifelse(y.pred == "yes", 1, 0)))
  return(k/n)
}

# Function to perform and get predictions
get_predictions = function(..BN, test, y.test, nodes) {
  # Fit parameters
  BN.fit = bn.fit(..BN, train)
  BN.grain = compile(as.grain(BN.fit))
  
  # Setup predictions
  n.test = length(y.test)
  predictions = numeric()
  
  # Make predictions
  for (i in 1:n.test) {
    
    # Extract state
    z = NULL
    for (j in nodes) {
      if (test[i,j] == "yes") z = c(z, "yes")
      else z = c(z, "no")
    }
    
    # Set evidence
    evidence = setEvidence(BN.grain, nodes = nodes, states = z)
    
    # Perform query
    query = querygrain(evidence, c("S"))
    
    # Make prediction
    if (query$S["yes"] >= 0.5) predictions[i] = "yes"
    else predictions[i] = "no"
  }
  
  return(predictions)
}

# Select the BN with best BIC score
BN.best = BN.structures[[which.max(BN.scores)]]
y.pred1 = get_predictions(BN.best, test, y.test, nodes)
CM1 = get_confusion_matrix(y.test, y.pred1)
#      PRED
# TRUE   no yes
#   no  358 147
#   yes 120 375
MCR1 = get_missclassifcation_rate(y.test, y.pred1)
# 0.267

# Compare to true BN
BN.true = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")
y.pred2 = get_predictions(BN.true, test, y.test, nodes)
CM2 = get_confusion_matrix(y.test, y.pred2)
#     PRED
# TRUE  no yes
#   no 358 147
#  yes 120 375
MCR2 =  get_missclassifcation_rate(y.test, y.pred2)
# 0.267

# Plot the two BNs for comparison
par(mfrow=c(1,2))
plot(BN.best, main="Selected")
plot(BN.true, main="True")

# Comment:
# Same result by markov blanket criteria where both the selected BN 
# and the true BN have edges between S and B+L, making S independent from
# the other nodes given B and L, thus yielding the same results in predictions

## Assignment 3
# Selected BN
# mb(BN.best)
# "L" "B"
y.pred1 = get_predictions(BN.best, test, y.test, mb(BN.best, c("S")))
CM1 = get_confusion_matrix(y.test, y.pred1)
#      PRED
# TRUE   no yes
#   no  358 147
#   yes 120 375
MCR1 = get_missclassifcation_rate(y.test, y.pred1)
# 0.267

# True BN
# mb(BN.true)
# "L" "B"
y.pred2 = get_predictions(BN.true, test, y.test, mb(BN.true, c("S")))
CM2 = get_confusion_matrix(y.test, y.pred2)
#     PRED
# TRUE  no yes
#   no 358 147
#  yes 120 375
MCR2 =  get_missclassifcation_rate(y.test, y.pred2)
# 0.267

# Comment: As stated in previous example, the networks yield the same result, 
# given the markov blanket values the resulting predictions are the same
# This is because the node S is independent all other nodes under the MB criteria.

## Assignment 4
BN.naive = empty.graph(c("A", "S", "T", "L", "B", "E", "X", "D"))
arc.set = matrix(c(
  "S", "A",
  "S", "T",
  "S", "L",
  "S", "B",
  "S", "E",
  "S", "X",
  "S", "D"
  ), ncol = 2, byrow = TRUE, dimnames = list(NULL, c("from", "to")))

arcs(BN.naive) = arc.set
y.pred3 = get_predictions(BN.naive, test, y.test, nodes)
CM3 = get_confusion_matrix(y.test, y.pred3)
#     PRED
# TRUE  no yes
#   no 389 116
#  yes 180 315
MCR3 =  get_missclassifcation_rate(y.test, y.pred3)
# 0.296
plot(BN.naive)

# Comment: Slightly worse performance than previous examples, given that the information gained 
# in this structure is in reality independent makes the model either overfit or just straight up incorrect

