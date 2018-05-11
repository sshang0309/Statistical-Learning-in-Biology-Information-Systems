# This file implemented the algorithms in the original paper

# Setup ----
rm(list=ls())
# !!CHANGE WORKING DIR TO MATCH YOUR CASE!!
setwd("/Users/shuqishang/Documents/courses/Statistical_Learning/proj/data/") 

require(RWeka) # C4.5
require(e1071) # Naive Bayes
require(class) # KNN
require(randomForest) # Random Forest
require(ipred) # Bagging
require(MASS) # lda, qda

# Different datasets should have different preprocessing for these several lines
# In the latter part, V2 should be changed to other dependent variables in other datasets
# The title of the graphs should also be changed for different datasets
# Selected features should be changed for different datasets
cancer = read.table('wpbc.data.txt', sep = ',', header = F)
for (row in 1:nrow(cancer)) {
  if ('?' %in% cancer[row, 'V35']) {
    cancer = cancer[-row,]
  }
}
cancer_without_id = cancer[, !(names(cancer) %in% c('V1', 'V3'))]

n_samples = nrow(cancer_without_id)
selected_features = c()

# Feature Selection
for (i in 1:5) {
  set.seed(i)
  sample_num = sample(n_samples, n_samples / 10)
  cancer_samp = cancer_without_id[sample_num,]
  tree_model = J48(V2 ~ ., data = cancer_samp)
  summ_tree = tree_model$classifier$prefix()
  
  s_l = strsplit(summ_tree, '\\[')
  feature_cnt = 0
  for (s in s_l[[1]]) {
    
    if (grepl(':', s)) {
      feature_s = strsplit(s, ':')
      feature = lapply(feature_s, head, n = 1)[[1]]
      selected_features = union(selected_features, feature)
      print(feature)
      feature_cnt = feature_cnt + 1
    }
    
    if (feature_cnt == 3) {
      break
    }
  }
}
# After feature Selection, we have the following selected features:
# 'spread1', 'D2', 'spread2', 'MDVP.Flo.Hz.', 'MDVP.Fo.Hz.', 'NHR'

sbc_acc = rep(0, 80)
nb_acc = rep(0, 80)
tree_acc = rep(0, 80)

# Run the three algorithms
for (i in 11:90) {
  for (j in 1:15) {
    cat('i =', i, 'j =', j, '\n')
    set.seed(j)
    train = sample(n_samples, n_samples / 100 * i)
    test = -train
    X_train = cancer_without_id[train,]
    X_test = cancer_without_id[test,]
    
    sbc_model = naiveBayes(V2 ~ V35 + V34 + V30 + V10 + V18, data = X_train)
    sbc_pred = predict(sbc_model, X_test, type = "class")
    sbc_acc[i-10] = sbc_acc[i-10] + sum(sbc_pred == X_test$V2) / length(sbc_pred)

    nb_model = naiveBayes(V2 ~ ., data = X_train)
    nb_pred = predict(nb_model, X_test, type = "class")
    nb_acc[i-10] = nb_acc[i-10] + sum(nb_pred == X_test$V2) / length(nb_pred)
    
    tree_model = J48(V2 ~ ., data = X_train)
    tree_pred = predict(tree_model, X_test, type = "class")
    tree_acc[i-10] = tree_acc[i-10] + sum(tree_pred == X_test$V2) / length(tree_pred)
  }
  sbc_acc[i-10] = sbc_acc[i-10] / 15
  nb_acc[i-10] = nb_acc[i-10] / 15
  tree_acc[i-10] = tree_acc[i-10] / 15
}

# Plot the gragh of original algorithms
plot(sbc_acc~c(11:90), col = 'red', cex = 0.3, main = "Breast Cancer Wisconsin (Prognostic)",
     xlab = 'Training Data (%)', ylab = 'Accuracy', ylim = c(0.59, 0.81))
points(c(11:90), nb_acc, col = 'blue', cex = 0.3)
points(c(11:90), tree_acc, col = 'darkgreen', cex = 0.3)
lines(sbc_acc~c(11:90), col = 'red')
lines(nb_acc~c(11:90), col = 'blue')
lines(tree_acc~c(11:90), col = 'darkgreen')
legend("topleft",
       c("SBC", "NBC", "C4.5"),
       col = c("red", "blue", "darkgreen"),
       text.col = c("red", "blue", "darkgreen"),
       lty = c(1, 1, 1))




