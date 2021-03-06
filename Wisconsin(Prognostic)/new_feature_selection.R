# This file implemented the new feature selection algorithms

# Setup ----
rm(list=ls())
# !!CHANGE WORKING DIR TO MATCH YOUR CASE!!
setwd("/Users/shuqishang/Documents/courses/Statistical_Learning/proj/data/")

require(RWeka) # C4.5
require(e1071) # SVM, Naive Bayes
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

# New Feature Selection for Random Forests
for (i in 1:5) {
  set.seed(i)
  sample_num = sample(n_samples, n_samples / 10)
  cancer_samp = cancer_without_id[sample_num,]
  forest_model = randomForest(V2 ~ ., data = cancer_samp, ntree = 300)
  idxs = order(forest_model$importance, decreasing = TRUE)
  
  feature_cnt = 0
  for (idx in idxs) {
    selected_features = union(selected_features, row.names(forest_model$importance)[idx])
    feature_cnt = feature_cnt + 1
    if (feature_cnt == 3) {
      break
    }
  }
}
# After feature Selection, we have the following newly selected features:
# V27 V24 V26 V31 V34 V35 V10 V16 V22 V29 V30 V18 V7

sbc_rf_acc = rep(0, 80)
sbc_acc = rep(0, 80)
nb_acc = rep(0, 80)
tree_acc = rep(0, 80)
forest_acc = rep(0, 80)

# Run the original 3 algorithms with the new feature selection algorithm
for (i in 11:90) {
  for (j in 1:15) {
    cat('i =', i, 'j =', j, '\n')
    set.seed(j)
    train = sample(n_samples, n_samples / 100 * i)
    test = -train
    X_train = cancer_without_id[train,]
    X_test = cancer_without_id[test,]
    
    sbc_rf_model = naiveBayes(V2 ~ V27 + V24 + V26 + V31 + V34 + V35 + V10 + V16 + V22 + V29 + V30 + V18 + V7,
                              data = X_train)
    sbc_rf_pred = predict(sbc_rf_model, X_test, type = "class")
    sbc_rf_acc[i-10] = sbc_rf_acc[i-10] + sum(sbc_rf_pred == X_test$V2) / length(sbc_rf_pred)

    sbc_model = naiveBayes(V2 ~ V35 + V34 + V30 + V10 + V18, data = X_train)
    sbc_pred = predict(sbc_model, X_test, type = "class")
    sbc_acc[i-10] = sbc_acc[i-10] + sum(sbc_pred == X_test$V2) / length(sbc_pred)

    nb_model = naiveBayes(V2 ~ ., data = X_train)
    nb_pred = predict(nb_model, X_test, type = "class")
    nb_acc[i-10] = nb_acc[i-10] + sum(nb_pred == X_test$V2) / length(nb_pred)

    tree_model = J48(V2 ~ ., data = X_train)
    tree_pred = predict(tree_model, X_test, type = "class")
    tree_acc[i-10] = tree_acc[i-10] + sum(tree_pred == X_test$V2) / length(tree_pred)

    forest_model = randomForest(V2 ~ ., data = X_train, ntree = 300)
    forest_pred = predict(forest_model, X_test)
    forest_acc[i-10] = forest_acc[i-10] + sum(forest_pred == X_test$V2) / length(forest_pred)
  }
  sbc_rf_acc[i-10] = sbc_rf_acc[i-10] / 15
  sbc_acc[i-10] = sbc_acc[i-10] / 15
  nb_acc[i-10] = nb_acc[i-10] / 15
  tree_acc[i-10] = tree_acc[i-10] / 15
  forest_acc[i-10] = forest_acc[i-10] / 15
}

# Plot the original 3 algorithms with the new feature selection algorithm
plot(sbc_acc~c(11:90), col = 'red', cex = 0.3, main = "Breast Cancer Wisconsin (Prognostic)",
     xlab = 'Training Data (%)', ylab = 'Accuracy', ylim = c(0.59, 0.81))
points(c(11:90), nb_acc, col = 'blue', cex = 0.3)
points(c(11:90), tree_acc, col = 'darkgreen', cex = 0.3)
points(c(11:90), sbc_rf_acc, col = 'purple', cex = 0.3)
points(c(11:90), forest_acc, col = 'orange', cex = 0.3)
lines(sbc_acc~c(11:90), col = 'red')
lines(nb_acc~c(11:90), col = 'blue')
lines(tree_acc~c(11:90), col = 'darkgreen')
lines(sbc_rf_acc~c(11:90), col = 'purple')
lines(forest_acc~c(11:90), col = 'orange')
legend("topleft", ncol = 2,
       c("SBC", "NBC", "C4.5", "SBC_RF", "Random Forest"),
       col = c("red", "blue", "darkgreen", "purple", "orange"),
       text.col = c("red", "blue", "darkgreen", "purple", "orange"),
       lty = c(1, 1, 1, 1, 1))




