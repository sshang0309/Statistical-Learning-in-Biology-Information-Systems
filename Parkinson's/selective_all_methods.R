# This file implemented all the classifiers with feature selection
# This file will use the variables from the other 3 files to plot the graph
# Please run the other 3 files first

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
# In the latter part, status should be changed to other dependent variables in other datasets
# The title of the graphs should also be changed for different datasets
# Selected features should be changed for different datasets
cancer = read.table('parkinsons.data.txt', sep = ',', header = T)
cancer$status = factor(cancer$status, levels = c(0, 1), labels = c('class0', 'class1'))
cancer_without_id = cancer[, !(names(cancer) %in% c('name'))]
independ_var = cancer[, !(names(cancer) %in% c('name', 'status'))]
select_var = cancer[, (names(cancer) %in% c('spread1', 'D2', 'spread2', 'MDVP.Flo.Hz.', 'MDVP.Fo.Hz.', 'NHR'))]

n_samples = nrow(cancer_without_id)

sknn_3_acc = rep(0, 80)
sknn_5_acc = rep(0, 80)
sknn_10_acc = rep(0, 80)

ssvm_acc = rep(0, 80)
slda_acc = rep(0, 80)
sqda_acc = rep(0, 70)

# Run all the classifiers with feature selection
for (i in 11:90) {
  for (j in 1:15) {
    cat('i =', i, 'j =', j, '\n')
    set.seed(j)
    train = sample(n_samples, n_samples / 100 * i)
    test = -train
    X_train = cancer_without_id[train,]
    X_test = cancer_without_id[test,]
    sknn_X_train = select_var[train,]
    sknn_X_test = select_var[test,]
    
    sknn_3_pred = knn(train = sknn_X_train, test = sknn_X_test, cl = X_train$status, k = 3)
    sknn_3_acc[i-10] = sknn_3_acc[i-10] + sum(sknn_3_pred == X_test$status) / length(sknn_3_pred)

    sknn_5_pred = knn(train = sknn_X_train, test = sknn_X_test, cl = X_train$status, k = 5)
    sknn_5_acc[i-10] = sknn_5_acc[i-10] + sum(sknn_5_pred == X_test$status) / length(sknn_5_pred)

    sknn_10_pred = knn(train = sknn_X_train, test = sknn_X_test, cl = X_train$status, k = 10)
    sknn_10_acc[i-10] = sknn_10_acc[i-10] + sum(sknn_10_pred == X_test$status) / length(sknn_10_pred)
    
    ssvm_model = svm(status ~ spread1 + D2 + spread2 + MDVP.Flo.Hz. + MDVP.Fo.Hz. + NHR, data = X_train)
    ssvm_pred = predict(ssvm_model, X_test)
    ssvm_acc[i-10] = ssvm_acc[i-10] + sum(ssvm_pred == X_test$status) / length(ssvm_pred)
    
    # slda_model = lda(status ~ spread1 + D2 + spread2 + MDVP.Flo.Hz. + MDVP.Fo.Hz. + NHR, data = X_train)
    # slda_pred = predict(slda_model, X_test)$class
    # slda_acc[i-10] = slda_acc[i-10] + sum(slda_pred == X_test$status) / length(slda_pred)
    # 
    # if (i > 20) { # if i < 20, some group is too small for 'qda'
    #   sqda_model = qda(status ~ spread1 + D2 + spread2 + MDVP.Flo.Hz. + MDVP.Fo.Hz. + NHR, data = X_train)
    #   sqda_pred = predict(sqda_model, X_test)$class
    #   sqda_acc[i-20] = sqda_acc[i-20] + sum(sqda_pred == X_test$status) / length(sqda_pred)
    # }
  }
  sknn_3_acc[i-10] = sknn_3_acc[i-10] / 15
  sknn_5_acc[i-10] = sknn_5_acc[i-10] / 15
  sknn_10_acc[i-10] = sknn_10_acc[i-10] / 15
  
  ssvm_acc[i-10] = ssvm_acc[i-10] / 15
  # slda_acc[i-10] = slda_acc[i-10] / 15
  # sqda_acc[i-20] = sqda_acc[i-20] / 15
}

# KNN, SKNN
plot(knn_3_acc~c(11:90), col = 'red', cex = 0.3, main = "Parkinson's Disease",
     xlab = 'Training Data (%)', ylab = 'Accuracy', ylim = c(0.75, 0.87))
points(c(11:90), knn_5_acc, col = 'blue', cex = 0.3)
points(c(11:90), knn_10_acc, col = 'darkgreen', cex = 0.3)
points(c(11:90), sknn_3_acc, col = 'darkred', cex = 0.3)
points(c(11:90), sknn_5_acc, col = 'darkblue', cex = 0.3)
points(c(11:90), sknn_10_acc, col = 'black', cex = 0.3)
lines(knn_3_acc~c(11:90), col = 'red')
lines(knn_5_acc~c(11:90), col = 'blue')
lines(knn_10_acc~c(11:90), col = 'darkgreen')
lines(sknn_3_acc~c(11:90), col = 'darkred')
lines(sknn_5_acc~c(11:90), col = 'darkblue')
lines(sknn_10_acc~c(11:90), col = 'black')
legend("bottomright", ncol = 2,
       c("KNN(k=3)", "KNN(k=5)", "KNN(k=10)", "SKNN(k=3)", "SKNN(k=5)", "SKNN(k=10)"),
       col = c("red", "blue", "darkgreen", "darkred", "darkblue", "black"),
       text.col = c("red", "blue", "darkgreen", "darkred", "darkblue", "black"),
       lty = c(1, 1, 1, 1, 1, 1))

# SVM, LDA, QDA, SSVM, SLDA, SQDA
plot(svm_acc~c(11:90), col = 'red', cex = 0.3, main = "Parkinson's Disease",
     xlab = 'Training Data (%)', ylab = 'Accuracy', ylim = c(0.79, 0.89))
# points(c(11:90), lda_acc, col = 'blue', cex = 0.3)
# points(c(21:90), qda_acc, col = 'darkgreen', cex = 0.3)
points(c(11:90), ssvm_acc, col = 'darkred', cex = 0.3)
# points(c(11:90), slda_acc, col = 'darkblue', cex = 0.3)
# points(c(21:90), sqda_acc, col = 'black', cex = 0.3)
lines(svm_acc~c(11:90), col = 'red')
# lines(lda_acc~c(11:90), col = 'blue')
# lines(qda_acc~c(21:90), col = 'darkgreen')
lines(ssvm_acc~c(11:90), col = 'darkred')
# lines(slda_acc~c(11:90), col = 'darkblue')
# lines(sqda_acc~c(21:90), col = 'black')
legend("bottomright",
       c("SVM", "SSVM"),
       col = c("red", "darkred"),
       text.col = c("red", "darkred"),
       lty = c(1, 1))

# Best few methods comparison
plot(ssvm_acc~c(11:90), col = 'red', cex = 0.3, main = "Parkinson's Disease",
     xlab = 'Training Data (%)', ylab = 'Accuracy', ylim = c(0.74, 0.96))
points(c(11:90), forest_acc, col = 'blue', cex = 0.3)
points(c(11:90), sbc_acc, col = 'darkgreen', cex = 0.3)
points(c(11:90), knn_3_acc, col = 'purple', cex = 0.3)
lines(ssvm_acc~c(11:90), col = 'red')
lines(forest_acc~c(11:90), col = 'blue')
lines(sbc_acc~c(11:90), col = 'darkgreen')
lines(knn_3_acc~c(11:90), col = 'purple')
legend("topleft",
       c("SSVM", "Random Forest", "SBC", "KNN(k=3)"),
       col = c("red", "blue", "darkgreen", "purple"),
       text.col = c("red", "blue", "darkgreen", "purple"),
       lty = c(1, 1, 1, 1))




