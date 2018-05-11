# This file implemented all the classifiers without feature selection

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
# require(nnet) # neural net
# require(adabag) # boosting

# Different datasets should have different preprocessing for these several lines
# In the latter part, status should be changed to other dependent variables in other datasets
# The title of the graphs should also be changed for different datasets
# Selected features should be changed for different datasets
cancer = read.table('parkinsons.data.txt', sep = ',', header = T)
cancer$status = factor(cancer$status, levels = c(0, 1), labels = c('class0', 'class1'))
cancer_without_id = cancer[, !(names(cancer) %in% c('name'))]
independ_var = cancer[, !(names(cancer) %in% c('name', 'status'))]

n_samples = nrow(cancer_without_id)

nb_acc = rep(0, 80)

knn_3_acc = rep(0, 80)
knn_5_acc = rep(0, 80)
knn_10_acc = rep(0, 80)

lda_acc = rep(0, 80)
qda_acc = rep(0, 70)
svm_acc = rep(0, 80)

tree_acc = rep(0, 80)
bag_acc = rep(0, 80)
forest_acc = rep(0, 80)

# logis_acc = rep(0, 80)
# boost_acc = rep(0, 80)
# nn_acc = rep(0, 80)

# Run all the classifiers
for (i in 11:90) {
  for (j in 1:15) {
    
    cat('i =', i, 'j =', j, '\n')
    set.seed(j)
    
    train = sample(n_samples, n_samples / 100 * i)
    test = -train
    X_train = cancer_without_id[train,]
    X_test = cancer_without_id[test,]
    
    knn_X_train = independ_var[train,]
    knn_X_test = independ_var[test,]
    # nn_X_train = knn_X_train
    # nn_X_test = knn_X_test
    # nn_X_train$status = class.ind(X_train$status)
    # nn_X_test$status = class.ind(X_test$status)
    
    nb_model = naiveBayes(status ~ ., data = X_train)
    nb_pred = predict(nb_model, X_test, type = "class")
    nb_acc[i-10] = nb_acc[i-10] + sum(nb_pred == X_test$status) / length(nb_pred)

    knn_3_pred = knn(train = knn_X_train, test = knn_X_test, cl = X_train$status, k = 3)
    knn_3_acc[i-10] = knn_3_acc[i-10] + sum(knn_3_pred == X_test$status) / length(knn_3_pred)

    knn_5_pred = knn(train = knn_X_train, test = knn_X_test, cl = X_train$status, k = 5)
    knn_5_acc[i-10] = knn_5_acc[i-10] + sum(knn_5_pred == X_test$status) / length(knn_5_pred)

    knn_10_pred = knn(train = knn_X_train, test = knn_X_test, cl = X_train$status, k = 10)
    knn_10_acc[i-10] = knn_10_acc[i-10] + sum(knn_10_pred == X_test$status) / length(knn_10_pred)

    # lda_model = lda(status ~ ., data = X_train)
    # lda_pred = predict(lda_model, X_test)$class
    # lda_acc[i-10] = lda_acc[i-10] + sum(lda_pred == X_test$status) / length(lda_pred)
    # 
    # if (i > 20) { # if i < 20, some group is too small for 'qda'
    #   qda_model = qda(status ~ ., data = X_train)
    #   qda_pred = predict(qda_model, X_test)$class
    #   qda_acc[i-20] = qda_acc[i-20] + sum(qda_pred == X_test$status) / length(qda_pred)
    # }
    
    svm_model = svm(status ~ ., data = X_train)
    svm_pred = predict(svm_model, X_test)
    svm_acc[i-10] = svm_acc[i-10] + sum(svm_pred == X_test$status) / length(svm_pred)

    tree_model = J48(status ~ ., data = X_train)
    tree_pred = predict(tree_model, X_test, type = "class")
    tree_acc[i-10] = tree_acc[i-10] + sum(tree_pred == X_test$status) / length(tree_pred)

    bag_model = bagging(status ~ ., data = X_train)
    bag_pred = predict(bag_model, X_test)
    bag_acc[i-10] = bag_acc[i-10] + sum(bag_pred == X_test$status) / length(bag_pred)

    forest_model = randomForest(status ~ ., data = X_train, ntree = 300)
    forest_pred = predict(forest_model, X_test)
    forest_acc[i-10] = forest_acc[i-10] + sum(forest_pred == X_test$status) / length(forest_pred)
    
    # Because simple glm fit doesn't converge, 
    # so we increase the number of iterations to let it converge
    # But the result is still so bad (accuaracy from 0.37 to 0.78)
    # logis_model = glm(status ~ ., data = X_train, family = "binomial", control=list(maxit=100))
    # logis_pred = predict(logis_model, X_test, type = 'response')
    # cur_acc_table = table(logis_pred, X_test$status)
    # logis_acc[i-10] = logis_acc[i-10] + sum(diag(cur_acc_table)) / sum(cur_acc_table)
    # 
    # This method is too slow
    # boost_model = boosting(status ~ ., data = X_train)
    # boost_pred = predict(boost_model, X_test)$class
    # boost_acc[i-10] = boost_acc[i-10] + sum(boost_pred == X_test$status) / length(boost_pred)
    # 
    # The result is bad even when hidden layer size is as large as 30
    # (roughly the same as number of features in original data)
    # nn_model = nnet(status ~ ., data = nn_X_train, size = 30, softmax = TRUE, trace = FALSE)
    # nn_pred = predict(nn_model, nn_X_test, type = "class")
    # nn_acc[i-10] = nn_acc[i-10] + sum(nn_pred == X_test$status) / length(nn_pred)
  }
  nb_acc[i-10] = nb_acc[i-10] / 15
  
  knn_3_acc[i-10] = knn_3_acc[i-10] / 15
  knn_5_acc[i-10] = knn_5_acc[i-10] / 15
  knn_10_acc[i-10] = knn_10_acc[i-10] / 15
  
  # lda_acc[i-10] = lda_acc[i-10] / 15
  # qda_acc[i-20] = qda_acc[i-20] / 15
  svm_acc[i-10] = svm_acc[i-10] / 15
  
  tree_acc[i-10] = tree_acc[i-10] / 15
  bag_acc[i-10] = bag_acc[i-10] / 15
  forest_acc[i-10] = forest_acc[i-10] / 15
  
  # logis_acc[i-10] = logis_acc[i-10] / 15
  # boost_acc[i-10] = boost_acc[i-10] / 15
  # nn_acc[i-10] = nn_acc[i-10] / 15
}

# KNN
plot(knn_3_acc~c(11:90), col = 'red', cex = 0.3, main = "Parkinson's Disease",
     xlab = 'Training Data (%)', ylab = 'Accuracy', ylim = c(0.75, 0.87))
points(c(11:90), knn_5_acc, col = 'blue', cex = 0.3)
points(c(11:90), knn_10_acc, col = 'darkgreen', cex = 0.3)
lines(knn_3_acc~c(11:90), col = 'red')
lines(knn_5_acc~c(11:90), col = 'blue')
lines(knn_10_acc~c(11:90), col = 'darkgreen')
legend("bottomright",
       c("KNN(k=3)", "KNN(k=5)", "KNN(k=10)"),
       col = c("red", "blue", "darkgreen"),
       text.col = c("red", "blue", "darkgreen"),
       lty = c(1, 1, 1))

# Random forest, Bagging, C4.5
plot(forest_acc~c(11:90), col = 'red', cex = 0.3, main = "Parkinson's Disease",
     xlab = 'Training Data (%)', ylab = 'Accuracy', ylim = c(0.77, 0.92))
points(c(11:90), bag_acc, col = 'blue', cex = 0.3)
points(c(11:90), tree_acc, col = 'darkgreen', cex = 0.3)
lines(forest_acc~c(11:90), col = 'red')
lines(bag_acc~c(11:90), col = 'blue')
lines(tree_acc~c(11:90), col = 'darkgreen')
legend("bottomright",
       c("Random Forest", "Bagging", "C4.5"),
       col = c("red", "blue", "darkgreen"),
       text.col = c("red", "blue", "darkgreen"),
       lty = c(1, 1, 1))

# SVM, LDA, QDA
# plot(svm_acc~c(11:90), col = 'red', cex = 0.3, main = "Parkinson's Disease",
#      xlab = 'Training Data (%)', ylab = 'Accuracy', ylim = c(0.9, 0.98))
# points(c(11:90), lda_acc, col = 'blue', cex = 0.3)
# points(c(21:90), qda_acc, col = 'darkgreen', cex = 0.3)
# lines(svm_acc~c(11:90), col = 'red')
# lines(lda_acc~c(11:90), col = 'blue')
# lines(qda_acc~c(21:90), col = 'darkgreen')
# legend("bottomright",
#        c("SVM", "LDA", "QDA"),
#        col = c("red", "blue", "darkgreen"),
#        text.col = c("red", "blue", "darkgreen"),
#        lty = c(1, 1, 1))




