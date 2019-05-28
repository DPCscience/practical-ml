setwd("~/Desktop/machine learning/iris-species/")
#https://rstudio-pubs-static.s3.amazonaws.com/304864_413b4745f10d46fab1adfff38333dd1c.html
#https://github.com/DavidKohler/IrisDataset/blob/master/Iris.ipynb
#https://datascienceplus.com/random-forests-in-r/
#http://www.sthda.com/english/wiki/print.php?id=234
#https://cran.rstudio.com/web/packages/randomForestExplainer/vignettes/randomForestExplainer.html
#https://journal.r-project.org/archive/2010/RJ-2010-006/RJ-2010-006.pdf

#The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.
#It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
#The columns in this dataset are:
#Id
#SepalLengthCm
#SepalWidthCm
#PetalLengthCm
#PetalWidthCm
#Species

library("randomForest")
library("MASS")
library("party")
#library("stats")
library("factoextra")
library("clValid")
library("class")
library("e1071")

library("neuralnet")
library("caret")
library("ggplot2")

####################################################################################################################################
#############################################       REGRESSION       ###############################################################
####################################################################################################################################

require(datasets)
data("iris")
table(iris$Species)

Y<- iris[,"Sepal.Width"] # select Target attribute
X<- iris[,"Sepal.Length"] # select Predictor attribute
#data exploration
head(X)
head(Y)
#plotting
plot(Y~X, col=X)
model1<- lm(Y~X)
model1 # provides regression line coefficients i.e. slope and y-intercept
plot(Y~X) # scatter plot between X and Y
abline(model1, col="blue", lwd=3) # add regression line to scatter plot to see relationship between X and Y
#prediction when we provide a predictor value
p1<- predict(model1,data.frame("X"=20))
p1

####try prediction selection other predictors and target attributes

############################################
##### RANDOM FOREST 
############################################
#mtry Number of variables randomly sampled as candidates at each split. 
#      Note that the default values are different for classification 
#      (sqrt(p) where p is number of variables in x) and regression (p/3)

#nodesize: Minimum size of terminal nodes

#sampsize: Size(s) of sample to draw

library(randomForest)
#Split iris data to Training data and testing data
set.seed(100)
ind <- sample(2,nrow(iris),replace=TRUE,prob=c(0.7,0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]
#run the random forest using the training data
iris_rf <- randomForest(Species~.,data=trainData,ntree=100,proximity=TRUE,importance=TRUE)
#prediction confusion matrix
irisPred_train<-predict(iris_rf)
table(trainData$Species,predict(iris_rf))
#accuracy
sum(trainData$Species == irisPred_train)/nrow(trainData)

#Random Forest model and  importance features
print(iris_rf)#compare the confusion matrix table with iris_rf$confusion with the train species table

plot(iris_rf)#plot the error rate
legend("top", colnames(iris_rf$err.rate),col=1:4,cex=0.8,fill=1:4)

#################
#Node purity is measured by Gini Index which is the the difference between RSS before and after the split on that variable
#The mean decrease in Gini coefficient is a measure of how each variable contributes to the homogeneity
#of the nodes and leaves in the resulting random forest.If the variable is useful, it tends to split mixed labeled nodes into pure single class nodes.
#################

importance(iris_rf)
varImpPlot(iris_rf)

#random forest for testing data check correct test
irisPred<-predict(iris_rf,newdata=testData)
table(testData$Species,irisPred)
table(testData$Species)
#calculate accuracy

#tuning Random Forest simple example
tune.rf <- tuneRF(iris[,-5],iris[,5])
print(tune.rf)

##############
#tune multiple hyperparameters
library("MASS")
##############
#train subset
train = sample(1:nrow(Boston), 400)
boston.train=Boston[train,]
#run the random forest model with parameters
set.seed(1)
bag.boston=randomForest(medv~.,data=boston.train,mtry=4,nodesize=8,samplesize=400,importance=TRUE)

# Establish a list of other possible values for mtry, nodesize and sampsize
mtry <- seq(4, ncol(boston.train) * 0.8, 2)
nodesize <- seq(2, 8, 2)
sampsize <- c(100,200,250)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry=mtry , nodesize=nodesize , sampsize=sampsize )

# Create an empty vector to store OOB error values
oob_err <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  
  # Train a Random Forest model
  model <- randomForest(medv ~ ., 
                        data = boston.train,
                        mtry = hyper_grid$mtry[i],
                        nodesize = hyper_grid$nodesize[i],
                        sampsize = hyper_grid$sampsize[i])
  val<-as.numeric(names(model$predicted))
  # Store MSE error for the model                      
  oob_err[i] <- mean((model$predicted-Boston$medv[val])^2)
  print(oob_err[i])
}

# Identify optimal set of hyperparmeters based on MSE value
opt_i <- which.min(oob_err)
print(hyper_grid[opt_i,])
#for classification it can be outofbag plotting try it with the iris dataset

#######
#forest visualization
#######
#An implementation of the random forest and bagging ensemble algorithms 
#utilizing conditional inference trees as base learners.

#install.packages("party")
library("party")
cf <- cforest(Species~., data=iris)
#select a tree from the forest, try to work out the number of trees and plot some of them
pt <- prettytree(cf@ensemble[[500]], names(cf@data@get("input"))) 
nt <- new("BinaryTree") 
nt@tree <- pt 
nt@data <- cf@data 
nt@responses <- cf@responses 
plot(nt)

#################################
# bagging and random forest
#################################

library(randomForest)
train = sample(1:nrow(Boston), nrow(Boston)/2)
boston.test=Boston[-train,"medv"]
set.seed(1)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE)
bag.boston
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)#Mean Squared Test Error
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25)
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)#Mean Squared Test Error
set.seed(1)
### The random forest uses an mtry that is low while the examples above used all the predictors available and becomes bagging
### the wealth level of the community (lstat) and the house size (rm) are by far the two most important variables.
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)#Mean Squared Test Error
importance(rf.boston)
varImpPlot(rf.boston)

####################################################################################################################################
#############################################       CLUSTERING       ###############################################################
####################################################################################################################################

############################################
##### K-MEANS CLUSTERING  
############################################
set.seed(100)
#library(stats)
iris.new<- iris[,c(1,2,3,4)]
iris.class<- iris[,"Species"]
head(iris.class)
head(iris.new)
table(iris.class)
normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

iris.new$Sepal.Length<- normalize(iris.new$Sepal.Length)
iris.new$Sepal.Width<- normalize(iris.new$Sepal.Width)
iris.new$Petal.Length<- normalize(iris.new$Petal.Length)
iris.new$Petal.Width<- normalize(iris.new$Petal.Width)

result<- kmeans(iris.new,3) #apply k-means algorithm with no. of centroids(k)=3
result$size # gives no. of records in each cluster
result$cluster #gives cluster vector showing the custer where each record falls
table(iris$Species,result$cluster)
#Total number of correctly classified instances are: 33 + 46 + 50= 129
#Total number of incorrectly classified instances are: 4 + 17= 21
#Accuracy = 129/(129+21) = 0.88 i.e our model has achieved 88% accuracy!

# Visualize
library("factoextra")
fviz_cluster(result, data = iris.new, choose.vars=iris.class,ellipse.type  = "convex")+
  theme_minimal()

par(mfrow=c(2,2), mar=c(5,4,2,2))
plot(iris.new[c(1,2)], col=result$cluster)# Plot to see how Sepal.Length and Sepal.Width data points have been distributed in clusters
plot(iris.new[c(1,2)], col=iris.class)# Plot to see how Sepal.Length and Sepal.Width data points have been distributed originally as per "class" attribute in dataset
plot(iris.new[c(3,4)], col=result$cluster)# Plot to see how Petal.Length and Petal.Width data points have been distributed in clusters
plot(iris.new[c(3,4)], col=iris.class)

############################################
#Finding the optimal number of clusters 
############################################

#elbow plots
#The plot of Within cluster sum of squares vs the number of clusters show us an elbow point at 3
#set.seed(200)
k.max <- 10
wss<- sapply(1:k.max,function(k){kmeans(iris[,3:4],k,nstart = 20,iter.max = 20)$tot.withinss})
wss
plot(1:k.max,wss, type= "b", xlab = "Number of clusters(k)", ylab = "Within cluster sum of squares")

#alternatively
library(factoextra)
iris.scaled <- scale(iris[, -5])

fviz_nbclust(iris.scaled, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)


############################################
##### HIERARCHICAL CLUSTERING
############################################
my_data <- scale(iris[, -5])

# Enhanced hierarchical clustering, cut in 3 groups
library("factoextra")
res.hc <- eclust(my_data, "hclust", k = 3, graph = FALSE) 

# Visualize
fviz_dend(res.hc, rect = TRUE, show_labels = FALSE)
fviz_silhouette(res.hc)

############################################
# How to choose the appropriate clustering algorithm for your data?.
############################################

library("clValid")
intern <- clValid(my_data, nClust = 2:6, 
                  clMethods = c("hierarchical","kmeans","pam"),
                  validation = "internal")
summary(intern)

################################################################################################################################################################################
####### ############################################ CLASSSIFICATION ###########################################################################################################
################################################################################################################################################################################


############################################
####### K NEAREST NEIGHBORS
############################################
#https://www.kaggle.com/xvivancos/knn-in-the-iris-data-set

dataNorm <- iris
dataNorm[,-5] <- scale(iris[,-5])

# 70% train and 30% test
set.seed(234)
ind <- sample(2, nrow(dataNorm), replace=TRUE, prob=c(0.7,0.3))
trainData <- dataNorm[ind==1,]
testData <- dataNorm[ind==2,]

#The knn() function has the following main arguments:
#train. Matrix or data frame of training set cases
#test. Matrix or data frame of test set cases
#cl. Factor of true classifications of training set
#k. Number of neighbours considered

library("class")

# Execution of k-NN with k=1
KnnTestPrediction_k1 <- knn(trainData[,-5], testData[,-5],
                            trainData$Species, k=1, prob=TRUE)

# Execution of k-NN with k=2
KnnTestPrediction_k2 <- knn(trainData[,-5], testData[,-5],
                            trainData$Species, k=2, prob=TRUE)

# Execution of k-NN with k=3
KnnTestPrediction_k3 <- knn(trainData[,-5], testData[,-5],
                            trainData$Species, k=3, prob=TRUE)

# Execution of k-NN with k=4
KnnTestPrediction_k4 <- knn(trainData[,-5], testData[,-5],
                            trainData$Species, k=4, prob=TRUE)

# Confusion matrix of KnnTestPrediction_k1
table(testData$Species, KnnTestPrediction_k1)
# Classification accuracy of KnnTestPrediction_k1
sum(KnnTestPrediction_k1==testData$Species)/length(testData$Species)*100

#To study graphically which value of k gives us the best classification, we can plot 
#Accuracy vs Choice of k.

KnnTestPrediction <- list()
accuracy <- numeric()

for(k in 1:100){
  
  KnnTestPrediction[[k]] <- knn(trainData[,-5], testData[,-5], trainData$Species, k, prob=TRUE)
  accuracy[k] <- sum(KnnTestPrediction[[k]]==testData$Species)/length(testData$Species)*100
  
}

plot(accuracy, type="b", col="dodgerblue", cex=1, pch=20,
     xlab="k, number of neighbors", ylab="Classification accuracy", 
     main="Accuracy vs Neighbors")

# Add lines indicating k with best accuracy
abline(v=which(accuracy==max(accuracy)), col="darkorange", lwd=1.5)

# Add line for max accuracy seen
abline(h=max(accuracy), col="grey", lty=2)

# Add line for min accuracy seen 
abline(h=min(accuracy), col="grey", lty=2)

#to get the exact number of K neighbors that give the highest accuracy
#which(accuracy==max(accuracy))

############################################
####### SUPPORT VECTOR MACHINE
############################################
#https://rischanlab.github.io/SVM.html
#https://medium.com/@ODSC/build-a-multi-class-support-vector-machine-in-r-abcdd4b7dab6
set.seed(314)    # Set seed for reproducible results
library("e1071")
data(iris)
n <- nrow(iris)  # Number of observations
ntrain <- round(n*0.75)  # 75% for training set
tindex <- sample(n, ntrain)   # Create a random index
train_iris <- iris[tindex,]   # Create training set
test_iris <- iris[-tindex,]   # Create test set

svm1 <- svm(Species~., data=train_iris, 
            method="C-classification", kernal="radial", 
            gamma=0.1, cost=10)
              
summary(svm1)
plot(svm1, train_iris, Petal.Width ~ Petal.Length,
     slice=list(Sepal.Width=3, Sepal.Length=4))

prediction <- predict(svm1, test_iris)
xtab <- table(test_iris$Species, prediction)
xtab

#(correct1+correct2+correct3)/ (nrow(test_iris)) # Compute prediction accuracy

#tuning
#http://rstudio-pubs-static.s3.amazonaws.com/371496_0fa7544028634e46b8d1efba7dc7a2f9.html

#try different SVM models
train<-train_iris

svm.model.poly <- svm(Species ~ ., data = train, kernel = 'polynomial')
table(Prediction = predict(svm.model.poly, train),Truth = train$Species)
#calculate accuracy!
svm.model.radial <- svm(Species ~ ., data = train, kernel = 'radial')
table(Prediction = predict(svm.model.radial, train),Truth = train$Species)
#calculate accuracy!
svm.model.sig <- svm(Species ~ ., data = train, kernel = 'sigmoid')
table(Prediction = predict(svm.model.sig, train),Truth = train$Species)
#calculate accuracy!

#Gamma and Cost Optimization
#use the tune.svm() function passing sequences of each parameter.
tuned.svm <- tune.svm(Species~., data = train, kernel = 'linear',
                      gamma = seq(1/2^nrow(iris),1, .01), cost = 2^seq(-6, 4, 2))

#tuned SVM
tuned.svm <- svm(Species ~ . , data = train, kernel = 'linear', gamma = 7.006492e-46, cost = 0.25)
table(Prediction = predict(tuned.svm, train),Truth = train$Species)
#calculate accuracy!

#e1017 package also comes with a function that will simply output the best possible SVM
best.svm <- best.svm(Species~. , data = train, kernel = 'linear')
table(Prediction = predict(best.svm, train), Truth = train$Species)


####Validation
test<-test_iris
best.svm.pred <- predict(best.svm, test)
table(Prediction = best.svm.pred, Truth = test$Species)
####accuracy
sum(test$Species == best.svm.pred)/length(test$Species)

###Now test all the possible SVM models using the tuned SVM training models as an example!

##########################################
#### NEURAL NETS
##########################################
#
#First, load the iris dataset and split the data into training and testing datasets:
data(iris)
ind = sample(2, nrow(iris), replace = TRUE, prob=c(0.7, 0.3))
trainset = iris[ind == 1,]
testset = iris[ind == 2,]

#Add the columns versicolor, setosa, and virginica based on the name matched value in the Species column:
trainset$setosa = trainset$Species == "setosa"
trainset$virginica = trainset$Species == "virginica"
trainset$versicolor = trainset$Species == "versicolor"
head(trainset)

#Next, train the neural network with the neuralnet function try two or three neurons in one hidden layer depending if the model converge. 
#Notice that the results may vary with each training, so you might not get the same result. 
#However, you can use set.seed at the beginning, so you can get the same result in every training process:
#hidden parameter=For example, a vector c(4,2,5) indicates a neural network with three hidden layers, and the numbers of neurons for the first, second and third layers are 4, 2 and 5,
network = neuralnet(versicolor + virginica + setosa~
                          Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, trainset,
                        hidden=2)
network

#Now, you can view the summary information by accessing the result.matrix attribute of the built neural network model:
network$result.matrix
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5009026/
#Lastly, you can view the generalized weight by accessing it in the network:
head(network$generalized.weights[[1]])
#You can visualize the trained neural network with the plot function:
plot(network)
#measure the prediction performance of the trained neural network:

#1. First, generate a prediction probability matrix based on a trained neural network and the testing dataset, testset:
net.predict = compute(network, testset[-5])$net.result
#2. Then, obtain other possible labels by finding the column with the greatest probability:
net.prediction = c("versicolor", "virginica", "setosa")[apply(net.predict, 1, which.max)]
#3. Generate a classification table based on the predicted labels and the labels of the testing dataset:
predict.table = table(testset$Species, net.prediction)
predict.table
#Output example
#prediction
#setosa versicolor virginica#
#setosa        20  0  0
#versicolor   0 19  1
#virginica    0  2  16
#4. use confusionMatrix to measure the prediction performance:
confusionMatrix(predict.table)
