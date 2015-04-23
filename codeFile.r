---
output: word_document
---
reuters = read.csv(file="C:/Users/gandhisneha711/Downloads/data/reutersCSV.csv/reutersCSV.csv",header=T,sep=",")



# ****......   Task I   ......***


library(NLP)
library(tm)
library(stringr) #for str_count
library(stringi)
library(SnowballC) #for stemming
library(topicmodels)
library(lda)

#Removing entries with empty articles
reuters.emptyList<-numeric(0)
for(i in 1:(dim(reuters)[1])) 
{
  rowTot <- sapply(unclass(reuters[i, 140][[1]]), sum)
  if(rowTot<2) 
  {
    reuters.emptyList<-c(reuters.emptyList,i)
  }
}
reuters.tp <- reuters[-reuters.emptyList,]

#Using only articles from the top 10 most populous topics
topicsList <- c("earn","acq","money.fx","grain","crude","trade","interest","ship","wheat","corn")
topicsList <- paste0("topic.",topicsList)

reuters.dt <- subset(reuters.tp, select=c(topicsList, 'purpose'))                   
reuters.dt <- reuters.dt[-which(rowSums(reuters.dt[,-11]) == 0),]
purpose <- reuters.dt[,11]
reuters.dt$purpose <- NULL

#Getting the frequency of each of the top 10 most populous topics
topicFreq <- colSums(reuters.dt)

#Getting classes
classes <- as.factor(colnames(reuters.dt)[max.col(t(t(reuters.dt) * topicFreq))])
#Drop the empty level in the factor. 
classes <- factor(classes)

#Getting the document names/id of all the documents we are going to use
reuters.DocNames <- row.names(reuters.dt)

#Creating the corpus containing all the documents to be considered
reuters.Corpus <- reuters.tp[reuters.DocNames,140]
reuters.Corpus <- Corpus(VectorSource(reuters.Corpus))
reuters.Dataframe<-data.frame(text=unlist(sapply(reuters.Corpus, `[`, "content")), stringsAsFactors=F)

#Text pre-processing
skipWords <- function(x)  removeWords(x, c(stopwords("english"), stopwords("SMART")))
funcs <- list(tolower, removePunctuation, removeNumbers, stripWhitespace, skipWords, stemDocument)
reuters.Corpus <- tm_map(reuters.Corpus, FUN = tm_reduce, tmFuns = funcs)
reuters.Corpus <- tm_map(reuters.Corpus, PlainTextDocument)

#Creating a document term matrix=>getting frequency of terms
dtm <- DocumentTermMatrix(reuters.Corpus, control = list(wordLengths = c(3,10), 
                          bounds = list(global = c(10,length(reuters.Corpus)-2))))
dtm$dimnames$Docs<-as.character(reuters.DocNames)
dtm.tfxidf<- weightTfIdf(dtm)




## ****......   Task II   ......***


x <- sort(colSums(as.matrix(dtm.tfxidf[which(purpose == 'train'),])), decreasing=TRUE)
n = 50
listOfFeatures <- names(x[1:n])

#Splitting the data into train and test set
set.train <- dtm.tfxidf[which(purpose == 'train'),]
set.test <- dtm.tfxidf[which(purpose == 'test'),]
capture.output(set.train <- as.data.frame(inspect((set.train[,listOfFeatures)->.null
set.train$type<-classes[which(purpose == 'train')]
capture.output(set.test <- as.data.frame(inspect((set.test[,listOfFeatures]))))->.null
set.test$type<-classes[which(purpose == 'test')]
set.complete <- rbind(set.train, set.test)

#Applying topic models for number of topics 20
N = 20
tm.all <- rbind(dtm[which(purpose == 'train'),], dtm[which(purpose == 'test'),])
lda.train <- LDA(tm.all, N)
topics.train <- topics(lda.train)

gamma.df <- as.data.frame(lda.train@gamma)
names(gamma.df) <- paste0("topic.",c(1:N))

#Adding topic models as a feature
d = dim(set.train)[1]
set.complete <- cbind(set.complete, gamma.df)
set.train <- cbind(set.train, gamma.df[1:d,])
set.test <- cbind(set.test, gamma.df[(d+1):dim(gamma.df)[1],])




# ****......   Task III   ......***


library(e1071)
library(randomForest)

#Naive bayes
nb.model <- naiveBayes(type ~ ., data = set.train, laplace = 0)
nb.pred <- predict(nb.model, set.test[, -(n+1)])
capture.output(nb.check <- table(pred = nb.pred, true = set.test[,n+1]))->.null
tune(naiveBayes, type ~ ., data = set.complete, cross = 10, best.model = TRUE)

#Support vector machines
svm.model <- svm(type ~ ., data = set.train)
svm.pred <- predict(svm.model, set.test[, -(n+1)])
capture.output(svm.check <- table(pred = svm.pred, true = set.test[,n+1]))->.null
tune(svm, type ~ ., data = set.complete, cross = 10, best.model = TRUE)

#Random Forest 
set.seed(17)
rf.model <- randomForest(type ~ ., data = set.train, ntree = 100)
rf.pred <- predict(rf.model, set.test[, -(n+1)])
capture.output(rf.check <- table(pred = rf.pred, true = set.test[,n+1]))->.null
tune.randomForest(type ~ ., data = set.complete, cross = 10, best.model = TRUE)

#10-fold cross validation
data <- set.complete
k = 10

#Sample from 1 to k, nrow times (the number of observations in the data)
data$id <- sample(1:k, nrow(data), replace = TRUE)
list <- 1:k

#Prediction and testset data frames that we add to with each iteration over the folds
pred.df <- data.frame()
ts.df <- data.frame()

for (i in 1:k)
{
  set.train.tmp <- subset(data, id %in% list[-i])
  set.test.tmp <- subset(data, id %in% c(i))
  model <- randomForest(type ~ ., data = set.train.tmp, ntree = 100)
  #Remove type
  temp.df <- as.data.frame(predict(model, set.test.tmp[,-which(names(set.test.tmp)=="type")]))
  #Append this iteration's predictions to the end of the prediction data frame
  pred.df <- rbind(pred.df, temp.df)
  
  #Keep only the "type" column
  ts.df <- rbind(ts.df, as.data.frame(set.test.tmp[,(N+1)]))
}


#Adding predictions and actual type
result <- cbind(pred.df, ts.df[, 1])
names(result) <- c("Predicted", "Original")
validation.check <- table(pred = result$Predicted, true = result$Original)
  
#estimating	accuracy,	precision	and	recall
measureFunc = function(data)
{
  n <- nrow(data)
  #true positives
  tp = array()
  #false positives
  fp = array()
  #false negatives
  fn = array()
  #accuracy
  accu = array()     
  #precision
  prec = array()
  recall = array()       
  for(i in 1:n)
  {
    tp[i] = data[i,i]
    fn[i] = sum(data[i,]) - tp[i]
    fp[i] = sum(data[,i]) - tp[i] 
    accu[i] = tp[i]/colSums(data)[i]
    prec[i] = tp[i]/(tp[i]+fn[i])
    recall[i] = tp[i]/(tp[i]+fp[i])
  }
  df1 <- data.frame(tp, fn, fp)
  df2 <- data.frame(accu, recall, prec, row.names=row.names(data))
  #colnames(df2)<-c('Accuracy','Recall','Pprecision')
  
  #calculating microaverage and macroaverage
  macroAvg = list()
  microAvg = list()
  microAvg$Recall <- sum(df1[,1])/(sum(df1[,c(1,3)]))
  microAvg$Precision <- sum(df1[,1])/(sum(df1[,c(1,2)]))
  macroAvg$Recall <- sum(df2[,3], na.rm = TRUE)/n
  macroAvg$Precision <- sum(df2[,2], na.rm = TRUE)/n
  output <- list("Performance Table:" = df2, "Micro-Average" = microAvg, "Macro-Average" = macroAvg)
  return(output)
}
measureFunc(nb.check)
measureFunc(svm.check)
measureFunc(rf.check)
measureFunc(validation.check)



## ****......   Task IV   ......***


#clustering
clus <- set.complete
clus$type <- NULL
sc <- scale(clus)
d <- dist(sc, method = "euclidean") # distance matrix

library(cluster)
library(flexclust)
library(fpc)

#k-means clustering
km <- kmeans(sc, 10)
km
km.table <- table(set.complete$type, km$cluster)
randIndex(km.table)
plotcluster(sc, km$cluster)

#CLARA
cl <- clara(sc, 10, samples=50)
cl.table <- table(set.complete$type, cl$clustering)
cl.table
randIndex(cl.table)
plotcluster(clus, cl$clustering)
pairs(clus[1:5], cl$clustering[1:5])

#hierarchical clustering
hc <- hclust(d, method="average")
plot(hc, hang = -1, labels=set.complete$type, cex=0.1)
hc.grps <- cutree(hc, k=10) # cut tree into 10 clusters
#making borders
rect.hclust(hc, k=10, border="red")
#taking a sample of 100 records from dataset
samp <- sample(1:dim(sc)[1], 100)
dataSample <- sc[samp,]
hc <- hclust(dist(dataSample), method="ave")
plot(hc, hang = -1, labels=set.complete$type[samp], cex=0.7)
grps <- cutree(hc, k=10) # cut tree into 10 clusters
#making borders
rect.hclust(hc, k=10, border="red")


#validation

#internal validation
library(clValid)
iVal <- clValid(sc, 10, clMethods=c("clara", "kmeans", "hierarchical"), validation="internal")
summary(iVal)
optimalScores(iVal)

#stability validation
sVal <- clValid(sc, 10, clMethods=c("clara", "kmeans", "hierarchical"), validation="stability")
summary(sVal)
optimalScores(sVal)

#comparing the two cluster solutions using distance matrix
cluster.stats(d, cl$clustering, km$cluster)
cluster.stats(d, cl$clustering, hc.grps)
cluster.stats(d, km$cluster, hc.grps)
