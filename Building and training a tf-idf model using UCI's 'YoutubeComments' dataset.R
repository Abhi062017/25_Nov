#Building and training a tf-idf model using UCI's 'YoutubeComments' dataset

#*******************************
#Part.I: Pre-processing........
#*******************************
#read file
spam.raw <- read.csv('YoutubeComments.csv', stringsAsFactors = FALSE)
View(spam.raw)
spam.raw <- spam.raw[,c(4,5)]
names(spam.raw) <- c('Content', 'Class')
spam.raw$Class <- as.factor(spam.raw$Class)
str(spam.raw)
length(which(!complete.cases(spam.raw))) # we deduce : no missing values
table(spam.raw$Class) 
prop.table(table(spam.raw$Class)) # 48.6% spams, 51.3% hams
# Let's get a feel for the distribution of 'Content' lengths
spam.raw$Content_Length <- nchar(spam.raw$Content)
View(spam.raw)
summary(spam.raw$Content_Length) # 75% of the Content_Length is <= 101 characters
# Visualize by adding segmentation for ham/spam:
library(ggplot2)
ggplot(spam.raw, aes(x = Content_Length, fill = Class)) +
  geom_histogram(binwidth = 4) +
  ggtitle('Distribution of Ham/Spam by Content Length') +
  theme_dark() +
  xlim(0,500) +
  xlab('Content Length') +
  ylab('Content Count')

#*******************************
#Part.II: Split in train/test
#*******************************
install.packages('caret', dependencies = T)
library(caret)
set.seed(32984)
indices <- createDataPartition(spam.raw$Class, times=1, p=0.7, list=FALSE)
class(indices)
nrow(indices) #1370 (70%)
nrow(spam.raw)#1956 (Total)
1370/1956  # validating our splits (indices / total observations in spam.raw)

train <- spam.raw[indices, ] # contains 70% of spam.raw
test <- spam.raw[-indices, ] # contains 30% of spam.raw

table(train$Class)
table(test$Class)
prop.table(table(train$Class))
prop.table(table(test$Class))

#**************************************
#Part.III: Text Mining using 'quanteda'
#**************************************
library(quanteda)
library(tm)
detach(package:tm)
train.tokens <- tokens(train$Content, what = 'word',
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens =  TRUE)
train.tokens <- tokens_tolower(train.tokens)
train.tokens <- tokens_select(train.tokens,stopwords(),selection = 'remove')
train.tokens <- tokens_wordstem(train.tokens, language = 'english')
train.tokens <- tokens_remove(train.tokens, " ", remove()) #stripWhiteSpaces
View(train.tokens)

#*************************************
#Part.IV: Building final df using dfm
#*************************************
#Building dfm
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE) 
dim(train.tokens.dfm)
View(train.tokens.matrix) #big ass doc, would take time to open-up,hence use head
head(train.tokens.dfm, n=10)
#Building Matrix of dfm
train.tokens.matrix <- as.matrix(train.tokens.dfm)
dim(train.tokens.matrix)
View(train.tokens.matrix[1:20,1:100]) #viewing partially, instead of the whole doc
colnames(train.tokens.matrix)[1:30]
#make a final dataframe by cbinding 'Class' column to the dfm
train.tokens.df <- cbind(Class = train$Class , as.data.frame(train.tokens.dfm))
length(train$Class)
dim(train.tokens.df)
View(train.tokens.df[1:10,1:15])
#Making syntactially valid names, using make.names()
colnames(train.tokens.df)[1:20] #returns the first 20 variable names
names(train.tokens.df) <- make.names(names(train.tokens.df))

#*************************************
#Part.V: training our rpart classifier
#*************************************
#set controls
set.seed(32984)
cv.folds <- createMultiFolds(train$Class, k=10, times=3) 
cv.ctrl <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 3,
                        index = cv.folds)
#train model
library(doSNOW)
cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)
rpart.cv.1 <- train(Class ~ .,   #Class ~ . => all the frequencies of Content words in train.tokens.df
                    data = train.tokens.df,
                    method = 'rpart',     # => is the algo to train our model
                    trControl = cv.ctrl,  # => trControl = cv.ctrl = trainControl(deduced above)
                    tuneLength = 7)       # => tells caret to use 7 different 
stopCluster(cl)
rpart.cv.1

#*************************************
#Part.VI: #*Building a TF-IDF df*
#*************************************
#Step.1: Defining TF,IDF,and,TF-IDF
#TF
term.frequency <- function(row) {row/sum(row)}
#IDF
inverse.doc.freq <- function(col)
{
  corpus.size <- length(col)
  doc.count <- length(which(col>0))
  log10(corpus.size/doc.count)
}
#TF-IDF
tf.idf <- function(tf,idf) {tf*idf}

#Step.2: Normalize all documents by applying TF
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)  #Notice, that apply() has transposed the matrix
View(train.tokens.df[1:20,1:40])

#Step.3: Calculate IDF vector, that we'll use for both (train data & test data)
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
class(train.tokens.idf)
head(train.tokens.idf)

#Step.4: Calculate the mighty 'TF-IDF' for our training corpus
train.tokens.tfidf <- apply(train.tokens.df, 2, tf.idf, train.tokens.idf)
class(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:5,1:5])
#transpose back the tfidf
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:5,1:5])

#Step.5: check for incomplete.cases
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
head(train$Content[incomplete.cases])#notice an emoji/special symbols
head(train.tokens.tfidf[incomplete.cases])
#Fixing these incomplete cases
train.tokens.tfidf[incomplete.cases, ] <- rep(0, ncol(train.tokens.tfidf))
head(train.tokens.tfidf[incomplete.cases])
tail(train.tokens.tfidf[incomplete.cases])
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))

#Step.6: Make the final df by cbinding 'Class' variable
train.tokens.tfidf.df <- cbind(Class = train$Class, as.data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))
dim(train.tokens.tfidf.df)
View(train.tokens.tfidf.df[1:100,1:30])

#*************************************
#Part.VI: train model
#*************************************
library(doSNOW)
cl <- makeCluster(1, type='SOCK')
registerDoSNOW(cl)
rpart.cv.2 <- train(Class ~ .,
                    data = train.tokens.tfidf.df,
                    method = 'rpart',
                    trControl = cv.ctrl, #Note: cv.ctrl is of : 'Our first model'
                    tuneLength = 7)
stopCluster(cl)
rpart.cv.2

# Clean-up unused objects in memory
gc()
