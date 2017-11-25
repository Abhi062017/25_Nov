#rpart classification on text mined UCI 'YoutubeComments' dataset
getwd()
sessionInfo()
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
summary(spam.raw$Content_Length) # 75% of the Content_Length is <= 95 characters

# Visualize by adding segmentation for ham/spam:
library(ggplot2)
ggplot(spam.raw, aes(x = Content_Length, fill = Class)) +
  geom_histogram(binwidth = 4) +
  ggtitle('Distribution of Ham/Spam by Content Length') +
  theme_dark() +
  xlim(0,500) +
  xlab('Content Length') +
  ylab('Content Count')

#************************************************************************************************************#
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
#*************************************************************************

library(quanteda)
# Step.1: Tokenize the Content messages
# Note: It returns a list.
train.tokens <- tokens(train$Content, what = 'word',
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens =  TRUE)

train.tokens[[358]] #compare this with below:
train$Content[358]

train.tokens <- tokens_tolower(train.tokens)
train.tokens[[358]]

train.tokens <- tokens_select(train.tokens,
                              stopwords(),
                              selection = 'remove')

length(stopwords()) #175 is the total no of built in stopwords,
train.tokens[[358]]

train.tokens <- tokens_wordstem(train.tokens, language = 'english')
train.tokens[[358]]

#building dfm
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE) 
View(train.tokens.dfm) #big ass document, may take a while to open-up.
dim(train.tokens.dfm)

train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20,1:100]) #viewing partially, instead of the whole doc
dim(train.tokens.matrix)

# Investigate the effects of stemming:
colnames(train.tokens.matrix)[1:30] #returns the colnames of first 30 columns

# **********************************************************************************************************#
# From our dfm ('bag of words' model), setup a feature dataframe with Class
train.tokens.df <- cbind(Class = train$Class , as.data.frame(train.tokens.dfm))
length(train$Class)
dim(train.tokens.df)
View(train.tokens.df[1:10,1:15])

# Making syntactially valid names, using make.names()
names(train.tokens.df)[1:20] #returns the first 20 variable names
names(train.tokens.df) <- make.names(names(train.tokens.df))

#train model:
set.seed(32984)
cv.folds <- createMultiFolds(train$Class, k=10, times=3) 
cv.ctrl <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 3,
                        index = cv.folds)

#invoke rpart single decision tree algo,
#and not 'rf',a multi-decision tree algo, bcoz 'rpart' is faster
#Step1: Invoking DoSNOW, to facilitate parallel processing
install.packages("devtools")
library(devtools)
devtools::install_github('topepo/caret/pkg/caret')
install.packages('doSNOW', dependencies = T)
library(doSNOW)
cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)
rpart.cv.1 <- train(Class ~ .,   #Class ~ . => all the frequencies of Content words in train.tokens.df
                    data = train.tokens.df,
                    method = 'rpart',     # => is the algo to train our model
                    trControl = cv.ctrl,  # => trControl = cv.ctrl = trainControl(deduced above)
                    tuneLength = 7)       # => tells caret to use 7 different 
stopCluster(cl)
# Check out our results
rpart.cv.1

#*********************************************************************************************************#
#*Calculating TF-IDF*
# TF(t,d) = frequency(t,d)/Summation(frequency(t,d))
# freq. of 'terms'/total no. of 'terms' in a document
# => Normalization
# IDF(t) = log10(N/Count(t))
# total no. of documents/no. of documents with that 'term' in it
# => Rationalization
# TF - IDF(t,d) = TF(t,d) * IDF(t)
#****************************************************************************************

# Step.1: Calculate TF,IDF,and, TF-IDF
# Calculating TF
term.frequency <- function(row) {row/sum(row)}
# Calculating IDF
inverse.doc.freq <- function(col)
{
  corpus.size <- length(col)
  doc.count <- length(which(col>0))
  log10(corpus.size/doc.count)
}
# Calculating TF-IDF
tf.idf <- function(tf,idf) {tf*idf}

# Step.2: Normalize all documents by applying TF
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)  #Notice, that apply() has transposed the matrix
View(train.tokens.df[1:20,1:40])

# Step.3: Calculate IDF vector, that we'll use for both (train data & test data)
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)

# Step.4: Calculate the mighty 'TF-IDF' for our training corpus
train.tokens.tfidf <- apply(train.tokens.df, 2, tf.idf, train.tokens.idf)
# Note: tf = train.tokens.df,
# Since the matrix was transposed by apply(), hence we'd run it on 2,
# which essentially is 1(row), because of transposition.
# tf.idf = the function for apply()
# idf = train.tokens.idf
class(train.tokens.tfidf)
dim(train.tokens.tfidf)
head(train.tokens.tfidf)

# After performing 'tfidf' (Step.4), each document(Content1, Content2,...Content n),
# is now rationalized (before it was just normalized (each document was normalized,
# based on it's length, so each doc could be compared on equal footing),
# now individual terms('go','jurong','crazi'...) are rationalized in the corpus,
# which says: look, those terms that appear more frequently in the documents,
# are going to be less useful(hence less value of those terms), than the ones
# which are appear less frequently(hence more value to those terms).
# Because the terms that appear less frequent in the documents,
# carry more predictive power.

# Step.5 : Transpose our mighty 'tfidf' back,
# because right now it's in 'term-frequency-document-frequency-matrix',
# where, term-frequencies are the rows, and documents are the columns.
# So we'd need to flip it back to ready it to train our Machine Model
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:5,1:5])

# Step.6: Check for Incomplete cases
# Why?
# Here's Why: when we invoke 'tfidf', we need to check for a particular
# degenerative case.
# What is that degen case?
# Here's What it is: After we've done all our pre-preocessing (i.e removing of
# all the stopwords, removing symbols, numbers,etc, after we've stemmed),
# it's entirely possible that there's nothing left. We basically might have
# removed everything from the string.
# Eg.: Imagine we had a string of emoticons(bunch of special chars/symobols).
# After the pre-processing,there's a strong probability that
# all the special chars/symbols may have been wiped off,
# hence there'd be nothing left but the empty string.
# So when we run 'tfidf' calculation on an empty string, we get error from R.
# error like 'NaN'.
# So to check if there's any Incomplete cases left or not, we'd do the following:
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train$Content[incomplete.cases] #notice an emoji

# Fixing these incomplete cases
train.tokens.tfidf[incomplete.cases, ] <- rep(0, ncol(train.tokens.tfidf))

# Verifying the same:
train.tokens.tfidf[incomplete.cases]
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))

# Step.7: Make a clean data frame using the same process as before
train.tokens.tfidf.df <- cbind(Class = train$Class, as.data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))
dim(train.tokens.tfidf.df)
View(train.tokens.tfidf.df[1:5,1:5])

# Step.8: Training the model.
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
