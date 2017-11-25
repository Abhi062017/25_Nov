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
train.tokens <- tokens_select(train.tokens,
                              stopwords(),
                              selection = 'remove')
length(stopwords()) #175 is the total no of built in stopwords,
train.tokens <- tokens_wordstem(train.tokens, language = 'english')
train.tokens[[358]]

#building dfm
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE) 
View(train.tokens.dfm) #big ass document, may take a while to open-up.
dim(train.tokens.dfm)

# **********************************************************************************************************#
#make final df by cbinding 'Class' variable to dfm
length(train$Class)
dim(train.tokens.dfm)
train.tokens.df <- cbind(Class = train$Class , as.data.frame(train.tokens.dfm))
dim(train.tokens.df)
View(train.tokens.df[1:10,1:15])
#Making syntactially valid names, using make.names()
names(train.tokens.df) <- make.names(names(train.tokens.df))

#train model:
set.seed(32984)
cv.folds <- createMultiFolds(train$Class, k=10, times=3) 
cv.ctrl <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 3,
                        index = cv.folds)
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

#free unused memory
gc()
