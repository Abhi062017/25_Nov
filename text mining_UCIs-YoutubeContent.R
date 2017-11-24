#text mining to predict spam/ham using UCI's "youtube comments" file
getwd()

#read file
youtube <- read.csv('YoutubeComments.csv', header = T, stringsAsFactors = F)
str(youtube)
summary(youtube)
View(youtube)#note: we just need 'Content' and 'Class', of all variables.
youtube <- youtube[,c(4,5)]
table(youtube$CLASS)
youtube$CLASS <- as.factor(youtube$CLASS)
colnames(youtube) <- c('Content', 'Class') #Didn't like them Bold letters
View(youtube)

#note: In some cases, there might be some emoticons/special char in Content,hence:
youtube.encoding <- youtube$Content
class(youtube.encoding)
youtube.encoding <- as.data.frame(sapply(youtube.encoding,
                           function(x) iconv(x, "latin1", "ASCII", sub="")))
View(youtube.encoding)
colnames(youtube.encoding) <- 'Content'

#Building corpus
install.packages('tm', dependencies = T)
library(tm)
youtube.corpus <- VCorpus(VectorSource(youtube.encoding))
View(youtube.corpus)

#text mining: lowercase,stopwords,punctuations,numbers,whitespaces,stemming
youtube.corpus.lower <- tm_map(youtube.corpus, content_transformer(tolower))
youtube.corpus.stopwords <- tm_map(youtube.corpus.lower,removeWords,stopwords())
youtube.corpus.punctuations <- tm_map(youtube.corpus.stopwords, removePunctuation)
youtube.corpus.numbers <- tm_map(youtube.corpus.punctuations, removeNumbers)
youtube.corpus.whitespaces <- tm_map(youtube.corpus.numbers, stripWhitespace)
youtube.corpus.stemming <- tm_map(youtube.corpus.whitespaces, stemDocument)

inspect(youtube.corpus.stemming)

#forming a wordcloud
install.packages("wordcloud", dependencies = T)
library(wordcloud)
wordcloud(youtube.corpus.stemming, max.words = 300,
          min.freq = 20, scale = c(2,.5),
          random.order = F, rot.per = .5,
          colors=brewer.pal(8, "Dark2"))

#creating a DTM sparse matrix
library(SnowballC)
youtube.dtm <- DocumentTermMatrix(youtube.corpus.stemming)
str(youtube.dtm)
class(youtube.dtm)
View(youtube.dtm)
