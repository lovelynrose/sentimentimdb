from sentiment.interface.getdataset import GetDatasetPath
from sentiment.interface.embedding import ReadWriteEmbedding
from sentiment.preprocessing.preparedata import PrepareDataImdb
from sentiment.preprocessing.stopwordremoval import StopwordRemoval


class ReadData:
    def __init__(self, filename, choice='imdb'):
        self.filename = filename
        self.choice = choice

    # returns each review as list
    def readDataReview(self, column_name):
        #print("readdatareview")
        df = GetDatasetPath(self.filename, choice=self.choice).readFileProcessed()
        lis = df[column_name].values.tolist()
        #print(lis)
        return lis

    def readNumWords(self):
        pathGet = ReadWriteEmbedding(self.filename, choice=self.choice)
        path = pathGet.EmbPath()
        f = open(path, "r")
        num_words = f.read()
        num_words = int(num_words)
        return num_words

    def readData(self):
        print("choice = ", self.choice)
        df=None

        read = PrepareDataImdb(self.filename, choice=self.choice)
        df = read.convertClass()
        lis = df['review'].values.tolist()
        sen = df['sentiment'].values.tolist()
        return lis, sen

    def readProcessedSentiment(self):
        review_lines = GetDatasetPath(self.filename, choice=self.choice)
        df = review_lines.readFileProcessed()
        lis = ReadData(self.filename, choice=self.choice).readDataReview('0')
        return lis


    def readProcessedReview(self):
        review_lines = GetDatasetPath(self.filename, choice=self.choice)
        lis = ReadData(self.filename, choice=self.choice).readDataReview('0')
        swr = StopwordRemoval(lis)
        stopwords = swr.stopwordsToRem()
        afterRem = swr.removefromStopwordList(stopwords)
        lines_processed = swr.stopwordRemoval(afterRem)
        return lines_processed
