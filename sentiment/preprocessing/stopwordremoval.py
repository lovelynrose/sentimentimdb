from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class StopwordRemoval:

    def __init__(self,lines):
        self.lines = lines

    def removefromStopwordList(self,lis):
        stop_words = set(stopwords.words('english'))
        for i in range(len(lis)):
            stop_words.remove(lis[i])
        return stop_words

    def stopwordRemoval(self , stop_words):
        review_lines = list()
        for line in self.lines:
            #print(line)
            tokens = word_tokenize(line)
            tokens = [w.lower() for w in tokens]
            words = [w for w in tokens if not w in stop_words]
            review_lines.append(words)
        return review_lines

    def stopwordsToRem(self):
        lis = ['not', 'than', 'against', 'can', 'no', 'nor', 'off', 'out']
        return lis

    def perform_removal(self):
        print("Removing Stopwords...")
        swr = StopwordRemoval(self.lines)
        stopwords = swr.stopwordsToRem()
        afterRem = swr.removefromStopwordList(stopwords)
        lines_processed = swr.stopwordRemoval(afterRem)
        return lines_processed
