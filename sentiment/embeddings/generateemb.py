import gensim

from sentiment.interface.embfilehandling import EmbfileHandling


class FastTextEmb:

    def __init__(self,review_lines, EMBEDDING_DIM, filename, choice='imdb'):
        self.review_lines = review_lines
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.filename = filename
        self.choice = choice

    def generate_Embeddings(self):
       obj = EmbfileHandling(self.filename, self.choice)
       if not(obj.fileCheck()):
            print("creating fasttext")
            model = gensim.models.FastText(sentences=self.review_lines , size=self.EMBEDDING_DIM , window=4, workers=4 ,min_count=1 ,iter = 10)
            obj.fileSave(model)

class W2vEmb:

    def __init__(self, review_lines, EMBEDDING_DIM, filename, choice='imdb'):
        self.review_lines = review_lines
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.filename = filename
        self.choice = choice

    def generate_Embeddings(self):
        obj = EmbfileHandling(self.filename, choice=self.choice)
        if not(obj.fileCheck()):
            model = gensim.models.Word2Vec(sentences=self.review_lines , size=self.EMBEDDING_DIM , window=4, workers=4 ,min_count=1 ,iter = 10)
            obj.fileSave(model)
