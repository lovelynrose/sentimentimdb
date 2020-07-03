import os
import config


class EmbfileHandling:
    def __init__(self,filename, choice='imdb'):
        self.filename = filename
        self.choice = choice

    def fileCheck(self):
        path = os.path.join(config.EMBEDDING, self.choice)
        if os.path.exists(os.path.join(path, self.filename)):
            return True
        else:
            return False

    def fileSave(self, model):
        path = os.path.join(config.EMBEDDING, self.choice)
        model.wv.save_word2vec_format(os.path.join(path, self.filename), binary=False)