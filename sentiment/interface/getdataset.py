import os
import pandas as pd
from numpy import loadtxt
from numpy import savetxt
import config
from keras.models import load_model



class GetDatasetPath:
    def __init__(self , filename, data = None, choice='imdb'):
        self.filename = filename
        self.data = data
        self.choice = choice

    def saveDataset(self, frame):
        path = os.path.join(config.DATASET, self.choice)
        if frame.empty:
            savetxt(os.path.join(path, self.filename), self.data, delimiter=',')
        else:
            frame.to_csv(os.path.join(path, self.filename), index=False)

    def loadDataset(self):
        path = os.path.join(config.DATASET, self.choice)
        data = loadtxt(os.path.join(path, self.filename), delimiter=',')
        return data

    def readFile(self):
        path = os.path.join(config.RAW, self.choice)
        df = pd.read_csv(os.path.join(path, self.filename))
        return df

    def readFileProcessed(self):
        path = os.path.join(config.DATASET, self.choice)
        df = pd.read_csv(os.path.join(path, self.filename))
        return df

    def embeddedfileCheck(self):
        path = os.path.join(config.EMBEDDING, self.choice)
        if os.path.exists(os.path.join(path, self.filename)):
            return True
        else:
            return False

    def embeddedfileSave(self, model):
        path = os.path.join(config.EMBEDDING, self.choice)
        model.wv.save_word2vec_format(os.path.join(path, self.filename), binary=False)

    def saveModel(self, model):
        path = os.path.join(config.MODELS, self.choice)
        model.save(os.path.join(path, self.filename))

    def loadModel(self):
        path = os.path.join(config.MODELS, self.choice)
        model = load_model(os.path.join(path, self.filename))
        return model



