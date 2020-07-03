import os
import numpy as np
from numpy import savetxt
from numpy import loadtxt
import config

class ReadWriteEmbedding:

    def __init__(self, filename, data=None, choice='imdb'):
        self.filename = filename
        self.data = data
        self.choice = choice

    def readEmb(self):
        path = os.path.join(config.EMBEDDING, self.choice)
        print("Read path = ",path)
        res = loadtxt(os.path.join(path, self.filename), delimiter=',')
        return res

    def writeEmb(self):
        path = os.path.join(config.EMBEDDING, self.choice)
        savetxt(os.path.join(path, self.filename), self.data, delimiter=',')

    def EmbPath(self):
        path = os.path.join(config.EMBEDDING, self.choice)
        return os.path.join(path, self.filename)
