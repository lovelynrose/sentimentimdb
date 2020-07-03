import pandas as pd
from numpy import asarray

from sentiment.interface.getdataset import GetDatasetPath
from sentiment.interface.embedding import ReadWriteEmbedding
from data.parameters.splitsets import SPLIT
from data.parameters.datasetnames import PROCESSED

class SaveData:
    def __init__(self, filename=None, choice='imdb'):
        self.filename = filename
        self.choice = choice

    def savePrepared(self, trainTestVal):
        empty_df = pd.DataFrame()
        store_res = GetDatasetPath(SPLIT['X_train_pad'], data=trainTestVal[0], choice=self.choice)
        store_res.saveDataset(empty_df)
        store_res = GetDatasetPath(SPLIT['y_train'], data=trainTestVal[1], choice=self.choice)
        store_res.saveDataset(empty_df)
        store_res = GetDatasetPath(SPLIT['X_test_pad'], data=trainTestVal[2], choice=self.choice)
        store_res.saveDataset(empty_df)
        store_res = GetDatasetPath(SPLIT['y_test'], data=trainTestVal[3], choice=self.choice)
        store_res.saveDataset(empty_df)

    def saveNumWords(self,word_index):
        print('Write number of Words in file...')
        pathGet = ReadWriteEmbedding(self.filename, choice=self.choice)
        path = pathGet.EmbPath()
        f = open(path, "w")
        num_words = str(word_index+1)
        f.write(num_words)
        f.close()

    def saveMat(self, mat):
        data = asarray(mat)
        print("Convert to csv...")
        saveEmb = ReadWriteEmbedding(self.filename, data=data, choice=self.choice)
        saveEmb.writeEmb()

    def saveData(self, res, sen):
        df1 = pd.DataFrame(res)
        sv = GetDatasetPath(PROCESSED['html_review'], choice=self.choice)
        sv.saveDataset(df1)
        df1 = pd.DataFrame(sen)
        sv = GetDatasetPath(PROCESSED['sentiment'], choice=self.choice)
        sv.saveDataset(df1)

    def saveDataReview(self, res):
        df1 = pd.DataFrame(res)
        sv = GetDatasetPath(self.filename, choice=self.choice)
        sv.saveDataset(df1)

