import os
import numpy as np
import itertools
import config
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.utils import to_categorical

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

from sentiment.interface.getdataset import GetDatasetPath
from sentiment.interface.readdata import ReadData
from sentiment.preprocessing.stopwordremoval import StopwordRemoval
from sentiment.training.attention import Attention

from data.parameters.datasetnames import PROCESSED, TEST
from data.parameters.modelnames import MODEL


class PredictModel:
    
    def __init__(self):
        self.test_samples_tokens_pad = None
        self.predictions = None
        self.tes_res = None
        
    def readProcessedReview(self, filename):
        review_lines = GetDatasetPath(filename, choice='imdb')
        df = review_lines.readFileProcessed()
        lis = ReadData(filename, choice='imdb').readDataReview('0')
        swr = StopwordRemoval(lis)
        stopwords = swr.stopwordsToRem()
        afterRem = swr.removefromStopwordList(stopwords)
        lines_processed  = swr.stopwordRemoval(afterRem)
        return lines_processed

    def readProcessedSentiment(self,filename):
        review_lines = GetDatasetPath(filename,choice='imdb')
        df = review_lines.readFileProcessed()
        lis = ReadData(filename, choice='imdb').readDataReview('0')
        self.tes_res = lis
        
    def prepTestData(self,lines_processed,tes_lis):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines_processed)
        test_samples_tokens = tokenizer.texts_to_sequences(tes_lis)
        self.test_samples_tokens_pad = pad_sequences(test_samples_tokens,maxlen=200)
        
    def prepModel(self):
        with CustomObjectScope({'Attention': Attention}):
            new_model = GetDatasetPath(MODEL['fasttext'], choice='imdb').loadModel()
        self.predictions = new_model.predict(self.test_samples_tokens_pad)
        
    def plot_confusion_matrix(self,cm, classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        print("Inside")
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        
    def resultsSingleClass(self):
        predictions = (self.predictions > 0.5).astype('int32')
        cm = confusion_matrix(predictions, self.tes_res)
        cm_plot_labels = ['positive','negative']
        self.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
        print('Accuracy:', accuracy_score(predictions, self.tes_res))
        print('F1 score:', f1_score(predictions, self.tes_res))
        print('Recall:', recall_score(predictions, self.tes_res))
        print('Precision:', precision_score(predictions, self.tes_res))
        
    def resultsMultiClass(self):
        pred = []
        for i in range(len(self.predictions)):
            temp = np.where(self.predictions[i] == np.amax(self.predictions[i]))
            pred.append(temp[0][0])
        predictions = to_categorical(pred)
        tes_res = to_categorical(self.tes_res)
        cm = multilabel_confusion_matrix(predictions, tes_res)
        print(cm)
        print(classification_report(predictions, tes_res))
        
if __name__ == "__main__":
    filenames = [PROCESSED['lemmatize'],TEST['review'], TEST['sentiment']]
    results = PredictModel()
    lines_processed = results.readProcessedReview(filenames[0])
    tes_lis = results.readProcessedReview(filenames[1])
    results.readProcessedSentiment(filenames[2])
    results.prepTestData(lines_processed,tes_lis)
    results.prepModel()
    print("choose the problem type\n1.single class\n2.multi class")
    prob = int(input ("Enter type :"))
    if prob == 1:
        results.resultsSingleClass()
    else:
        results.resultsMultiClass()
