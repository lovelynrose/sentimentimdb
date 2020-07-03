from keras.utils import CustomObjectScope
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    accuracy_score, f1_score, confusion_matrix

from sentiment.training.attention import Attention
from sentiment.interface.getdataset import GetDatasetPath
from data.parameters.modelnames import MODEL
from sentiment.visualization.confusionmatrix import EvaluationResults



class PredictModel:

    def __init__(self, choice=None, model=MODEL['fasttext']):
        self.choice = choice
        self.model = model

    def prepModel(self, test_samples_tokens_pad):
        with CustomObjectScope({'Attention': Attention}):
            new_model = GetDatasetPath(self.model, choice=self.choice).loadModel()
        predictions = new_model.predict(test_samples_tokens_pad)
        return predictions

    def resultsSingleClass(self, predictions, tes_res):
        #print(predictions)
        predictions = (predictions > 0.5).astype('int32')
        cm = confusion_matrix(predictions, tes_res)
        cm_plot_labels = ['positive', 'negative']
        EvaluationResults().plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
        print('Accuracy:', accuracy_score(predictions, tes_res))
        print('F1 score:', f1_score(predictions, tes_res))
        print('Recall:', recall_score(predictions, tes_res))
        print('Precision:', precision_score(predictions, tes_res))

