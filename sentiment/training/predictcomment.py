import os
import config
from keras.models import load_model
from keras.utils import CustomObjectScope
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from sentiment.interface.readdata import ReadData
from sentiment.training.attention import Attention
from data.parameters.datasetnames import PROCESSED

lemmatizer = WordNetLemmatizer()

class Sentiment:

    def __init__(self, choice='imdb'):
        self.choice=choice

    def commentSentiment(self, model, comment):
        print("Read data..")
        lines_processed = ReadData(PROCESSED['lemmatize'], choice=self.choice).readProcessedReview()
        print("tokenize...")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines_processed)
        tes_res = [0]
        print("Comment = ",comment, type(comment))
        test_sample_1 = comment
        test_samples = [test_sample_1]
        test_samples_tokens = tokenizer.texts_to_sequences(test_samples)
        test_samples_tokens_pad = pad_sequences(test_samples_tokens,maxlen=200)
        print("Load Model...")
        with CustomObjectScope({'Attention': Attention}):
            new_model = load_model(os.path.join(config.MODELS, model))
        probability = new_model.predict(x=test_samples_tokens_pad)
        print("Probability = ", probability)
        predictions = (probability > 0.5).astype('int32')
        print("Class = ", type(predictions))
        if predictions == 0:
            sent = "Negative"
            print(sent)
        else:
            sent = "Positive"
            print(sent)
        return sent, probability

obj = Sentiment()
model = 'imdb\\model_fasttext.h5'
obj.commentSentiment(model,"good")