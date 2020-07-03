import pandas as pd
import os
import config

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences



class PrepareDataImdb:
    def __init__(self, filename, choice):
        self.filename = filename
        self.choice = choice

    def convertClass(self):
        path = os.path.join(config.RAW, self.choice)
        df = pd.read_csv(os.path.join(path, self.filename))
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        return df


class PrepareData:

    def __init__(self,filename):
        self.filename = filename

    def prepTestData(self, lines_processed, tes_lis):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines_processed)
        test_samples_tokens = tokenizer.texts_to_sequences(tes_lis)
        test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=200)
        print(test_samples_tokens_pad)
        return test_samples_tokens_pad

