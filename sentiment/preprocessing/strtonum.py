from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np


class StrToNum:
    def __init__(self,review_lines,max_length):
        self.review_lines = review_lines
        self.max_length = max_length

    def convert_str_to_nums(self):
        tokenizer_obj = Tokenizer()
        tokenizer_obj.fit_on_texts(self.review_lines)
        sequences = tokenizer_obj.texts_to_sequences(self.review_lines)
        word_index = tokenizer_obj.word_index
        review_pad = pad_sequences(sequences , maxlen=self.max_length)
        return np.matrix(review_pad), len(word_index)

