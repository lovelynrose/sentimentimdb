from tensorflow.python.keras.preprocessing.text import Tokenizer


class Tokenize:


    def tokenizer(self, lines_processed):
        print("Tokenize...")
        tokenizer_obj = Tokenizer()
        tokenizer_obj.fit_on_texts(lines_processed)
        sequences = tokenizer_obj.texts_to_sequences(lines_processed)
        word_index = tokenizer_obj.word_index
        return lines_processed, word_index
