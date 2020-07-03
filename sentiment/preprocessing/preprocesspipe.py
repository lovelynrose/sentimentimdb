from abc import ABC, abstractmethod


from sentiment.preprocessing.specialcharremoval import RemoveSpcChar
from sentiment.preprocessing.lemmatize import Lemmatize
from sentiment.preprocessing.expandcontraction import ExpandContractions
from sentiment.preprocessing.htmlremoval import HTMLTagRemoval

class Handler(ABC):
    def __init__(self, filename, choice = 'imdb',next_process=None):
        self.next_process = next_process
        self.filename = filename
        self.choice = choice

    @abstractmethod
    def preprocess_next(self):
        pass

class HTMLHandler(Handler):

    def preprocess_next(self):
        print("html", self.filename)
        print("choice : ", self.choice)
        HTMLTagRemoval(choice=self.choice).perform_removal(self.filename)
        self.next_process.preprocess_next()

class StopwordHandler(Handler):

    def preprocess_next(self):
        print("stopword : ")
        if False:
            pass
        elif self.next_process is not None:
            self.next_process.preprocess_next()

class ContractionsHandler(Handler):

    def preprocess_next(self):
        print("contractions : ", self.filename)
        ExpandContractions(choice=self.choice).perform_expand(self.filename)
        self.next_process.preprocess_next()

class LemmatizeHandler(Handler):
    def preprocess_next(self):
        print("lemmatize : ", self.filename)
        print(self.next_process)
        Lemmatize(choice=self.choice).perform_expand(self.filename)

class SpecialCharactersHandler(Handler):
    def preprocess_next(self):
        print("Special characters : ", self.filename)
        RemoveSpcChar(choice=self.choice).perform_expand(self.filename)
        self.next_process.preprocess_next()
