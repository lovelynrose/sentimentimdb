from abc import ABC, abstractmethod

from sentiment.imdb.imdb import Imdb


class CreatorDataset(ABC):

    @abstractmethod
    def factory_method(self):
        pass


class ConcreteCreatorImdb(CreatorDataset):

    def factory_method(self):
        return Imdb()


class ConcreteCreatorAmazon(CreatorDataset):

    def factory_method(self):
        return Amazon()

