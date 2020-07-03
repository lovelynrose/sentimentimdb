from abc import ABC, abstractmethod


class Dataset(ABC):

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def data_preparation(self):
        pass

    @abstractmethod
    def embedding(self):
        pass

    @abstractmethod
    def createmodel(self):
        pass
