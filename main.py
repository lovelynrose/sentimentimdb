from sentiment.createdataset import ConcreteCreatorImdb

def data_imdb():
    obj_imdb = ConcreteCreatorImdb()
    ob_imdb = obj_imdb.factory_method()
    #ob_imdb.preprocess()
    #ob_imdb.data_preparation()
    #ob_imdb.embedding()
    #ob_imdb.createmodel()

    ob_imdb.choice = "imdb\\test"
    ob_imdb.preprocess()
    ob_imdb.choice = "imdb"
    ob_imdb.predictmodel()

if __name__ == '__main__':
    data_imdb()

