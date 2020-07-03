from numpy import asarray

from data.parameters.datasetnames import DATASET, PROCESSED, EMBEDDING, TEST, TEST_DATASET
from data.parameters.modelnames import MODEL
from data.parameters.numwords import WORDS

from sentiment.abcdataset import Dataset
from sentiment.preprocessing.preprocesspipe import HTMLHandler
from sentiment.preprocessing.preprocesspipe import SpecialCharactersHandler, ContractionsHandler, LemmatizeHandler
from sentiment.preprocessing.stopwordremoval import StopwordRemoval
from sentiment.preprocessing.tokenize import Tokenize
from sentiment.preprocessing.strtonum import StrToNum
from sentiment.preprocessing.preparedata import PrepareData
from sentiment.interface.savedata import SaveData
from sentiment.training.splitdataset import TrainTestSplit
from sentiment.interface.getdataset import GetDatasetPath
from sentiment.interface.embedding import ReadWriteEmbedding
from sentiment.interface.readdata import ReadData
from sentiment.training.createmodel import CreateModel
from sentiment.training.trainmodel import TrainModel
from sentiment.training.predictmodel import PredictModel
from sentiment.embeddings.prepareembedding import PrepareEmbeddingMatrix
from sentiment.embeddings.generateemb import FastTextEmb, W2vEmb




class Imdb(Dataset):

    def __init__(self, choice='imdb', output=1):
        self.choice = choice
        self.output = output
        print("Constructor : ",self.choice)


    def preprocess(self):
        lemmatize = LemmatizeHandler(PROCESSED['spl_char'], next_process=None, choice=self.choice)
        special_characters = SpecialCharactersHandler(PROCESSED['contractions'], next_process=lemmatize, choice=self.choice)
        contractions = ContractionsHandler(PROCESSED['html_review'], next_process=special_characters, choice=self.choice)
        html = HTMLHandler(DATASET['imdb'], next_process=contractions, choice=self.choice)
        if self.choice == 'imdb\\test':
            print("in test")
            html = HTMLHandler(TEST_DATASET['name'], next_process=contractions, choice=self.choice)
        html.preprocess_next()

    def data_preparation(self):
        preprocessFinal = ReadData(PROCESSED['lemmatize']).readDataReview('0')
        stop_removed = StopwordRemoval(preprocessFinal).perform_removal()
        review_pad, word_index = StrToNum(stop_removed, 200).convert_str_to_nums()
        SaveData(WORDS['num_words']).saveNumWords(word_index)
        df = GetDatasetPath(PROCESSED['sentiment']).readFileProcessed()
        X_train_pad, y_train, X_test_pad, y_test = TrainTestSplit().train_test(review_pad, df, 0.2)
        trainTestVal = [X_train_pad, y_train, X_test_pad, y_test]
        SaveData().savePrepared(trainTestVal)

    def embedding(self):
        lines_processed = ReadData(PROCESSED['lemmatize']).readDataReview('0')
        stop_removed = StopwordRemoval(lines_processed).perform_removal()
        lines_processed, word_index = Tokenize().tokenizer(stop_removed)
        num_words = ReadData(WORDS['num_words']).readNumWords()

        print("Embedding as fastext...")
        embprep = FastTextEmb(lines_processed, 100, EMBEDDING['fast_text'])
        embprep.generate_Embeddings()
        embMat = PrepareEmbeddingMatrix(EMBEDDING['fast_text'], 100, num_words)
        embeddings_index = embMat.get_EmbeddingIndex()
        mat = embMat.get_embMatrix(word_index, embeddings_index)
        data = asarray(mat)
        print("Convert to csv...")
        saveEmb = ReadWriteEmbedding(EMBEDDING['fast_emb'], data)
        saveEmb.writeEmb()

        print("Embedding as w2v...")
        embprep = W2vEmb(lines_processed, 100, EMBEDDING['w2v_text'])
        embprep.generate_Embeddings()
        embMat = PrepareEmbeddingMatrix(EMBEDDING['w2v_text'], 100, num_words)
        embeddings_index = embMat.get_EmbeddingIndex()
        mat = embMat.get_embMatrix(word_index, embeddings_index)
        data = asarray(mat)
        print("Convert to csv...")
        saveEmb = ReadWriteEmbedding(EMBEDDING['w2v_emb'], data)
        saveEmb.writeEmb()

        print("Embedding as glove...")
        embMat = PrepareEmbeddingMatrix(EMBEDDING['glove_text'], 100, num_words)
        embeddings_index = embMat.get_EmbeddingIndex()
        mat = embMat.get_embMatrix(word_index, embeddings_index)
        data = asarray(mat)
        print("Convert to csv...")
        saveEmb = ReadWriteEmbedding(EMBEDDING['glove_emb'], data)
        saveEmb.writeEmb()

    def createmodel(self):
        print("Getting training and validation files...")
        dataObj = GetDatasetPath('x_train.csv')
        X_train_pad = dataObj.loadDataset()
        dataObj = GetDatasetPath('x_test.csv')
        X_test_pad = dataObj.loadDataset()
        dataObj = GetDatasetPath('y_test.csv')
        y_test = dataObj.loadDataset()
        dataObj = GetDatasetPath('y_train.csv')
        y_train = dataObj.loadDataset()
        print("Getting embedding file...")

        print("Fasttext embedding - model...")
        embObj = ReadWriteEmbedding(EMBEDDING['fast_emb'])
        embedding_matrix = embObj.readEmb()
        print("Creating model...")
        mod = CreateModel(200, 100)
        model = mod.my_model(embedding_matrix, 1)
        #model = mod.existing_model(embedding_matrix, 1)
        print("Training model...")
        train = TrainModel(128, 50)
        model = train.model_train(model, X_train_pad, y_train, X_test_pad, y_test)
        print("Saving model...")
        GetDatasetPath(MODEL['fasttext']).saveModel(model)

        #elif num == 2:
        print("w2v embedding - model...")
        embObj = ReadWriteEmbedding(EMBEDDING['w2v_emb'])
        embedding_matrix = embObj.readEmb()
        print("Creating model...")
        mod = CreateModel(200, 100)
        model = mod.existing_model(embedding_matrix, 1)
        print("Training model...")
        train = TrainModel(128, 50)
        model = train.model_train(model, X_train_pad, y_train, X_test_pad, y_test)
        print("Saving model...")
        GetDatasetPath(MODEL['w2v']).saveModel(model)
        #elif num == 3:

        print("Glove embedding - model...")
        embObj = ReadWriteEmbedding(EMBEDDING['glove_emb'])
        embedding_matrix = embObj.readEmb()
        print("Creating model...")
        mod = CreateModel(200, 100)
        model = mod.existing_model(embedding_matrix, 1)
        print("Training model...")
        train = TrainModel(128, 50)
        model = train.model_train(model, X_train_pad, y_train, X_test_pad, y_test)
        print("Saving model...")
        GetDatasetPath(MODEL['glove']).saveModel(model)


    def predictmodel(self):
        lines_processed = ReadData(PROCESSED['lemmatize'], choice=self.choice).readProcessedReview()
        tes_lis = ReadData(TEST['review'], choice=self.choice).readProcessedReview()
        tes_sen = ReadData(TEST['sentiment'], choice=self.choice).readProcessedSentiment()
        test_samples_tokens_pad = PrepareData(filename=None).prepTestData(lines_processed, tes_lis)

        print("Predicting for Fasttext...")
        results = PredictModel(choice=self.choice, model=MODEL['fasttext'])
        predictions = results.prepModel(test_samples_tokens_pad)
        if self.output == 1:
            results.resultsSingleClass(predictions, tes_sen)
        else:
            results.resultsMultiClass(predictions, tes_sen)

        print("Predicting for W2v...")
        results = PredictModel(choice=self.choice, model=MODEL['w2v'])
        predictions = results.prepModel(test_samples_tokens_pad)
        if self.output == 1:
            results.resultsSingleClass(predictions, tes_sen)
        else:
            results.resultsMultiClass(predictions, tes_sen)

        print("Predicting for Glove...")
        results = PredictModel(choice=self.choice, model=MODEL['glove'])
        predictions = results.prepModel(test_samples_tokens_pad)
        if self.output == 1:
            results.resultsSingleClass(predictions, tes_sen)
        else:
            results.resultsMultiClass(predictions, tes_sen)


