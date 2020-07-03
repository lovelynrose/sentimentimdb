import numpy as np
from sentiment.interface.embedding import ReadWriteEmbedding


class PrepareEmbeddingMatrix:

    def __init__(self, filename, embedding_dim, num_words, choice='imdb'):
        self.filename = filename
        self.EMBEDDING_DIM = embedding_dim
        self.num_words = num_words
        self.choice = choice

    def get_EmbeddingIndex(self):
        embeddings_index = {}
        pathGet = ReadWriteEmbedding(self.filename, choice=self.choice)
        path = pathGet.EmbPath()
        f = open(path, encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def get_embMatrix(self, word_index, embeddings_index):
        embedding_matrix = np.zeros((self.num_words, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            if i > self.num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
