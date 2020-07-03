import types

#import tensorflow as tf
from keras.layers import Input, Dense , Bidirectional ,LSTM
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.layers import Concatenate , Lambda , Add
from keras.models import Model
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dropout
from keras.initializers import Constant



from sentiment.training.attention import Attention

class CreateModel:
   def __init__(self, maxlength, embeddingDim):
      self.maxlength = maxlength
      self.EMBEDDING_DIM = embeddingDim


   def sequential(self, num_words):
      model = Sequential()
      model.add(Embedding(num_words, self.EMBEDDING_DIM, input_length=self.maxlength, trainable=True))
      model.add(Bidirectional(LSTM(25)))
      model.add(Dropout(0.5))
      model.add(Dense(1, activation='sigmoid'))
      model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
      return model

   def existing_model(self, embedding_matrix, output_neurons):
      inp = Input(shape=(self.maxlength,))
      model = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                        input_length=self.maxlength, weights=[embedding_matrix], trainable=False)(inp)
      lstm = Bidirectional(LSTM(50, return_sequences=False), merge_mode='concat')(model)
      outputs = Dense(output_neurons, activation='sigmoid', trainable=True)(lstm)
      model = Model(inp, outputs)
      model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
      return model

   def my_model(self, embedding_matrix, output_neurons):
      inp = Input(shape=(self.maxlength,))
      model = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                        input_length=self.maxlength, weights=[embedding_matrix], trainable=False)(inp)
      lstm = Bidirectional(LSTM(50, return_sequences=True), merge_mode='concat')(model)
      prod1 = Lambda(lambda x: x * 0.1)(model)
      prod2 = Lambda(lambda x: x * 0.9)(lstm)
      res = Add()([prod1, prod2])
      att_out = Attention()(res)
      outputs = Dense(output_neurons, activation='sigmoid', trainable=True)(att_out)
      model = Model(inp, outputs)
      model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
      return model
