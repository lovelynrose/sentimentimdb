from keras.layers import Input, Dense , Bidirectional ,LSTM
from keras.models import Model
from keras.layers import Embedding


class CreateModel:
   def __init__(self, maxlength, embeddingDim):
      self.maxlength = maxlength
      self.EMBEDDING_DIM = embeddingDim


   def existing_model(self, embedding_matrix, output_neurons):
      inp = Input(shape=(self.maxlength,))
      model = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                        input_length=self.maxlength, weights=[embedding_matrix], trainable=False)(inp)
      lstm = Bidirectional(LSTM(50, return_sequences=False), merge_mode='concat')(model)
      outputs = Dense(output_neurons, activation='sigmoid', trainable=True)(lstm)
      model = Model(inp, outputs)
      model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
      return model

