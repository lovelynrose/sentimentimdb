from keras.callbacks import EarlyStopping


class TrainModel:

    def __init__(self,batchsize,epochs):
        self.batchsize = batchsize
        self.epochs = epochs

    def model_train(self,model ,X_train_pad , y_train ,X_test_pad , y_test):
        my_callbacks = [EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=1, mode='min')]
        model.fit(X_train_pad , y_train , batch_size=self.batchsize , epochs=self.epochs ,validation_data=(X_test_pad , y_test) , verbose=2 , callbacks = my_callbacks)
        return model