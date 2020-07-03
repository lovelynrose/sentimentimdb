import numpy as np

class TrainTestSplit:

    def train_test(self, review_pad, df, vs):
        print("Splitting Datasets to X_train_pad, y_train, X_test_pad, y_test...")
        VALIDATION_SPLIT = vs
        indices = np.arange(review_pad.shape[0])
        np.random.shuffle(indices)
        review_pad = review_pad[indices]
        sentiment = df['0'].values
        sentiment = sentiment[indices]
        num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])
        X_train_pad = review_pad[:-num_validation_samples]
        y_train = sentiment[:-num_validation_samples]
        X_test_pad = review_pad[-num_validation_samples:]
        y_test = sentiment[-num_validation_samples:]
        return X_train_pad, y_train, X_test_pad, y_test


