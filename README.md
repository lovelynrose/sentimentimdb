# Check effect of using Word Embeddings - Fasttext, w2v, glove
Sentimentimdb checks the effect of Word Embeddings when performing sentiment analysis on the IMDB dataset
## Process Flow
Run the main.py file to perform the following
1. Preprocess the file
  a. Remove HTML Tags
  b. Expand contractions
  c. Remove special characters
  d. Lemmatize
2. Prepare the Data
  a. Remove stopwords
  b. Convert strings to numbers
  c. Save the number of words
  d. Split into training and test data
3. Word Embedding using
  a. Fasttext
  b. Word2Vec
  c. GLove
4. Create model using LSTM
5. Save model in .h5 format
6. Use the model for prediction
7. Evaluate the model
8. Data Visualization
   a. Visualize the confusion matrix
   b. Visualize the sentiment of comment using guagemeter
## Unique Features
### Usage of Design Patterns
  1. Factory Pattern - for imdb object creation
  2. Strategy Pattern - for lemmatization using nltk or spacy
  3. Chain of responsibility - to create the preprocess pipeline
### Object Oriented Programming
### Directory Structure as suggested by machine learning enthusiasts 
## Execution
1. Download the training set from the following link : https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews.
Place the unzipped csv file it in the folder data/raw/imdb
2. Download the imdb dataset for testing from the following link : https://www.kaggle.com/c/word2vec-nlp-tutorial/data?select=testData.tsv.zip.
Place the unzipped tsv file in data/raw/imdb/test
3. Download glove.6B.zip from the link : https://nlp.stanford.edu/projects/glove/.
Unzip and place the file glove.6B.100d.txt in the folder data\processed\embeddings
4. pip install requirements.txt
5. Run : python main.py
