A training and a deployment of RNN for email spam classfication.


## Performance metrics

> Precision: 0.9937
> 
> Recall: 0.9967
> 
> accuracy: 0.9949
> 
> loss: 0.0215
> 
> val_Precision: 0.9917
> 
> val_Recall: 0.9531
> 
> val_accuracy: 0.9711
> 
> val_loss: 0.0967


# Techniques

## Before model training

1. Remove special characters (symbols, numbers) using `word.translate(str.maketrans('', '', string.punctuation))`
2. `from sklearn.model_selection import train_test_split`

   Casual train test split.
3. `from nltk.tokenize import word_tokenize`

   Splits sentences into words.
4. `from tensorflow.keras.preprocessing.text import Tokenizer`

   Builds vocabulary (word_index) & assigns integer IDs by frequency ranking on `fit_on_texts`.
   > most frequent = 1, second most = 2, ...
   - Ability to filter rare words using `num_words`, which only consider the most frequent ones
   - Ability to handle unknown words in test data using `oov_token`

   Lastly, convert tokens to sequences(integer index) based on the vocabulary to represent the text data, which suits a neural network, using `texts_to_sequences(data)`.
5. `from keras.preprocessing.sequence import pad_sequences`

   Sequences from above may have varying length. Use `pad_sequences` to make them uniform.

## During model training

1. `from tensorflow.keras.layers import LsTM`
   
   Long Short-Term Memory is added as a layer, so the machine could connect the existing words to better understand the logic.
   > It isn't good
   - The quote above is better understood when the model sees the `isn't`
   
   **Note**: LSTM-based model learns **sequential patterns** of random length through its recurrent connection, which implicitly capture _bigrams_, _trigrams_, and even longer dependencies.
3. `from tensorflow.keras.callbacks import EarlyStopping`

   - Stop the training after the learning has reached a plateau (N consecutive epochs without improvement).
   - Reverts to the weights from the epoch with the best performance. Otherwise, the model would keep the possibly overfitted weights from the last epoch.


# Techniques NOT implemented

The techniques below could possibly improve the performance if implemented.
1. Lemmatization
2. Stopwords removal


# Usage

1. Create a python virtual environment to house the packages requied.
   ```
   python -m venv <EnvName>
   ```
2. Clone the project to the virtual environment.
   ```
   cd <EnvName>
   ```
   ```
   git clone https://github.com/c-0w0/RNN-Email-Spam-Classification.git
   ```
   
## Deployment(Optional)

3. To deploy the project in docker:
   First, we get into the `docker` folder by executing command lines:
   ```
   cd RNN-Email-Spam-Classification\docker
   ```
   - Start **Docker Desktop** app.
   - Build image.
   ```
   docker build -t spam-classifier .
   ```
   - Run container OR you may use the UI in **Docker Desktop**'s _**Images**_ tab.
   ```
   docker run -p 5000:5000 spam-classifier
   ```
5. Visit the page: http://localhost:5000 OR you may use the UI in **Docker Desktop**'s _**Images**_ tab.
