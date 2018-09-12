from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import gzip


class Data(object):
    """The Signal dataset."""

    def __init__(self, path, **kwargs):
        self.load_data(path, **kwargs)

    def load_data(self, path, input_size=100, seed=123, pos=False,
                  test_size=0.1, val_size=0.1, mode='binary', **kwargs):
        """Loads the Signal dataset.

        # Arguments:
            path: path to signal dataset.
            input_size: max number of words to include. Words are ranked
                by how often they occur (in the training set) and only
                the most frequent words are kept
            seed: random seed for sample shuffling.
            pos: whether to use pos tags from dataset.
            test_size: size of test data.
            val_size: size of validation data.
            mode: mode for Tokenizer, one of "binary", "count", "tfidf", "freq"
        """
        self.label_encoder = LabelEncoder()
        self.tokenizer = Tokenizer(num_words=input_size, filters='\t\n')

        with gzip.open(path, mode='rt', encoding='utf-8') as f:

            df = pd.read_csv(f, sep='\t', usecols=['source', 'text'])

            df['source'] = self.label_encoder.fit_transform(df['source'])

            if not pos:
                df['text'] = df['text'].map(lambda x: " ".join(word.split('_')[0] for word in x.split()))

            x_train, x_test, y_train, y_test = train_test_split(
                df['text'], df['source'], test_size=test_size, random_state=seed)
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=val_size, random_state=seed
            )

            self.tokenizer.fit_on_texts(x_train)

            self.x_train = self.tokenizer.texts_to_matrix(x_train, mode=mode)
            self.x_val = self.tokenizer.texts_to_matrix(x_val, mode=mode)
            self.x_test = self.tokenizer.texts_to_matrix(x_test, mode=mode)

            self.y_train = to_categorical(y_train.values, num_classes=10)
            self.y_val = to_categorical(y_val.values, num_classes=10)
            self.y_test = to_categorical(y_test.values, num_classes=10)

    def get_training_data(self):
        """Returns training data.

        # Returns:
            Tuple of Numpy arrays `(x_train, y_train)`.
        """
        return self.x_train, self.y_train

    def get_validation_data(self):
        """Returns validation data.

        # Returns:
            Tuple of Numpy arrays `(x_val, y_val)`.
        """
        return self.x_val, self.y_val

    def get_test_data(self):
        """Returns test data.

        #Returns:
            Tuple of Numpy arrays `(x_test, y_test)`.
        """
        return self.x_test, self.y_test
