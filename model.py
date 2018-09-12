from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import InputLayer
from keras import regularizers


class Model(object):
    "Base for neural network model."

    def __init__(self, **kwargs):
        self.build_model(**kwargs)

    def build_model(self, layers=[], input_size=100, loss='categorical_crossentropy', optimizer='adam', **kwargs):
        """Builds model according to configuration provided in layers list.

        # Arguments
            layers: list of dictionaries with parameters for different layers.
            input_size: size of input vector.
            loss: loss function.
            optimizer: optimizer used to train model.
        """

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(input_size,)))

        for layer in layers:
            layer_type = layer.get('type')

            # Add fully connected layer
            if layer_type in ('dense', 'fc', 'fully_connected'):
                regularizer = layer.get('regularizer')

                if regularizer and regularizer in ('l1'):
                    regularizer = regularizers.l1
                elif regularizer and regularizer in ('l2'):
                    regularizer = regularizers.l2

                self.model.add(Dense(
                    layer.get('units'),
                    activation=layer.get('activation', 'relu'),
                    kernel_regularizer=regularizer))

            # Add batch normalization layer
            elif layer_type in ('batch_norm', 'bn'):
                self.model.add(BatchNormalization())

            # Add dropout layer
            elif layer_type in ('dropout', 'dp'):
                self.model.add(Dropout(layer.get('rate', 0.5)))
            else:
                raise ValueError

        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['accuracy']
        )
