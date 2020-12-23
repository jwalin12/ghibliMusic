# Local import
# ...

# Third party import
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, LSTM

class MusicGenerator(Model):
    """The tensorflow neural network architecture.

    ...

    Uses:
      LSTM
    """

    def __init__(self, network_input, n_vocab):
        """A tensorflow model that uses multiple LSTMs to generate musical 
        output. 

        ...

        WIP: Inspiration by Skuli's architecture at the moment.

        Source:
        Towards Data Science - How to Generate Music using LSTM  Neural
        Networks in Keras by Sigurour Skuli

        Arguments:
            None
        """

        super(MusicGenerator, self).__init__()

        self.lstm1 = LSTM(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True,
        )
        self.dropout1 = Dropout(0.3)
        self.lstm2 = LSTM(
            512,
            return_sequences=True,
        )
        self.dropout2 = Dropout(0.3)
        self.lstm3 = LSTM(
            256
        )
        self.dense1 = Dense(256)
        self.dropout3 = Dropout(0.3)
        self.dense2 = Dense(n_vocab)

        self.activation1 = Activation('softmax')


    def call(self, x):
        """Executes the neural network.

        ...

        Arguments:
        x   : The input.
        """

        x = self.lstm1(x)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)
        x = self.lstm3(x)
        x = self.dense1(x)
        x = self.dropout3(x)
        x = self.dense2(x)

        x = self.activation1(x)

        return x
