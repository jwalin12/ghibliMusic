# Local imports
from transformer import Transformer
from word2vec import word2vec

# Third-party imports
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, LSTM

class MusicGenerator(Model):
    """The tensorflow neural network architecture.

    ...

    Uses:
      ...
    """

    def __init__(self, w2c_object):
        """A tensorflow model that uses a Transformer architecture to
        generate musical output.

        Arguments:
            n_vocab         : ...


        Returns:
            self.call   : Execution method.
        """

        super(MusicGenerator, self).__init__()

        input_vocab_size = len(w2c_object.forward_map.keys()) + 2
        target_vocab_size = len(w2c_object.back_map.keys()) + 2

        self.w2c = w2c_object
        self.t1 = Transformer(
            2, 
            512, 
            8, 
            100, 
            input_vocab_size,
            target_vocab_size,
            pe_input=input_vocab_size,
            pe_target=target_vocab_size,
            rate=0.1
        )
        self.dropout = Dropout(0.25)
        self.dense = Dense(w2c_object.n_notes)
        self.activation = Activation('softmax')

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
