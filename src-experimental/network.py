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

    def __init__(self, w2v_object):
        """A tensorflow model that uses a Transformer architecture to
        generate musical output.

        Arguments:
            n_vocab         : ...


        Returns:
            self.call   : Execution method.
        """

        super(MusicGenerator, self).__init__()

        input_vocab_size = len(w2v_object.forward_map.keys()) + 2
        target_vocab_size = len(w2v_object.back_map.keys()) + 2

        self.w2v = w2v_object
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
        self.dense = Dense(w2v_object.n_notes)
        self.activation = Activation('softmax')

    def call(self, inp, tar_inp, training, enc_padding_mask, combined_mask, dec_padding_mask):
        """Executes the neural network.

        ...

        Arguments:
        x   : The input.
        """

        x = self.t1(inp, tar_inp, 
                    training, 
                    enc_padding_mask, 
                    combined_mask, 
                    dec_padding_mask)[0]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)

        return x