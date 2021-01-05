# Local imports
from transformer import Transformer
from word2vec import word2vec

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, LSTM
import numpy as np

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
        self.dense = Dense(w2v_object.n_elements)
        self.activation = Activation('softmax')

    def call(self, inp, tar_inp, training, enc_padding_mask, combined_mask, dec_padding_mask):
        """Executes the neural network.

        ...

        Arguments:
        x   : The input.
        """
        assert not np.any(np.isnan(inp))
        assert not np.any(np.isnan(tar_inp))

        x = self.t1(inp, tar_inp, 
                    training, 
                    enc_padding_mask, 
                    combined_mask, 
                    dec_padding_mask)[0]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)

        return x

    '''Returns a one-hot np array from an input tensor, where it will be 1 at the argmax idx.'''

    def force_one_hot(self,inp):
        hot_idx = tf.argmax(inp)
        arr = np.zeros(tf.shape(inp))
        arr[hot_idx] = 1
        return arr


    '''Maps from network output to a note.'''

    def eval(self, inp):
        one_hot_inp = self.force_one_hot(inp)
        return self.w2v.back_map.get(one_hot_inp)



