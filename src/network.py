# Local import
# ...

# Third party import
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, LSTM





'''Attention head class. Matrices are trainable variables. Applies Attention Mechanism to
 embedding vector.'''
class attention_head:
    def __init__(self, embedding_length, output_len):
        self.wQ = tf.Variable(tf.random.normal([embedding_length, output_len]))
        self.wK = tf.Variable(tf.random.normal([embedding_length, output_len]))
        self.wV = tf.Variable(tf.random.normal([embedding_length, output_len]))

    def __call__(self, x):
        queries = tf.matmul(x, self.wQ)
        keys = tf.matmul(x, self.wK)
        values = tf.matmul(x, self.wV)
        dk = tf.sqrt(tf.shape(keys)[0])
        return tf.matmul(tf.nn.softmax((tf.matmul(queries, keys)) / dk), values)





'''Encoder class. Goes through self attention and feed forward neural steps.'''
class encoder(Layer):
    def __init__(self, embedding_length, output_len, nheads = 1, wO_len = 4):
        super(encoder, self).__init__()

        self.nheads = nheads
        self.heads = []
        for _ in range(nheads):
            self.heads.append(attention_head.__init__(embedding_length, output_len))
        self.wO = None
        if (self.nheads >1):
            self.wO = tf.Variable(tf.random.normal([nheads*output_len,wO_len]))

'''Does self attention. Normalizes and adds residual. Does Feed Forward Neural Network.
Adds and normalizes. Assume input is already embedded.'''
    def __call__(self,x):
        Z = tf.Tensor([], dtype = float, value_index=0)
        for head in self.heads:
            tf.concat([Z,head(x)])
        atten_to_norm = tf.matmul(Z, self.wO)+ x #attention step
        atten_normed = tf.keras.layers.LayerNormalization(atten_to_norm) #adding and normalizing

        #TODO: feedforward network



















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
