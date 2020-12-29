# Local import
from src.multihead_attention import MultiHeadAttention
from src.positional_encoding import positional_encoding

# Third party import
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Layer

##### HELPER FUNCTION #####
def pointwise_feed_forward_network(d_model, dff):
    """Construct a pointwise feed-forward neural network.

    Args:
        d_model     : Number of neurons in the last Dense layer.
        dff         : Number of neurons in the first Dense layer.

    Returns:
        A tf.keras.Sequential model.
    """

    return tf.keras.Sequential([
        Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
###########################


class EncoderLayer(Layer):
    """A single encoder layer.

    Consists of two sublayers:
        multi-head self-attention layer
        positionwise feed-forward subnet layer

    
    Source:
        Google - Attention Is All You Need
        TensorFlow - Transformer Model for Language Understanding
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """Instantiate an EncoderLayer object.

        Args:
            d_model     : Number of neurons for each Dense layer.
            num_heads   : Number of attention heads.
            dff         : Number of neurons in the first Dense layer.
            rate        : Dropout rate.

        Returns:
            self.call   : Execution method.
        """

        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = pointwise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """Executes the encoder layer.

        ...

        Args:
            x           : Input data.
            training    : Boolean training flag.
            mask        : Float tensor with shape broadcastable to 
                            (..., seq_len_q, seq_len_k).

        Returns:
            out2    : Output data.
        """

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2



class Encoder(Layer):
    """An encoder.
    
    Source:
        Google - Attention Is All You Need
        TensorFlow - Transformer Model for Language Understanding
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        """Instantiate an Encoder object.

        Args:
            num_layers                  : Number of EncoderLayers.
            d_model                     : Number of neurons for each Dense layer.
            num_heads                   : Number of attention heads.
            dff                         : Number of neurons in the first Dense layer.
            input_vocab_size            : Size of input vocabulary.
            positional_encoding         : Function for positional encoding.
            maximum_position_encoding   : Maximum positional encoding value.
            rate                        : Dropout rate.

        Returns:
            self.call   : Execution method.
        """


        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """Execute the encoder.

        Args:
            x           : Input encoder data.
            training    : Boolean training flag.
            mask        : Float tensor with shape broadcastable to 
                            (..., seq_len_q, seq_len_k).

        Returns:
            x   : Input decoder data.
        """

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
