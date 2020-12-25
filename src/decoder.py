# Local import
from multihead_attention import MultiHeadAttention

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


class DecoderLayer(Layer):
    """A single encoder layer.

    Consists of three sublayers:
        masked multi-head self-attention layer
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

        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = pointwise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Executes the encoder layer.

        ...

        Args:
            x               : Input data.
            enc_output      : Output from the encoder.
            training        : Boolean training flag.
            look_ahead_mask : Tensor that prevents the Transformer from
                                including all words in generation; only 
                                the preceding words.
            padding_mask    : Tensor that provides padding for input data.

        Returns:
            out3                : Output data
            attn_weights_block1 : Attention weights from the masked MHA layer.
            attn_weights_block2 : Attention weights from MHA layer 
        """

        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2



class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                positional_encoding, maximum_position_encoding, rate=0.1):
        """Instantiate a Decoder object.

        Args:
            num_layers                  : Number of DecoderLayers.
            d_model                     : Number of neurons for each Dense layer.
            num_heads                   : Number of attention heads.
            dff                         : Number of neurons in the first Dense layer.
            target_vocab_size           : Size of target vocabulary.
            positional_encoding         : Function for positional encoding.
            maximum_position_encoding   : Maximum positional encoding value.
            rate                        : Dropout rate.

        Returns:
            self.call   : Execution method.
        """

        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Execute the decoder.

        Args:
            x               : Input data.
            enc_output      : Output from the encoder.
            training        : Boolean training flag.
            look_ahead_mask : Tensor that prevents the Transformer from
                                including all words in generation; only 
                                the preceding words.
            padding_mask    : Tensor that provides padding for input data.

        Returns:
            x                   : Input linear data.
            attention_weights   : Attention weights from the masked and 
                                    unmasked MHA layer.
        """

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                    look_ahead_mask, padding_mask)

        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
