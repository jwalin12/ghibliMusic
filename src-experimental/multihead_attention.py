# Local imports
# ...

# Third-party imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer

class MultiHeadAttention(Layer):
    """Multi-Head Attention Layer

    A layer that computes the attention function by mapping a query and a set
    of key-value pairs to an output where the query, keys and values are 
    vectors.

    Source:
        Google - Attention Is All You Need
        TensorFlow - Transformer Model for Language Understanding
    """

    def __init__(self, d_model, num_heads):
        """Instantiate a MultiHeadAttention object.

        Args:
            d_model     : Number of neurons for each Dense layer.
            num_heads   : Number of attention heads.

        Returns:
            self.call   : Execution method.
        """

        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        # Ensures that d_model is integer divisible by num_heads.
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).

        Args:
            x           : Input data.
            batch_size  : Number of samples in batch size.

        Returns:
            A tensor such that the shape is 
            (batch_size, num_heads, seq_length, depth).
        """

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        """Executes the multi-head attention layer.

        Args:
            v       : Value matrix.
            k       : Key matrix.
            q       : Query matrix.
            mask    : Float tensor with shape broadcastable to 
                        (..., seq_len_q, seq_len_k).

        Returns:
            output              : Output data.
            attention_weights   : Attention weights from the MHA layer.
        """

        def scaled_dot_product_attention(q, k, v, mask):
            """Calculate the attention weights.

            Q, K, and V must have matching leading dimensions.
            K and V must have matching penultimate dimension, i.e.: seq_len_k == seq_len_v.
            The mask has different shapes depending on its type(padding or look ahead) 
            but it must be broadcastable for addition.

            Args:
                q       : Query shape.
                k       : Key shape.
                v       : Value shape.
                mask    : Float tensor with shape broadcastable to 
                            (..., seq_len_q, seq_len_k).

            Returns:
                output              : Output data.
                attention_weights   : Attention weights from the MHA layer.
            """

            matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

            # scale matmul_qk
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

            # add the mask to the scaled tensor.
            if mask is not None:
                scaled_attention_logits += (mask * -1e9)  

            # softmax is normalized on the last axis (seq_len_k) so that the scores
            # add up to 1.
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

            output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

            return output, attention_weights



        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
