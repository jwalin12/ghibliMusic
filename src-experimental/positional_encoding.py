# Local imports
# ...

# Third-party imports
import numpy as np
import tensorflow as tf

def positional_encoding(position, d_model):
    """Determines a relative and/or absolute position of the token in an
    encoding space. This is done because no convolution or reccurence are used.

    Source:
        Google - Attention Is All You Need
        Google - MUSIC TRANSFORMER: GENERATING MUSIC WITH LONG-TERM STRUCTURE

    Args:
        position    : position of the input
        d_model     : max depth of the model
    
    Returns:
        A tensor of positional encoding vectors that must be added to the embedding vector to create
        a final input vector, using pos_encoding[:, :seq_len, :].
    """

    def get_angles(pos, i, d_model):
        """Computes an angle using the token and index information
        for positional encoding space.

        Args:
            position: position of the input
            d_model : depth of the model
            i: dimension of embedding

        Returns:
            The token and index defined in positional encoding space.
        """

        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

        return pos * angle_rates



    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
