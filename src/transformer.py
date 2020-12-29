# Local import
from encoder import Encoder
from decoder import Decoder

# Third party import
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        """Instantiate a Transformer object.

        Args:
            num_layers          : Number of EncoderLayers/DecoderLayers.
            d_model             : Number of neurons for each Dense layer.
            num_heads           : Number of attention heads.
            dff                 : Number of neurons in the first Dense layer.
            input_vocab_size    : Size of input vocabulary.
            target_vocab_size   : Size of target vocabulary.
            pe_input            : ...
            pe_target           : ...
            rate                : Dropout rate.

        Returns:
            self.call   : Execution method.
        """
                    
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                            input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                            target_vocab_size, pe_target, rate)

        self.final_layer = Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """Execute the Transformer

        Args:
            inp                 : Input data.
            tar                 : Target data.
            training            : Boolean training flag.
            enc_padding_mask    : Tensor that provides padding for 
            look_ahead_mask     : Tensor that prevents the Transformer from
                                    including all words in generation; only 
                                    the preceding words.
            dec_padding_mask    : Tensor that provides padding for input decoder data.

        Returns:
            self.call   : Execution method.
        """

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
        