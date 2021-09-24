import tensorflow as tf
from selfattention import SelfAttention
from transformerblock import TransformerBlock

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, head, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, head)
        #self.norm = tf.keras.layers.LayerNormalization()
        self.transformer_block = TransformerBlock(embed_size, head, dropout=dropout, forward_expansion=forward_expansion)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, key, value, src_mask, trg_mask):
        attention = self.attention(inputs, inputs, inputs, trg_mask)
        # skip connection
        query = self.dropout(attention + inputs)

        output = self.transformer_block(value, key, query, src_mask)

        return output
