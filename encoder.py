from transformerblock import TransformerBlock
import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, num_layer, head, device, forward_expansion, dropout, max_length): # max_len dataset
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.position_embedding = tf.keras.layers.Embedding(max_length, embed_size)

        self.layer = [TransformerBlock(embed_size, head, dropout=dropout, forward_expansion=forward_expansion) for _ in range(num_layer)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, mask):
        N, seq_length = inputs.shape
        position = tf.broadcast_to(tf.range(seq_length), (N, seq_length))

        output = self.dropout(self.word_embedding(inputs) + self.position_embedding(position))

        for layer in self.layer:
            output = layer(output, output, output, mask)

        return output

