import tensorflow as tf
from decoderblock import DecoderBlock

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, num_layer, head, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.position_embedding = tf.keras.layers.Embedding(max_length, embed_size)
        self.head = head
        self.embed_size = embed_size
        self.head_dim = embed_size // head

        self.layer = [
            DecoderBlock(embed_size, head, forward_expansion, dropout) for _ in range(num_layer)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc_out = tf.keras.layers.Dense(120, activation='linear')
        self.softmax = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs, enc_out, src_mask, trg_mask):
        N, seq_length = inputs.shape
        position = tf.broadcast_to(tf.range(0, seq_length), (N, seq_length))
        x = self.dropout(self.word_embedding(inputs)+self.position_embedding(position))

        for layer in self.layer:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        output = self.fc_out(x)
        output = self.softmax(output)

        return output
