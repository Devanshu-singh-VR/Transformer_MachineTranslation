from selfattention import SelfAttention
import tensorflow as tf

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, head, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, head)
        #self.norm1 = tf.keras.layers.LayerNormalization()
        #self.norm2 = tf.keras.layers.LayerNormalization()

        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_size, activation='linear'),
            tf.keras.layers.Dense(forward_expansion*embed_size, activation='linear'),
            tf.keras.layers.Dense(embed_size, activation='linear')
        ])
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # this one is skip connection
        x = self.dropout(attention+query)

        forward = self.feed_forward(x)
        output= self.dropout(forward+x)

        return output