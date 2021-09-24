import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, head):
        super(SelfAttention, self).__init__()
        self.head = head
        self.embed_size = embed_size
        self.head_dim = embed_size // head

        assert (self.head_dim * head == embed_size), 'size of head_dim is not matching'

        self.query = tf.keras.layers.Dense(self.head_dim, activation='linear', use_bias=False)
        self.value = tf.keras.layers.Dense(self.head_dim, activation='linear', use_bias=False)
        self.key = tf.keras.layers.Dense(self.head_dim, activation='linear', use_bias=False)
        self.fc_layer = tf.keras.layers.Dense(self.embed_size, activation='linear')

    def call(self, value, key, query, mask):
        # Number of training examples
        N = query.shape[0]
        query_len, value_len, key_len = query.shape[1], value.shape[1], key.shape[1]

        # Reshape according to the number of examples and words
        query = tf.reshape(query, (N, query_len, self.head, self.head_dim))
        value = tf.reshape(value, (N, value_len, self.head, self.head_dim))
        key = tf.reshape(key, (N, key_len, self.head, self.head_dim))

        query = self.query(query)
        value = self.value(value)
        key = self.key(key)

        # energy shape: (N, head, query_len, key_len) try to imagine the shape in mind
        energy = tf.einsum("nqhd, nkhd->nhqk", query, key)

        if mask is not None:
            energy = energy * mask
            energy = tf.where(tf.equal(energy, 0), -1e20, energy)

        attention = tf.keras.activations.softmax(energy, axis=-1)
        # attention shape: (N, head, query_len, key_len)
        # value shape:(N, value_len, head, head_dim)
        # output: (N, query_len, head, head_dim)
        output = tf.reshape(tf.einsum("nhql, nlhd->nqhd", attention, value), (N, query_len, self.head*self.head_dim))

        output = tf.keras.activations.linear(output)

        return output





        





