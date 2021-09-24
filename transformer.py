import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256
                 , num_layer=6, forward_expansion=4, head=8, dropout=0.9, device='cuda', enc_max_length=100, dec_max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(vocab_size, embed_size, num_layer, head, device, forward_expansion, dropout, enc_max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layer, head, forward_expansion, dropout, device, dec_max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.head = head

    def make_src_mask(self, src):
        src = src.numpy()
        src_mask = (src != self.src_pad_idx) * 1.000
        src_mask = np.expand_dims(src_mask, axis=1)
        src_mask = np.expand_dims(src_mask, axis=1)
        return src_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = np.broadcast_to(np.tril(np.ones((trg_len, trg_len))), (N, self.head, trg_len, trg_len))
        return trg_mask

    def call(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, src_mask, trg_mask)

        return output


if __name__ == "__main__":
    device = tf.device('cuda')

    x = tf.constant([[1, 5, 6, 4, 3, 9, 1, 0, 0], [1, 8, 7, 3, 4, 5, 0, 0, 0]])
    trg = tf.constant([[1, 7, 4, 3, 5, 9, 2], [1, 5, 6, 2, 4, 7, 6]])

    src_pad_idx = 0
    trg_pad_idx = 1
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx)

    print('target max: ', model.make_trg_mask(trg).shape)
    print('source mask: ', model.make_src_mask(x).shape)

    output = model(x, trg)
    print(output.shape)
