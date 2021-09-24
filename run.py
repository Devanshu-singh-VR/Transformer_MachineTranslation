import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import numpy as np
import tensorflow as tf
from transformer import Transformer
from data_lit import PreProcessing
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = PreProcessing('hello')

''' Load the tokenizer data '''
with open('train_tokenizer.pickle', 'rb') as handle: # train_tokenizer path
    eng_token = pickle.load(handle)

with open('label_tokenizer.pickle', 'rb') as handle: # label_tokenizer path
    hindi_token = pickle.load(handle)

''' Lode the maximum training feature length '''
file = open('max_data.txt', 'r') # max_data path
fil = open('max_len_eng.txt', 'r')
fi = open('max_len_hindi.txt', 'r')

max_len, max_eng, max_hindi = int(file.read()), int(fil.read()), int(fi.read())

src_pad_idx = 0
trg_pad_idx = 1

Eng_vocab_size = len(eng_token.word_index)+1
hindi_vocab_size = len(hindi_token.word_index)+1

model = Transformer(vocab_size=Eng_vocab_size, trg_vocab_size=hindi_vocab_size, src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, enc_max_length=max_eng, dec_max_length=max_hindi)

model.load_weights('transformer/')

while True:
    test = str(input('question: '))
    test = data.Filter_sentence(test, 'english')

    whole = [] # collect the " " split sentence words
    for i in test.split(' '):
        # throw an exception if user input word not present in the vocabulary of the train data
        whole.append(eng_token.word_index[i])

    sentence = pad_sequences([whole], maxlen=max_eng, padding='post')
    test = tf.convert_to_tensor(sentence)

    start = np.ones((1,1))

    answer = ''
    for i in range(max_hindi):
        pred = model(test, start)
        pred = np.argmax(pred[0][i])

        answer += hindi_token.index_word[pred] + ' '

        if hindi_token.index_word[pred] == '<end>':
            break

        start = np.hstack((start, np.array(pred).reshape(-1, 1)))

    print('answer: ', answer)