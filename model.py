import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pickle
import tensorflow as tf
from data_lit import PreProcessing
from transformer import Transformer

#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=True)
#session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

Path = 'chatbot.txt'
Process = PreProcessing(Path)

english, hindi = Process.Load()

english = [Process.Filter_sentence(str(a), 'english') for a in english]
hindi = [Process.Filter_sentence(str(a), 'hindi') for a in hindi]

english, english_token = Process.WordToVec(english)
hindi, hindi_token = Process.WordToVec(hindi)

''' Save the tokenize data '''
with open('train_tokenizer.pickle', 'wb') as handle:
    pickle.dump(english_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_tokenizer.pickle', 'wb') as handle:
    pickle.dump(hindi_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

length_of_data = len(english)

''' Save the maximum length of training feature'''
file = open('max_data.txt', 'w')
file.write(str(length_of_data))
fil = open('max_len_eng.txt', 'w')
fil.write(str(english.shape[1]))
fi = open('max_len_hindi.txt', 'w')
fi.write(str(hindi.shape[1]))
file.close()
fil.close()
fi.close()

Buffer_size = length_of_data
Batch_size = 64
StepsPerEpochs = length_of_data//Batch_size
vocab_size = len(english_token.word_index)+1 # +1 due to the extra entry of 0 during padding
trg_vocab_size = len(hindi_token.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((english, hindi)).shuffle(Buffer_size)
dataset = dataset.batch(Batch_size, drop_remainder=True)

#print(english.shape)
#print(hindi.shape)

src_pad_idx = 0
trg_pad_idx = 1

model = Transformer(vocab_size=vocab_size, trg_vocab_size=trg_vocab_size, src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, enc_max_length=english.shape[1], dec_max_length=hindi.shape[1])

model.load_weights('transformer/')

EPOCHS = 10

# if we have [0, 0, 1, 0 ,0 ,0 ..] like label vector then we can use categorical_crossentropy, other wise sparse
losses = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.0001)


def loss_function(real, pred):
    mask_local = tf.math.logical_not(tf.math.equal(real, 0))
    loss_fn = losses(real, pred)

    mask = tf.cast(mask_local, dtype=loss_fn.dtype)
    loss_fn *= mask
    return tf.reduce_mean(loss_fn)

for i in range(1, EPOCHS+1):
    total_loss = 0

    for (Batch, (train, label)) in enumerate(dataset.take(StepsPerEpochs)):
        loss = 0

        label_ot = label[:, 1:]
        label = label[:, :-1]

        with tf.GradientTape() as tape:
            output = model(train, label)
            for idx in range(output.shape[1]):
                loss += loss_function(label_ot[:, idx], output[:, idx])

        batch_loss = (loss/ int(label.shape[1]))

        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    total_loss += batch_loss

    print('Epoch:{:3d} Loss:{:.4f}'.format(i, total_loss / StepsPerEpochs))

model.save_weights('transformer/')
