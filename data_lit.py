import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PreProcessing:
    def __init__(self, path):
        self.path = path

    def Load(self):
        '''
        data = pd.read_csv(self.path)
        english = data.iloc[:64, 1]
        hindi = data.iloc[:64, 2]
        return english, hindi
        '''
        file = open(self.path, 'r').read()
        lists = [f.split('\t') for f in file.split('\n')]

        questions = [x[0] for x in lists]
        answers = [x[1] for x in lists]

        return questions, answers

    def Filter_sentence(self, line, language):
        '''
        if language == 'english':
            text = text.lower().strip()
        else:
            text = text.strip()

        text = re.sub(r"([?.!,¿()])", r" \1 ", text)
        text = '<start> ' + text + ' <end>'

        return text
        '''
        line = line.lower().strip()

        line = re.sub(r"([?.!,¿()])", r" \1 ", line)  # create the space between words and [?.!,¿] these signs
        # line = re.sub(r'[" "]+', " ", line) # remove the extra space between the words
        line = re.sub(r"[^a-zA-Z?.!,¿()]+", " ",
                      line)  # allow only alphabets and [?.!,¿] these symbols or remove the digits.
        line = line.strip()
        line = '<start> ' + line + ' <end>'  # join the <start> and <end> tags at both ends of the sentence.

        return line


    def WordToVec(self, text):
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(text)

        token = tokenizer.texts_to_sequences(text)
        token = pad_sequences(token, padding='post')

        return token, tokenizer

if __name__ == '__main__':
    print('Preprocessing')