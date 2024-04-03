from collections import Counter
import pickle
import json
import nltk
import numpy as np
from tqdm import tqdm

def preprocess_charades(glove):
    vocab = {'w2id': dict(), 'id2vec': [], 'counter': Counter()}
    with open('data/charades/train.json') as fp:
        train = json.load(fp)
    with open('data/charades/test.json') as fp:
        test = json.load(fp)

    vocab['w2id']['<PAD>'] = 0
    vocab['id2vec'].append(np.zeros(300, dtype=np.float32))
    for data in train + test:
        sentence = data[3]
        for word in nltk.tokenize.word_tokenize(sentence):
            word = word.lower()
            if word not in glove:
                continue
            if word not in vocab['w2id']:
                vocab['w2id'][word] = len(vocab['id2vec'])
                vocab['id2vec'].append(glove[word])
            vocab['counter'][word] += 1

    with open('data/charades/glove.pkl', 'wb') as fp:
        pickle.dump(vocab, fp)


if __name__ == '__main__':
    with open('glove.840B.300d.txt') as fp:
        lines = fp.readlines()
    glove = dict()
    for line in tqdm(lines):
        line = line.split()
        word, vec = ' '.join(line[:-300]), list(map(float, line[-300:]))
        glove[word] = np.array(vec).astype(np.float32)
    preprocess_charades(glove)
