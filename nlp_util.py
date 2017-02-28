from itertools import permutations

import random
import nltk as nltk
import numpy as np
import os.path
from sklearn.decomposition import PCA



def load_glove_model(glove_file):
    print("Loading Glove Model from {}".format(glove_file))
    embeddings = {}

    if os.path.isfile(glove_file):
        with open(glove_file, 'r') as f:
            for line in f:
                row = line.strip().split(' ')
                embeddings[row[0]] = (row[1:])

    print("Loaded {} GloVe rows!".format(len(embeddings)))
    return embeddings


def save_glove_embeddings(vocab, embeddings, file_name):
    print("Saving GloVe vocab to: {}".format(file_name))
    with open(file_name, 'w') as f:
        for word, word_id in vocab.items():
            if word is not "<OOV>" and word is not "<PAD>":
                f.write("{} {}\n".format(word, " ".join(map(str, embeddings[word_id]))))
    print("Saved {} rows for GloVe embeddings".format(len(embeddings)))


class Story:
    def __init__(self, sentences, order):
        self.sentences = sentences
        self.order = order

    def max_sent_len(self):
        return max([len(s) for s in self.sentences])


class Stories:
    def __init__(self, stories=None):
        if stories is None:
            self.stories = []
            self.max_sent_len = 0
        else:
            self.stories = stories
            self.max_sent_len = max([s.max_sent_len() for s in stories])

    def add_story(self, story):
        self.stories.append(story)
        self.max_sent_len = max(self.max_sent_len, story.max_sent_len())

    def randomize_get_batch(self, batch_size):
        random.shuffle(self.stories)
        return Stories(self.stories[0:batch_size])

    def shuffle(self):
        random.shuffle(self.stories)

    def get_batch(self, s, e):
        return Stories(self.stories[s:e])

    def as_np_sents(self, padding_symbol=0):
        np_sentences_matrix = np.full([len(self.stories), 5, self.max_sent_len], padding_symbol, dtype=np.int32)
        for i, story in enumerate(self.stories):
            for j, sent in enumerate(story.sentences):
                np_sentences_matrix[i, j, 0:len(sent)] = sent

        return np_sentences_matrix

    def num_stories(self):
        return len(self.stories)

    def as_np_orders(self):
        return np.array([story.order for story in self.stories], dtype=np.int32)

    def as_np_sent_lens(self):
        np_sentence_lens = np.empty([len(self.stories), 5], dtype=np.int32)
        for i, story in enumerate(self.stories):
            for j, sent in enumerate(story.sentences):
                np_sentence_lens[i, j] = len(sent);

        return np_sentence_lens


class Embeddings:
    def __init__(self, emb):
        self.embeddings = emb

    def as_np_array(self, ndim=None):

        emb_np = np.asarray(self.embeddings, dtype=float)

        if ndim is not None:
            assert ndim <= len(self.embeddings[0])

            pca = PCA(n_components=ndim)
            pca.fit(emb_np)
            return pca.transform(emb_np)
        else:
            return emb_np


def pipeline(data, vocab=None, ext_embeddings=None):
    stories = Stories()
    missing_word_count = 0
    is_ext_vocab = True
    embeddings = None
    if vocab is None:
        is_ext_vocab = False
        vocab = {'<PAD>': 0, '<OOV>': 1}

    for instance in data:
        sents = []
        for sentence in instance['story']:
            sent = []
            for token in nltk.word_tokenize(sentence):
                ltok = token.lower()
                if not is_ext_vocab and ltok not in vocab:
                    vocab[ltok] = len(vocab)

                if ltok not in vocab:
                    token_id = vocab['<OOV>']
                else:
                    token_id = vocab[ltok]
                sent.append(token_id)

            sents.append(sent)
        stories.add_story(Story(sents, instance['order']))

    if ext_embeddings is not None:
        embeddings = [None for x in range(len(vocab))]
        embd_len = len(list(ext_embeddings.values())[0])
        for word, word_id in vocab.items():
            if word == '<PAD>':
                embeddings[word_id] = [0 for x in range(embd_len)]
            elif word == '<OOV>':
                embeddings[word_id] = [1 for x in range(embd_len)]
            else:
                if word in ext_embeddings:
                    embeddings[word_id] = ext_embeddings[word]
                else:
                    embeddings[word_id] = [1 for x in range(embd_len)]
                    missing_word_count += 1

    print("Missing embeddings for {} words".format(missing_word_count))
    return stories, vocab, Embeddings(embeddings)
