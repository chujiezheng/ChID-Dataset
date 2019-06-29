# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import random
import re
import jieba
import time
from utils import Vocabulary

random.seed(time.time())


class DataManager:
    def __init__(self):
        self.vocab = Vocabulary()
        self.ans = {}
        for line in open("../data/train_answer.csv"):
            line = line.strip().split(',')
            self.ans[line[0]] = int(line[1])

        print("*** Finish building vocabulary")


    def get_num(self):
        num_word, num_idiom = len(self.vocab.id2word) - 2, len(self.vocab.id2idiom) - 1
        print("Numbers of words and idioms: %d %d" % (num_word, num_idiom))
        return num_word, num_idiom


    def _prepare_data(self, temp_data):
        cans = temp_data["candidates"]
        cans = [self.vocab.tran2id(each, True) for each in cans]

        for text in temp_data["content"]:
            content = re.split(r'(#idiom\d+#)', text)

            doc = []
            loc = []
            labs = []
            tags = []

            for i, segment in enumerate(content):
                if re.match(r'#idiom\d+#', segment) is not None:
                    tags.append(segment)
                    if segment in self.ans:
                        labs.append(self.ans[segment])
                    loc.append(len(doc))
                    doc.append(self.vocab.tran2id('#idiom#'))
                else:
                    doc += [self.vocab.tran2id(each) for each in jieba.lcut(segment)]

            yield doc, cans, labs, loc, tags


    def train(self, dev=False):
        if dev:
            file = open("../data/train.txt")
            lines = file.readlines()[:10000]
        else:
            file = open("../data/train.txt")
            lines = file.readlines()[10000:]
            random.shuffle(lines)
        for line in lines:
            temp_data = eval(line)
            for doc, cans, labs, loc, tags in self._prepare_data(temp_data):
                yield doc, cans, labs, loc, tags


    def test(self, file):
        for line in open(file):
            temp_data = eval(line)
            for doc, cans, _, loc, tags in self._prepare_data(temp_data):
                yield doc, cans, loc, tags


    def get_embed_matrix(self):  # DataManager
        np.random.seed(37)
        def embed_matrix(file, dic, dim=200):
            fr = open(file, encoding="utf8")
            wv = {}
            for line in fr:
                vec = line.split(" ")
                word = vec[0]
                if word in dic:
                    vec = [float(value) for value in vec[1:]]
                    assert len(vec) == dim
                    wv[dic[word]] = vec
                    # which indicates the order filling in wv is the same as id2idiom/id2word

            lost_cnt = 0
            matrix = []
            for i in range(len(dic)):
                if i in wv:
                    matrix.append(wv[i])
                else:
                    lost_cnt += 1
                    matrix.append(np.random.uniform(-0.1, 0.1, [dim]))

            return matrix, lost_cnt

        if os.path.exists("newWordvector.txt"):
            self.word_embed_matrix, lost_word = embed_matrix("newWordvector.txt", self.vocab.word2id)
        else:
            self.word_embed_matrix = np.random.rand(len(self.vocab.word2id), 200)
            lost_word = len(self.vocab.word2id)

        if os.path.exists("newIdiomvector.txt"):
            self.idiom_embed_matrix, lost_idiom = embed_matrix("newIdiomvector.txt", self.vocab.idiom2id)
        else:
            self.idiom_embed_matrix = np.random.rand(len(self.vocab.idiom2id), 200)
            lost_idiom = len(self.vocab.idiom2id)

        self.word_embed_matrix = np.array(self.word_embed_matrix, dtype=np.float32)
        self.idiom_embed_matrix = np.array(self.idiom_embed_matrix, dtype=np.float32)
        print("*** %d idioms and %d words not found" % (lost_idiom, lost_word))

        print("*** Embed matrixs built")
        return self.word_embed_matrix, self.idiom_embed_matrix
