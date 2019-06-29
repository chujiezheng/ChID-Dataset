import numpy as np


class Vocabulary:
    def __init__(self):
        with open("idiomList.txt") as f:
            id2idiom = eval(f.readline())

        self.id2idiom = ["<PAD>"] + id2idiom
        self.idiom2id = {}
        for id, idiom in enumerate(self.id2idiom):
            self.idiom2id[idiom] = id

        with open("wordList.txt") as f:
            id2word = eval(f.readline())

        self.id2word = ["<PAD>", "<UNK>"] + id2word
        self.word2id = {}
        for id, word in enumerate(self.id2word):
            self.word2id[word] = id


    def tran2id(self, token, is_idiom=False):
        if is_idiom:
            return self.idiom2id[token]
        else:
            if token in self.word2id:
                return self.word2id[token]
            else:
                return self.word2id["<UNK>"]



def caculate_acc(original_labels, pred_labels):
    """
    :param original_labels: look like [[list1], [list2], ...], num of list == batch size
        length of each list is not determined, for example, it may be 3, 4, 6
    :param pred_labels: [[pred_list1], [pred_list2], ...]
        length of each pred_list is padding to 10, we just care the first several ones
    :return: an array, looks like
    """

    acc_blank = np.zeros((2, 2), dtype=np.float32)
    acc_array = np.zeros((2), dtype=np.float32)

    for id in range(len(original_labels)): # batch_size
        ori_label = original_labels[id]
        pre_label = list(pred_labels[id])


        x_index = 0 if len(ori_label) == 1 else 1

        for real, pred in zip(ori_label, pre_label):
            acc_array[1] += 1
            acc_blank[x_index, 1] += 1

            if real == pred:
                acc_array[0] += 1
                acc_blank[x_index, 0] += 1

    return acc_array, acc_blank

