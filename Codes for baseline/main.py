import os
import numpy as np
import random
import json
import time
from DataManager import DataManager
from utils import caculate_acc
import tensorflow as tf
from Flags import get_flags


FLAGS = get_flags()
if not "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


if FLAGS.model == "ar":
    from Models.AR import Model
elif FLAGS.model == "lm":
    from Models.LM import Model
elif FLAGS.model == "sar":
    from Models.SAR import Model
else:
    raise EOFError

train_dir = FLAGS.train_dir + "/" + FLAGS.model
if not os.path.exists(FLAGS.train_dir):
    os.mkdir(FLAGS.train_dir)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)


dataManager = DataManager()
num_words, num_idioms = dataManager.get_num()
word_embed_matrix, idiom_embed_matrix = dataManager.get_embed_matrix()



def prepare_batch_data(document, candidates, ori_labels, ori_locs):
    # padding docs
    batch_size = len(document)
    doc_length = [len(doc) for doc in document]
    max_length = max(doc_length)
    mask = np.zeros((batch_size, max_length), dtype=np.float32)
    for i in range(batch_size):
        document[i] = document[i] + [0] * (max_length - doc_length[i])
        mask[i, :doc_length[i]] = 1
    document = np.array(document, dtype=np.int32)
    doc_length = np.array(doc_length, dtype=np.int32)

    # candidates
    num_choices = len(candidates[0][0])

    max_num_labels = max([len(each) for each in candidates])
    assert max_num_labels == max([len(each) for each in ori_labels])
    for i in range(batch_size):
        candidates[i] += [[0 for _ in range(num_choices)]] * (max_num_labels - len(candidates[i]))
    candidates = np.array(candidates, dtype=np.int32)

    # process labels and locs
    labels = np.zeros((batch_size, max_num_labels, num_choices), dtype=np.int32)
    for id in range(batch_size):
        temp_labels = ori_labels[id]
        for i in range(len(temp_labels)):
            for j in range(num_choices):
                #labels[id, i, j] = -1 / (num_choices - 1) if j != temp_labels[i] else 1
                labels[id, i, j] = -1 if j != temp_labels[i] else 1

    locs = np.zeros((batch_size, max_num_labels, max_length), dtype=np.int32)
    for id in range(batch_size):
        temp_locs = ori_locs[id]
        for i in range(len(temp_locs)):
            locs[id, i, temp_locs[i]] = 1

    return document, doc_length, candidates, labels, locs, ori_labels, mask


def valid(sess, model, mode="dev", record=False):
    if record:
        file = open("pred_result_" + FLAGS.model + "_" + mode + ".txt", "w")

    total_data = dataManager.valid(mode)
    acc_array = np.zeros((2), dtype=np.float32)
    count, total_loss, cnt = 0, 0., 0
    document, candidates, ori_labels, ori_locs = [], [], [], []

    acc_blank = np.zeros((2, 2), dtype=np.float32)

    for doc, can, lab, loc in total_data:
        document.append(doc)
        candidates.append(can)
        ori_labels.append(lab)
        ori_locs.append(loc)
        cnt += 1

        if cnt == FLAGS.batch_size * 2:
            document, doc_length, candidates, labels, locs, ori_labels, mask = \
                prepare_batch_data(document, candidates, ori_labels, ori_locs)

            valid_result = model.train_step(sess, document, doc_length, candidates, labels, locs, False, mask)
            count += 1
            total_loss += valid_result[0]
            caculate_result = caculate_acc(ori_labels, valid_result[1])
            acc_array += caculate_result[0]
            acc_blank += caculate_result[1]

            if record:
                for id in range(len(ori_labels)):
                    ori_label = ori_labels[id]
                    pred_result = valid_result[1][id]

                    for pred in pred_result[: len(ori_label)]:
                        file.write(str(int(pred)) + " ")
                    file.write("\n")

            document, candidates, ori_labels, ori_locs = [], [], [], []
            cnt = 0

    if cnt > 0:
        document, doc_length, candidates, labels, locs, ori_labels, mask = \
            prepare_batch_data(document, candidates, ori_labels, ori_locs)

        valid_result = model.train_step(sess, document, doc_length, candidates, labels, locs, False, mask)
        count += 1
        total_loss += valid_result[0]
        caculate_result = caculate_acc(ori_labels, valid_result[1])
        acc_array += caculate_result[0]
        acc_blank += caculate_result[1]

        if record:
            for id in range(len(ori_labels)):
                ori_label = ori_labels[id]
                pred_result = valid_result[1][id]

                for pred in pred_result[: len(ori_label)]:
                    file.write(str(int(pred)) + " ")
                file.write("\n")


    if record:
        file.close()

    acc = acc_array[0] / acc_array[1]
    acc_single = acc_blank[0, 0] / (acc_blank[0, 1] + 1e-12)
    acc_multi = acc_blank[1, 0] / (acc_blank[1, 1] + 1e-12)

    avg_loss = total_loss / count

    print(
        "*** " + mode.upper() + "  acc %.5f  loss %.3f  single_acc %.5f  multi_acc %.5f" % (acc, avg_loss, acc_single, acc_multi))

    return acc, avg_loss


def train(sess, model):
    #if not os.path.exists("train_process"):
    #    os.mkdir("train_process")
    #f = open("train_process/" + FLAGS.model + ".txt", "w")

    acc_array = np.zeros((2), dtype=np.float32)
    st_time, count, total_loss, cnt = time.time(), 0, 0., 0

    if tf.train.get_checkpoint_state(train_dir):
        best_dev_acc = valid(sess, model, "dev")[0]
    else:
        best_dev_acc = 0.1
    prev_loss = [1e15, 1e15]

    document, candidates, ori_labels, ori_locs = [], [], [], []
    for _ in range(10):
        total_data = dataManager.train()

        for doc, can, lab, loc in total_data:
            document.append(doc)
            candidates.append(can)
            ori_labels.append(lab)
            ori_locs.append(loc)
            cnt += 1

            if cnt == FLAGS.batch_size:
                document, doc_length, candidates, labels, locs, ori_labels, mask = \
                    prepare_batch_data(document, candidates, ori_labels, ori_locs)

                train_result = model.train_step(sess, document, doc_length, candidates, labels, locs, True, mask)
                total_loss += train_result[0]
                acc_array += caculate_acc(ori_labels, train_result[1])[0]
                count += 1

                if model.global_step.eval() % 100 == 0:
                    temp_avg_time = (time.time() - st_time) / count
                    temp_avg_loss = total_loss / count
                    acc = acc_array[0] / acc_array[1]
                    record_str = "step %d  lr %.6f  acc %.4f  time %.3f  loss %.3f" % \
                                 (model.global_step.eval(), model.lr.eval(), acc, temp_avg_time, temp_avg_loss)
                    print(record_str)

                    #f.write(record_str + "\n")
                    #acc_list.append(acc)

                    if model.global_step.eval() % 1000 == 0:
                        dev_acc, dev_loss = valid(sess, model, "dev")
                        if dev_acc > best_dev_acc:
                            model.saver.save(sess, "%s/checkpoint" % train_dir, global_step=model.global_step)
                            best_dev_acc = dev_acc
                            best_iter = model.global_step.eval()
                            print("**  New best iteration  %d" % best_iter)
                        print("*   Best iteration  %d" % best_iter)

                        if dev_loss > max(prev_loss):
                            sess.run(model.lr_decay)
                        prev_loss = [prev_loss[1], dev_loss]

                    acc_array = np.zeros((2), dtype=np.float32)
                    st_time, count, total_loss = time.time(), 0, 0.

                document, candidates, ori_labels, ori_locs = [], [], [], []
                cnt = 0


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        model = Model(learning_rate=FLAGS.lr, init_word_embed=word_embed_matrix, init_idiom_embed=idiom_embed_matrix)

        if FLAGS.is_train:
            model.print_parameters()

            if tf.train.get_checkpoint_state(train_dir):
                print("Reading model parameters from %s" % train_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            else:
                print("Created model with fresh parameters.")
                tf.global_variables_initializer().run()

            train(sess, model)

        else:
            model_path = tf.train.latest_checkpoint(train_dir)
            print("restore from %s" % model_path)
            model.saver.restore(sess, model_path)
            
            valid(sess, model, "dev", True)



if __name__ == "__main__":
    main()