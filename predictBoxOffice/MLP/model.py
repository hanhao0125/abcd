import xlrd

import numpy as np
from functools import reduce
import tensorflow as tf
import math
import os
from sklearn import decomposition as decom
import xlwt
from xlutils.copy import copy


def table_to_matrix(table, with_col_row_num=False):  # transform table into matrix encoded with utf-8
    row_num = table.nrows
    data = []
    for i in range(row_num):
        data.append(table.row_values(i))

    def to_utf8(y):
        return y.encode('utf-8') if not isinstance(y, float) and not isinstance(y, int) else y

    data = [[to_utf8(y) for y in x] for x in data]
    return data


def build_dict(mat, i, j):  # gen dict by the ith and jth columns
    _dict = {}
    count = 0
    for ls in mat:
        if count != 0:
            _dict[ls[i]] = ls[j]
        else:
            count = count + 1
    # print _dict.get('罗志祥')
    # print _dict.__len__()
    return _dict


def tran_dir_act_by_rank(data, rank_dict, type_dict, director_col=7, actor_col=8, type_col=9):
    t = 0
    for ls in data:
        if t != 0:
            if rank_dict.get(ls[director_col]) == None:
                ls[director_col] = 6
            else:
                ls[director_col] = rank_dict.get(ls[director_col])  # convert the director into rank

            def add(x, y):
                return x + y if x != None and y != None \
                    else x if x != None and y == None \
                    else y if x == None and y != None \
                    else 0

            # convert the actor and director's#  rank into the cell
            ls[actor_col] = reduce(add, map(rank_dict.get, ls[actor_col].split()))
            # convert the type'rank into the cell
            ls[type_col] = reduce(add, map(type_dict.get, ls[type_col].split())) / float(ls[type_col].split().__len__())
        else:
            t = t + 1
    print(data)


def gen_train_data(data, train_size=190, out_dim=5, use17=False):
    Data = np.asarray(data)
    Data = Data[1:, 1:]  # get rid of names and column names
    Data = np.asarray(Data, np.float32)
    # PCA here
    # a = Data[:, 0:1]
    # b = Data[:, 1:]
    # pca = decom.PCA(n_components=8)
    # pca.fit(b)
    # b = pca.transform(b)
    # Data = np.concatenate((a, b), 1)
    # PCA here end
    dim = Data[0].__len__()
    for j in range(dim):  # normalization
        if j != 0:
            min = Data[0][j]
            max = Data[0][j]
            for i in range(Data.__len__()):
                min = Data[i][j] if min > Data[i][j] else min
                max = Data[i][j] if max < Data[i][j] else max
            print(min, max)
            for i in range(Data.__len__()):
                Data[i][j] = (Data[i][j] - min) / (max - min)

    blocks = Data.__len__() / out_dim  # gen labels
    for i in range(Data.__len__()):
        if i / blocks < out_dim - 1:
            Data[i][0] = i / blocks
        else:
            Data[i][0] = out_dim - 1

    if not use17:
        np.random.shuffle(Data)

    pre_mat = []
    for i in range(Data.__len__()):
        pre = np.zeros([out_dim], dtype=np.float32)
        pre[int(Data[i][0])] = 1
        pre_mat.append(pre)
    # for da in Data: #one_hot label
    #     pre = [0, 0, 0]
    #     if da[0] <= 430000.0 and da[0] > 48000.0:
    #         pre[0] = 1
    #     elif da[0] <= 49000.0 and da[0] > 20000.0:
    #         pre[1] = 1
    #     elif da[0] <= 20000.0 and da[0] > 0.0:
    #         pre[2] = 1
    #     pre_mat.append(pre)
    pre_mat = np.asarray(pre_mat)

    if use17:
        train_data = []
        test_data = []
        test_label = []
        for i in range(Data.__len__()):
            if Data[i][1] == 1:
                test_data.append(Data[i])
                test_label.append(pre_mat[i])
            else:
                train_data.append(Data[i])
        return np.asarray(train_data), np.asarray(test_data)[:, 2:], np.asarray(
            test_label)  # tempary remove month and year
    else:
        train_data = Data[0:train_size, :]  # slice the data into data of train and test
        test_data = Data[train_size:, 1:]
        test_label = pre_mat[train_size:, :]
        return train_data, test_data, test_label


def get_shuffle_batch(Data_, out_dim=5):  # presume the Data_ has not been sliced
    Data = Data_
    pre_mat = []
    for i in range(Data.__len__()):
        pre = np.zeros([out_dim], dtype=np.float32)
        pre[int(Data[i][0])] = 1
        pre_mat.append(pre)
    # for da in Data: #one_hot label
    #     pre = [0, 0, 0]
    #     if da[0] <= 430000.0 and da[0] > 48000.0:
    #         pre[0] = 1
    #     elif da[0] <= 49000.0 and da[0] > 20000.0:
    #         pre[1] = 1
    #     elif da[0] <= 20000.0 and da[0] > 0.0:
    #         pre[2] = 1
    #     pre_mat.append(pre)
    pre_mat = np.asarray(pre_mat)
    return Data[:, 2:], pre_mat  # tempary remove month and year


def cross_validate(y, y_hat):
    def max(ls):
        t = 0
        for i in range(ls.__len__()):
            if ls[i] > ls[t]:
                ls[t] = 0
                t = i
            else:
                if i != 0:
                    ls[i] = 0
        ls[t] = 1
        return ls

    y_ = [max(i) for i in y_hat]
    # print y_
    a = np.zeros((y[0].__len__(), y[0].__len__()))
    for i in range(y.__len__()):
        for j in range(y[i].__len__()):
            if y[i][j] == 1:
                for k in range(y_[i].__len__()):
                    if y_[i][k] == 1:
                        a[j][k] = a[j][k] + 1
    return a


class MLP(object):
    def __init__(self, sess, name="default", input_dim=20, output_dim=10, hidden_depth=8, stddev=0.02,
                 learning_rate=0.008, log_dir='logs', data_dir='box_office.xlsx', scope=None):
        self.sess = sess
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_depth = hidden_depth
        self.stddev = stddev
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.data_dir = data_dir

        self.bo_aver_dict = None
        self.bo_total_dict = None
        self.fan_dict = None
        self.rank_dict = None
        self.type_dict = None

        self.train_data, self.train_label, self.test_data, self.test_label = None, None, None, None
        self.train_step, self.accuracy, self.x, self.y, self.loss, self.y_hat = None, None, None, None, None, None
        self.scope = scope
        self.data = None
        self.output_label = None
        self.ground_truth_label = None
        self.out_pro = None
        self.var = []

    def set_data(self, train_data, train_label, test_data, test_label):
        self.train_label = train_label
        self.train_data = train_data
        self.test_data = test_data
        self.test_label = test_label

    def init2(self):
        self.train_step, self.accuracy, self.x, self.y, self.loss, self.y_hat, \
        self.output_label, self.ground_truth_label, self.out_pro = self.build_MLP()

    def init4(self):
        self.train_step, self.accuracy, self.x, self.y, self.loss, self.y_hat, \
        self.output_label, self.ground_truth_label, self.out_pro = self.build_MLP1()

    def init3(self, train_data, train_label, test_data, test_label):
        self.set_data(train_data, train_label, test_data, test_label)
        self.input_dim = train_data[0].__len__()
        self.train_step, self.accuracy, self.x, self.y, self.loss, self.y_hat, \
        self.output_label, self.ground_truth_label, self.out_pro = self.build_MLP1()

    def init1(self, train_data, train_label, test_data, test_label):
        self.set_data(train_data, train_label, test_data, test_label)
        self.input_dim = train_data[0].__len__()
        self.train_step, self.accuracy, self.x, self.y, self.loss, self.y_hat, \
        self.output_label, self.ground_truth_label, self.out_pro = self.build_MLP()

    def run1(self, steps=20000, name="", is_save=False, is_test=False):
        saver = tf.train.Saver()
        if is_save:
            saver.restore(self.sess,
                          save_path="save/" + self.name + "_" + name + "/" + self.name + "_" + name + ".ckpt")
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, self.scope))
        writer = tf.summary.FileWriter("logs", self.sess.graph)

        max_ = 0.0
        acc_ = 0.0
        pre_train_acc = 0.0
        count_pre = 0
        y_ = None
        output_label = None
        ground_truth_label = None
        out_pro = None
        for i in range(steps):
            if not is_save:
                self.sess.run(self.train_step, feed_dict={self.x: self.train_data, self.y: self.train_label})
            if i % 75 == 0:
                acc, y_, output_label, ground_truth_label, out_pro = self.sess.run(
                    [self.accuracy, self.y_hat, self.output_label, self.ground_truth_label, self.out_pro],
                    feed_dict={self.x: self.test_data, self.y: self.test_label})
                writer.add_summary(self.sess.run(merged, feed_dict={self.x: self.test_data, self.y: self.test_label}),
                                   i)
                acc_ = acc
                max_ = acc if acc > max_ else max_
                train_acc = self.sess.run(self.accuracy, feed_dict={self.x: self.train_data, self.y: self.train_label})
                if pre_train_acc == train_acc:
                    count_pre = count_pre + 1
                    if count_pre > 0:
                        cross = cross_validate(self.test_label, self.sess.run(self.y_hat,
                                                                              feed_dict={self.x: self.test_data,
                                                                                         self.y: self.test_label}))
                        temp1 = np.concatenate((cross, np.zeros([1, cross.__len__()])), 0)
                        cr = np.concatenate((temp1, np.zeros([cross.__len__() + 1, 1])), 1)
                        for k in range(cross.__len__()):
                            c = 0
                            for j in range(cross[0].__len__()):
                                c = cross[k][j] + c
                            cr[k][cross[k].__len__()] = round(cross[k][k] / c, 2)
                        for k in range(cross[0].__len__()):
                            c = 0
                            for j in range(cross.__len__()):
                                c = cross[j][k] + c
                            cr[cross[k].__len__()][k] = round(cross[k][k] / c, 2)
                        cr[cr.__len__() - 1][cr.__len__() - 1] = round(acc, 2)
                        break
                else:
                    count_pre = 0
                    pre_train_acc = train_acc
                print("the steps, loss, train_accuracy, test_accuracy and maximum accuracy is :%d, %f, %f, %f, %f" % (
                    i,
                    self.sess.run(self.loss, feed_dict={self.x: self.train_data, self.y: self.train_label}),
                    train_acc,
                    acc,
                    max_
                ))
                cross = cross_validate(self.test_label, self.sess.run(self.y_hat,
                                                                      feed_dict={self.x: self.test_data,
                                                                                 self.y: self.test_label}))
                temp1 = np.concatenate((cross, np.zeros([1, cross.__len__()])), 0)
                cr = np.concatenate((temp1, np.zeros([cross.__len__() + 1, 1])), 1)
                for k in range(cross.__len__()):
                    c = 0
                    for j in range(cross[0].__len__()):
                        c = cross[k][j] + c
                    cr[k][cross[k].__len__()] = round(cross[k][k] / c, 2)
                for k in range(cross[0].__len__()):
                    c = 0
                    for j in range(cross.__len__()):
                        c = cross[j][k] + c
                    cr[cross[k].__len__()][k] = round(cross[k][k] / c, 2)
                cr[cr.__len__() - 1][cr.__len__() - 1] = round(acc, 2)
                print(cr)
        if not is_save and not is_test:
            model_dir = "save/" + self.name + "_" + name
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            model_name = self.name + "_" + name + ".ckpt"
            saver.save(self.sess,
                       model_dir + "/" + model_name)  # save the model
        return acc_, y_, output_label, ground_truth_label, out_pro, cr

    def get_data(self, train_size=190, use17=False):
        if not use17:
            data_ = xlrd.open_workbook(self.data_dir, encoding_override='utf-8')
            table = data_.sheet_by_index(0)
            self.data = table_to_matrix(table=table)
            print(np.asarray(self.data).shape)
            bo = table_to_matrix(data_.sheet_by_index(1))
            print(np.asarray(bo).shape)
            fan = table_to_matrix(data_.sheet_by_index(2))
            print(np.asarray(fan).shape)
            rank = table_to_matrix(data_.sheet_by_index(3))
            print(np.asarray(rank).shape)
            type_ = table_to_matrix(data_.sheet_by_index(6))
            print(np.asarray(rank).shape)
            self.bo_aver_dict = build_dict(bo, 0, 3)
            self.bo_total_dict = build_dict(bo, 0, 2)
            self.fan_dict = build_dict(fan, 0, 1)
            self.rank_dict = build_dict(rank, 0, 1)
            self.type_dict = build_dict(type_, 0, 2)
            tran_dir_act_by_rank(data=self.data, rank_dict=self.rank_dict, type_dict=self.type_dict)
            self.train_data_label, self.test_data, self.test_label = gen_train_data(data=self.data,
                                                                                    train_size=train_size,
                                                                                    out_dim=self.output_dim)
            self.input_dim = self.train_data_label[0].__len__() - 1
            self.train_data, self.train_label = get_shuffle_batch(Data_=self.train_data_label,
                                                                  out_dim=self.output_dim)
        else:
            data_ = xlrd.open_workbook(self.data_dir, encoding_override='utf-8')
            table = data_.sheet_by_index(0)
            self.data = table_to_matrix(table=table)
            type_ = table_to_matrix(data_.sheet_by_name("电影类型票房统计"))
            rank = table_to_matrix(data_.sheet_by_name("导演演员评分等级"))
            self.rank_dict = build_dict(rank, 0, 1)
            self.type_dict = build_dict(type_, 0, 2)
            tran_dir_act_by_rank(data=self.data, rank_dict=self.rank_dict, type_dict=self.type_dict,
                                 director_col=11, actor_col=12, type_col=13)
            self.train_data_label, self.test_data, self.test_label = gen_train_data(data=self.data,
                                                                                    train_size=train_size,
                                                                                    out_dim=self.output_dim,
                                                                                    use17=True)
            self.input_dim = self.train_data_label[0].__len__() - 2  # remove month and year
            self.train_data, self.train_label = get_shuffle_batch(Data_=self.train_data_label,
                                                                  out_dim=self.output_dim)

        print("d")

    def build_MLP1(self):
        with tf.name_scope(self.name + "_input_layer"):
            x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name=self.name + "_x")
            x_ = x
            y = tf.placeholder(tf.float32, shape=[None, self.output_dim])

        in_ = self.input_dim
        # self.hidden_depth = 10
        with tf.name_scope(self.name + "_hidden_layer"):
            for i in range(self.hidden_depth - 1):
                hidden_width = self.input_dim if i == 0 else 12
                w = tf.get_variable(
                    name=self.name + "_w_" + str(i),
                    dtype=tf.float32,
                    shape=[hidden_width, 12],
                    initializer=tf.random_normal_initializer(stddev=self.stddev)
                )
                b = tf.get_variable(
                    name=self.name + "_b_" + str(i),
                    dtype=tf.float32,
                    shape=[12],
                    initializer=tf.zeros_initializer()
                )
                x = tf.nn.tanh(tf.matmul(x, w) + b, name=self.name + "_out_" + str(i))
                in_ = round(math.sqrt(in_ * self.output_dim))
            w = tf.get_variable(
                name=self.name + "_w_" + str(self.hidden_depth - 1),
                dtype=tf.float32,
                shape=[12, self.output_dim],
                initializer=tf.random_normal_initializer(stddev=self.stddev)
            )
            b = tf.get_variable(
                name=self.name + "_b_" + str(self.hidden_depth - 1),
                dtype=tf.float32,
                shape=[self.output_dim],
                initializer=tf.zeros_initializer()
            )
            y_hat = tf.nn.softmax(tf.matmul(x, w) + b, name=self.name + "_output")
            out = tf.matmul(x, w) + b
            out_pro = out / tf.reduce_sum(out)
            tf.summary.histogram(self.name + "y_hat", y_hat)

        with tf.name_scope(name=self.name + "_output_layer"):
            loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]),
                                  name=self.name + "_cross_entropy")
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
            tf.summary.scalar(self.name + "_loss", loss)

        with tf.name_scope(name=self.name + "_train_step"):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        with tf.name_scope(name=self.name + "_predict"):
            output_label = tf.argmax(y_hat, 1)
            ground_truth_label = tf.argmax(y, 1)
            predict = tf.equal(ground_truth_label, output_label, name=self.name + "_predict")
            accuracy = tf.reduce_mean(tf.cast(predict, tf.float32), name=self.name + "_accuracy")
            tf.summary.scalar(self.name + "_accuracy", accuracy)
        return train_step, accuracy, x_, y, loss, y_hat, output_label, ground_truth_label, out_pro

    def build_MLP(self):
        with tf.name_scope(self.name + "_input_layer"):
            x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name=self.name + "_x")
            x_ = x
            y = tf.placeholder(tf.float32, shape=[None, self.output_dim])

        in_ = self.input_dim
        # self.hidden_depth = 10
        with tf.name_scope(self.name + "_hidden_layer"):
            for i in range(self.hidden_depth - 1):
                w = tf.get_variable(
                    name=self.name + "_w_" + str(i),
                    dtype=tf.float32,
                    shape=[self.input_dim, self.input_dim],
                    initializer=tf.random_normal_initializer(stddev=self.stddev)
                )
                b = tf.get_variable(
                    name=self.name + "_b_" + str(i),
                    dtype=tf.float32,
                    shape=[self.input_dim],
                    initializer=tf.zeros_initializer()
                )
                self.var.append(w)
                self.var.append(b)
                x = tf.nn.tanh(tf.matmul(x, w) + b, name=self.name + "_out_" + str(i))
                in_ = round(math.sqrt(in_ * self.output_dim))
            w = tf.get_variable(
                name=self.name + "_w_" + str(self.hidden_depth - 1),
                dtype=tf.float32,
                shape=[self.input_dim, self.output_dim],
                initializer=tf.random_normal_initializer(stddev=self.stddev)
            )
            b = tf.get_variable(
                name=self.name + "_b_" + str(self.hidden_depth - 1),
                dtype=tf.float32,
                shape=[self.output_dim],
                initializer=tf.zeros_initializer()
            )
            self.var.append(w)
            self.var.append(b)
            y_hat = tf.nn.softmax(tf.matmul(x, w) + b, name=self.name + "_output")
            out = tf.matmul(x, w) + b
            out_pro = out / tf.reduce_sum(out)
            tf.summary.histogram(self.name + "y_hat", y_hat)

        with tf.name_scope(name=self.name + "_output_layer"):
            loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]),
                                  name=self.name + "_cross_entropy")
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
            tf.summary.scalar(self.name + "_loss", loss)

        with tf.name_scope(name=self.name + "_train_step"):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        with tf.name_scope(name=self.name + "_predict"):
            output_label = tf.argmax(y_hat, 1)
            ground_truth_label = tf.argmax(y, 1)
            predict = tf.equal(ground_truth_label, output_label, name=self.name + "_predict")
            accuracy = tf.reduce_mean(tf.cast(predict, tf.float32), name=self.name + "_accuracy")
            tf.summary.scalar(self.name + "_accuracy", accuracy)
        return train_step, accuracy, x_, y, loss, y_hat, output_label, ground_truth_label, out_pro

    def init(self, use17=False):
        if use17:
            self.get_data(use17=use17, train_size=235)
        else:
            self.get_data(use17=use17)
        self.train_step, self.accuracy, self.x, self.y, self.loss, \
        self.y_hat, self.output_label, self.ground_truth_label, self.out_pro = self.build_MLP()

    def run(self, steps=20000, init_=True, name="", train_size=190, use17=False):
        if init_:
            self.init()
        self.train_data_label, self.test_data, self.test_label = gen_train_data(data=self.data,
                                                                                train_size=train_size,
                                                                                out_dim=self.output_dim, use17=use17)
        self.train_data, self.train_label = get_shuffle_batch(Data_=self.train_data_label,
                                                              out_dim=self.output_dim)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, self.scope))
        writer = tf.summary.FileWriter("logs", self.sess.graph)
        saver = tf.train.Saver()

        max_ = 0.0
        acc_ = 0.0
        pre_train_acc = 0.0
        count_pre = 0
        y_ = None
        for i in range(steps):
            self.sess.run(self.train_step, feed_dict={self.x: self.train_data, self.y: self.train_label})
            if i % 50 == 0:
                # result = self.sess.run(merged, feed_dict={self.x: self.train_data, self.y: self.train_label})
                # writer.add_summary(result, i)
                acc = self.sess.run(self.accuracy, feed_dict={self.x: self.test_data, self.y: self.test_label})
                y_ = self.sess.run(self.y_hat, feed_dict={self.x: self.test_data, self.y: self.test_label})
                writer.add_summary(self.sess.run(merged, feed_dict={self.x: self.test_data, self.y: self.test_label}),
                                   i)
                acc_ = acc
                if acc > max_:
                    max_ = acc
                    if not os.path.exists("models/" + self.name + "/" + name):
                        os.mkdir("models/" + self.name + "/" + name)
                    saver.save(self.sess, "models/" + self.name + "/" + name + "/" + "steps_" + str(i))
                train_acc = self.sess.run(self.accuracy, feed_dict={self.x: self.train_data, self.y: self.train_label})
                if pre_train_acc == train_acc:
                    count_pre = count_pre + 1
                    if count_pre > 0:
                        break
                else:
                    count_pre = 0
                    pre_train_acc = train_acc

                print("the steps, loss, train_accuracy, test_accuracy and maximum accuracy is :%d, %f, %f, %f, %f" % (
                    i,
                    self.sess.run(self.loss, feed_dict={self.x: self.train_data, self.y: self.train_label}),
                    train_acc,
                    acc,
                    max_
                ))
                cross = cross_validate(self.test_label, self.sess.run(self.y_hat,
                                                                      feed_dict={self.x: self.test_data,
                                                                                 self.y: self.test_label}))
                temp1 = np.concatenate((cross, np.zeros([1, cross.__len__()])), 0)
                cr = np.concatenate((temp1, np.zeros([cross.__len__() + 1, 1])), 1)

                for k in range(cross.__len__()):
                    c = 0
                    for j in range(cross[0].__len__()):
                        c = cross[k][j] + c
                    cr[k][cross[k].__len__()] = round(cross[k][k] / c, 2)
                for k in range(cross[0].__len__()):
                    c = 0
                    for j in range(cross.__len__()):
                        c = cross[j][k] + c
                    cr[cross[k].__len__()][k] = round(cross[k][k] / c, 2)
                cr[cr.__len__() - 1][cr.__len__() - 1] = round(acc, 2)
                print(cr)

        return acc_, y_


def add(x_, y):
    return x_ + y


def mean_accuracy():
    acc_category = []
    for j in range(9):
        with tf.name_scope("model_" + str(j + 2)) as scope:
            model = MLP(sess=tf.Session(config=config), name="model_" + str(j + 2), output_dim=j + 2, scope=scope)
            model.init()
            if not os.path.exists("models/" + "model_" + str(j + 2)):
                os.mkdir("models/" + "model_" + str(j + 2))
            acc = []
            if j == 1:
                print(j)
            for i in range(20):
                acc_, y_ = model.run(20000, init_=False, name=str(i))
                acc.append(acc_)
            print(acc)

            acc_mean = reduce(add, acc) / acc.__len__()
            acc_category.append(acc_mean)
    print(acc_category)
    x = tf.placeholder(tf.float32, name="category")
    tf.summary.scalar("category", x)
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loges", sess.graph)
    for i in range(acc_category.__len__()):
        result = sess.run(merged, feed_dict={x: acc_category[i]})
        writer.add_summary(result, i)


def train_size_descent():
    with tf.name_scope("train_size_test") as scope:
        model = MLP(tf.Session(config=config), name="train_size_test", output_dim=5, scope=scope)
        model.init()
        if not os.path.exists("models/train_size_test"):
            os.mkdir("models/train_size_test")
        acc = []

        for i in range(2):
            a = []
            for j in range(20):
                train_size = 210 + i * 10
                acc_, y_ = model.run(20000, False, train_size=train_size, name="train_size_" + str(train_size))
                a.append(acc_)
            acc.append(reduce(add, a) / a.__len__())
        print(acc)


def print_to_board():
    ac = [0.87888898849487307, 0.81666669845581052, 0.68222227096557619, 0.6077778816223145, 0.57000002861022947,
          0.46000003814697266, 0.43444447517395018, 0.41000003814697267, 0.37111117839813235]
    x = tf.placeholder(tf.float32, name="category")
    tf.summary.scalar("category", x)
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loges", sess.graph)
    for i in range(ac.__len__()):
        result = sess.run(merged, feed_dict={x: ac[i]})
        writer.add_summary(result, round(i + 2))


def print_to_board1():
    ac = [0.6100001335144043, 0.63200006484985349, 0.5985715389251709, 0.60000004768371584, 0.62090911865234377,
          0.59615383148193357, 0.59400005340576167, 0.60294117927551272, 0.61526327133178715, 0.60047626495361328,
          0.60739126205444338, 0.59520001411437984, 0.55111107826232908, 0.54931030273437498, 0.55419359207153318,
          0.50484848022460938, 0.50885710716247556]
    x = tf.placeholder(tf.float32, name="decrease_train_size")
    tf.summary.scalar("decrease_train_size", x)
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loges", sess.graph)
    for i in range(ac.__len__()):
        result = sess.run(merged, feed_dict={x: ac[i]})
        writer.add_summary(result, round(220 - 10 * i))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def result():
    with tf.name_scope("mnt") as scope:
        model = MLP(sess=tf.Session(config=config), name="model_", output_dim=5, scope=scope, data_dir="17.xlsx")
        model.init(use17=True)
        ac = []
        for i in range(1):
            acc_, y_ = model.run(name="haha", init_=False, train_size=235, use17=True)
            ac.append(acc_)
            work = xlwt.Workbook()
            sheet = work.add_sheet("result")
            for k in range(y_.__len__()):
                for j in range(y_[k].__len__()):
                    y_[k][j] = round(y_[k][j], 2)
                    sheet.write(k, j, float(y_[k][j]))
            work.save("result.xlsx")
            print(y_)
        print(reduce(add, ac) / ac.__len__())
    a = [0.55555558, 0.33333334, 0.33333334, 0.5, 0.22222222, 0.22222222, 0.3888889, 0.22222222, 0.33333334, 0.33333334,
         0.27777779, 0.11111111, 0.22222222, 0.16666667, 0.33333334, 0.27777779, 0.3888889, 0.22222222, 0.33333334,
         0.27777779]

    print(reduce(add, a) / a.__len__())


def add_new_feature():
    movie_dict = build_dict(
        table_to_matrix(xlrd.open_workbook("haha.xls", encoding_override="utf-8").sheet_by_index(0)), 0, 9)

    rb = xlrd.open_workbook("17.xlsx", encoding_override="utf-8")
    rs = rb.sheet_by_index(0)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    data_set = table_to_matrix(rs)
    for i in range(1, data_set.__len__()):
        ws.write(i, 16, movie_dict.get(data_set[i][0]))
    wb.save("17s.xlsx")


# model.run(20000)
if __name__ == "__main__":
    with tf.name_scope("asdf") as scope:
        model = MLP(name="17plusc", sess=tf.Session(config=config), output_dim=5, scope=scope,
                    data_dir="17s (copy).xlsx")
        if not os.path.exists("models/17plusc"):
            os.mkdir("models/17plusc")
        model.init(use17=True)
        ac = []
        for i in range(10):
            acc, _ = model.run(name="you are shock!!", train_size=235, use17=True, init_=False)
            ac.append(acc)
        print(reduce(add, ac) / ac.__len__())
# print(str(table_to_matrix(rb.sheet_by_index(0))[0][0], encoding="utf-8") == "电影名称")
