import sqlite3 as sql
import numpy as np
from sklearn import preprocessing as pre
import tensorflow as tf
import os
import pickle
import xlwt
import sys
import model as mod
# todo this is totally fuck, try to make neural_network as a module, but it did't work
abs_path = '/home/hanhao/abc/predictBoxOffice/neural_network'


def prepare_data(out_dim, gen_label=False, static_box=False, prepare_features=False, covs=None):
    genre_dict = build_genre()
    conn = sql.connect(database="/mnt/new_train.db")
    cur = conn.cursor()
    year = 0
    if gen_label:
        sql_str = "select director,actor," if prepare_features else "select name, director,actor,genre,"
        index = 0
        for i in covs:
            feat = i[0].decode("UTF-8")
            if feat == "year":
                year = index + 3
            if feat == "competitio":
                sql_str = sql_str + "competition,"
                index = index + 1
            elif feat == "screen_3da":
                sql_str = sql_str + "screen_3day,"
                index = index + 1
            elif feat == "screen_30d":
                sql_str = sql_str + "screen_30day,"
                index = index + 1
            elif feat != "actor" and feat != "director" and feat != "budget" and feat != "location":
                index = index + 1
                sql_str = sql_str + i[0].decode("UTF-8") + ","
        sql_str = sql_str + "box_office from train_movie where year>2012 order by box_office desc"
        print(sql_str)
        cur.execute(sql_str)
    else:
        cur.execute(
            "select director,actor,year,month, location,search,holiday,competition,budget,screen_3day,screen_30day,"
            "box_office from train_movie where year>2012 order by box_office desc")  # need to specify the sql carefully
    raw_data_ = np.asarray(cur.fetchall())
    raw_data = raw_data_ if prepare_features else raw_data_[:, 1:]
    # print(raw_data.shape)
    blocks = raw_data.__len__() / out_dim
    pre_mat = []
    fo = open("actor.data", "rb")
    map_t = pickle.load(fo)
    fuck = ["director", "actor"]
    for row, k in zip(raw_data, range(raw_data.__len__())):
        if static_box:  # the section of the classification is static
            # (such as, sections range from 0 to 10000 and 10000 to 20000)
            pre_ = np.zeros([10], dtype=np.float32)
            if float(row[-1]) < 5000.0:
                pre_[0] = 1
            elif 5000.0 <= float(row[-1]) < 10000.0:
                pre_[1] = 1
            elif 10000.0 <= float(row[-1]) < 15000.0:
                pre_[2] = 1
            elif 15000.0 <= float(row[-1]) < 20000.0:
                pre_[3] = 1
            elif 20000.0 <= float(row[-1]) < 30000.0:
                pre_[4] = 1
            elif 30000.0 <= float(row[-1]) < 40000.0:
                pre_[5] = 1
            elif 40000.0 <= float(row[-1]) < 50000.0:
                pre_[6] = 1
            elif 50000.0 <= float(row[-1]) < 100000.0:
                pre_[7] = 1
            elif 100000.0 <= float(row[-1]) < 200000.0:
                pre_[8] = 1
            elif 200000.0 <= float(row[-1]):
                pre_[9] = 1

            # if float(row[-1]) < 10000.0:
            #     pre_[0] = 1
            # elif 10000.0 <= float(row[-1]) < 20000.0:
            #     pre_[1] = 1
            # elif 20000.0 <= float(row[-1]) < 30000.0:
            #     pre_[2] = 1
            # elif 30000.0 <= float(row[-1]) < 40000.0:
            #     pre_[3] = 1
            # elif 40000.0 <= float(row[-1]) < 50000.0:
            #     pre_[4] = 1
            # elif 50000.0 <= float(row[-1]) < 100000.0:
            #     pre_[5] = 1
            # elif 100000.0 <= float(row[-1]) < 200000.0:
            #     pre_[6] = 1
            # elif 200000.0 <= float(row[-1]):
            #     pre_[7] = 1

            # pos = round(float(row[-1]) / 10000)
            # if pos < 9:
            #     pre_[round(float(row[-1]) / 10000)] = 1
            # else:
            #     pre_[9] = 1
            pre_mat.append(pre_)
        else:  # non-static section
            if k / blocks < out_dim - 1:  # gen labelsll
                row[-1] = round(k / blocks)
            else:
                row[-1] = out_dim - 1
            pre_ = np.zeros([out_dim], dtype=np.float32)
            pre_[int(row[-1])] = 1
            pre_mat.append(pre_)

        for i in range(0, 2):  # transform the director and actor into numerous variable
            name_list = row[i].split(",")
            l = []
            for j in name_list:
                lists = map_t.get(j).get(fuck[i])
                l.append(np.mean(lists))
            row[i] = np.mean(l)
            # name_list = row[i].split(",")
            # count = 0.0
            # for j in name_list:
            #     cur.execute("select best_scores from " + fuck[i] + " where name='" + j + "'")
            #     temp = cur.fetchone()
            #     if temp is not None:
            #         best_scores = temp[0].split(",")
            #         count_s = 0.0
            #         for score in best_scores:
            #             try:
            #                 count_s = count_s + float(score)
            #             except ValueError:
            #                 count_s = count_s
            #         count = count + count_s / best_scores.__len__()
            # counts = count / name_list.__len__()
            # print(counts)
            # row[i] = counts
        if not prepare_features:
            genre_list = row[2].split(",")
            temp = []
            for i in genre_list:
                temp.append(genre_dict.get(i))
            row[2] = np.mean(temp)
    cur.close()
    conn.close()
    data = np.asarray(raw_data[:, 0:-1], dtype=np.float32)
    scaler = pre.MinMaxScaler()
    data = scaler.fit_transform(data)
    if gen_label:
        test_data = []
        test_label = []
        train_data = []
        train_label = []
        test_data_ = []
        if gen_label:
            for i in range(data.__len__()):  # choose the 2017 as the test data
                if data[i][year] == 1:
                    test_data.append(data[i])
                    test_data_.append(
                        [raw_data_[i][0], raw_data_[i][-1]])  # the name and the box_office of the test data
                    test_label.append(pre_mat[i])
                else:
                    train_data.append(data[i])
                    train_label.append(pre_mat[i])
        else:
            for i in range(data.__len__()):
                if data[i][2] == 1:
                    test_data.append(data[i])
                    test_label.append(pre_mat[i])
                else:
                    train_data.append(data[i])
                    train_label.append(pre_mat[i])
        test_data = np.asarray(test_data, dtype=np.float32)
        test_label = np.asarray(test_label, dtype=np.float32)
        train_data = np.asarray(train_data, dtype=np.float32)
        train_label = np.asarray(train_label, dtype=np.float32)
        return train_data, train_label, test_data, test_label, test_data_
    else:
        return data


def add(x_, y):
    return x_ + y


def return_covs():
    conn = sql.connect(database="/mnt/new_train.db")
    cur = conn.cursor()
    cur.execute("select box_office from train_movie where year>2012 order by box_office desc")
    temp = cur.fetchall()
    x = np.asarray(temp, np.float32).reshape((1, temp.__len__()))
    features = ["director", "actor", "year", "month", "location", "search", "holiday", "competition", "budget",
                "screen_3day", "screen_30day"]
    cur.close()
    conn.close()
    raw_data = prepare_data(5, False, prepare_features=True).transpose()
    print(raw_data.shape)
    covs = []
    covs1 = []
    dtype = [('feature', "S10"), ("cov", float)]
    for feature, i in zip(raw_data, range(raw_data.__len__())):
        cov = np.abs(np.cov(x, feature)[1, 0])
        cov_name = (features[i], cov)
        covs1.append(cov)
        covs.append(cov_name)
    print(covs)
    covs = np.array(covs, dtype=dtype)
    covs = np.sort(covs, order="cov")
    print(covs)
    return covs


def test():
    covs = return_covs()
    with tf.name_scope("hellokugou") as scope:
        if not os.path.exists("models/hellokugou"):
            os.mkdir("models/hellokugou")
        sess = tf.Session()
        model = mod.MLP(name="hellokugou", sess=sess, scope=scope, output_dim=10)
        train_data, train_label, test_data, test_label, test_data_ = prepare_data(10, gen_label=True, static_box=True,
                                                                                  covs=covs)
        model.init1(train_data, train_label, test_data, test_label)
        ac = []
        # for i in range(0, 20):
        acc, y_, output_label, ground_truth_label, out_pro = model.run1(name="with_genre")
        ac.append(acc)
        total = np.asarray(np.concatenate(
            (test_data_, np.asmatrix(ground_truth_label).transpose(), np.asmatrix(output_label).transpose(), y_),
            1))
        print(total)
        print(acc)
        print(np.mean(ac))
        # print(np.asarray("战狼2,丛林有情狼".split(",")))


def restore():
    with tf.name_scope("hellokugou") as scope:
        covs = return_covs()
        sess = tf.Session()
        model = mod.MLP(name="hellokugou", sess=sess, scope=scope, output_dim=10)
        train_data, train_label, test_data, test_label, test_data_ = prepare_data(10, gen_label=True, static_box=True,
                                                                                  covs=covs)
        model.init1(train_data, train_label, test_data, test_label)
        acc, y_, output_label, ground_truth_label, out_pro = model.run1(name="with_genre", steps=1, is_save=True)
        scaler = pre.MinMaxScaler()
        out_pro = np.asarray(out_pro, dtype=np.float32).transpose()
        out_pro = scaler.fit_transform(out_pro).transpose()
        sums = np.asmatrix(np.sum(out_pro, axis=1))
        out_pro = out_pro / sums.transpose()
        total = np.asarray(np.concatenate(
            (test_data_, np.asmatrix(ground_truth_label).transpose(), np.asmatrix(output_label).transpose(),
             np.asmatrix(out_pro, dtype=np.float32)),
            1))
        print(total)
        print(acc)
        # print(out_pro)
        # print(np.sum(out_pro, axis=1))
    return total, out_pro


def save_detailed_result():
    total, out_pro = restore()
    normal_sort(total, compare_complex)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("test")
    for row, i in zip(total, range(total.__len__())):
        for col, j in zip(row, range(row.__len__())):
            sheet.write(i, j, col)
    workbook.save("save/10_with2.xls")


def compare(x, y):
    return -1 if x > y else 1


def compare_complex1(x, y):
    pre_ = np.zeros([10], dtype=np.float32)
    pre_[0] = 0.0
    pre_[1] = 5000.0
    pre_[2] = 10000.0
    pre_[3] = 15000.0
    pre_[4] = 20000.0
    pre_[5] = 30000.0
    pre_[6] = 40000.0
    pre_[7] = 50000.0
    pre_[8] = 100000.0
    pre_[9] = 200000.0
    x_mean = np.sum(np.asarray(x[4:x.__len__()], dtype=np.float32) * pre_)
    y_mean = np.sum(np.asarray(y[4:y.__len__()], dtype=np.float32) * pre_)
    return -1 if x_mean > y_mean else 1


def compare_complex(x, y):
    x_max = 0
    x_n = 4
    y_max = 0
    y_n = 4
    for i in range(4, x.__len__()):
        if float(x[i]) > x_max:
            x_n = i
            x_max = float(x[i])
        if float(y[i]) > y_max:
            y_n = i
            y_max = float(y[i])
    return -1 if x_n > y_n else 1 if x_n < y_n else -1 if x_max > y_max else 1


def normal_sort(total, f):
    length = total.__len__()
    for i in range(length):
        for j in range(length - 1):
            if f(total[j], total[j + 1]) < 0:
                total[[j, j + 1], :] = total[[j + 1, j], :]
    return total


def build_genre():
    conn = sql.connect(database="/mnt/new_train.db")
    cur = conn.cursor()
    sql_str = "select box_office,genre from movie"
    cur.execute(sql_str)
    data = np.asarray(cur.fetchall())
    genre_dict = {}
    for row in data:
        genres = row[1].split(",")
        for genre in genres:
            if genre_dict.get(genre) is None:
                genre_dict[genre] = float(row[0])
            else:
                genre_dict[genre] = genre_dict[genre] + float(row[0])
    cur.close()
    conn.close()
    return genre_dict


def predict_box(model=None, sess=None, film_name=None, genre=None, director=None, actor=None, holiday=None, year=None,
                month=None,
                competition=0, topic=None, screen3d=0, screen30d=0):

    fo = open(os.path.normpath('%s/%s' % (abs_path, 'actor.data')), "rb")
    map_t = pickle.load(fo)
    fuck = ["director", "actor"]
    genre_dict = build_genre()
    conn = sql.connect(database="/mnt/new_train.db")
    cur = conn.cursor()
    sql_str = "select director,actor,genre,holiday,search,month,competition,year,screen_3day,screen_30day" \
              " from train_movie where year>2012 order by box_office desc"
    cur.execute(sql_str)
    raw_data = np.asarray(cur.fetchall())
    for row, k in zip(raw_data, range(raw_data.__len__())):
        for i in range(0, 2):  # transform the director and actor into numerous variable
            name_list = row[i].split(",")
            l = []
            for j in name_list:
                lists = map_t.get(j).get(fuck[i])
                l.append(np.mean(lists))
            row[i] = np.mean(l)
        genre_list = row[2].split(",")
        temp = []
        for i in genre_list:
            temp.append(genre_dict.get(i))
        row[2] = np.mean(temp)
    raw_data = np.asmatrix(raw_data, np.float32)

    min_ = np.asarray(np.min(raw_data, axis=0))
    max_ = np.asarray(np.max(raw_data, axis=0))
    director_list = director.split(",")
    l = []
    for j in director_list:
        lists = map_t.get(j).get("director")
        l.append(np.mean(lists))
    director_rep = np.mean(l)
    actor_list = actor.split(",")
    l = []
    for j in actor_list:
        lists = map_t.get(j).get("actor")
        l.append(np.mean(lists))
    actor_rep = np.mean(l)
    l = []
    genre_list = genre.split(",")
    for j in genre_list:
        l.append(genre_dict.get(j))
    genre_rep = np.mean(l)
    total = [director_rep, actor_rep, genre_rep, holiday, topic, month, competition, year, screen3d, screen30d]
    total_rep = np.asarray((total - min_) / (max_ - min_))

    with tf.name_scope("hellokugou") as scope:
        return sess.run([model.y_hat], feed_dict={model.x: total_rep})[0][0]


def restore_model():
    save_path = os.path.normpath('%s/%s' % (abs_path, 'save/hellokugou_'))
    with tf.name_scope("hellokugou") as scope:
        sess = tf.Session()
        model = mod.MLP(name="hellokugou", sess=sess, scope=scope, output_dim=10, input_dim=10)
        model.init2()
        saver = tf.train.Saver()
        name = "with_genre"
        saver.restore(sess, save_path=save_path + name + ".ckpt")
        return sess, model

# print(predict_box("速度与激情8", director="F·加里·格雷", actor="范·迪塞尔,道恩·强森,查理兹·塞隆,杰森·斯坦森,米歇尔·罗德里格兹",
#                   genre="动作,犯罪", holiday=3, topic=362, screen3d=520000, screen30d=2333999, year=2017, month=4,
#                   competition=3))
# print(build_genre())
# test()
# restore()
# a = normal_sort(restore(), compare_complex)
# for ll in a:
#     print(ll)
# save_detailed_result()
# print(normal_sort([1, 2, 5, 2, 9, 2542, 3, -1], compare))
# print(compare_complex(['战狼2', '561456.0', '9', '9', '1.5930887251741632e-14', '3.20423809013759e-13',
#                        '2.4234830808822494e-10', '6.94801238765308e-09', '4.4827388592239e-06',
#                        '1.231633405041066e-06', '5.72001084583033e-17', '0.002741813426837325',
#                        '2.0251152932360128e-07', '0.9972522854804993'],
#                       ['乘风破浪', '104600.0', '9', '9', '8.269662732374172e-12', '7.820659675417119e-09',
#                        '2.965578289604309e-07', '3.320349328816974e-09', '0.03206559643149376',
#                        '0.0569404698908329', '0.2624549865722656', '0.1366627961397171',
#                        '0.15435688197612762', '0.3575189411640167']))
