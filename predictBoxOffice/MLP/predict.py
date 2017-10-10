# -*- coding: utf-8 -*-
import sqlite3 as sql
import numpy as np
import model as mod
import tensorflow as tf
import pickle
import os
abs_path = '/home/hanhao/abc/predictBoxOffice/MLP'


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
                genre_dict[genre] = [float(row[0]), 1]
            else:
                temp = genre_dict[genre][0] + float(row[0])
                count = genre_dict[genre][1] + 1
                genre_dict[genre] = [temp, count]
    for i in genre_dict:
        genre_dict[i][0] = round(genre_dict[i][0] / genre_dict[i][1], 1)
    cur.close()
    conn.close()
    return genre_dict


def gen_models():
    name1 = "10category_top3acc"
    model_name1 = "MLP"
    total_name1 = model_name1 + "_" + name1
    name2 = "8category_top3acc_extend"
    model_name2 = "MLP1"
    total_name2 = model_name2 + "_" + name2
    with tf.name_scope(model_name1) as scope1:
        sess1 = tf.Session()
        model1 = mod.MLP(name=model_name1, sess=sess1, scope=scope1, output_dim=10, input_dim=10)
        model1.init2()
        saver1 = tf.train.Saver(model1.var)
        save_path = abs_path + "/save/" + total_name1 + "/" + total_name1 + ".ckpt"
        saver1.restore(sess1, save_path=save_path)
    with tf.name_scope(model_name2) as scope2:
        sess2 = tf.Session()
        model2 = mod.MLP(name=model_name2, sess=sess2, scope=scope2, output_dim=8, input_dim=5)
        model2.init2()
        saver2 = tf.train.Saver(model2.var)
        save_path1 = abs_path + "/save/" + total_name2 + "/" + total_name2 + ".ckpt"
        saver2.restore(sess2, save_path=save_path1)
    return sess1, model1, sess2, model2


def predict_box(features=None, model=None, sess=None, is_3rd=True):
    # print(features[0])  # movie name
    fo = open(os.path.normpath('%s/%s' % (abs_path, 'actor.data')), "rb")
    map_t = pickle.load(fo)
    fuck = ["director", "actor"]
    genre_dict = build_genre()
    conn = sql.connect(database="/mnt/new_train.db")
    cur = conn.cursor()
    sql_str = "select director,actor,genre,year,holiday,search,month,competition,screen_3day,screen_30day" \
              " from train_movie where year>2012 order by box_office desc" \
        if is_3rd else "select director,actor,genre,year,month " \
                       "from train_movie where year>2012 order by box_office desc"
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
    director_list = features[1].split(",")
    l = []
    for j in director_list:
        lists = map_t.get(j).get("director")
        l.append(np.mean(lists))
    features[1] = np.mean(l)
    actor_list = features[2].split(",")
    l = []
    for j in actor_list:
        lists = map_t.get(j).get("actor")
        l.append(np.mean(lists))
    features[2] = np.mean(l)
    l = []
    genre_list = features[3].split(",")
    for j in genre_list:
        l.append(genre_dict.get(j))
    features[3] = np.mean(l)
    total = np.asarray(features[1:], dtype=np.float32)
    total_rep = np.asarray((total - min_) / (max_ - min_))
    return sess.run([model.y_hat], feed_dict={model.x: total_rep})[0][0]

sess11, model11, sess22, model22 = gen_models()
# sess11, model11, sess22, model22 = gen_models()
# print(predict_box(["速度与激情8", "F·加里·格雷", "范·迪塞尔,道恩·强森,查理兹·塞隆,杰森·斯坦森,米歇尔·罗德里格兹",
#                    "动作,犯罪", 2017, 3, 362, 4, 3, 520000, 2333999], model11, sess11, is_3rd=True))
# print(predict_box(["速度与激情8", "F·加里·格雷", "范·迪塞尔,道恩·强森,查理兹·塞隆,杰森·斯坦森,米歇尔·罗德里格兹",
#                    "动作,犯罪", 2017, 3], model22, sess22, is_3rd=False))
