from numpy import *
from math import e
from math import log
def load_data_set(filename):
    data_mat = []; label_mat = []
    fr = open(filename, 'r', encoding = 'utf-8')
    k = 0
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        i = len(cur_line)
        #k += 1
        #print(k)
        for j in range(i - 1):
            if j == 0:
                continue
            else:
                line_arr.append(float(cur_line[j]))
        line_arr.append(float(1))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat

def data_format(data_set, label_set):
    for j in range(len(data_set[0])):
        data_max = data_set[0][j]
        data_min = data_set[0][j]
        for i in range(len(label_set)):
            if data_set[i][j] > data_max:
                data_max = data_set[i][j]
            if data_set[i][j] < data_min:
                data_min = data_set[i][j]
        distance = data_max - data_min
        if distance > 1:
            for k in range(len(data_set)):
                data_set[k][j] = (data_set[k][j] - data_min) / distance
    for j in range(len(label_set)):
        if label_set[j] > 20:
            label_set[j] = 8
        else:
            label_set[j] = int(label_set[j] / 3) + 1

    return data_set, label_set

def get_train_and_test(data_mat, label_mat, train_num):
    train_data_set = data_mat.copy()
    train_label_set = label_mat.copy()
    test_data_set = []
    test_label_set = []
    for i in range(train_num):
        j = int(random.uniform(0,len(train_data_set)))
        test_data_set.append(train_data_set[j])
        test_label_set.append(train_label_set[j])
        del train_data_set[j]
        del train_label_set[j]
    return array(train_data_set), array(train_label_set), array(test_data_set), array(test_label_set)

def stand_regres(x_arr, y_arr):
    x_data = []
    y_data = []
    for i in range(len(x_arr)):
        cur_x = []
        for j in range(1, len(x_arr[i])):
            cur_x.append(x_arr[i][j])
        x_data.append(cur_x)
        y_data.append(y_arr[i][1])
    x_mat = mat(x_data); y_mat = mat(y_data).T
    xtx = x_mat.T * x_mat
    if linalg.det(xtx) == 0.0:
        print('this matrix is singular, can\'t do inverse')
        return
    ws = xtx.I * (x_mat.T * y_mat)
    return ws


def boxoffice_test(ws, test_data, test_label):
    test_set = []
    test_label_set = []
    for i in range(len(test_data)):
        line = []
        for j in range(1, len(test_data[i])):
            line.append(test_data[i][j])
        test_set.append(line)
        test_label_set.append(test_label[i][1])
    result_label = mat(test_set) * ws
    result_set = list(map(int, list(result_label)))
    error_count = 0
    for i in range(len(test_label_set)):
        #if ((result_set[i] - test_label_set[i])) / test_label_set[i] > 0.1:
        if (((abs(result_set[i] - test_label_set[i])) / test_label_set[i]) > 0.2):
            error_count += 1
    return error_count / len(test_label_set)


def result_test(y_hat, y_data):
    count = 0
    for i in range(len(y_hat)):
        if  (y_data[i] - y_hat[i]) == 0:
            count += 1
    return count / len(y_hat)