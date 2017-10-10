import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlwt

from sklearn import datasets, linear_model, metrics
import xlrd
import random
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE, SelectKBest
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

from matplotlib.font_manager import *


def analyse_feature():
    '''
        10:screen_3day       good
        11:screen_30day      best
        12:search            no use
        16:director          good
        17:actor             good
    '''
    data = pd.read_excel('new_data.xlsx')
    y = data[[5]].values
    x = data[[13]].values
    x = np.arange(len(y))

    plt.figure('data')
    # plt.plot(x, y, 'g*')
    plt.bar(x, sorted(y), width=0.4)
    plt.show()


def analyse_movie_type():
    data = pd.read_excel('13-17.xlsx')
    feature = [7, 5]
    x = data[feature].values
    dic = {}

    for i in range(len(x)):
        dic[x[i][0]] = x[i][1]
    print(dic)
    data = pd.read_excel('14-16.xlsx')
    x = data[['票房', '类型']].values
    for i in range(len(x)):
        x[i][1] = dic.get(x[i][1].split(' ')[0], 0)
    plt.figure('data')
    plt.plot(x[:, 1], x[:, 0], '*')
    plt.show()


def director_avg_box_office(director_names):
    data = pd.read_excel('all_info.xls', sheetname='电影目录')
    data = data[[4, 5]].values
    box_office = {}
    for i in range(len(director_names)):
        sum_box_office = 0
        count = 0
        for j in range(len(data)):
            if type(data[j][1]) is float:
                continue
            if director_names[i] in data[j][1]:
                sum_box_office += data[j][0]
                count += 1
        if count == 0:
            box_office[director_names[i]] = 0
        else:
            box_office[director_names[i]] = sum_box_office / count
    print(box_office)
    return box_office


def plot_pred_y(predicted, y):
    plt.figure()
    # plt.plot(range(len(predicted)), predicted, 'b+', label="predict")
    # plt.plot(range(len(predicted)), y, 'r*', label="test")
    # plt.scatter(y, predicted)
    # plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)

    x = np.arange(len(y))
    width = 0.4
    predicted = np.array(predicted)
    plt.bar(x, y, width=width, label='truth')
    plt.bar(x + width, predicted, width=width, label='predicted')
    plt.xlabel("nums")
    plt.ylabel('box office')
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.show()


def extract_feature():
    data = pd.read_excel('fuck1')
    feature = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 18]
    x = data[feature]
    y = data[[1]]
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=2)
    rfe.fit_transform(x, y)
    print(rfe.support_)
    print(rfe.ranking_)


def movie_data():
    data1 = pd.read_excel('13-17.xlsx')
    '''
        11 16 17 :           0.44
        11 16 17 13:         0.45 not stable
        11 16 17 13 14:      0.43 stable
        11 16 17 13 14 9 :   0.42 stable
        2,3,4 is no use
        + 15 0.42
    '''
    feature = [0, 11, 13, 14, 16, 17]
    x = data1[feature]
    y = data1[[5]]
    # x = preprocessing.Imputer(missing_values=0, strategy='mean').fit_transform(x)
    ss = preprocessing.MinMaxScaler()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=random.randint(1, 100),
                                                                        test_size=0.2)
    test_x = X_test.iloc[:, 0].values
    X_train = X_train[[11, 13, 14, 16, 17]]
    X_test = X_test[[11, 13, 14, 16, 17]]

    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    # rfe = RFE(linreg)
    # rfe.fit(X_train, y_train)
    #
    # for i in range(len(feature)-1):
    #     print('%d:%d' % (feature[i], rfe.ranking_[i]))
    # predicted = cross_val_predict(linreg, X_test, y_test, cv=10)
    # print(linreg.coef_)
    # print(linreg.intercept_)
    # print(linreg.score(X_test, y_test))
    predicted = linreg.predict(X_test)

    test_y = y_test.values
    predicted = predicted.tolist()
    # print(np.sqrt(metrics.mean_squared_error(test_y, predicted)))

    error_rate = 0.2
    num = 0
    truth = []
    error = []
    name = []
    for i in range(len(predicted)):
        if predicted[i][0] < 0:
            predicted[i][0] = -predicted[i][0]
        if abs(predicted[i][0] - test_y[i]) < test_y[i] * error_rate:
            num += 1
        else:
            truth.append(float(test_y[i][0]))
            error.append(float(predicted[i][0]))
            name.append(test_x[i])
    print(num / len(test_y))
    # plot_pred_y(error, truth)
    # return name, truth, error, test_x.tolist()
    return num / len(test_y)


# generate the error list where error_rate > 0.3
def make_error(signle=False):
    if signle:
        print(movie_data())
    else:
        sum_acc = 0
        data = xlwt.Workbook()
        data2 = xlwt.Workbook()
        for i in range(20):
            # sum_acc += movie_data()
            name, truth, error, test_x = movie_data()
            table = data.add_sheet('iter_' + str(i + 1))
            table2 = data2.add_sheet('iter_' + str(i + 1))
            for j in range(len(name)):
                table.write(j, 0, name[j])
                table.write(j, 1, truth[j])
                table.write(j, 2, error[j])
            for j in range(len(test_x)):
                table2.write(j, 0, test_x[j])
        data.save('error_list.xls')
        data2.save('test_y.xls')


def avg_acc():
    sum_acc = 0

    for i in range(20):
        sum_acc += movie_data()
    print(sum_acc / 20)


def year_test(year):
    data1 = pd.read_excel('new_data.xlsx', sheetname=year)
    '''
        11 16 17 : 0.44
        11 16 17 13: 0.45 not stable
        11 16 17 13 14: 0.43 stable
        11 16 17 13 14 9 : 0.42 stable
        2,3,4 is no use
    '''
    feature = [11, 13, 14, 16, 17]
    x = data1[feature]
    y = data1[[5]]
    print('------------%s-------------' % year)
    print(liner_res_acc(x, y))
    print('------------%s-------------' % year)


def diff_year_test():
    year_test('2011')
    year_test('2012')
    year_test('2013')
    year_test('2014')
    year_test('2015')
    year_test('2016')
    year_test('2017')


def liner_res_acc(x, y):
    # x = preprocessing.Imputer(missing_values=0, strategy='mean').fit_transform(x)
    x = preprocessing.MinMaxScaler().fit_transform(x)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=random.randint(1, 100),
                                                                        test_size=0.2)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print(linreg.coef_)
    print(linreg.intercept_)
    predicted = linreg.predict(X_test)
    test_y = y_test.values
    predicted = predicted.tolist()
    error_rate = 0.2
    num = 0
    for i in range(len(predicted)):
        if abs(predicted[i][0] - test_y[i]) < test_y[i] * error_rate:
            num += 1
    print(num / len(test_y))
    # plot_pred_y(predicted,test_y)
    return num / len(test_y)


def plot_diff_year():
    year = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    acc = [0.25, 0.26, 0.55, 0.5, 0.47, 0.47, 0.36]
    plt.plot(year, acc)
    plt.show()


def write_error_to_excel(name, truth, error, file_name):
    data = xlwt.Workbook()
    table = data.add_sheet(file_name)
    for i in range(len(name)):
        table.write(i, 0, name[i])
        table.write(i, 1, truth[i])
        table.write(i, 2, error[i])
    data.save('error_list.xls')


def count_error():
    sheetname = ['iter_' + str(i + 1) for i in range(20)]
    count = {}
    all_count = {}
    box_office = {}
    for sn in sheetname:
        data = pd.read_excel('error_list.xls', sheetname=sn)
        data2 = pd.read_excel('test_y.xls', sheetname=sn)

        for index, row in data2.iterrows():
            row = list(row)
            all_count[row[0]] = all_count.get(row[0], 0) + 1
        for index, row in data.iterrows():
            row = list(row)
            if row[0] not in box_office:
                box_office[row[0]] = []
            box_office[row[0]].append(row[2])
            count[row[0]] = count.get(row[0], 0) + 1
    error = {k: v for k, v in count.items() if v > 5 and all_count[k] > 5}
    # for k, v in error.items():
    #     print(k, v, all_count[k])
    # print(error)
    # total_width, n = 0.8, 2
    # width = total_width / n
    # x = np.arange(len(error))
    # x = x - (total_width - width) / 2
    # keys = list(error.keys())
    # print(keys)
    # print(x)
    # import matplotlib
    # myfont = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    # matplotlib.rcParams['axes.unicode_minus'] = False
    #
    # plt.bar(x, [all_count[k] for k in keys], width=width,label='occurrences')
    # plt.bar(x + width, [error[k] for k in keys], width=width, label='error_count')
    # for i in range(len(keys)):
    #     plt.text(x[i], all_count[keys[i]] + 0.2, keys[i],ha='center',va='top',fontproperties=myfont,rotation=30)
    # plt.legend()
    # plt.show()

    return list(error.keys()), box_office


'''
    9 location
    10 screen_3
    11 screen_30        x
    12 search
    13 holiday          x
    14 competition      x
    15 budget
    16 director         x
    17 actor            x
'''


def analyse_error():
    error_name, box_office_p = count_error()
    print(error_name)
    data = pd.read_excel('13-17.xlsx')
    refer_dict = {}
    analyse_movie = {}
    for en in error_name:
        analyse_movie[en] = data[data[0] == en]
        box_office = data[data[0] == en][5].values[0]
        refer_movie = data[(abs(data[5] - box_office) / box_office <= 0.1)]
        refer_dict[en] = refer_movie[~refer_movie[0].isin(error_name)]
    myfont = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    matplotlib.rcParams['axes.unicode_minus'] = False
    test = ['魔境仙踪', '哆啦A梦：大雄的南极冰冰凉大冒险', '喜羊羊与灰太狼之飞马奇遇记', '黑猫警长之翡翠之星', '超脑48小时', '别惹我', '你的名字。', '分歧者3：忠诚世界',
            '星际迷航3：超越星辰']
    rd = refer_dict[test[3]]
    am = analyse_movie[test[3]]
    plt.bar(np.arange(len(rd)), rd[17], label='others')
    plt.bar(-1, am[17], label='self')
    plt.text(-2, am[17] + 0.3, test[3], fontproperties=myfont)
    plt.legend()
    '''
    plot the predict and truth assign by movie_name
    '''
    # plt.bar(np.arange(len(box_office_p[test[1]])), box_office_p[test[1]],label='predict')
    # plt.bar(-1, am[5],label='truth')
    # plt.legend()
    # plt.title(test[1], fontproperties=myfont)

    # total_width, n = 0.8, 2
    # width = total_width / n
    # x = np.arange(len(error_name))
    # x = x - (total_width - width) / 2
    # keys = list(refer_dict.keys())
    # print(keys)
    # plt.bar(x, [refer_dict[k][11].mean() for k in keys], width=width)
    # plt.bar(x + width, [analyse_movie[k][11] for k in keys], width=width)

    plt.show()


def analyse_theme():
    sys.path.append('/home/hanhao/abc/predictBoxOffice')
    print(sys.path)
    from predictBoxOffice import models
    theme = '动画'
    movies = models.Movie.query.filter(models.Movie.movie_type.find('动画') != -1).all()
    print(len(movies))
if __name__ == '__main__':
    analyse_theme()
    print('aaa')
