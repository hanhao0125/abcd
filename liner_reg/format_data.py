import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
import xlwt


def actor_avg_box_office():
    data = pd.read_excel('13-17.xlsx')
    data = data[[5,7]].values
    actors = pd.read_excel('movie data/actor data.xls')
    actors = actors[['演员姓名']].values
    box_office = {}
    for a in actors:
        sum_box_office = 0
        count = 0
        for d in data:
            if type(d[1]) is float:
                continue
            if a[0] in d[1]:
                sum_box_office += d[0]
                count += 1
        if count == 0:
            box_office[a[0]] = 0
        else:
            box_office[a[0]] = sum_box_office / count
    return box_office


def director_avg_box_office():
    data = pd.read_excel('13-17.xlsx')
    data = data[[5, 6]].values
    directors = pd.read_excel('movie data/director data.xls')
    directors = directors[['导演姓名']].values
    box_office = {}
    for a in directors:
        sum_box_office = 0
        count = 0
        for d in data:
            if type(d[1]) is float:
                continue
            if a[0] in d[1]:
                sum_box_office += d[0]
                count += 1
        if count == 0:
            box_office[a[0]] = 0
        else:
            box_office[a[0]] = sum_box_office / count
    return box_office


def get_data():
    director_avg = director_avg_box_office()
    actor_avg = actor_avg_box_office()
    movies = pd.read_excel('13-17.xlsx')
    movies['director_feature'] = 0
    movies['actor_feature'] = 0
    movies = movies.values
    # director index:6 actor index :7
    for m in movies:
        if type(m[6]) is float:
            m[-2] = m[5]
        else:
            directors = m[6].split(',')
            sum_box_office = 0

            for d in directors:
                sum_box_office += director_avg.get(d,m[5])
            m[-2] = sum_box_office / len(directors)
        if type(m[7]) is float:
            m[-1] = m[5]
        else:
            sum_box_office = 0
            actors = m[7].split(',')[:3]
            for a in actors:
                sum_box_office += actor_avg.get(a, m[5])
            m[-1] = sum_box_office / len(actors)
    print(movies.shape)
    data = xlwt.Workbook()
    table = data.add_sheet('name')
    shape = movies.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            table.write(i, j, movies[i][j])
    data.save('new_train.xls')


def cls():
    '''
        
    :return: director level by score and box_office 
    '''
    data = pd.read_excel('movie data/director data(1).xls')
    data = data[['导演姓名', '历史最好5部片']].values
    directors_info = []
    directors_name = []
    for i in range(len(data)):
        directors_name.append(data[i][0])
        d = []
        if type(data[i][1]) is float:
            d.append(0)
        else:
            c = 0
            for s in data[i][1].split(','):
                c += float(s)
            d.append(c / len(data[i][1].split(',')))
        directors_info.append(d)

    avg_box_office = director_avg_box_office(directors_name)

    for i in range(len(directors_info)):
        directors_info[i].append(avg_box_office[directors_name[i]])
    from sklearn.cluster import KMeans
    directors_info = np.array(directors_info)
    # plt.scatter([i for i in range(len(directors_info))],directors_info[:,1])
    kmeans = KMeans(n_clusters=5)  # n_clusters:number of cluster
    colors = ['b', 'g', 'r', 'c', 'y']
    markers = ['o', 's', 'D', '*', '+']
    directors_info = preprocessing.MinMaxScaler(feature_range=(0, 10)).fit_transform(directors_info)
    y = kmeans.fit_predict(directors_info)
    director_label = {}
    for i in range(len(y)):
        director_label[directors_name[i]] = y[i]
    data = xlwt.Workbook()
    table = data.add_sheet('name')
    n = 0
    print(director_label)
    for k, v in director_label.items():
        table.write(n, 0, k)
        table.write(n, 1, int(v))
        n += 1
    data.save('director_level1.xls')
    print(sorted(director_label.items(), key=lambda x: x[1], reverse=True))
    x1 = directors_info[:, 0]
    x2 = directors_info[:, 1]
    label_count = {}
    print(kmeans.cluster_centers_)
    for i in range(len(y)):
        label_count[y[i]] = label_count.get(y[i], 0) + 1
    print(label_count)
    # for i, l in enumerate(kmeans.labels_):
    #     plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
    plt.scatter(x1, x2, c=y)
    plt.xlabel('movie score')
    plt.ylabel('box office')
    plt.show()


def director_actor_feature():
    data = xlrd.open_workbook('14-16.xlsx')
    movies = data.sheets()[0]
    avg_box_office = data.sheets()[2]
    box_office = {}
    director_feature = {}
    actor_feature = {}
    for i in range(1, avg_box_office.nrows):
        box_office[avg_box_office.row_values(i)[0]] = avg_box_office.row_values(i)[3]
    for i in range(1, movies.nrows):
        if movies.row_values(i)[13] not in box_office:
            director_feature[movies.row_values(i)[0]] = movies.row_values(i)[1]
        else:
            director_feature[movies.row_values(i)[0]] = box_office[movies.row_values(i)[13]]
        sum_box_office = 0
        actor_len = len(movies.row_values(i)[14].split(' '))
        for a in movies.row_values(i)[14].split(' '):
            if a not in box_office:
                actor_len -= 1
                continue
            sum_box_office += box_office[a]
        if actor_len == 0:
            actor_feature[movies.row_values(i)[0]] = movies.row_values(i)[1]
            continue
        actor_feature[movies.row_values(i)[0]] = sum_box_office / actor_len
    return director_feature, actor_feature


def get_movie_type():
    data = xlrd.open_workbook('14-16.xlsx')
    movies = data.sheets()[0]
    types = data.sheets()[5]
    types_dict = {}
    types_box_office = {}
    for i in range(1, types.nrows):
        types_box_office[types.row_values(i)[0]] = types.row_values(i)[2]
    for i in range(1, movies.nrows):
        box_office = 0
        num = 0
        for t in movies.row_values(i)[15].split(' '):
            if t not in types_box_office:
                continue
            num += 1
            box_office += types_box_office[t]
        box_office /= num
        types_dict[movies.row_values(i)[0]] = box_office
    return types_dict


def get_level():
    data = xlrd.open_workbook('14-16.xlsx')
    level_data = data.sheets()[4]
    level_dict = {}
    for i in range(1, level_data.nrows):
        level_dict[level_data.row_values(i)[0]] = level_data.row_values(i)[3]
    return level_dict


def avg_level_for_actor_in_movie():
    data = xlrd.open_workbook('14-16.xlsx')
    movie = data.sheets()[0]
    level_dict = get_level()
    actor_level = {}
    for i in range(1, movie.nrows):
        level = 0
        num = 0
        for a in movie.row_values(i)[14].split(' '):
            if a not in level_dict:
                continue
            num += 1
            level += level_dict[a]
        actor_level[movie.row_values(i)[0]] = int(level / num)
    return actor_level


def format_movie_data():
    file = xlwt.Workbook()
    table = file.add_sheet('features', cell_overwrite_ok=True)
    levels = get_level()
    type_box_office = get_movie_type()
    d_a = director_actor_feature()
    data = xlrd.open_workbook('14-16.xlsx')
    movie = data.sheets()[0]
    actor_level = avg_level_for_actor_in_movie()
    for i in range(1, movie.nrows):
        value = movie.row_values(i)
        for j in range(len(value)):
            table.write(i, j, value[j])
        table.write(i, 17, levels.get(value[13], 2))
        table.write(i, 18, type_box_office[value[0]])
        table.write(i, 19, actor_level[value[0]])
        table.write(i, 20, d_a[0][value[0]])
        table.write(i, 21, d_a[1][value[0]])
    file.save('fuck1')


# company and movie: no business
def company():
    data = pd.read_excel("movie data/movie's companies.xls")[['name', '发行','boxoffice']].values
    company_info = pd.read_excel("movie data/companies's info.xls")[['name','average boxoffice']].values
    movie_company = {}
    com_info = {}
    boxoffice_com = {}
    for i in range(len(company_info)):
        com_info[company_info[i][0]] = company_info[i][1]
    for i in range(len(data)):
        boxoffice_com[data[i][2]] = com_info[data[i][1]]
        if data[i][1] not in com_info:
            print(data[i][0])
            continue
        movie_company[data[i][0]] = com_info[data[i][1]]
    plt.plot(list(boxoffice_com.keys()),list(boxoffice_com.values()), 'g*')
    plt.show()

if __name__ == '__main__':
    get_data()