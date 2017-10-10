import xlrd
import xlwt
from predictBoxOffice import db, models

DEFAULT_LEVEL = 2


def write_movie_to_db():
    data = xlrd.open_workbook('movie_data.xlsx')
    table = data.sheets()[0]
    rows = table.nrows
    print(rows)
    for i in range(1, rows):
        value = table.row_values(i)
        movie = models.Movie(value[0], value[1] / 1000.0, value[2], value[3], value[4], value[7], value[8]
                             , value[9] * 10, value[12], value[13], value[14], value[15], value[16])
        db.session.add(movie)
        db.session.commit()


def avg_box_office1():
    data = xlrd.open_workbook('movie_data.xlsx')
    box_office_data = data.sheets()[1]
    box_office_dict = {}
    for i in range(1, box_office_data.nrows):
        box_office_dict[box_office_data.row_values(i)[0]] = box_office_data.row_values(i)[3]
    return box_office_dict


def test_data():
    cost_dict, movie_type_dict = cost_and_type()
    level_dict = get_director_level()
    avg_box_office = avg_box_office1()
    for k, v in cost_dict.items():
        if k not in avg_box_office:
            print('%s not in box_office_dict' % k)
        elif k not in movie_type_dict:
            print('%s not in movie_type_dict' % k)
        elif k not in level_dict:
            print('%s not in level_dict' % k)


def get_director_level():
    data = xlrd.open_workbook('movie_data.xlsx')
    level_data = data.sheets()[3]
    level_dict = {}
    for i in range(1, level_data.nrows):
        level_dict[level_data.row_values(i)[0]] = level_data.row_values(i)[3] + 1
    return level_dict


def cost_and_type():
    data = xlrd.open_workbook('movie_data.xlsx')
    movie = data.sheets()[0]
    movie_data = []
    for i in range(1, movie.nrows):
        movie_data.append(movie.row_values(i))
    data = movie_data
    cost_dict = {}
    num_dict = {}
    movie_type_dict = {}
    # index 13 is director,index 9 is cost
    for i in range(len(data)):
        num_dict[data[i][13]] = num_dict.get(data[i][13], 0) + 1
        cost_dict[data[i][13]] = cost_dict.get(data[i][13], 0) + data[i][9]
        if data[i][13] in movie_type_dict:
            types = data[i][15].split(' ')
            for t in types:
                if movie_type_dict[data[i][13]].find(t) == -1:
                    movie_type_dict[data[i][13]] = movie_type_dict[data[i][13]] + ' ' + t
        else:
            movie_type_dict[data[i][13]] = data[i][15]
    for k, v in cost_dict.items():
        cost_dict[k] = v / num_dict[k]
    return cost_dict, movie_type_dict


def write_director_to_db():
    cost_dict, movie_type_dict = cost_and_type()
    level_dict = get_director_level()
    avg_box_office = avg_box_office1()
    for k, v in cost_dict.items():
        director = models.Director(k, avg_box_office[k] / 1000, cost_dict[k] * 10, level_dict[k], movie_type_dict[k])
        db.session.add(director)
        db.session.commit()


def write_actor_to_db():
    cost_dict, movie_type_dict = cost_and_type()
    level_dict = get_director_level()
    avg_box_office = avg_box_office1()
    for k, v in avg_box_office.items():
        if k not in cost_dict:
            actor = models.Actor(k, v / 1000, level_dict.get(k, 1), '')
            db.session.add(actor)
            db.session.commit()


def actor_movie_type(name, movie_data):
    movie_type = ''
    for m in movie_data:
        if m.starring.find(name) != -1:
            types = m.movie_type.split(' ')
            for t in types:
                if movie_type.find(t) == -1:
                    movie_type = movie_type + ' ' + t
    return movie_type


def write_old_data():
    write_movie_to_db()
    write_director_to_db()
    write_actor_to_db()
    update_actor()


# update movie_type to actor
def update_actor():
    movies = models.Movie.query.all()
    actors = models.Actor.query.all()
    for a in actors:
        a.movie_type = actor_movie_type(a.name, movies)
        db.session.add(a)
        db.session.commit()


def add_new_movie():
    data = xlrd.open_workbook('data_excel/new_data.xlsx')
    movies = data.sheets()[0]
    n = 0
    a_n = 0
    d_n = 0
    for i in range(1, movies.nrows):
        value = models.Movie.query.filter(models.Movie.name == movies.row_values(i)[0]).all()
        if value is not None:
            if len(value) != 0:
                print('repeat data!')
                continue
        value = movies.row_values(i)
        directors = value[6].split(',')
        actors = value[7].split(',')

        try:
            box_office = float(value[5])
            box_office /= 1000
        except:
            print('movie %s has no box_office, insert to db failed!' % value[0])
            continue

        movie = models.Movie(value[0], box_office, value[2], value[3], value[4], False, False, -1, False,
                             value[6],
                             value[7], value[8], -1)
        print(movie)
        n += 1
        db.session.add(movie)

        # add director from new movie
        for d in directors:
            director = models.Director.query.filter(models.Director.name == d).all()

            if len(director) == 0:
                # todo how to define default level when a new director or actor record coming without level
                # if table director have no related record, then insert only with name,
                # next update step will complete other info.
                k = models.Director(d, 0, 0, DEFAULT_LEVEL, '')
                print(d)
                db.session.add(k)
                d_n += 1

        for s in actors:
            actor = models.Actor.query.filter(models.Actor.name == s).all()
            if len(actor) == 0:
                k = models.Actor(s, 0, DEFAULT_LEVEL, '')
                print(k)
                db.session.add(k)
                a_n += 1
        db.session.commit()
    print(n)
    print(d_n)
    print(a_n)
    # update_actor_and_director()


def format_movie_data():
    file = xlwt.Workbook()
    table = file.add_sheet('movies', cell_overwrite_ok=True)

    data = xlrd.open_workbook('movie_data_v2.xlsx')
    movie = data.sheets()[0]
    for i in range(movie.nrows):
        value = movie.row_values(i)
        directors = value[6].replace(' ', '').split(',')[::2]
        actors = value[7].replace(' ', '').split(',')[::2]
        value[8] = value[8].replace(' ', '')
        format_directors = ''
        format_actors = ''
        for d in directors:
            format_directors = format_directors + d + ','
        value[6] = format_directors[:-1]
        for a in actors:
            format_actors = format_actors + a + ','
        value[7] = format_actors[:-1]
        for j in range(len(value)):
            table.write(i, j, value[j])
    file.save('movie_without_2017.xlsx')


# return array data:which d[0,1,2] is director_name,level,avg_score
def director_level1():
    data = xlrd.open_workbook('movie_data1.xlsx')
    directors = data.sheets()[1]
    data = []
    for i in range(1, directors.nrows):
        d = directors.row_values(i)
        avg_score = 0
        if len(directors.row_values(i)[7]) == 0:
            avg_score = 0
        else:
            value = list(map(lambda x: float(x), d[7].split(',')))
            avg_score = sum(value) / len(value)
        data.append([d[0], d[6], avg_score])
    data = sorted(data, key=lambda x: x[2], reverse=True)
    label_count = 5
    nums = int(len(data) / label_count)
    director_level = {}
    k = 0
    for i in range(len(data)):
        director_level[data[i][0]] = label_count
        if k == nums:
            k = 0
            label_count -= 1
        k += 1
    return director_level


def actor_level1():
    data = xlrd.open_workbook('movie_data1.xlsx')
    directors = data.sheets()[2]
    data = []
    for i in range(1, directors.nrows):
        d = directors.row_values(i)
        avg_score = 0
        d[7] = d[7].replace(' ', '')
        if len(d[7]) == 0:
            avg_score = 0
        else:
            value = list(map(lambda x: float(x), d[7].split(',')))
            avg_score = sum(value) / len(value)
        data.append([d[0], d[6], avg_score])
    data = sorted(data, key=lambda x: x[2], reverse=True)
    label_count = 5
    nums = int(len(data) / label_count)
    k = 0
    actor_level = {}
    for i in range(len(data)):
        actor_level[data[i][0]] = label_count
        if k == nums:
            k = 0
            label_count = label_count - 1
        k = k + 1

    return actor_level


# update director and actor when new movie data added
def update_actor_and_director():
    director_level = director_level1()
    actor_level = actor_level1()
    directors = models.Director.query.all()
    for d in directors:
        movies = models.Movie.query.filter(models.Movie.director.contains(d.name)).all()
        if movies is None:
            print('director % s is not found in table movies' % d.name)
            continue
        sum_box_office = 0
        sum_cost = 0
        types = d.movie_type
        for m in movies:
            movie_types = m.movie_type.split(',')
            for mt in movie_types:
                if types.find(mt) == -1:
                    types = types + ',' + mt
            sum_box_office += m.box_office
            sum_cost += m.production_cost

        avg_box_office = sum_box_office / len(movies)
        avg_cost = sum_cost / len(movies)
        if types != d.movie_type or avg_box_office != d.avg_box_office:
            print('1movie_name = %s avg_box_office= %f movie_type = %s level=%s' % (
            d.name, avg_box_office, types, director_level.get(d.name, DEFAULT_LEVEL)))
            print('1movie_name = %s avg_box_office= %f movie_type = %s' % (d.name, d.avg_box_office, d.movie_type))
            d.avg_box_office = avg_box_office
            d.level = director_level.get(d.name,DEFAULT_LEVEL)
            d.movie_type = types
            d.avg_cost = avg_cost
    actors = models.Actor.query.all()
    for a in actors:
        movies = models.Movie.query.filter(models.Movie.starring.contains(a.name)).all()
        if movies is None:
            print('actor %s is not found in table movies' % a.name)
            return
        sum_box_office = 0
        types = ''
        for m in movies:
            types = a.movie_type
            movie_types = m.movie_type.split(',')
            for mt in movie_types:
                if types.find(mt) == -1:
                    types = types + ',' + mt
            sum_box_office += m.box_office
        avg_box_office = sum_box_office / len(movies)
        if types != a.movie_type or avg_box_office != a.avg_box_office:
            print('2movie_name = %s avg_box_office= %f movie_type = %s level=%s' % (
            a.name, avg_box_office, types, actor_level.get(a.name, DEFAULT_LEVEL)))
            print('2movie_name = %s avg_box_office= %f movie_type = %s' % (a.name, a.avg_box_office, a.movie_type))

            a.avg_box_office = avg_box_office
            a.movie_type = types
            a.level = actor_level.get(a.name, DEFAULT_LEVEL)

    # db.session.commit()



def movie_without_cost():
    data = xlrd.open_workbook('movie_data1.xlsx')
    movies = data.sheets()[0]
    data = []
    for i in range(1, movies.nrows):
        m = models.Movie.query.filter(models.Movie.name == movies.row_values(i)[0]).all()
        if len(m) == 0:
            data.append(movies.row_values(i))

    file = xlwt.Workbook()
    table = file.add_sheet('movies', cell_overwrite_ok=True)

    for i in range(len(data)):
        for j in range(len(data[i])):
            table.write(i, j, data[i][j])
    file.save('movie_without_cost.xlsx')


# update movie_type for actor and director from movie
def update_movie_type():
    directors = models.Director.query.all()
    for d in directors:
        movies = models.Movie.query.filter(models.Movie.director.contains(d.name)).all()
        types = ''
        for m in movies:
            movie_types = m.movie_type.split(',')
            for mt in movie_types:
                if types.find(mt) == -1:
                    types += ',' + mt
        d.movie_type = types[1:]
    actors = models.Actor.query.all()
    for a in actors:
        movies = models.Movie.query.filter(models.Movie.starring.contains(a.name)).all()
        types = ''
        for m in movies:
            movie_types = m.movie_type.split(',')
            for mt in movie_types:
                if types.find(mt) == -1:
                    types += ',' + mt
        a.movie_type = types[1:]

    db.session.commit()


# add forward num to db
def add_forward_num_to_db():
    movies = models.Movie.query.all()
    data = xlrd.open_workbook('haha.xls')
    data = data.sheets()[0]
    w = {}
    for i in range(1,data.nrows):
        value = data.row_values(i)
        w[value[0]] = value[9]
    for m in movies:
        if m.name not in w:
            continue
        m.forward_num = w[m.name]
    db.session.commit()

if __name__ == '__main__':
    pass
