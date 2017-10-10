import xlrd
from predictBoxOffice import db, models

DEFAULT_LEVEL = 3


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


# update movie_type to actor
def update_actor():
    movies = models.Movie.query.all()
    actors = models.Actor.query.all()
    for a in actors:
        a.movie_type = actor_movie_type(a.name, movies)
        db.session.add(a)
        db.session.commit()


def add_new_movie():
    data = xlrd.open_workbook('new_movie_data.xlsx')
    movies = data.sheets()[0]
    for i in range(1, movies.nrows):
        value = models.Movie.query.filter(models.Movie.name == movies.row_value(i).name).all()
        if value is not None:
            if len(value) > 1:
                print('repeat data!')
                print(value)
                continue
        value = movies.row_value(i)
        movie = models.Movie(value[0], value[1] / 1000.0, value[2], value[3], value[4], value[7] * 10, value[8]
                             , value[9], value[12], value[13], value[14], value[15], value[16])
        db.session.add(movie)
        db.session.commit()
        director = models.Director.query.filter(models.Director.name == movie.director).all()
        if director is None:
            # todo how to define default level when a new director or actor record coming without level
            # if table director have no related record, then insert only with name,
            # next update step will complete other info.
            director = models.Director(movie.director, 0, 0, DEFAULT_LEVEL, '')
            db.session.add(director)
            db.session.commit()
        starrings = movie.starring.split(' ')
        for s in starrings:
            actor = models.Actor.query.filter(models.Actor.name == s).all()
            if actor is None:
                actor = models.Actor(s, 0, DEFAULT_LEVEL, '')
                db.session.add(actor)
                db.session.commit()


# update avg_box_office for director and actor when new movie data added
def update_avg_box_office():
    directors = models.Director.query.all()
    for d in directors:
        movies = models.Movie.query.filter(models.Movie.director == d.name).all()
        if movies is None:
            print('director % s is not found in table movies' % d.name)
            return
        sum_box_office = 0
        sum_cost = 0
        types = ''
        for m in movies:
            types = d.movie_type
            movie_types = m.movie_type.split(' ')
            for mt in movie_types:
                if types.find(mt) == -1:
                    types = types + ' ' + mt
            sum_box_office += m.box_office
            sum_cost += m.production_cost

        avg_box_office = sum_box_office / len(movies)
        avg_cost = sum_cost / len(movies)
        if types != d.movie_type:
            print('movie_name = %s avg_box_office= %f movie_type = %s' % (d.name, avg_box_office, types))
            print('movie_name = %s avg_box_office= %f movie_type = %s' % (d.name, d.avg_box_office, d.movie_type))
        models.Director.query.update(
            {'avg_box_office': avg_box_office, 'avg_cost': avg_cost, 'movie_type': types})
        db.session.commit()
    actors = models.Actor.query.all()
    for a in actors:
        movies = models.Movie.query.filter(models.Movie.starring.contains(a.name)).all()
        # movies = list(filter(lambda x: x.starring.find(a.name) != -1, movies))
        if movies is None:
            print('actor %s is not found in table movies' % a.name)
            return
        sum_box_office = 0
        types = ''
        for m in movies:
            types = a.movie_type
            movie_types = m.movie_type.split(' ')
            for mt in movie_types:
                if types.find(mt) == -1:
                    types = types + ' ' + mt
            sum_box_office += m.box_office
        avg_box_office = sum_box_office / len(movies)
        if types != a.movie_type or avg_box_office != a.avg_box_office:
            print('movie_name = %s avg_box_office= %f movie_type = %s' % (a.name, avg_box_office, types))
            print('movie_name = %s avg_box_office= %f movie_type = %s' % (a.name, a.avg_box_office, a.movie_type))
        models.Actor.query.update({'avg_box_office': avg_box_office, 'movie_type': types})
        db.session.commit()


if __name__ == '__main__':
   update_actor()
