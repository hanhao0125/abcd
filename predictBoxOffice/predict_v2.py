from flask import Blueprint, jsonify, request
from predictBoxOffice import result_type as rt
import sys
import matplotlib.pyplot as plt

ERROR_RATE = 0.2
box_office_test = Blueprint('box_office_test', __name__)


def type_level_dict(data, mtd):
    level_dict = {}
    num_dict = {}
    for m in mtd:
        for d in data:
            if d.movie_type.find(m[0]) != -1:
                num_dict[m[0]] = num_dict.get(m[0], 0) + 1
                level_dict[m[0]] = level_dict.get(m[0], 0) + d.level

    for k, v in level_dict.items():
        level_dict[k] = round(v / num_dict[k])
    return level_dict


@box_office_test.route('/test_2_v2')
def query_director():
    from predictBoxOffice import models
    level = request.args.get('level')
    box_office = request.args.get('boxOffice')
    theme = request.args.get('theme')
    cost = request.args.get('cost')

    if level is None or box_office is None or theme is None:
        return rt.necessary_param_needed()

    try:
        box_office = int(box_office)
        level = int(level)
        if not 1 <= box_office <= 8:
            return rt.param_cross_border()
        box_office = transform_box_office(box_office)
    except:
        return rt.param_type_error()

    if cost is not None:
        try:
            cost = int(cost)
            if not 1 <= cost <= 9:
                return rt.param_cross_border()
            cost = transform_cost(cost)
        except:
            return rt.param_type_error()
    else:
        cost = (-1, sys.maxsize)

    movies = models.Movie.query.filter(
        (box_office[0] <= models.Movie.box_office) & (models.Movie.box_office <= box_office[1])) \
        .filter((cost[0] <= models.Movie.production_cost) & (models.Movie.production_cost <= cost[1])) \
        .filter(models.Movie.movie_type.contains(theme)) \
        .all()
    directors = directors_from_movie(movies, level)

    refer_movie = {}
    for d in directors:
        refer_movie[d.name] = list(map(lambda x: x.name, filter(lambda x: x.director.find(d.name) != -1, movies)))
    directors = sorted(directors, key=lambda x: abs(x.avg_box_office - (box_office[1] - box_office[0])))
    print(directors)
    basic_rate = 0.5
    box_office_rate = 0.2
    theme_rate = 0.2
    cost_rate = 0.1

    match_rate = {}
    is_internal = {}

    directors_len = len(directors)
    for i in range(directors_len):
        if directors[i].is_internal:
            is_internal[directors[i].name] = '1'
        else:
            is_internal[directors[i].name] = '0'
        match_rate[directors[i].name] = basic_rate + ((directors_len - i) / directors_len) * box_office_rate
        types = directors[i].movie_type.split(',')
        types_len = len(types)
        for t in range(types_len):
            if types[t] == theme:
                match_rate[directors[i].name] += ((types_len - t) / types_len) * theme_rate
                break

    if cost[0] != -1:
        directors = sorted(directors, key=lambda x: abs(x.avg_cost - (cost[1] - cost[0])))
        for i in range(directors_len):
            match_rate[directors[i].name] += ((directors_len - 1 - i) / directors_len) * cost_rate
    else:
        for d in directors:
            match_rate[d.name] += cost_rate

    if len(match_rate) == 0:
        return rt.no_response_data()

    match_rate = sorted(match_rate.items(), key=lambda x: x[1], reverse=True)
    result = []
    for m in match_rate:
        res = {
            'name': m[0],
            'nationality': is_internal[m[0]],
            'matching': (str(m[1] * 100))[:4] + '%',
            'referMovie': refer_movie[m[0]]
        }
        result.append(res)
    return jsonify({
        'code': '001',
        'msg': 'success',
        'directorList': result
    })


@box_office_test.route('/test_3_v2')
def query_actor():
    from predictBoxOffice import models
    level = request.args.get('level')
    box_office = request.args.get('boxOffice')
    theme = request.args.get('theme')
    cost = request.args.get('cost')

    if level is None or box_office is None or theme is None:
        return rt.necessary_param_needed()

    try:
        box_office = int(box_office)
        level = int(level)
        if not 1 <= box_office <= 8:
            return rt.param_cross_border()
        box_office = transform_box_office(box_office)
    except:
        return rt.param_type_error()

    if cost is not None:
        try:
            cost = int(cost)
            if not 1 <= cost <= 9:
                return rt.param_cross_border()
            cost = transform_cost(cost)
        except:
            return rt.param_type_error()
    else:
        cost = (-1, sys.maxsize)

    movies = models.Movie.query.filter(
        (box_office[0] <= models.Movie.box_office) & (models.Movie.box_office <= box_office[1])) \
        .filter((cost[0] <= models.Movie.production_cost) & (models.Movie.production_cost <= cost[1])) \
        .filter(models.Movie.movie_type.contains(theme)) \
        .all()
    actors = actors_from_movie(movies, level)
    refer_movie = {}
    for d in actors:
        refer_movie[d.name] = list(map(lambda x: x.name, filter(lambda x: x.starring.find(d.name) != -1, movies)))
    actors = sorted(actors, key=lambda x: abs(x.avg_box_office - (box_office[1] - box_office[0])))

    basic_rate = 0.5
    box_office_rate = 0.2
    theme_rate = 0.2
    cost_rate = 0.1

    match_rate = {}
    is_internal = {}

    actors_len = len(actors)
    for i in range(actors_len):
        if actors[i].is_internal:
            is_internal[actors[i].name] = '1'
        else:
            is_internal[actors[i].name] = '0'
        match_rate[actors[i].name] = basic_rate + ((actors_len - i) / actors_len) * box_office_rate
        types = actors[i].movie_type.split(',')
        types_len = len(types)
        for t in range(types_len):
            if types[t] == theme:
                match_rate[actors[i].name] += ((types_len - t) / types_len) * theme_rate
                break
    # do not consider cost for actor (no related data)
    for a in actors:
        match_rate[a.name] += cost_rate

    if len(match_rate) == 0:
        return rt.no_response_data()

    match_rate = sorted(match_rate.items(), key=lambda x: x[1], reverse=True)
    result = []
    for m in match_rate:
        res = {
            'name': m[0],
            'nationality': is_internal[m[0]],
            'matching': (str(m[1] * 100))[:4] + '%',
            'referMovie': refer_movie[m[0]]
        }
        result.append(res)
    return jsonify({
        'code': '001',
        'msg': 'success',
        'actorList': result
    })


def directors_level(directors, mtd, is_director=True):
    nums = {}
    if is_director:
        threshold = 0.2
    else:
        threshold = 0.2
    for k in mtd:
        for d in directors:
            if k[0] in d.movie_type:
                if k[0] not in nums:
                    nums[k[0]] = {}
                nums[k[0]][d.level] = nums[k[0]].get(d.level, 0) + 1
    for n in nums.values():
        counts = sum(list(n.values()))
        for k, v in n.copy().items():
            if v / counts < threshold:
                del n[k]
    theme_levels = {}

    for theme, levels in nums.items():
        level_rate = []
        counts = sum(list(levels.values()))
        for level, count in levels.items():
            l = {
                'level': str(level),
                'rate': str(round(count / counts, 2))
            }
            level_rate.append(l)
        if is_director:
            theme_levels[theme] = level_rate
        else:
            theme_levels[theme] = level_rate

    return theme_levels


def plot_bar(mtd):
    plt.bar(range(len(mtd)), list(map(lambda x: x[1], mtd)))
    plt.show()


@box_office_test.route('/test_1_v2')
def create_creator_test():
    from predictBoxOffice import models
    default_level = 2
    box_office = request.args.get('boxOffice')
    cost = request.args.get('cost')
    theme = request.args.get('theme')

    if box_office is None:
        return rt.necessary_param_needed()
    try:
        box_office = int(box_office)
        if not 1 <= box_office <= 8:
            return rt.param_cross_border()
        box_office = transform_box_office(box_office)
    except Exception:
        return rt.param_type_error()

    if cost is not None:
        try:
            cost = int(cost)
            if not 1 <= cost <= 9:
                return rt.param_cross_border()
            cost = transform_cost(cost)
        except:
            return rt.param_type_error()
    else:
        cost = (-1, sys.maxsize)

    movies = models.Movie.query.filter(
        (box_office[0] <= models.Movie.box_office) & (models.Movie.box_office <= box_office[1])) \
        .filter((cost[0] <= models.Movie.production_cost) & (models.Movie.production_cost <= cost[1])).all()
    movie_type_dict = {}
    directors, actors = director_actor(movies)
    for i in range(len(movies)):
        types = movies[i].movie_type.split(',')
        for t in types:
            movie_type_dict[t] = movie_type_dict.get(t, 0) + 1

    mtd = sorted(movie_type_dict.items(), key=lambda x: x[1], reverse=True)[:6]
    if not len(mtd):
        return rt.no_response_data()

    director_level = directors_level(directors, mtd)
    actor_level = directors_level(actors, mtd, is_director=False)

    if theme is not None:
        if theme not in list(map(lambda x: x[0], mtd)):
            return rt.no_response_data()
        else:
            res = {
                'theme': theme,
                'directorLevels': director_level[theme],
                'actorLevels': actor_level[theme]
            }
            return jsonify({
                'code': '001',
                'msg': 'success',
                'schemes': res
            })

    result = []
    for i in range(len(mtd)):
        res = {
            'theme': mtd[i][0],
            'directorLevels': director_level[mtd[i][0]],
            'actorLevels': actor_level[mtd[i][0]]
        }
        result.append(res)
    return jsonify({
        'code': '001',
        'msg': 'success',
        'schemes': result
    })


def directors_from_movie(movies, level):
    from predictBoxOffice import models
    directors = {}
    for i in range(len(movies)):
        d = movies[i].director.split(',')
        for k in d:
            j = models.Director.query.filter(models.Director.name == k) \
                .filter(models.Director.level == level) \
                .limit(1).all()
            if len(j) == 0:
                continue
            directors[k] = j[0]

    return list(directors.values())


def actors_from_movie(movies, level):
    from predictBoxOffice import models
    actors = {}
    for i in range(len(movies)):
        # only consider the starring
        d = movies[i].starring.split(',')[:2]
        for k in d:
            j = models.Actor.query.filter(models.Actor.name == k) \
                .filter(models.Actor.level == level) \
                .limit(1).all()
            if len(j) == 0:
                continue
            actors[k] = j[0]

    return list(actors.values())


def director_actor(movies):
    from predictBoxOffice import models
    directors = {}
    actors = {}
    for i in range(len(movies)):
        d = movies[i].director.split(',')
        for k in d:
            j = models.Director.query.filter(models.Director.name == k).limit(1).all()
            if len(j) == 0:
                continue
            directors[k] = j[0]
        a = movies[i].starring.split(',')
        for k in a:
            j = models.Actor.query.filter(models.Actor.name == k).limit(1).all()
            if len(j) == 0:
                continue
            actors[k] = j[0]
    return list(directors.values()), list(actors.values())


def transform_box_office(param):
    return {
        1: (0, 10),
        2: (10, 20),
        3: (20, 30),
        4: (30, 40),
        5: (40, 50),
        6: (50, 100),
        7: (100, 200),
        8: (200, sys.maxsize)
    }.get(param)
    # return {
    #         1: (0, 4),
    #         2: (4, 7),
    #         3: (7, 10),
    #         4: (10, 14),
    #         5: (14, 20),
    #         6: (20, 40),
    #         7: (40, 70),
    #         8: (70, sys.maxsize)
    #     }.get(param)


def transform_cost(param):
    return {
        1: (0, 1),
        2: (1, 2),
        3: (2, 3),
        4: (3, 5),
        5: (5, 8),
        6: (8, 12),
        7: (12, 20),
        8: (20, 30),
        9: (30, sys.maxsize)
    }.get(param)


def analyse_box_office():
    from predictBoxOffice import models
    movies = models.Movie.query.all()
    box_office = list(map(lambda x: x.box_office, movies))
    box_office = sorted(box_office)
    count = int(len(box_office) / 8)
    print(count)
    print(len(movies))
    count_dict = {}
    for b in box_office:
        if 0 < b <= 10:
            count_dict[(0, 10)] = count_dict.get((0, 10), 0) + 1
        elif 10 < b <= 20:
            count_dict[(10, 20)] = count_dict.get((10, 20), 0) + 1
        elif 20 < b <= 30:
            count_dict[(20, 30)] = count_dict.get((20, 30), 0) + 1
        elif 30 < b <= 40:
            count_dict[(30, 40)] = count_dict.get((30, 40), 0) + 1
        elif 40 < b <= 50:
            count_dict[(40, 50)] = count_dict.get((40, 50), 0) + 1
        elif 50 < b <= 100:
            count_dict[(50, 100)] = count_dict.get((50, 100), 0) + 1
        elif 100 < b <= 200:
            count_dict[(100, 200)] = count_dict.get((100, 200), 0) + 1
        elif 200 < b <= sys.maxsize:
            count_dict[(200, sys.maxsize)] = count_dict.get((200, sys.maxsize), 0) + 1
    k = 0
    print(count_dict)
    for i in box_office:
        if k == count:
            print(i)
            k = 0
            continue
        k += 1
    plt.bar(range(len(box_office)), box_office)
    # plt.show()


if __name__ == '__main__':
    analyse_box_office()
