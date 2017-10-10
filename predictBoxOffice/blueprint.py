# -*- coding: utf-8 -*-
from flask import Blueprint, jsonify, request
from functools import wraps
from predictBoxOffice import result_type as rt
import sys
import datetime
import json

sys.path.append('/home/hanhao/abc/predictBoxOffice/MLP')
from .MLP.predict import predict_box as neural_network_model

predict_box_office = Blueprint('predictBoxOffice', __name__)
from .MLP.predict import sess11, sess22, model11, model22
from .bayes.NaiveBayesPredict import get_prob_list as bayes_model


@predict_box_office.errorhandler(404)
def service_not_found():
    return rt.service_not_found()


@predict_box_office.errorhandler(500)
def server_error():
    return rt.server_error()


def is_authenticated(channel_code):
    from config import CHANNEL_CODE
    return channel_code == CHANNEL_CODE


def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated(request.args.get('channelCode')):
            return rt.unauthorized()
        return f(*args, **kwargs)

    return decorated_function


@predict_box_office.route('/queryDirector')
@auth_required
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
        if not 0 <= box_office <= 8:
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


@predict_box_office.route('/queryActor')
@auth_required
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
        if not 0 <= box_office <= 8:
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


@predict_box_office.route('/createCreator')
@auth_required
def create_creator_test():
    from predictBoxOffice import models
    box_office = request.args.get('boxOffice')
    cost = request.args.get('cost')
    theme = request.args.get('theme')

    if box_office is None:
        return rt.necessary_param_needed()
    try:
        box_office = int(box_office)
        if not 0 <= box_office <= 8:
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

    director_level = directors_actors_level(directors, mtd)
    actor_level = directors_actors_level(actors, mtd, is_director=False)

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
        0: (0, 5),
        1: (5, 10),
        2: (10, 20),
        3: (20, 30),
        4: (30, 40),
        5: (40, 50),
        6: (50, 100),
        7: (100, 200),
        8: (200, sys.maxsize)
    }.get(param)


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
        d = movies[i].starring.split(',')
        for k in d:
            j = models.Actor.query.filter(models.Actor.name == k) \
                .filter(models.Actor.level == level) \
                .limit(1).all()
            if len(j) == 0:
                continue
            actors[k] = j[0]

    return list(actors.values())


def directors_actors_level(directors_actors, mtd, is_director=True):
    nums = {}
    if is_director:
        threshold = 0.2
    else:
        threshold = 0.2
    for k in mtd:
        for d in directors_actors:
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


@predict_box_office.route('/predictIssueBoxOffice')
@auth_required
def stage2():
    film_name = request.args.get('filmName')
    cost = request.args.get('cost')
    theme = request.args.get('theme')
    director = request.args.get('director')
    actor = request.args.get('actor')
    release_date = request.args.get('releaseDate')
    if cost is None or theme is None or director is None or actor is None or release_date is None:
        return rt.necessary_param_needed()
    try:
        cost = int(cost)
    except:
        return rt.param_type_error()
    nn_res = neural_network_res(film_name=film_name, director=director, actor=actor,
                                genre=theme, release_date=release_date, is_3rd=False)
    b_res = bayes_res(director=director, actor=actor,
                      genre=theme, release_date=release_date, cost=cost)
    dt_res = [0, 5]
    box_office = vote(nn_res, b_res, dt_res)
    return jsonify({
        'code': '001',
        'msg': 'success',
        'data': {
            'boxOfficeMin': box_office[0],
            'boxOfficeMax': box_office[1]
        }})


@predict_box_office.route('/predictReleaseBoxOffice')
@auth_required
def stage31():
    film_name = request.args.get('filmName')
    cost = request.args.get('cost')
    theme = request.args.get('theme')
    director = request.args.get('director')
    actor = request.args.get('actor')
    competition = request.args.get('competition')
    topic = request.args.get('topic')
    screen3 = request.args.get('screen3')
    screen30 = request.args.get('screen30')
    release_date = request.args.get('releaseDate')

    if cost is None or theme is None or director is None or actor is None or release_date is None \
            or topic is None or screen3 is None or competition is None:
        return rt.necessary_param_needed()
    if screen30 is None:
        screen30 = 0
    try:
        cost = float(cost)
        competition = int(competition)
        topic = float(topic)
        screen3 = int(screen3)
        screen30 = int(screen30)
    except:
        return rt.param_type_error()
    nn_res = neural_network_res(film_name=film_name, director=director, actor=actor,
                                genre=theme, release_date=release_date, topic=topic,
                                screen3d=screen3, screen30d=screen30, competition=competition, is_3rd=True)
    b_res = bayes_res(director=director, actor=actor,
                      genre=theme, release_date=release_date, search=topic,
                      screen3days=screen3, screen30days=screen30, competition=competition, stage=3)
    dt_res = [0, 5]
    box_office = vote(nn_res, b_res, dt_res)
    return jsonify({
        'code': '001',
        'msg': 'success',
        'data': {
            'boxOfficeMin': box_office[0],
            'boxOfficeMax': box_office[1],
            'probability': str((nn_res[1] + b_res[1]) / 2)[:4]
        }})


@predict_box_office.route('/predictReleaseBoxOfficeCompare', methods=['POST'])
@auth_required
def stage32():
    films = json.loads(str(request.get_data(), encoding='utf-8'))['data']
    rank = {}
    for f in films:
        check_param(f)
        nn_res = neural_network_res(film_name=f['filmName'], director=f['director'], actor=f['actor'],
                                    genre=f['theme'], release_date=f['releaseDate'], competition=f['competition'],
                                    topic=f['topic'], screen3d=f['screen3'], screen30d=f.get('screen30', 0),
                                    is_3rd=True)
        b_res = bayes_res(director=f['director'], actor=f['actor'],
                          genre=f['theme'], release_date=f['releaseDate'], search=f['topic'],
                          screen3days=f['screen3'], screen30days=f.get('screen30', 0), competition=f['competition'],
                          stage=3)
        dt_res = [0, 5]
        box_office = vote(nn_res, b_res, dt_res)
        rank[f['filmName']] = box_office[0]
    rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)
    result = []
    for i in range(len(rank)):
        res = {
            'filmName': rank[i][0],
            'ranking': i + 1
        }
        result.append(res)
    return jsonify({
        'code': '001',
        'msg': 'success',
        'filmList': result
    })


def vote(nn_res, b_res, dt_res):
    nn_res = tuple(nn_res[0])
    b_res = tuple(b_res[0])
    dt_res = tuple(dt_res)
    print(nn_res,b_res, dt_res)
    d = {}
    d[nn_res] = 1
    d[b_res] = d.get(b_res, 0) + 1
    d[dt_res] = d.get(dt_res, 0) + 1
    d = sorted(d.items(), key=lambda x: (x[1], x[0][0]), reverse=True)
    print(d)
    return d[0][0]


def neural_network_res(film_name=None, genre=None, director=None, actor=None, release_date=None,
                       competition=0, topic=None, screen3d=0, screen30d=0, is_3rd=False):
    topic_border = 762383 - 8
    try:
        date = [int(i) for i in release_date.split('.')]
        hs = holidays(date[0], date[1], date[2])
    except:
        return rt.param_type_error()
    if is_3rd:
        res = neural_network_model(
            [film_name, director, actor, genre, date[0], hs, topic_border * topic, date[1], competition, screen3d,
             screen30d], model11, sess11, is_3rd=True)
        box_office = box_office_level_stage3(res.argmax())
    else:
        res = neural_network_model([film_name, director, actor, genre, date[0], hs], model22, sess22, is_3rd=False)
        # res = neural_network_model(["速度与激情8", "F·加里·格雷", "范·迪塞尔,道恩·强森,查理兹·塞隆,杰森·斯坦森,米歇尔·罗德里格兹",
        #             "动作,犯罪", 2017, 3], model22, sess22, is_3rd=False)
        print(res)
        box_office = box_office_level_stage2(res.argmax())
    return box_office, round(res.max(), 2)


def bayes_res(director, actor, genre, screen3days=-1, release_date=None, cost=0,
              screen30days=-1, search=-1, competition=-1, stage=2
              ):
    topic_border = 762383 - 8
    try:
        date = [int(i) for i in release_date.split('.')]
        hs = holidays_name(date[0], date[1], date[2])
    except:
        return rt.param_type_error()

    prob_list, box_range = bayes_model(director=director, actor=actor,
                                       genre=genre, holiday=hs, year=date[0], search=search * topic_border,
                                       budget=cost, screen3days=screen3days, screen30days=screen30days,
                                       competition=competition,
                                       stage=stage)

    return box_range[prob_list[0][0]], round(prob_list[0][1], 2)


def check_param(film):
    if film['cost'] is None or film['theme'] is None or film['director'] is None or film['actor'] is None or \
                    film['releaseDate'] is None \
            or film['topic'] is None or film['screen3'] is None or film['competition'] is None:
        return rt.necessary_param_needed()


def box_office_level_stage2(index):
    return {
        0: (0, 10),
        1: (10, 20),
        2: (20, 30),
        3: (30, 40),
        4: (40, 50),
        5: (50, 100),
        6: (100, 200),
        7: (200, sys.maxsize)
    }.get(index)


def box_office_level_stage3(index):
    return {
        0: (0, 5),
        1: (5, 10),
        2: (10, 15),
        3: (15, 20),
        4: (20, 30),
        5: (30, 40),
        6: (40, 50),
        7: (50, 100),
        8: (100, 200),
        9: (200, sys.maxsize)
    }.get(index)


class Lunar(object):
    # ******************************************************************************
    # 下面为阴历计算所需的数据,为节省存储空间,所以采用下面比较变态的存储方法.
    # ******************************************************************************
    # 数组g_lunar_month_day存入阴历1901年到2050年每年中的月天数信息，
    # 阴历每月只能是29或30天，一年用12（或13）个二进制位表示，对应位为1表30天，否则为29天
    g_lunar_month_day = [
        0x4ae0, 0xa570, 0x5268, 0xd260, 0xd950, 0x6aa8, 0x56a0, 0x9ad0, 0x4ae8, 0x4ae0,  # 1910
        0xa4d8, 0xa4d0, 0xd250, 0xd548, 0xb550, 0x56a0, 0x96d0, 0x95b0, 0x49b8, 0x49b0,  # 1920
        0xa4b0, 0xb258, 0x6a50, 0x6d40, 0xada8, 0x2b60, 0x9570, 0x4978, 0x4970, 0x64b0,  # 1930
        0xd4a0, 0xea50, 0x6d48, 0x5ad0, 0x2b60, 0x9370, 0x92e0, 0xc968, 0xc950, 0xd4a0,  # 1940
        0xda50, 0xb550, 0x56a0, 0xaad8, 0x25d0, 0x92d0, 0xc958, 0xa950, 0xb4a8, 0x6ca0,  # 1950
        0xb550, 0x55a8, 0x4da0, 0xa5b0, 0x52b8, 0x52b0, 0xa950, 0xe950, 0x6aa0, 0xad50,  # 1960
        0xab50, 0x4b60, 0xa570, 0xa570, 0x5260, 0xe930, 0xd950, 0x5aa8, 0x56a0, 0x96d0,  # 1970
        0x4ae8, 0x4ad0, 0xa4d0, 0xd268, 0xd250, 0xd528, 0xb540, 0xb6a0, 0x96d0, 0x95b0,  # 1980
        0x49b0, 0xa4b8, 0xa4b0, 0xb258, 0x6a50, 0x6d40, 0xada0, 0xab60, 0x9370, 0x4978,  # 1990
        0x4970, 0x64b0, 0x6a50, 0xea50, 0x6b28, 0x5ac0, 0xab60, 0x9368, 0x92e0, 0xc960,  # 2000
        0xd4a8, 0xd4a0, 0xda50, 0x5aa8, 0x56a0, 0xaad8, 0x25d0, 0x92d0, 0xc958, 0xa950,  # 2010
        0xb4a0, 0xb550, 0xb550, 0x55a8, 0x4ba0, 0xa5b0, 0x52b8, 0x52b0, 0xa930, 0x74a8,  # 2020
        0x6aa0, 0xad50, 0x4da8, 0x4b60, 0x9570, 0xa4e0, 0xd260, 0xe930, 0xd530, 0x5aa0,  # 2030
        0x6b50, 0x96d0, 0x4ae8, 0x4ad0, 0xa4d0, 0xd258, 0xd250, 0xd520, 0xdaa0, 0xb5a0,  # 2040
        0x56d0, 0x4ad8, 0x49b0, 0xa4b8, 0xa4b0, 0xaa50, 0xb528, 0x6d20, 0xada0, 0x55b0,  # 2050
    ]

    # 数组gLanarMonth存放阴历1901年到2050年闰月的月份，如没有则为0，每字节存两年
    g_lunar_month = [
        0x00, 0x50, 0x04, 0x00, 0x20,  # 1910
        0x60, 0x05, 0x00, 0x20, 0x70,  # 1920
        0x05, 0x00, 0x40, 0x02, 0x06,  # 1930
        0x00, 0x50, 0x03, 0x07, 0x00,  # 1940
        0x60, 0x04, 0x00, 0x20, 0x70,  # 1950
        0x05, 0x00, 0x30, 0x80, 0x06,  # 1960
        0x00, 0x40, 0x03, 0x07, 0x00,  # 1970
        0x50, 0x04, 0x08, 0x00, 0x60,  # 1980
        0x04, 0x0a, 0x00, 0x60, 0x05,  # 1990
        0x00, 0x30, 0x80, 0x05, 0x00,  # 2000
        0x40, 0x02, 0x07, 0x00, 0x50,  # 2010
        0x04, 0x09, 0x00, 0x60, 0x04,  # 2020
        0x00, 0x20, 0x60, 0x05, 0x00,  # 2030
        0x30, 0xb0, 0x06, 0x00, 0x50,  # 2040
        0x02, 0x07, 0x00, 0x50, 0x03  # 2050
    ]

    START_YEAR = 1901

    # 月份
    # lm = '正二三四五六七八九十冬腊'
    lm = '010203040506070809101112'
    # 日份
    # ld = '初一初二初三初四初五初六初七初八初九初十十一十二十三十四十五十六十七十八十九二十廿一廿二廿三廿四廿五廿六廿七廿八廿九三十'
    ld = '010203040506070809101112131415161718192021222324252627282930'

    # 节气

    def __init__(self, dt=None):
        '''初始化：参数为datetime.datetime类实例，默认当前时间'''
        self.localtime = dt if dt else datetime.datetime.today()

    def ln_year(self):  # 返回农历年
        year, _, _ = self.ln_date()
        return year

    def ln_month(self):  # 返回农历月
        _, month, _ = self.ln_date()
        return month

    def ln_day(self):  # 返回农历日
        _, _, day = self.ln_date()
        return day

    def ln_date(self):  # 返回农历日期整数元组（年、月、日）（查表法）
        delta_days = self._date_diff()

        # 阳历1901年2月19日为阴历1901年正月初一
        # 阳历1901年1月1日到2月19日共有49天
        if (delta_days < 49):
            year = self.START_YEAR - 1
            if (delta_days < 19):
                month = 11
                day = 11 + delta_days
            else:
                month = 12
                day = delta_days - 18
            return (year, month, day)

        # 下面从阴历1901年正月初一算起
        delta_days -= 49
        year, month, day = self.START_YEAR, 1, 1
        # 计算年
        tmp = self._lunar_year_days(year)
        while delta_days >= tmp:
            delta_days -= tmp
            year += 1
            tmp = self._lunar_year_days(year)

        # 计算月
        (foo, tmp) = self._lunar_month_days(year, month)
        while delta_days >= tmp:
            delta_days -= tmp
            if month == self._get_leap_month(year):
                (tmp, foo) = self._lunar_month_days(year, month)
                if delta_days < tmp:
                    # return (0, 0, 0)
                    return year, month, delta_days + 1
                delta_days -= tmp
            month += 1
            (foo, tmp) = self._lunar_month_days(year, month)

        # 计算日
        day += delta_days
        return year, month, day

    def ln_date_str(self):  # 返回农历日期字符串，形如：农历正月初九
        _, month, day = self.ln_date()
        month1 = self.lm[(month - 1) * 2:month * 2]
        if month1.startswith('0'):
            month1 = month1[1]
        day1 = self.ld[(day - 1) * 2:day * 2]
        if day1.startswith('0'):
            day1 = day1[1]
        return int(month1), int(day1)

    def _date_diff(self):
        '''返回基于1901/01/01日差数'''
        return (self.localtime - datetime.datetime(1901, 1, 1)).days

    def _get_leap_month(self, lunar_year):
        flag = self.g_lunar_month[(lunar_year - self.START_YEAR) // 2]
        if (lunar_year - self.START_YEAR) % 2:
            return flag & 0x0f
        else:
            return flag >> 4

    def _lunar_month_days(self, lunar_year, lunar_month):
        if lunar_year < self.START_YEAR:
            return 30

        high, low = 0, 29
        iBit = 16 - lunar_month

        if lunar_month > self._get_leap_month(lunar_year) and self._get_leap_month(lunar_year):
            iBit -= 1

        if self.g_lunar_month_day[lunar_year - self.START_YEAR] & (1 << iBit):
            low += 1

        if lunar_month == self._get_leap_month(lunar_year):
            if self.g_lunar_month_day[lunar_year - self.START_YEAR] & (1 << (iBit - 1)):
                high = 30
            else:
                high = 29

        return high, low

    def _lunar_year_days(self, year):
        days = 0
        for i in range(1, 13):
            (high, low) = self._lunar_month_days(year, i)
            days += high
            days += low
        return days


def holidays(year, month, day):
    days = 0
    if month == 10:
        if 1 <= day <= 7:
            days = 8 - day
    elif month == 5:
        if 1 <= day <= 3:
            days = 4 - day
    elif month == 1:
        if 1 <= day <= 3:
            days = 4 - day
    elif month == 7 or month == 8:
        days = 7
    else:
        month, day = Lunar(datetime.datetime(year, month, day)).ln_date_str()
        if month == 12:
            if 25 <= day <= 30:
                days = 7
        elif month == 1:
            if 1 <= day <= 7:
                days = 7
        elif month == 5:
            if 5 <= day <= 7:
                days = 3
        elif month == 8:
            if 14 <= day <= 16:
                days = 3
    return days


def holidays_name(year, month, day):
    # '春节', '端午,暑期', '国庆,中秋', '暑期', '端午',
    # '国庆', '清明', '五一', '元旦', '中秋', '其他'
    if month == 10:
        if 1 <= day <= 7:
            return '国庆'
    elif month == 5:
        if 1 <= day <= 3:
            return '五一'
    elif month == 1:
        if 1 <= day <= 3:
            return '元旦'
    elif month == 7 or month == 8:
        return '暑期'
    else:
        month, day = Lunar(datetime.datetime(year, month, day)).ln_date_str()
        if month == 12:
            if 25 <= day <= 30:
                return '春节'
        elif month == 1:
            if 1 <= day <= 7:
                return '春节'
        elif month == 5:
            if 5 <= day <= 7:
                return '端午,暑期'
        elif month == 8:
            if 14 <= day <= 16:
                return '中秋'
    return '其他'
