import xlrd
import xlwt
from predictBoxOffice import db, models
import pandas as pd

# write actor and director info to excel from db
def actor_director_to_excel():
    data = xlwt.Workbook()
    table = data.add_sheet('director')
    directors = models.Director.query.all()
    for i in range(len(directors)):
        table.write(i, 0, directors[i].name)
        table.write(i, 1, directors[i].level)
        table.write(i, 2, directors[i].movie_type)
        table.write(i, 3, directors[i].avg_box_office)
        table.write(i, 4, directors[i].avg_cost)
    actors = models.Actor.query.all()
    table = data.add_sheet('actor')
    for i in range(len(actors)):
        table.write(i, 0, actors[i].name)
        table.write(i, 1, actors[i].level)
        table.write(i, 2, actors[i].movie_type)
        table.write(i, 3, actors[i].avg_box_office)
    data.save('director_actor.xlsx')


def update_is_internal():
    actor_is_internal = {}
    director_is_internal = {}
    data = pd.read_excel('data_excel/actor and director.xlsx')
    data = data[[0, 5]].values
    for d in data:
        if d[1] == 1:
            director_is_internal[d[0]] = 1
        else:
            director_is_internal[d[0]] = 0
    data = pd.read_excel('data_excel/actor and director.xlsx', sheetname='actor')
    data = data[[0, 4]].values
    for d in data:
        if d[1] == 1:
            actor_is_internal[d[0]] = 1
        else:
            actor_is_internal[d[0]] = 0
    print(actor_is_internal)
    print(director_is_internal)
    actors = models.Actor.query.all()
    for a in actors:
        a.is_internal = actor_is_internal[a.name]
    directors = models.Director.query.all()
    for d in directors:
        d.is_internal = director_is_internal[d.name]
    db.session.commit()


# update director level with new K-means: use scores and box office
# data from data_excel/director_level.xls
def update_director_level():

    director_level = {}
    data = pd.read_excel('data_excel/director_level1.xls')
    data = data[[0, 1]].values
    for d in data:
        director_level[d[0]] = d[1]
    directors = models.Director.query.all()
    for d in directors:
        d.level = director_level.get(d.name, 2)
    db.session.commit()
if __name__ == '__main__':
    update_director_level()
