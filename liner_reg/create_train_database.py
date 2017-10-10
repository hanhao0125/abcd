from sqlalchemy import Column, String, Integer, Float, SmallInteger, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import xlrd

Base = declarative_base()
engine = create_engine('sqlite:///train.db')

DBSession = sessionmaker(bind=engine)


# all info of movie
class Movie(Base):
    __tablename__ = 'movie'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(20))
    index = Column(Integer)
    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    box_office = Column(Float)
    director = Column(String(100))
    actor = Column(String(200))
    genre = Column(String(100))
    location = Column(SmallInteger)
    screen_3day = Column(Integer)
    screen_30day = Column(Integer)
    search = Column(Integer)
    holiday = Column(SmallInteger)
    competition = Column(SmallInteger)
    budget = Column(Float)


class TrainMovie(Base):
    __tablename__ = 'train_movie'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(20))
    index = Column(Integer)
    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    box_office = Column(Float)
    director = Column(String(100))
    actor = Column(String(200))
    genre = Column(String(100))
    location = Column(SmallInteger)
    screen_3day = Column(Integer)
    screen_30day = Column(Integer)
    search = Column(Integer)
    holiday = Column(SmallInteger)
    competition = Column(SmallInteger)
    budget = Column(Float)


# actor data
class Actor(Base):
    __tablename__ = 'actor'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(32))
    douban_id = Column(String(32))
    sex = Column(String(2))
    birthplace = Column(String(64))
    birth_year = Column(Integer)
    profession = Column(String(32))
    fans_num = Column(Integer)
    best_scores = Column(String(32))


# director data
class Director(Base):
    __tablename__ = 'director'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(32))
    douban_id = Column(String(32))
    sex = Column(String(2))
    birthplace = Column(String(64))
    birth_year = Column(Integer)
    profession = Column(String(32))
    fans_num = Column(Integer)
    best_scores = Column(String(32))


# movie's search
class Search(Base):
    __tablename__ = 'search'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    num = Column(Integer)


# screens number total
class Screen(Base):
    __tablename__ = 'screen'
    id = Column(Integer, primary_key=True, autoincrement=True)
    index = Column(Integer)
    screen_3day = Column(Integer)
    screen_30day = Column(Integer)


# movie's cost
class Cost(Base):
    __tablename__ = 'movie_cost'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    cost = Column(Float)


# movie's companies
class Company(Base):
    __tablename__ = 'company'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    index = Column(Integer)
    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    box_office = Column(Float)
    production = Column(String(100))
    issue = Column(String(200))
    joint_production = Column(String(300))
    joint_issue = Column(String(300))


# companies's info
class Company_Info(Base):
    __tablename__ = 'company_info'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    type = Column(String(20))
    box_office = Column(Float)
    num = Column(Integer)
    avg_box_office = Column(Float)


def init_db():
    Base.metadata.create_all(engine)


def drop_db():
    Base.metadata.drop_all(engine)


session = DBSession()


def write_movie():
    data = xlrd.open_workbook('movie data/all info of movie.xls').sheets()[0]
    for i in range(1, data.nrows):
        value = data.row_values(i)
        movie = Movie(name=value[0], index=value[1], year=value[2]
                      , month=value[3], day=value[4], box_office=value[5],
                      director=value[6], actor=value[7], genre=value[8], location=value[9], screen_3day=value[10],
                      screen_30day=value[11], search=value[12], holiday=value[13], competition=value[14],
                      budget=value[15])
        session.add(movie)
    session.commit()
    session.close()


def write_train_movie():
    data = xlrd.open_workbook('movie data/TrainMovieData.xls').sheets()[0]
    for i in range(1, data.nrows):
        value = data.row_values(i)
        movie = TrainMovie(name=value[0], index=value[1], year=value[2]
                           , month=value[3], day=value[4], box_office=value[5],
                           director=value[6], actor=value[7], genre=value[8], location=value[9], screen_3day=value[10],
                           screen_30day=value[11], search=value[12], holiday=value[13], competition=value[14],
                           budget=value[15])
        session.add(movie)
    session.commit()
    session.close()


def write_actor():
    data = xlrd.open_workbook('movie data/actor data(1).xls').sheets()[0]
    for i in range(1, data.nrows):
        value = data.row_values(i)
        actor = Actor(name=value[0], douban_id=value[1], sex=value[2]
                      , birthplace=value[3], birth_year=value[4], profession=value[5],
                      fans_num=value[6], best_scores=value[7])
        session.add(actor)
    session.commit()
    session.close()


def write_director():
    data = xlrd.open_workbook('movie data/director data(1).xls').sheets()[0]
    for i in range(1, data.nrows):
        value = data.row_values(i)
        director = Director(name=value[0], douban_id=value[1], sex=value[2]
                            , birthplace=value[3], birth_year=value[4], profession=value[5],
                            fans_num=value[6], best_scores=value[7])
        session.add(director)
    session.commit()
    session.close()


def write_search():
    data = xlrd.open_workbook("movie data/movie's search.xls").sheets()[0]
    for i in range(1, data.nrows):
        value = data.row_values(i)
        search = Search(name=value[0], num=value[1])
        session.add(search)
    session.commit()
    session.close()


def write_screen():
    data = xlrd.open_workbook("movie data/screens number total.xls").sheets()[0]
    for i in range(1, data.nrows):
        value = data.row_values(i)
        session.add(Screen(index=value[0], screen_3day=value[1], screen_30day=value[2]))
    session.commit()
    session.close()


def write_cost():
    data = xlrd.open_workbook("movie data/movie's cost.xls").sheets()[0]
    for i in range(1, data.nrows):
        value = data.row_values(i)
        session.add(Cost(name=value[0], cost=value[1]))
    session.commit()
    session.close()


def write_company():
    data = xlrd.open_workbook("movie data/movie's companies.xls").sheets()[0]
    for i in range(1, data.nrows):
        value = data.row_values(i)
        session.add(
            Company(name=value[0], index=value[1], year=value[2], month=value[3], day=value[4], box_office=value[5],
                    production=value[6], issue=value[7], joint_production=value[8], joint_issue=value[9]))
    session.commit()
    session.close()


def write_company_info():
    data = xlrd.open_workbook("movie data/companies's info.xls").sheets()[0]
    for i in range(1, data.nrows):
        value = data.row_values(i)
        session.add(
            Company_Info(name=value[0], type=value[1], box_office=value[2], num=value[3], avg_box_office=value[4]))
    session.commit()
    session.close()


def write_all():
    write_movie()
    write_cost()
    write_screen()
    write_director()
    write_actor()
    write_company()
    write_company_info()
    write_search()
    write_train_movie()


if __name__ == '__main__':
    write_all()
