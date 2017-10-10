import requests
import xlrd

# 测试百度

def data():
    data = xlrd.open_workbook('../data_excel/movie_data.xlsx').sheets()[0]
    params = []
    for i in range(1, data.nrows):
        d = data.row_values(i)
        print(data.row_values(i))
        a = []
        year = int(d[2])
        m = int(d[3])
        day = int(d[4])
        date = str(year) + '.' + str(m) + '.' + str(day)
        a.append(d[0])
        a.append(d[1])
        a.append(d[13])
        a.append(d[14])
        a.append(d[15])
        a.append(date)
        params.append(a)
    print(params)
    return params


def data3():
    data = xlrd.open_workbook('../data_excel/movie_data.xlsx').sheets()[0]
    params = []
    for i in range(1, data.nrows):
        d = data.row_values(i)
        print(data.row_values(i))
        a = []
        year = int(d[2])
        m = int(d[3])
        day = int(d[4])
        date = str(year) + '.' + str(m) + '.' + str(day)
        a.append(d[0])
        a.append(d[1])
        a.append(d[13])
        a.append(d[14])
        a.append(d[15])
        a.append(date)
        a.append(d[6])
        a.append(d[16])
        a.append(d[10])

        params.append(a)
    print(params)
    return params


def data4():
    data = xlrd.open_workbook('../data_excel/13-17.xlsx').sheets()[0]
    params = []
    for i in range(1, data.nrows):
        d = data.row_values(i)
        print(data.row_values(i))
        a = []
        year = int(d[2])
        m = int(d[3])
        day = int(d[4])
        date = str(year) + '.' + str(m) + '.' + str(day)
        a.append(d[0])
        a.append(d[5])
        a.append(d[6])
        a.append(d[7])
        a.append(d[8])

        a.append(date)
        a.append(d[14])
        a.append(d[12])
        a.append(d[10])

        params.append(a)
    print(params)
    return params
def baidu_func(url):
    res = {}
    import random
    d = random.sample(data(), 40)
    for i in d:
        try:
            params = {
                "channelCode": '0001',
                "filmName": i[0],
                "director": i[2].replace(' ',','),
                "actor": i[3].replace(' ',','),
                "theme": i[4].replace(' ',','),
                "releaseDate": i[5],
                "cost": 3
            }
            req = requests.get(url, params=params)
            res[i[0]] = (req.json()['data']['boxOfficeMin'] ,req.json()['data']['boxOfficeMax'], i[1] / 1000)
        except:
            print('error')
    print(res)


def stage3(url):
    res = {}
    import random
    d = random.sample(data4(), 40)
    print(d)
    for i in d:
        try:
            params = {
                    "channelCode": '0001',
                    "filmName": i[0],
                    "director": i[2],
                    "actor": i[3],
                    "theme": i[4],
                    "releaseDate": i[5],
                    "cost": 3,
                    "screen3": int(i[-1]),
                    "competition": int(i[-3]),
                    "topic": i[-2] / 762383
                }
            print(params)
            req = requests.get(url, params=params)
            print(req.json())
            res[i[0]] = ((req.json()['data']['boxOfficeMin'], req.json()['data']['boxOfficeMax']), i[1] / 1000)
        except:
            print('error')
    print(res)
if __name__ == '__main__':
    url = "http://120.77.172.14:5000/predictBoxOffice/predictReleaseBoxOffice"
    stage3(url)
