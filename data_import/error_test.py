from predictBoxOffice import models
import matplotlib.pyplot as plt


def box_office_error():
    directors = models.Director.query.all()
    error_dict = {}
    error_list = []
    for c in range(2, 11):
        for i in range(7, 338):
            temp = sorted(directors, key=lambda x: abs(x.avg_box_office - i))[:c]
            error_dict[i] = sum(map(lambda x: abs(x.avg_box_office - i), temp)) / c
        error_list.append(error_dict)
        error_dict = {}
    plt.figure(figsize=(10, 6))
    for i in range(len(error_list)):
        x = []
        y = []
        for k, v in error_list[i].items():
            x.append(k)
            y.append(v)
        plt.plot(x, y, label='n = %d' % (i + 2))
        plt.legend()
    x = [146 for k in range(100)]
    y = [k for k in range(100)]
    plt.plot(x, y, label='divide')
    plt.legend()
    plt.xlabel('predict_box_office / ten million')
    plt.ylabel('avg error / ten million')
    plt.show()


def cost_error():
    directors = models.Director.query.order_by(models.Director.avg_cost).all()
    error_dict = {}
    error_list = []
    for c in range(2,11):
        for i in range(int(directors[0].avg_cost),int(directors[-1].avg_cost)):
            temp = sorted(directors, key=lambda x: abs(x.avg_cost - i))[:c]
            error_dict[i] = sum(map(lambda x: abs(x.avg_cost - i), temp)) / c
        error_list.append(error_dict)
        error_dict = {}
    plt.figure(figsize=(10, 6))
    for i in range(len(error_list)):
        x = []
        y = []
        for k, v in error_list[i].items():
            x.append(k)
            y.append(v)
        plt.plot(x, y, label='n = %d' % (i + 2))
        plt.legend()
    x = [146 for k in range(100)]
    y = [k for k in range(100)]
    plt.plot(x, y, label='divide')
    plt.legend()
    plt.xlabel('predict_box_cost / ten million')
    plt.ylabel('avg error / ten million')
    plt.show()

if __name__ == '__main__':
    cost_error()
