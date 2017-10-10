import pickle
import math
abs_path = '/home/hanhao/abc/predictBoxOffice/bayes/'


holiday_dic = {
    '春节':10, '端午,暑期':9, '国庆,中秋':8, '暑期':7, '端午':6,
    '国庆':5, '清明':4, '五一':3, '元旦':2, '中秋':1, '其他':0
}

# 读取概率文件
def read_pickle_data(filename):
    fo = open(abs_path + filename,'rb')
    data = pickle.load(fo)
    fo.close()
    return data

# 产生预测结果概率表
def predict_boxoffice_naive(prob_list_2, test_set, prob_c):
    real_pre_list = []
    result_with_prob = []
    for i in range(len(test_set)):
        result_prob = []
        feature = test_set[i]
        for elem in prob_c:
            elem_prob = math.log(prob_c[elem])
            for j in range(1, len(feature)):
                # print((0, i))
                # print(prob_list_2[(0, i)])
                # print(prob_list_2[(0, i)][(feature[0], feature[i])])
                try:
                    elem_prob += math.log(prob_list_2[(0, j)][(elem, feature[j])])
                except:
                    elem_prob += 0
            result_prob.append([elem, elem_prob])
        result = max(result_prob, key = lambda b: b[1])
        result_with_prob.append(sorted(result_prob,key=lambda b:b[1], reverse=True))
        real_pre_list.append([feature[0], result[0]])
    return real_pre_list, result_with_prob

#得到合理的概率表
def soft_max(prob_list):
    # print(prob_list)
    pro_sum = sum([math.e ** prob[1] for prob in prob_list])
    # print(pro_sum)
    real_pro_list = [math.e ** prob[1] / pro_sum for prob in prob_list]
    # print(real_pro_list)
    label_list = [prob[0] for prob in prob_list]
    # print(label_list)
    label_prob_list = [[label_list[i], real_pro_list[i]] for i in range(len(real_pro_list))]
    return label_prob_list

# 将人名转化为值
def get_name_value(j, duty, movie_info):
    actor_dic = read_pickle_data('actor.data')
    genre_dic = read_pickle_data('genre.data')
    movie_dic = read_pickle_data('movie box-office.data')
    name = movie_info[j]
    num = 0
    box_ave = 0
    if duty != 'genre':
        for i in name.split(','):
            try:
                movie_list = actor_dic[i][duty]
            except:
                movie_list = []
            for index in movie_list:
                num += 1
                try:
                    box_ave += movie_dic[index]['boxoffice']
                except:
                    print('couldn\'t find %s infomation'%i)
        try:
            movie_info[j] = box_ave/num
        except:
            movie_info[j] = 0
    elif duty == 'genre':
        for i in movie_info[j].split(','):
            num += 1
            box_ave += genre_dic[i][2]
        try:
            movie_info[j] = box_ave/num
        except:
            movie_info[j] = 0
    # print(box_ave, num, box_ave/num)

# ['0 boxoffice', '1 year', '2 director', '3 actor',
    # '4 genre', '5 location', '6 screen 3day', '7 screen 30day',
    # '8 search', '9 holiday', '10 competition', '11 budget']

# 输入需要的参数，可以返回一张概率表，和预测值区间
def get_prob_list(director, actor, genre, year=2017, budget=0, location=0, screen3days = -1,
                  screen30days = -1, search = -1, holiday = '其他', competition = -1, stage = 2
                  ):
    movie_info = [year, director, actor, genre, location,
                  screen3days, screen30days, search, holiday_dic[holiday], competition, budget]

    class_index = [1, 2, 3, 5, 6, 7, 10]
    get_name_value(1, 'director', movie_info)
    get_name_value(2, 'actor', movie_info)
    get_name_value(3, 'genre', movie_info)
    for j in class_index:
        class_number = read_pickle_data('class value %d.data' % (j + 1))
        for k in range(len(class_number)):
            if movie_info[j] <= -1:
                break
            if movie_info[j] <= class_number[k]:
                movie_info[j] = k
                break
            if movie_info[j] > class_number[-1]:
                movie_info[j] = len(class_number) - 1
                break
    movie_info = [1] + movie_info
    if stage == 2:
        prob_list = read_pickle_data('second stage prob list.data')
        box_office_dict = {1: [0, 10], 2: [10, 20], 3: [20, 30],
                           4: [30, 40], 5: [40, 50], 6: [50, 100],
                           7: [100, 200], 8: [200]}
    else:
        prob_list = read_pickle_data('third stage prob list.data')
        box_office_dict = {1: [0, 5], 2: [5, 10], 3: [10, 15],
                           4: [1.50, 20], 5: [20, 30], 6: [40, 50],
                           7: [40, 50], 8: [50, 100], 9:[100, 200],
                           10: [200, ]}
    prob_c = prob_list['prob_c']
    prob_list_2 = prob_list['prob_list_2']

    result_list_2, result_list_prob = predict_boxoffice_naive(prob_list_2, [movie_info], prob_c)


    # print(box_office_dict[result_list_2[0][1]])
    # print(result_list_prob)
    label_prob_list = soft_max(result_list_prob[0])
    return label_prob_list, box_office_dict

if __name__ == '__main__':
    # '春节', '端午,暑期', '国庆,中秋', '暑期', '端午',
    # '国庆', '清明', '五一', '元旦', '中秋', '其他'

    prob_list, box_range = get_prob_list(director = '王晶,关智耀',
                                         actor = '甄子丹,刘德华,姜皓文,郑则仕,刘浩龙',
                                        budget=0, competition=5,
                                         genre='犯罪,动作', holiday='国庆,中秋', stage=2)
    print('概率表',prob_list[:3])
    print('分类结果和区间1', prob_list[0][0], box_range[prob_list[0][0]])
    print('分类结果和区间2', prob_list[1][0], box_range[prob_list[1][0]])
    print('分类结果和区间3', prob_list[2][0], box_range[prob_list[2][0]])



