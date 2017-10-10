if __name__ == '__main__':
    from predict import model22,model11,sess22,sess11,predict_box

    print(predict_box(["速度与激情8", "F·加里·格雷", "范·迪塞尔,道恩·强森,查理兹·塞隆,杰森·斯坦森,米歇尔·罗德里格兹",
                       "动作,犯罪", 2017, 3, 362, 4, 3, 520000, 2333999], model11, sess11, is_3rd=True))
    print(predict_box(["速度与激情8", "F·加里·格雷", "范·迪塞尔,道恩·强森,查理兹·塞隆,杰森·斯坦森,米歇尔·罗德里格兹",
                       "动作,犯罪", 2017, 3], model22, sess22, is_3rd=False))
    print(predict_box(["速度与激情8", "F·加里·格雷", "范·迪塞尔,道恩·强森,查理兹·塞隆,杰森·斯坦森,米歇尔·罗德里格兹",
                       "动作,犯罪", 2017, 3, 362, 4, 3, 520000, 2333999], model11, sess11, is_3rd=True))
    print(predict_box(["速度与激情8", "F·加里·格雷", "范·迪塞尔,道恩·强森,查理兹·塞隆,杰森·斯坦森,米歇尔·罗德里格兹",
                       "动作,犯罪", 2017, 3], model22, sess22, is_3rd=False))