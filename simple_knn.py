'''
使用knn来判定一个新的电影是什么类型的电影
'''

import math


'''
movie_data里面是已知类型的电影及其各个镜头的数据，其中第一个数字是
喜剧镜头个数，第二个数字是爱情镜头个数，第三是打斗镜头个数
'''
movie_data={"宝贝当家":[45,2,9,"喜剧片"],
            "美人鱼":[21,17,5,"喜剧片"],
            "澳门风云3":[54,9,11,"喜剧片"],
            "功夫熊猫":[39,0,31,"喜剧片"],
            "谍影重重":[5,2,57,"动作片"],
            "叶问3":[3,2,65,"动作片"],
            "伦敦陷落":[2,3,55,"动作片"],
            "特工":[6,4,21,"动作片"],
            "奔爱":[7,46,4,"爱情片"],
            "夜孔雀":[9,39,8,"爱情片"],
            "代理情人":[9,39,2,"爱情片"],
            "步步惊心":[8,34,17,"爱情片"]}

#x是待分类电影的各个镜头的个数统计
x =[23,3,17]
KNN = []

#计算新电影到各训练样本的欧式距离，排序取前五并输出
for key,v in movie_data.items():
    distance = math.sqrt((x[0]-v[0])**2 + (x[1]-v[1])**2 + (x[2]-v[2])**2)
    KNN.append([key,round(distance,2)])  #保留distance的两位小数

KNN.sort(key=lambda dis:dis[1])
KNN=KNN[:5]

for movie_and_dis in KNN:
    print(str(movie_and_dis)+'\n')


#计算前k的样本所在类别的频率并统计类别
comedy = 0
action = 0
love_movie = 0

for movie in KNN:
    for key,value in movie_data.items():
        if movie[0] == key:
            if value[3] == "喜剧片":
                comedy +=1
            elif value[3] == "动作片":
                action += 1
            elif value[3] == "爱情片":
                love_movie += 1
            else:
                print("Error")
print("comedy:",comedy)
print("action:",action)
print("love_movie",love_movie)


#输出结果
if comedy>action:
    movie_type = "comedy"
elif love_movie>action:
    movie_type = "love movie"
else:
    move_type = "action"
print("New movie is prabably a ",movie_type)




                






