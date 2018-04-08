# author = Li_ting_wei
import matplotlib.pyplot as plt
import pandas as pda
# 上层做数据分析用的,主要是做表格数据呈现
from sklearn.model_selection import train_test_split
# 加载数据方式1：
filename = "E:\iris.csv"
data = pda.read_csv(filename)
# 读取excel格式数据文件（不同格式有不同读取命令）
x = data.iloc[:, 0:4].as_matrix()
# 索引1-4列组成矩阵，逗号前冒号显示索引所有的行
y = data.iloc[:, 4:5].as_matrix()
# 索引第5列组成矩阵
# print(x, y, len(x), len(y)) 测试代码
'''
#加载数据方式2:（直接在python中调用）
from sklearn import datasets
irisdata = datasets.load_iris()
x = irisdata.data 
y = irisdata.target 
'''
# 使用KNN算法进行分类方式1：（自编函数）
accuracy_list = []
def KNN(k,x_test,x_train,y_trainlist):
    # k为最近邻个数
    # x_test 为测试数据
    # x_train 为训练数据
    # y_trainlist为分类结果列表
    predict_list = []
    # 定义KNN输出索引空列表（方便后面数据可视化）
    for test_num in range(len(x_test)):
        # test样本循环
        distance_list = []
        # 定义距离空列表
        for train_num in range(len(x_train)):
            # test样本与每个训练样本距离循环
            dif = x_test[test_num] - x_train[train_num]
            # test样本与train样本数组差值
            sqdif = dif ** 2
            # 求test样本与train样本距离平方
            sqdis = 0
            for i in range(len(sqdif)):
                sqdis += sqdif[i]
            distance = sqdis**0.5
                # 得到test样本与train样本欧氏距离
            distance_list.append(distance)
        # print(distance_list,len(distance_list)) # 测试代码（以上没有问题）
        vote_type = []
        # 定义近邻空列表
        for i in range(k):
            # k个最近邻点循环
            train_num = len(x_train)
            for j in range(train_num):
                a = min(distance_list)
                if distance_list[j] == a:
                    # 在train点中寻找最近邻点
                    vote_type.append(y_trainlist[j])
                    # 最近邻点分类类型
                    distance_list[j] = max(distance_list)
                    # 将选出的最近邻置为最大，方便后面寻找最近邻
                    break
                    # 选到一个最近邻就跳出寻找过程
        # print(len(vote_type), vote_type) # 测试代码
        numlist = []
        # 定义匹配次数空列表
        for i in range(p):
            # 在最近邻所分类型中循环出最多类型
            num = 0
            for j in range(k):
                if vote_type[j] == new_list[i]:
                    num += 1
                else:
                    num += 0
            numlist.append(num)
        # print(numlist) # 测试循环结果是否正确
        for i in range(p):
            if numlist[i] == max(numlist):
                # print('此测试样本预测为' + new_list[i] )
                predict_list.append(i)
                break
            continue
    # print(predict_list) # 测试KNN输出索引是否正确
    true_list = []
    # 定义真实索引空列表（方便后面数据可视化）
    for i in range(len(y_test)):
        # print(y_test[i][0]) # 测试KNN算法预测与真实结果的准确率（此次测试为100%）
        for j in range(p):
            if y_test[i][0] == new_list[j]:
                true_list.append(j)
                break
            continue
    # print(true_list) # 测试真实输出索引是否正确
    n = 0
    for i in range(len(y_test)):
        if predict_list[i] == true_list[i]:
            n += 1
            continue
        else:
            n += 0
    # print(n) # 测试预测正确个数输出是否正确
    accuracy = n / len(y_test)
    # print(accuracy)
    accuracy_list.append(accuracy)
    # print(accuracy_list)
    '''
    # 以下为数据可视化代码，不属于算法代码
    plt.plot(predict_list, linewidth=5, label='prediction')
    # label添加此曲线标注
    plt.title("KNN-Predict Method(K = 3,test_size=97)", fontsize=24)
    plt.xlabel('Num', fontsize=20)
    plt.ylabel('Type', fontsize=20)
    plt.plot(true_list, linewidth=2, label='verity')
    # label添加此曲线标注
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(loc='upper left')
    # 用以显示每条曲线的标注
    plt.text(122,2.15,r'accuracy='+ str(accuracy),color='red',fontsize=25 )
    # 正确率在图片中表示，(122,2.15 是text文本在图中坐标，自己调，字体大小也可调)
    plt.show()
    '''
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.97, random_state=20)
# 随机划分训练集和测试集
y_trainlist = []
# 定义训练数据分类空列表（由于y_train后面运用不便，此处构造一个列表）
for i in range(len(y_train)):
    j = y_train[i][0]
    y_trainlist.append(j)
new_list = list(set(y_trainlist))
# 将所有分类集合去重后化为列表，方便后面使用
# print(new_list) # 测试集合是否正确
p = len(set(y_trainlist))
KNN(3, x_test, x_train, y_trainlist)
# 查看给定值的预测情况(区别于下面的循环给值)
'''
size = 0
# 定义测试集占总数据比例
for j in range(97):
    # 以不同比例循环，得到不同比例下的准确率
    size += 0.01
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=20)
    # 随机划分训练集和测试集
    y_trainlist = []
    # 定义训练数据分类空列表（由于y_train后面运用不便，此处构造一个列表）
    for i in range(len(y_train)):
        j = y_train[i][0]
        y_trainlist.append(j)
    new_list = list(set(y_trainlist))
    # 将所有分类集合去重后化为列表，方便后面使用
    # print(new_list) # 测试集合是否正确
    p = len(set(y_trainlist))
    # print(p) # 检测共有多少种类型
    # print(y_trainlist)  # 测试代码
    KNN(1, x_test, x_train, y_trainlist)
listone = accuracy_list
# K=1时正确率列表
accuracy_list = []
size = 0
# 定义测试集占总数据比例
for j in range(97):
    # 以不同比例循环，得到不同比例下的准确率
    size += 0.01
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=20)
    # 随机划分训练集和测试集
    y_trainlist = []
    # 定义训练数据分类空列表（由于y_train为ndarray 后面运用不便，此处构造一个列表）
    for i in range(len(y_train)):
        j = y_train[i][0]
        y_trainlist.append(j)
    new_list = list(set(y_trainlist))
    # 将所有分类集合去重后化为列表，方便后面使用
    # print(new_list) # 测试集合是否正确
    p = len(set(y_trainlist))
    # print(p) # 检测共有多少种类型
    # print(y_trainlist)  # 测试代码
    KNN(3, x_test, x_train, y_trainlist)
listwo = accuracy_list
# K=3时的正确率列表
# print(accuracy_list,len(accuracy_list)) # 测试准确率列表输出是否正确
plt.plot(listone, linewidth=2, color='red', label='K=1')
plt.plot(listwo, linewidth=2, color='blue', label='K=3')
plt.title('Prediction_accuracy', fontsize=25)
plt.xlabel('Test_size', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.legend(loc='upper right')
plt.show()
# 不同K值准确率数据可视化
'''
#使用KNN算法进行分类（2）
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)
y2=model.predict(x_test)
print(y2)
'''