## 简介  
这次项目主要以督促大家学习，提供一起交流学习的机会为目的。尝试使用各种算法(由易到难)解决手写体数字识别难题。  
大家把每日学习心得都可以写进日报，包括学习到的理论算法知识，网上get到的好的代码，好的特征处理方法，或者自己看到的好的文章。  
# 2018-04-08  
### @GoodTroyTang  
1.已完成  
  (1) 学习数据降维度  
2.在数字识别上运用数据降维。  
3.随笔  
~~~ python
def analyse_data(dataMat):
    # 求均值
    meanVals = np.mean(dataMat, axis=0)
    # 减去均值
    meanRemoved = dataMat-meanVals
    # 求协方差。
    covMat = np.cov(meanRemoved, rowvar=0)
    # 用numpy里面的模块求特征值和特征向量。
    eigvals, eigVects = np.linalg.eig(np.mat(covMat))
    # 对特征值进行从小到大排序
    eigValInd = np.argsort(eigvals)
    
    topNfeat = 100
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    cov_all_score = float(sum(eigvals))
    sum_cov_score = 0
    
    # 遍历输出下面的主成分，方差占比以及累积方差。
    for i in range(0, len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' % (format(i+1, '2.0f'), 
            format(line_cov_score/cov_all_score*100, '4.2f'), format(sum_cov_score/cov_all_score*100, '4.1f')))  
 ~~~ 



 
# 2018-04-08   
### @joxhehe   
KNN算法简介   
## 1.	算法概述   
kNN算法又称为k近邻分类(k-nearest neighbor classification)算法。   
最简单平凡的分类器也许是那种死记硬背式的分类器，记住所有的训练数据，对于新的数据则直接和训练数据匹配，如果存在相同属性的训练数据，则直接用它的分类来作为新数据的分类。这种方式有一个明显的缺点，那就是很可能无法找到完全匹配的训练记录。   
kNN算法则是从训练集中找到和新数据最接近的k条记录，然后根据他们的主要分类来决定新数据的类别。该算法涉及3个主要因素：训练集、距离或相似的衡量、k的大小。   
## 2.	算法要点   
### 2.1	计算步骤如下：   
1）	算距离：计算测试数据与各个训练数据之间的距离；   
2）	做排序：按照距离的递增关系进行排序；   
3）	找邻居：选取距离最小的K个点；   
4）	算频率：确定前K个点所在类别的出现频率；   
5）	做分类：返回前K个点中出现频率最高的类别作为测试数据的预测分类。　　　　
### 2.2 类别的判定  　
1）	投票决定：少数服从多数，近邻中哪个类别的点最多就分为该类。   
2）	加权投票法：根据距离的远近，对近邻的投票进行加权，距离越近则权重越大（权重为距离平方的倒数）。  




# 2018-04-09   
### @xiaocaijizzz
复习了西瓜书的knn概念
准备把仓库里的有用资料拉下来学习一下，入手较慢，会抓紧==
