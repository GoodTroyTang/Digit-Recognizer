# 简介  
kaggle入门题目，训练数据已经处理成向量并与标签一一对应，判断测试数据对应的标签。  
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
    meanVals = np.mean(dataMat, axis=0)  
    meanRemoved = dataMat-meanVals  
    covMat = np.cov(meanRemoved, rowvar=0)  
    eigvals, eigVects = np.linalg.eig(np.mat(covMat))  
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
