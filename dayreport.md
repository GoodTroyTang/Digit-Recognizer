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
### @Fuyingtao
**今日完成**
## kNN算法部分
了解kNN算法的基本原理  
Kaggle入门：练手项目参考(https://blog.csdn.net/bbbeoy/article/details/73274931).
## Python部分
Python基本知识  
&space;1） Python中文编码解码原理[参考博客]（https://www.cnblogs.com/OldJack/p/6658779.  html）;（http://python.jobbole.com/81244/）.  
2） Python中的变量作用域问题[参考博客](http://www.jb51.net/article/86766.htm).  
3） Python中class和object对应的变量方法类型[参考博客](https://www.cnblogs.com/20150705-yilushangyouni-Jacksu/p/6238187.html).  
**明日计划**
实现kNN的Python实现  
[参考博客](https://www.cnblogs.com/erbaodabao0611/p/7588840.html)



# 2018-04-09   
### @xiaocaijizzz
复习了西瓜书的knn概念。
准备把仓库里的有用资料拉下来学习一下，入手较慢，会抓紧==。

# 2018-04-10
### @JKTao
## 概率的定义

在测度论中，概率被定义为事件域上的测度，这种定义回答了“概率应满足什么条件”，回避了“概率是什么”的本质问题。对“概率是什么”的解释与公理体系无关，无论采用哪一种解释，概率公理都是相容的。作为一种哲学层面的概念建构，这个问题的分歧是不可免的。统计学习和模式识别的经典教材《ESL》《PRML》可分别被归纳到上述两个学派的观点中，《MLAPP》则法乎其中，在对三本书的方法有具体认知之前，理清概率的定义是必要的。

统计推断的一般模式是，样本 <a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/svg.latex?X" title="X" /></a> 的分布或者概率密度<a href="https://www.codecogs.com/eqnedit.php?latex=f_{\theta}(x)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f_{\theta}(x)" title="f_{\theta}(x)" /></a>依赖于参数<a href="https://www.codecogs.com/eqnedit.php?latex=\theta(\theta\in\Theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta(\theta\in\Theta)" title="\theta(\theta\in\Theta)" /></a>，根据样本对<a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /></a>作估计。频率学派和贝叶斯学派的分歧在于对参数空间的认知上，前者认为，<a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /></a>对于同一个数据发生过程是**固定且未知**的常数，与样本观测值<a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/svg.latex?X" title="X" /></a>无关，也就没有后验概率这个该念，因此对于概率的计算只存在于样本空间<a href="https://www.codecogs.com/eqnedit.php?latex=\Omega" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\Omega" title="\Omega" /></a>中；后者则认为，参数<a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /></a>也是一个具有分布的**随机变量**，由现有样本<a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/svg.latex?X" title="X" /></a>和参数<a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /></a>的先验分布可以推理出参数<a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /></a>的后验分布，即<a href="https://www.codecogs.com/eqnedit.php?latex=P(\theta_i|X)\propto&space;P(X|\theta_i)P(\theta_i)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P(\theta_i|X)\propto&space;P(X|\theta_i)P(\theta_i)" title="P(\theta_i|X)\proptoP(X|\theta_i)P(\theta_i)" /></a>，这里所谓的先与后只是对知晓样本前后的时序界定，对于概率的计算发生在参数空间<a href="https://www.codecogs.com/eqnedit.php?latex=\Theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\Theta" title="\Theta" /></a>中，而不涉及样本分布的概率计算，因为样本<a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/svg.latex?X" title="X" /></a>是已知的观测结果。

频率学派在统计推断中的核心理论就是大数定理，它指出了大样本情况下样本均值对期望的趋近性质。中心极限定理补充指出，以均值估计期望时，估计的精度和<a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{N}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\sqrt{N}" title="\sqrt{N}" /></a>成正比，其中<a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?N" title="N" /></a>为样本数量，区间估计则给出了一定精度需求下截取的参数区间。

从信息论的视角来看，贝叶斯概率有非常直观的解释，它表示观察者本身掌握的信息状态，亦称为主观概率，所谓参数的概率分布，是观察者根据目前的信息量对不同的参数值的合理程度估计，即**确信度**。如果观察者在一无所知的情况下估计参数的先验分布，这时参数的先验分布就是掌握的信息量为0时所做的最大信息熵假设。

在两种观点的概率定义下，贝叶斯公式都是可用的，频率学派也可以利用关于参数的先验信息，如果先验信息来自于历史样本，那么对先验信息的处理完全可以纳入到频率学派的框架中，如经验贝叶斯方法。是否存在对先验信息的整合**不是**这两个学派观点的本质差别，这是一种流传甚广的谬误，关键是概率的计算发生在参数空间还是样本空间。在频率学派的观点下，不存在参数分布这一概念，因此也不会利用参数的概率分布去估计参数值，这才是两种方法的根本差别。



[1] 陈希孺《数理统计简史》

[2] Christopher Bishop《Pattern Recognition and Machine Learning》

