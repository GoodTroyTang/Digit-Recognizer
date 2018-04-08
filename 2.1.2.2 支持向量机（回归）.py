# 从sklearn.datasets 导入波士顿房价数据读取器。
from sklearn.datasets import load_boston
# 从读取房价数据存储在变量boston中
boston = load_boston()
# 输出数据描述
print(boston.DESCR)

# 从sklearn.cross_validation导入数据分割器
from sklearn.cross_validation import train_test_split

import numpy as np

X = boston.data
y = boston.target

# 随机采样25%的数据构建测试样本，其余作为训练样本。
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 33,test_size = 0.25)

# 分析回归目标值的差异。
print("max target value is",np.max(boston.target))
print("min target value is",np.max(boston.target))
print("average target value is",np.mean(boston.target))

# 上述对数据的初步查验发现预测目标房价之间的差异较大，因此对特征及目标值进行标准化处理。
# 从sklearn.preprocessing 导入数据标准化模块。
from sklearn.preprocessing import StandardScaler
#分别初始化特征和目标值的标准化器。
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

y_train = ss_y.fit_transform(y_train.reshape(-1,1))
y_test = ss_y.transform(y_test.reshape(-1,1))

# 代码39：使用三种不同核函数配置的支持向量机回归模型进行训练，并且分别对测试数据做出预测
# 从sklearn.svm中导入支持向量机(回归)模型。
from sklearn.svm import SVR

# 使用线性函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
linear_svr = SVR(kernel = 'linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

# 使用多项式核函数配置的支持向量机进行回归训练，并且对训练样本进行预测
poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict = poly_svr.predict(X_test)

# 对三种核函数配置下的支持向量机回归模型在相同测试集上进行性能评估。
# 使用R-squared MSE和MAE指标对三种配置的支持向量机(回归)模型在相同测试集上进行性能评估

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
# 使用r2_score模块，并输出评估结果。
print('The value of R-squared of linear SVR is',linear_svr.score(X_test,y_test))

# 使用mean_squared_error模块，并输出评估结果。
print('The mean squard error of linear SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))

# 使用mean_absolute_error模块，并输出评估结果
print('The mean absoluate error of linear SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))


print('The value of R-squared of Poly SVR is',poly_svr.score(X_test,y_test))

# 使用mean_squared_error模块，并输出评估结果。
print('The mean squard error of Poly SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))

# 使用mean_absolute_error模块，并输出评估结果
print('The mean absoluate error of Poly SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))


print('The value of R-squared of RBF SVR is',rbf_svr.score(X_test,y_test))

# 使用mean_squared_error模块，并输出评估结果。
print('The mean squard error of RBF SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))

# 使用mean_absolute_error模块，并输出评估结果
print('The mean absoluate error of RBF SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))



