import numpy as np
import operator
import os
import csv

# 加载训练数据
def loadTrainData():
	l = []
	with open('train.csv') as file:
		lines = csv.reader(file)
		for line in lines:
			l.append(line)  # 42001*785
	l.remove(l[0])
	l = np.array(l)
	label = l[:, 0]
	data = l[:, 1:]
	return toInt(data), toInt(label)  # label 1*42000  data 42000*784


# 加载测试数据
def loadTestData():
	l = []
	with open('test.csv') as file:
		lines = csv.reader(file)
		for line in lines:
			l.append(line)
	# 28001*784
	l.remove(l[0])
	data = np.array(l)
	return toInt(data)  # data 28000*784


# 将读取到的str像素转化为数字
def toInt(arr):
	newArray=np.zeros(arr.shape)
	if len(arr.shape)==2:
		for i in list(range(arr.shape[0])):
			for j in list(range(arr.shape[1])):
				newArray[i][j] = int(arr[i][j])
	else:
		for i in list(range(arr.shape[0])):
			newArray[i]=int(arr[i])
	return newArray

# 归一化特征值，防止确保无论数值大小每一个特征的影响一样
# 在这里因为像素范围一致，所以无需归一化
def auto_norm(data_set):
	min_val = data_set.min(0)
	max_val = data_set.max(0)
	ranges = max_val - min_val
	if ranges==0:
		ranges=1
	norm_data_set = np.zeros(data_set.shape)
	m = data_set.shape[0]
	norm_data_set = data_set - np.tile(min_val, (m, 1))
	norm_data_set = norm_data_set/np.tile(ranges, (m, 1))
	return norm_data_set, ranges, min_val

# dataSet:m*n   labels:m*1  inX:1*n
# 对于输入向量进行k近邻分类
def classify(inx, dataset, labels, k):
	dataSetSize = dataset.shape[0]
	diffMat = np.tile(inx, (dataSetSize, 1)) - dataset
	sqDiffMat = np.array(diffMat) ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		votelabel = labels[sortedDistIndicies[i]]
		classCount[votelabel] = classCount.get(votelabel, 0) + 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

# 保存最后的分类矩阵到result.csv中
def saveResult(result):
	with open('result.csv', 'w+',newline='') as myFile:
		myWriter = csv.writer(myFile)
		myWriter.writerow(["ImageId","Label"])
		for i in result:
			myWriter.writerow(i)


def handwritingClassTest():
	trainData, trainLabel = loadTrainData()
	testData = loadTestData()
	m, n = np.shape(testData)
	resultList = []
	# m是28000太大了，运行不来运行不来，先来10个
	for i in range(10):
		classifierResult = classify(testData[i], trainData, trainLabel, 5)
		resultList.append([i+1,classifierResult])
		print("the %d number is %d"%(i+1,classifierResult))
	saveResult(resultList)


# 运行手写数字识别程序
handwritingClassTest()
