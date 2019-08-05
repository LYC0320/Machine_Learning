import sys
import numpy as np
import pandas as pd
import csv
import math, os

def preprocessing():
	raw_data = pd.read_csv("./data/train.csv")
	raw_data_test = pd.read_csv("./data/test.csv")

	f = open("./data/train_Y.csv", "w", newline = "")
	wr = csv.writer(f)
	title = ["income"]
	wr.writerow(title)

	for i in range(len(raw_data)):
		if(raw_data["income"][i] == " <=50K"):
			content = str(0)
		else:
			content = str(1)
		wr.writerow(content)

	raw_data = raw_data.drop(["income"], axis = 1)

	raw_data_all = pd.concat([raw_data, raw_data_test], axis = 0, ignore_index = True, sort = False)
	raw_data_all_tmp = raw_data_all
	raw_data_all = raw_data_all.drop(["education_num", "sex"], axis = 1)

	allData_X = raw_data_all

	listObjColName = [col for col in raw_data_all.columns if raw_data_all[col].dtypes == "object"]
	listNotObjColName = [col for col in raw_data_all.columns if raw_data_all[col].dtypes != "object"]

	for col in listObjColName:
		allData_X = allData_X.drop([col], axis = 1)

	for col in listObjColName:
		allData_X = allData_X.join(pd.get_dummies(raw_data_all[col], prefix = col))

	allData_X.insert(2, "sex", (raw_data_all_tmp["sex"] == " Male").astype(np.int))
	
	f2 = open("./data/train_X.csv", "w", newline = "")
	wr = csv.writer(f2)
	wr.writerow(allData_X.columns)

	for i in range(len(raw_data)):
		content = allData_X.loc[i]
		wr.writerow(content)

	f3 = open("./data/test_X.csv", "w", newline = "")
	wr = csv.writer(f3)
	wr.writerow(allData_X.columns)

	for i in range(len(raw_data), len(allData_X)):
		content = allData_X.loc[i]
		wr.writerow(content)

def f_u_s(mean, sigma, D, x):
	pi = 3.14
	tmpX_u = (x - mean).reshape(-1, 1)
	tmpExp = math.exp(-0.5 * np.dot(np.dot(np.transpose(tmpX_u).reshape(1, -1), np.linalg.inv(sigma)), tmpX_u))
	if tmpExp == 0:
		tmpExp = 0.00000000001
	return (1 / (2 * pi) ** (D / 2)) * (1 / (abs(np.linalg.det(sigma)) ** 0.5)) * tmpExp

def normalize(train_X, test_X):
	all_X = np.concatenate((train_X, test_X), axis = 0)

	mean = np.mean(all_X, axis = 0)
	std = np.std(all_X, axis = 0)

	all_X = (all_X - mean) / std

	train_normal_X = all_X[0 : train_X.shape[0], :]
	test_normal_X = all_X[train_X.shape[0] : , :]

	return train_normal_X, test_normal_X
	

def sigmoid(z):
	s = np.empty(shape = (z.shape[0], 1), dtype = float)
	for i in range(z.shape[0]):
		s[i] = 1 / (1 + math.exp(-z[i]))
	return s

def crossEntropy(train_Y, s):
	return -(np.dot(np.transpose(train_Y).reshape(1, -1), np.log(s)) + np.dot(np.transpose(1 - train_Y).reshape(1, -1), np.log(1 - s)))

def probabilistic_generative(train_X_0, train_X_1, test_X, correct_ans):
	size_0 = len(train_X_0)
	size_1 = len(train_X_1)

	mean_0 = np.mean(train_X_0, axis = 0)
	mean_1 = np.mean(train_X_1, axis = 0)

	sigma_0 = np.zeros(shape = (len(train_X_0[0]), len(train_X_0[0])), dtype = float)
	sigma_1 = np.zeros(shape = (len(train_X_0[0]), len(train_X_0[0])), dtype = float)

	for i in range(size_0):
		tmpX = (train_X_0[i][:] - mean_0)
		sigma_0 += np.dot(tmpX.reshape(-1, 1), np.transpose(tmpX).reshape(1, -1))

	for i in range(size_1):
		tmpX = (train_X_1[i][:] - mean_1)
		sigma_1 += np.dot(tmpX.reshape(-1, 1), np.transpose(tmpX).reshape(1, -1))
	
	sigma_0 /= size_0
	sigma_1 /= size_1

	sigma_all = sigma_0 * (size_0 / (size_0 + size_1)) + sigma_1 * (size_1 / (size_0 + size_1))

	predict = []

	for i in range(test_X.shape[0]):
		tmpX = test_X[i][:]
		p_x_c0 = f_u_s(mean_0, sigma_all, test_X.shape[1], tmpX)
		p_x_c1 = f_u_s(mean_1, sigma_all, test_X.shape[1], tmpX)
		p_c0 = (size_0 / (size_0 + size_1))
		p_c1 = (size_1 / (size_0 + size_1))
		if(p_x_c0 * p_c0 / (p_x_c0 * p_c0 + p_x_c1 * p_c1) > 0.5):
			predict.append(0)
		else:
			predict.append(1)

	
	correct_ans["predict"] = predict
	correct_ans.to_csv("./data/predict_PG.csv")

	diff = correct_ans["label"] - predict
	arrayDiff = np.array(diff)

	correct = 0
	for i in range(test_X.shape[0]):
		if arrayDiff[i] == 0:
			correct += 1
	f = open("./data/predict_PG.csv", "a", newline = "")
	wr = csv.writer(f)
	wr.writerow(["Accuracy: " + str(round(correct / test_X.shape[0], 2))])

def logistic_regression(train_X, train_Y, test_X, correct_ans):
	train_X = np.concatenate(((np.ones(shape = (train_X.shape[0], 1), dtype = float)), train_X), axis = 1)
	test_X = np.concatenate(((np.ones(shape = (test_X.shape[0], 1), dtype = float)), test_X), axis = 1)
	arrayW = np.zeros(shape = (train_X.shape[1], 1), dtype = float)
	learning_rate = 0.00001
	train_Y = train_Y.reshape(-1, 1)
	predict = []

	for T in range(300):
		z = np.dot(train_X, arrayW)
		s = sigmoid(z)

		arrayW -= learning_rate * np.transpose(np.dot(-np.transpose(train_Y - s).reshape(1, -1), train_X)).reshape(-1, 1)
		if(T % 50 == 0):
			print("T = ", T)
			print("Loss = ", crossEntropy(train_Y, s) / train_Y.shape[0])

	# testing
	z = np.dot(test_X, arrayW)
	s = sigmoid(z)

	for i in range(len(s)):
		if s[i] < 0.5:
			predict.append(0)
		else:
			predict.append(1)
	
	correct_ans["predict"] = predict
	correct_ans.to_csv("./data/predict_LR.csv")

	diff = correct_ans["label"] - predict
	arrayDiff = np.array(diff)

	correct = 0
	for i in range(test_X.shape[0]):
		if arrayDiff[i] == 0:
			correct += 1
	f = open("./data/predict_LR.csv", "a", newline = "")
	wr = csv.writer(f)
	wr.writerow(["Accuracy: " + str(round(correct / test_X.shape[0], 2))])
	

def main():

	if(__name__ == "__main__"):

		preprocessing()
		train_X = np.genfromtxt("./data/train_X.csv", delimiter = ",")
		train_Y = np.genfromtxt("./data/train_Y.csv", delimiter = ",")
		test_X = np.genfromtxt("./data/test_X.csv", delimiter = ",")
		correct_ans = pd.read_csv("./data/correct_answer.csv", index_col = 0)

		train_X = train_X[1:, :]
		train_Y = train_Y[1:]
		test_X = test_X[1:, :]
		train_X_0 = []
		train_X_1 = []

		for i in range(train_Y.shape[0]):
			if train_Y[i] == 0:
				train_X_0 += [train_X[i, :]]
			else:
				train_X_1 += [train_X[i, :]]

		probabilistic_generative(train_X_0, train_X_1, test_X, correct_ans)

		tmp = normalize(train_X, test_X)
		train_normal_X = tmp[0]
		test_normal_X = tmp[1]
		
		logistic_regression(train_normal_X, train_Y, test_normal_X, correct_ans)

main()