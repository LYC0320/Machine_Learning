import sys
import numpy as np
import pandas as pd
import csv

#read data from train.csv
raw_data = np.genfromtxt("./data/train.csv", delimiter = ",")

data = raw_data[1:, 3:]
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 0

month_to_data = {} ##{key: month, value: data}

for month in range(12):
	sample = np.empty(shape = (18, 480))
	for day in range(20):
		for hour in range(24):
			sample[:, day * 24 + hour] = data[month * 360 + day * 18 : month * 360 + day * 18 + 18, hour]

	month_to_data[month] = sample

# declare x, y data structure
train_x = np.empty(shape = (6 * 471, 18 * 9), dtype = float)
train_y = np.empty(shape = (6 * 471, 1), dtype = float)
test_x_validation = np.empty(shape = (6 * 471, 18 * 9), dtype = float)
test_y_validation = np.empty(shape = (6 * 471, 1), dtype = float)

for month in range(12):
	for day in range(20):
		for hour in range(24):
			if day == 19 and hour == 15:
				break
			if month <= 5:
				train_x[month * 471 + day * 24 + hour, :] = month_to_data[month][:, day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
				train_y[month * 471 + day * 24 + hour][0] = month_to_data[month][9][day * 24 + 9 + hour]
			else:
				test_x_validation[(month - 6) * 471 + day * 24 + hour, :] = month_to_data[month][:, day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
				test_y_validation[(month - 6) * 471 + day * 24 + hour][0] = month_to_data[month][9][day * 24 + 9 + hour]

train_x = np.concatenate((train_x, test_x_validation), axis = 0)
train_y = np.concatenate((train_y, test_y_validation), axis = 0)

# swap train and validation
'''
tmp = train_x
tmp2 = train_y

train_x = test_x_validation
train_y = test_y_validation

test_x_validation = tmp
test_y_validation = tmp2
'''

# normalization
mean = np.mean(train_x, axis = 0)
std = np.std(train_x, axis = 0)

mean_v = np.mean(test_x_validation, axis = 0)
std_v = np.std(test_x_validation, axis = 0)

for i in range(train_x.shape[0]):
	for j in range(train_x.shape[1]):
		if std[j] != 0:
			train_x[i][j] = (train_x[i][j] - mean[j]) / std[j]
		if std[j] != 0 and i < 471 * 6:
			test_x_validation[i][j] = (test_x_validation[i][j] - mean[j]) / std[j] # why

# training
w = np.zeros(shape = (train_x.shape[1] + 1, 1), dtype = float)
learning_rate = np.array([[200]] * w.shape[0])
train_x = np.concatenate((np.ones((train_x.shape[0], 1)), train_x), axis = 1)
test_x_validation = np.concatenate((np.ones((test_x_validation.shape[0], 1)), test_x_validation), axis = 1)
adagrad = np.zeros(shape = (w.shape[0], 1))

for T in range(10000):
	yp = train_x.dot(w)
	if(T % 500 == 0):
		print("T = ", T)
		print("Loss: ", np.power(np.sum(np.power(train_y - yp, 2)) / train_y.shape[0], 0.5))
	gradient = (-2) * np.transpose(train_x).dot(train_y - yp)
	adagrad += gradient ** 2
	w = w - gradient * learning_rate / np.sqrt(adagrad)

# read test_X.csv
raw_test_data = np.genfromtxt("./data/test_X.csv", delimiter = ",")
test_data = raw_test_data[:, 2:]
where_are_NaNs = np.isnan(test_data)
test_data[where_are_NaNs] = 0

test_x = np.ones(shape = (int(test_data.shape[0] / 18), w.shape[0]), dtype = float)

for row in range(0, test_data.shape[0], 18):
	test_x[int(row / 18), 1 : ] = test_data[row : row + 18, : ].reshape(1, -1)

mean_test = np.mean(test_x, axis = 0)
std_test = np.std(test_x, axis = 0)


for i in range(test_x.shape[0]):
	for j in range(1, test_x.shape[1]):
		if std[j - 1] != 0:
			test_x[i][j] = (test_x[i][j] - mean[j - 1]) / std[j - 1] #why


# predict PM2.5(test_X.csv)
test_y = test_x.dot(w)

f = open("./data/test_Y.csv", "w", newline = "")
wr = csv.writer(f)
title = ["id", "value"]
wr.writerow(title)

for i in range(240):
	content = ["id_" + str(i), test_y[i][0]]
	wr.writerow(content)

# predict PM2.5(test_x_validation)
test_v_y = test_x_validation.dot(w)

f2 = open("./data/test_v_Y.csv", "w", newline = "")
wr2 = csv.writer(f2)
title2 = ["id", "value"]
wr2.writerow(title2)

f3 = open("./data/test_c_Y.csv", "w", newline = "")
wr3 = csv.writer(f3)
title3 = ["id", "value"]
wr3.writerow(title3)

for i in range(test_v_y.shape[0]):
	content = ["id_" + str(i), test_v_y[i][0]]
	content2 = ["id_" + str(i), test_y_validation[i][0]]
	wr2.writerow(content)
	wr3.writerow(content2)

print("Testing loss:", np.power(np.sum(np.power(test_y_validation - test_v_y, 2)) / test_v_y.shape[0], 0.5))
