import tensorflow
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

img_row = 48
img_col = 48
class_num = 7
modelPath = "./data/myModel.h5"
predictionPath = "./data/prediction.csv"

def showImg(img):
	plt.gray()
	plt.imshow(img)
	plt.show()

raw_data_train = pd.read_csv("./data/train.csv")
raw_data_test = pd.read_csv("./data/test.csv")
train_y = raw_data_train["label"]
tmp_train_x = np.array(raw_data_train.drop(["label"], axis = 1))
tmp_test_x = np.array(raw_data_test.drop(["id"], axis = 1))

trainSize = int(tmp_train_x.shape[0] * 3 / 4)

train_x = np.empty(shape = (trainSize, img_row, img_col, 1), dtype = int)
validation_x = np.empty(shape = (tmp_train_x.shape[0] - trainSize, img_row, img_col, 1), dtype = int)

test_x = np.empty(shape = (tmp_test_x.shape[0], img_row, img_col, 1), dtype = int)

train_y = np.array(train_y)
validation_y = train_y[trainSize :]
train_y = train_y[: trainSize]

train_y = keras.utils.to_categorical(train_y, class_num)
validation_y = keras.utils.to_categorical(validation_y, class_num)

for i in range(tmp_train_x.shape[0]):
	imgVector = np.fromstring(tmp_train_x[i][0], dtype = int, sep = " ")
	if i < trainSize:
		train_x[i] = imgVector.reshape(48, 48, 1)
	else:
		validation_x[i - trainSize] = imgVector.reshape(48, 48, 1)

for i in range(test_x.shape[0]):
	imgVector = np.fromstring(tmp_test_x[i][0], dtype = int, sep = " ")
	test_x[i] = imgVector.reshape(48, 48, 1)

train_x = train_x / 255
validation_x = validation_x / 255
test_x = test_x / 255

if not os.path.exists(modelPath):
	model = Sequential()
	model.add(Conv2D(25, kernel_size = (3, 3), activation = "relu", input_shape = (img_row, img_col, 1)))
	model.add(MaxPooling2D(2, 2))
	model.add(Conv2D(50, kernel_size = (3, 3), activation = "relu"))
	model.add(MaxPooling2D(2, 2))
	model.add(Flatten())
	model.add(Dense(1280, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(class_num, activation = "softmax"))

	model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
	model.fit(train_x, train_y, batch_size = 100, epochs = 20)
	score = model.evaluate(validation_x, validation_y)

	print("Total loss = ", score[0])
	print("Accuracy = ", score[1])

	model.save(modelPath)
else:
	model = load_model(modelPath)
	prediction = model.predict(test_x)

	f = open(predictionPath, "w", newline = "")
	w = csv.writer(f)
	title = ["id", "label"]
	w.writerow(title)

	for i in range(prediction.shape[0]):
		content = [i, np.argmax(prediction[i])]
		w.writerow(content)