import tensorflow
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_y) = mnist.load_data()

img_row, img_col = train_x.shape[1], train_x.shape[2]

train_x = train_x.reshape(train_x.shape[0], img_row * img_col)
test_x = test_x.reshape(test_x.shape[0], img_row * img_col)

train_x = train_x.astype("float32")
test_x = test_x.astype("float32")

train_x /= 255
test_x /= 255

train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

model = Sequential()
model.add(Dense(input_dim = img_row * img_col, units = 500))
model.add(Activation("sigmoid"))

model.add(Dense(units = 500))
model.add(Activation("sigmoid"))

model.add(Dense(units = 10))
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(train_x, train_y, batch_size = 100, epochs = 20)
score = model.evaluate(test_x, test_y)

print("Total loss = ", score[0])
print("Accuracy = ", score[1])