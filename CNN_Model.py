import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.utils import Sequence
from cv2 import cv2
import numpy as np

#
# class DataGenerator(Sequence):
#   def __init__(self, x_data, y_data, batch_size):
#     self.x, self.y = x_data, y_data
#     self.batch_size = batch_size
#     self.num_batches = np.ceil(len(x_data) / batch_size)
#     self.batch_idx = np.array_split(range(len(x_data)), self.num_batches)
#
#   def __len__(self):
#     return len(self.batch_idx)
#
#   def __getitem__(self, idx):
#     batch_x = self.x[self.batch_idx[idx]]
#     batch_y = self.y[self.batch_idx[idx]]
#     return batch_x, batch_y
#


def load_data():
    folders = os.listdir("./dataset")
    str_counterpart = {"add" : "+", "div" : "/", "left_bracket" : "(", "mul" : "*", "right_bracket" : ")", "sub": "-"}
    label_str = []
    data = []
    data_labels = []
    zero_label = [0 for i in range(len(folders))]
    for folder in folders:
        if folder in str_counterpart:
            label_str.append(str_counterpart[folder])
        else:
            label_str.append(folder)
    for folder in folders:
        pics = os.listdir("./dataset/"+folder)
        for pic in pics:
            img = cv2.cvtColor(cv2.imread("./dataset/" + folder + "/" + pic),cv2.COLOR_BGR2GRAY)
            data.append(img)
            i = folders.index(folder)
            label = zero_label.copy()
            label[i] = 1
            data_labels.append(label)
    return data, data_labels

def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid

data_list, data_labels_list = load_data()
data = np.array(data_list)
data_labels = np.array(data_labels_list,dtype=np.float32)

x_train, x_valid, y_train, y_valid = split_data(data, data_labels)

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation="relu",input_shape=(32, 32, 1)))
# model.add(MaxPooling2D((2, 2),strides=2))

model.add(Conv2D(64, kernel_size=3, activation="relu"))
# model.add(MaxPooling2D((2, 2),strides=2))

model.add(Flatten())
model.add(Dropout(0.5))
# model.add(Dense(40, activation="relu"))
model.add(Dense(16, activation="softmax"))
model.summary()

model.compile(optimizer='adam',loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(x_train,y_train,epochs=30)
model.evaluate(x_valid,y_valid)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
