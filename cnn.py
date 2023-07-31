import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import csv
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def read_data(dic, f):
  img = []
  img_name = []
  label = [] 
  with open(dic+f) as img_file:
    rows = csv.reader(img_file)

    for row in rows:
      img_name.append(row[0])
      label.append(int(row[1]))
  # print(img_name[:5], label[:5])
  for i in range(len(img_name)):
    #img.append([])
    img_arr = cv2.imread(dic+img_name[i])[...,::-1] #convert BGR to RGB format
    resized_arr = cv2.resize(img_arr, (28, 28)) # Reshaping images to preferred size
    img.append(resized_arr)
  
  return np.array(img), np.array(label)

def img_pre(image, label):
  image = image/255
  # label = np_utils.to_categorical(label)
  return image, label

"""### create a CNN model to train"""

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#model.summary()

# image, labels = read_data('0416/data/', 'data.csv')
# img, label = img_pre(image, labels)

# train_images, test_images, train_labels, test_labels = train_test_split(img, labels, test_size=0.33, random_state=42)

train_img, train_lab = read_data('0416/data/', 'data.csv')
train_images, train_labels = img_pre(train_img, train_lab)
print(train_labels.shape)
test_img_p, test_lab_p = read_data('0416/test_0_cut/', 'cutdata.csv')
test_images_p, test_labels_p = img_pre(test_img_p, test_lab_p)
test_img_b, test_lab_b = read_data('0416/ball/test_0_cut/', 'cutdata.csv')
test_images_b, test_labels_b = img_pre(test_img_b, test_lab_b)
print(test_images_p.shape, test_labels_p.shape, test_images_b.shape, test_labels_b.shape)
test_images, test_labels = np.vstack((test_images_p, test_images_b)), np.array(list(test_labels_p)+list(test_labels_b))
print(test_images.shape, test_labels.shape)

model.compile(optimizer='adam', 
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50, 
                    validation_data=(test_images, test_labels))

model.save('CNN_pattern8_0503.h5')



prediction = np.argmax(model.predict(test_images_p), axis=1)
print(classification_report(test_labels_p, prediction))

prediction = np.argmax(model.predict(test_images_b), axis=1)
print(classification_report(test_labels_b, prediction))