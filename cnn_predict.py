import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
#from tmp import a

model = tf.keras.models.load_model('CNN_pattern8_0416.h5')

# model.summary()

def load_img(dic):
  img = []
  img_name = []
  center = []
  with open(dic+'cut.txt') as f:
    for line in f.readlines():
      a, x, y = line.split()
      img_name.append(a)
      center.append([x, y])

  for name in img_name:
    if "txt" not in name:
      #img.append([])
      img_arr = cv2.imread(dic+name)[...,::-1] #convert BGR to RGB format
      resized_arr = cv2.resize(img_arr, (28, 28)) # Reshaping images to preferred size
      img.append(resized_arr/255)
  
  return np.array(img), np.array(center), img_name


def cnn_pred(dic):
  """
    classifity the pattern
  """    
  test_img, center, name = load_img(dic)

  # print(test_img.shape)
  #print(name[0])

  prediction = np.argmax(model.predict(test_img), axis=1)
  # print(prediction)

  #test_labels = np.loadtxt(dic+'ans.txt')
  #test_labels = a
  #print(classification_report(test_labels, prediction))
  #f_c = count_false(prediction)
  #print(f_c)
  
  with open(dic + 'predict.txt', 'w') as f:
    for i in range(len(prediction)):
      #if prediction[i] == test_labels[i]:  
        f.write(name[i]+' '+str(prediction[i])+" "+str(center[i][0])+" "+str(center[i][1])+'\n')

  f.close()

# for i in range(6):
# dic = '0420/test_0_cut/'
# cnn_pred(dic)