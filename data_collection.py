import numpy as np
import random
import cv2
import csv
import os
from preprocess import pre

def img_pre(img):
  binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  kernel = np.ones((3, 3), np.uint8)
  erosion = cv2.erode(binary_img, kernel, iterations=1)
  dilation = cv2.dilate(erosion, kernel, iterations=1)
  return dilation    

def ROTATION(img, center, angle):
  rows, cols = img.shape
  center = (rows/2, cols/2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotate = cv2.warpAffine(img, M, (rows, cols))
  return rotate

def SCALE(img, scale_percent):
  width = int(img.shape[1] * scale_percent)
  height = int(img.shape[0] * scale_percent)
  dim = (width, height)
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  return resized

def cut(org_img, pre_img, pixel, folder, num, idx):
  start = idx
  gap = pixel//5
  length = int(pixel/2 + gap)
  if os.path.exists(folder) == False:
    os.makedirs(folder)
  num_labels, labels_img = cv2.connectedComponents(pre_img)
  for i in range(1, num_labels):
    #original = np.zeros([pixel+2*gap, pixel+2*gap])
    #shift = np.zeros([4, pixel+gap, pixel+gap])
    p_matrix = np.where(labels_img == i)
    if p_matrix[0].size < (pixel*pixel)/5:
      continue
    
    min_x = p_matrix[0].min()
    max_x = p_matrix[0].max()
    min_y = p_matrix[1].min()
    max_y = p_matrix[1].max()
    if (max_x-min_x) >= pixel or (max_y-min_y) >= pixel:
      continue
    
    #if min_x < 450 or max_x > 600 or min_y < 1000 or max_y > 1100:
    #  continue    
    #cv2_imshow(img[min_x:max_x, min_y:max_y])
    
    #original
    #for j in range(p_matrix[0].size):
      #original[p_matrix[0][j]-min_x+gap][p_matrix[1][j]-min_y+gap] = 255 if img[p_matrix[0][j]][p_matrix[1][j]] > 0 else 0

    center_x, center_y = int((max_x-min_x)/2 + min_x), int((max_y-min_y)/2 + min_y)
    #original = org_img[low_x : int(center_x + pixel/2), low_y : int(center_y + pixel/2)]
    original = org_img[((center_x - length) if (center_x - length)>0 else 0) : center_x + length, ((center_y - length) if (center_y - length)>0 else 0) : center_y + length]
    # cv2.imshow('cut', original)
    # cv2.waitKey(100)
    
    center = ((pixel+gap)//2, (pixel+gap)//2)
    
    cv2.imwrite(folder+str(idx)+'.jpg', original)
    idx += 1
  
    #rotation -15 ~ 15
    r1 = ROTATION(original, center, -15)
    cv2.imwrite(folder+str(idx)+'.jpg', r1)
    idx += 1
    r1 = ROTATION(original, center, -10)
    cv2.imwrite(folder+str(idx)+'.jpg', r1)
    idx += 1
    r1 = ROTATION(original, center, -5)
    cv2.imwrite(folder+str(idx)+'.jpg', r1)
    idx += 1
    r1 = ROTATION(original, center, 5)
    cv2.imwrite(folder+str(idx)+'.jpg', r1)
    idx += 1
    r1 = ROTATION(original, center, 10)
    cv2.imwrite(folder+str(idx)+'.jpg', r1)
    idx += 1
    r1 = ROTATION(original, center, 15)
    cv2.imwrite(folder+str(idx)+'.jpg', r1)
    idx += 1
    
    s1 = SCALE(original, 0.6)
    cv2.imwrite(folder+str(idx)+'.jpg', s1)
    idx += 1
    s1 = SCALE(original, 0.7)
    cv2.imwrite(folder+str(idx)+'.jpg', s1)
    idx += 1
    s1 = SCALE(original, 0.8)
    cv2.imwrite(folder+str(idx)+'.jpg', s1)
    idx += 1
    s1 = SCALE(original, 0.9)
    cv2.imwrite(folder+str(idx)+'.jpg', s1)
    idx += 1
    s1 = SCALE(original, 1.1)
    cv2.imwrite(folder+str(idx)+'.jpg', s1)
    idx += 1
    s1 = SCALE(original, 1.2)
    cv2.imwrite(folder+str(idx)+'.jpg', s1)
    idx += 1
    s1 = SCALE(original, 1.3)
    cv2.imwrite(folder+str(idx)+'.jpg', s1)
    idx += 1

    blur = cv2.blur(original, (5,5))
    cv2.imwrite(folder+str(idx)+'.jpg', blur)
    idx += 1

    
  
  with open(folder+'data.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(start, idx):
      writer.writerow([str(i)+'.jpg', str(num)])
  
  return idx

if __name__ == '__main__':
  bg = cv2.imread('0416/ball/bg.jpg', 0)
  #for j in range(1, 4):
  #print('test_'+str(j))    
  dic = '0416/ball/'
  idx = 12000
  for i in range(8):
    #print(i)
    img = cv2.imread(dic + str(i) + '.jpg', 0)
    # sub = img
    sub = (cv2.subtract(img, bg))
    # cv2.imshow('show', sub)
    # cv2.waitKey(100)
    
    idx = cut(sub, pre(sub, 5, 3, 1), 20, '0416/data/', i, idx)
    print(idx)

