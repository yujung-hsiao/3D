import cv2
import numpy as np
import os
from preprocess import pre

def cut(org_img, img, pixel, folder, x_plus=0, y_plus=0):
  """
    cut each pattern of the image

    Args:
      org_img (2D array): the orginal image without background
      img (2D array): image with preprocessing
      pixel (int): the number of the pattern expected size
      folder (str): the folder store each pattern
      x_plus (int): x of the first pattern
      y_plus (int): y of the first pattern
    
    Returns:
      idx (int): the number of patterns
  """    
  idx = 0
  gap = pixel//5
  length = int(pixel/2 + gap)
  if os.path.exists(folder) == False:
    os.makedirs(folder)
  fp = open(folder+'cut.txt', 'w')
  num_labels, labels_img = cv2.connectedComponents(img)
  # print(num_labels)
  for i in range(1, num_labels):
    #original = np.zeros([pixel+2*gap, pixel+2*gap])
    p_matrix = np.where(labels_img == i)
    if p_matrix[0].size < (pixel*pixel)/10:
      continue
    
    min_x = p_matrix[0].min()
    max_x = p_matrix[0].max()
    min_y = p_matrix[1].min()
    max_y = p_matrix[1].max()
    # if min_x > 700:
    #   continue    
    #if max_x > 600 or min_x < 100 or max_y > 600 or min_y < 200:
     #   continue
    #if (max_x-min_x) >= pixel or (max_y-min_y) >= pixel:
    #    continue
    
    center_x = int((max_x-min_x)/2 + min_x)+x_plus
    center_y = int((max_y-min_y)/2 + min_y)+y_plus
    
    #original
    #for j in range(p_matrix[0].size):
    #  original[p_matrix[0][j]-min_x+gap][p_matrix[1][j]-min_y+gap] = 255 if img[p_matrix[0][j]][p_matrix[1][j]] > 0 else 0
    original = org_img[((center_x - length) if (center_x - length)>0 else 0) : center_x + length, ((center_y - length) if (center_y - length)>0 else 0) : center_y + length]
    
    if len(np.where(original>100)[0]) < (pixel*pixel)/3:
      continue    
    '''
    cv2.imshow('show', original)
    cv2.waitKey(100)
    '''
    cv2.imwrite(folder+str(idx)+'.jpg', original)
    
    fp.write(str(idx)+'.jpg '+str(center_x)+' '+str(center_y)+'\n')
    idx += 1

  fp.close()

  return idx
  

# dic = '0602'
# name = str(0)
# bg = cv2.imread(f'{dic}/bg.png', 0)
# img = cv2.imread(f'{dic}/{name}.png',0)

# sub = cv2.subtract(img, bg)
# cv2.imshow('show', sub[100:350, 250:500])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# n = cut(sub, pre(sub[100:350, 250:500]), 25, f'{dic}/test_{name}_cut/', 100, 250)
# print(n)

# img = cv2.imread(f'0725/test1/fps180_white/Image0006.jpg')
# bg = cv2.imread('0725/bg.jpg', 0)
# sub = cv2.subtract(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), bg)


# n = cut(sub, pre(sub), 20, f'0725/test1/fps180_white/test_6_cut/', 0, 0)
# print(n)