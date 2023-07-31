import cv2
import numpy as np
from cut import cut
from preprocess import pre, split
from cnn_predict import cnn_pred
from direction import Compute_Direction
from match import match
from draw import DrawCenter, DrawDirection, DrawFinal

import warnings
warnings.filterwarnings("ignore")

# cut
size = 'f90' 
color = ['blue', 'white']
dic = f'0725/test1/fps180_{color[0]}/'
bg = cv2.imread('0725/bg.jpg', 0)

tmp = 0

for i in range(18, 140):
    if i < 10:
        name = f'00{i}'
    elif 9 < i < 100:
        name = f'0{i}'
    else: name = str(i)
    # bg = cv2.imread(f'0713/bg.png', 0)
    img = cv2.imread(f'{dic}/Image0{name}.jpg')
    sub = cv2.subtract(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), bg)
    
    # too dark too little information
    # if (np.mean(sub.ravel())) < 1:
    #     continue
    # if i != 1 and (tmp == sub).all():
    #         continue
    # tmp = sub

    # cv2.imshow('show', pre(sub))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    n = cut(sub, pre(sub), 20, f'{dic}/test_{name}_cut/', 0, 0)
    print(n)
    
    cnn_pred(f'{dic}/test_{name}_cut/')
    Compute_Direction(f'{dic}/test_{name}_cut/')
    match(dic, name)
    
    # present result
    # DrawCenter(dic, name, False, img)
    # DrawDirection(dic, name, img)
    DrawFinal(dic, name, False, img, 10)