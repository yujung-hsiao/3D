import cv2
import numpy as np
import csv

def DrawCenter(dic, name, dot, img):
    '''
        Mark the center of the cut pattern

        Args:
            dic (str): the folder of the 
            name (str): the name of the test case
            dot (bool): mark by circle or dot
            img (array): the test image
    '''
    file = dic + '/test_' + name + '_cut/cut.txt'
    center = []
    #n = []
    with open(file) as f:
        for lines in f.readlines():
            p, c_x, c_y = lines.split(" ")
            #n.append(p)
            center.append([float(c_x), float(c_y)])
    # img = cv2.imread(dic +'pre'+ name + '.jpg') 
    for i in range(len(center)):
        c = np.array([int(center[i][1]), int(center[i][0])])
        if dot:
            cv2.circle(img, c, 2, (0,0,255), 2)
            #cv2.putText(img, n[i].replace('.jpg', ''), c, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
        else:
            length = 10
            cv2.rectangle(img, c-length, c+length, (0,0,255), 2)
            

    cv2.imwrite(f'{dic}/{name}_cut.jpg', img)


def DrawDirection(dic, name, img):
    '''
        Mark the direction of the cut pattern

        Args:
            dic (str): the folder of the 
            name (str): the name of the test case
            img (array): the test image
    '''
    # img = cv2.imread(f'{dic}/{name}.jpg')
    file = dic + '/test_' + name + '_cut/predict_direction.txt'
    center = []
    ver = []
    hor = []
    with open(file) as f:
        for lines in f.readlines():
            p, v1, v2, h1, h2, c_x, c_y = lines.split(" ")
            center.append([float(c_x), float(c_y)])
            ver.append([float(v1), float(v2)])
            hor.append([float(h1), float(h2)])
    ver = np.array(ver, int)
    hor = np.array(hor, int)
    for i in range(len(center)):
        c = np.array([int(center[i][1]), int(center[i][0])])
        cv2.line(img, c, c+ver[i]*5, (255,0,0), 2)
        cv2.line(img, c, c+hor[i]*5, (0,255,0), 2)

    cv2.imwrite(f'{dic}/{name}_dir.jpg', img) 


def DrawFinal(dic, name, dot, img, l):
    '''
        Mark the pattern that are decoded

        Args:
            dic (str): the folder of the 
            name (str): the name of the test case
            dot (bool): mark by circle or dot
            img (array): the test image
            l (int): the length of circle
    '''
    file = dic + '/test_' + name + '_cut/project.txt'
    # img = cv2.imread(dic + name + '.jpg')
    '''
    row = []
    for r in csv.reader(f):
        row.append(r)
    for i in range(len(row)):
        color = (0,255,0)
        c_x = int(float(row[i][2]))
        c_y = int(float(row[i][3]))
        cv2.rectangle(img, (c_y-l, c_x-l), (c_y+l, c_x+l),color, 1)'''
    
    center = []
    with open(file) as f:
        for lines in f.readlines():
            c1, c2, c_x, c_y = lines.split(" ")
            center.append([float(c_x), float(c_y)])
    
    for i in range(len(center)):
        c = np.array([int(center[i][1]), int(center[i][0])])
        if dot:
            cv2.circle(img, c, 2, (0,0,255), 2)
        else:
            length = np.array([l, l])
            cv2.rectangle(img, c-length, c+length, (0,0,255), 2)

    cv2.imwrite(f'{dic}/{name}_final.jpg', img) 


# dic = '0720'
# for i in range(0, 1):
#     name = str(i)
#     # bg = cv2.imread(f'{dic}/bg.png', 0)
#     img = cv2.imread(f'{dic}/{name}.png')
#     # DrawCenter(dic, name, False, img)
#     DrawFinal(dic, name, False, img, 10)
    # sub = cv2.subtract(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), bg)

    # cv2.imshow('show', sub)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()