import numpy as np
import csv
import cv2
import os
import math
from sklearn.neighbors import NearestNeighbors

def readHash(file):
    '''
        Read the hash table
    '''
    hash_table = {}
    with open(file) as f:
        for lines in f.readlines():
            a, b = lines.split(":")
            tmp = a.replace('(', '').replace(')', '').replace(' ','').split(',')
            tmp2 = b.replace('(', '').replace(')', '').replace(' ','').replace('\n','').split(',')
            #hash_table[(int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]))] = (float(tmp2[0]), float(tmp2[1]))
            hash_table[tuple(float(x) for x in tmp)] = (float(tmp2[0]), float(tmp2[1]))
    return hash_table

def readFile(file):
    '''
        Read the prediction and the direction of the test case
    '''
    pattern = []
    center = []
    ver = []
    hor = []
    with open(file) as f:
        for lines in f.readlines():
            p, v1, v2, h1, h2, c_x, c_y = lines.split(" ")
            pattern.append(int(p))
            center.append([float(c_x), float(c_y)])
            ver.append([float(v1), float(v2)])
            hor.append([float(h1), float(h2)])
    return np.array(pattern), np.array(center), np.array(ver), np.array(hor)

def Norm(x):
    return (x-min(x))/(max(x)-min(x))

def FindNeighbor(nbr, ver, hor):
    '''
        Define the up, down, left, right neighbor, and return their number

        Args
            nbr (list): the list of nearest neighbors
            ver (array): the vertical vector point to right direction
            hor (array): the horizonal vector point to up direction

        Returns
            up, down, left, right : each neighbor's number (pattern number)
            [up_n, d_n, l_n, r_n] : each neighbor's image index
    '''
    num = np.zeros((len(nbr), 5))
    for i in range(len(nbr)):
        x = np.array([nbr[i][1], nbr[i][2]])
        x = x / np.linalg.norm(x)
        num[i][0] = np.dot(x, ver)
        num[i][1] = np.dot(x, hor)
        num[i][2] = math.sqrt(nbr[i][1]**2 + nbr[i][2]**2)
        num[i][3] = nbr[i][0]
        num[i][4] = nbr[i][3]
    drop_index = np.where(Norm(num[:, 2])>0.8)
    num = np.delete(num, drop_index, 0)
    up_index, down_index, left_index, right_index = np.argmin(num[:, 0]), np.argmax(num[:, 0]), np.argmin(num[:, 1]), np.argmax(num[:, 1])
    up, down, left, right = num[up_index][3], num[down_index][3], num[left_index][3], num[right_index][3]
    up_n, d_n, l_n, r_n = num[up_index][4], num[down_index][4], num[left_index][4], num[right_index][4]
    return up, down, left, right, [up_n, d_n, l_n, r_n]

def neareastNeighbor(now, position, num):
    dis = []
    for i in range(len(position)):
        dis.append(((position[now][0] - position[i][0])**2 + (position[now][1] - position[i][1])**2))
    nbr_n = []
    for i in range(num):
        min_index = np.argmin(dis)
        dis[min_index] = float('inf')
        if min_index == now:
            i -= 1
            continue
        nbr_n.append(min_index)
    return nbr_n

def Draw(image, center, file, name, i):
    if os.path.exists(file) == False:
        os.mkdir(file)
    img = cv2.imread(image)
    center = np.array(center, int)
    cv2.circle(img, center[0], 2, (0,0,255), 2)
    for a in range(1, len(center)):
        cv2.circle(img, center[a], 2, (255,0,0), 2)
    cv2.imwrite(file+str(i)+'_'+name+'.jpg', img)
    '''cv2.imshow('show', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''




def match(f, name):
    dic = f + '/test_' + name + '_cut/'
    # print(dic)
    img = f + name + '.jpg'
    #original pattern hash table
    pattern_hash = readHash('pic/test_hash.txt')
    
    #detect project pattern : pattern number center direction
    pattern_detect, center, ver, hor = readFile(dic+'predict_direction.txt')
    detect_hash = {}
    
    #k nearest neighbors
    NN = NearestNeighbors(n_neighbors=9)
    NN.fit(center)
    
   
    #parameter for debug
    draw_wrong = 0
    detect_draw = []
    same_neighbor = 0
    draw_neighbor = []
    wrong_file = dic + 'wrong/'
    
    for i in range(len(pattern_detect)):
        nbr = []
        nbr_n = NN.kneighbors([center[i]], return_distance=False)[0]
        #nbr_n = neareastNeighbor(i, center, 8)
        for tmp in nbr_n:
            if tmp == i:
                continue
            x = center[tmp][0] - center[i][0]
            y = center[tmp][1] - center[i][1]
            nbr.append([pattern_detect[tmp], x, y, tmp])
        up, down, left, right, dir_num = FindNeighbor(nbr, ver[i], hor[i])
        
        #draw neighbor
        if i in draw_neighbor and draw_wrong:
            c = []
            c.append([center[i][1], center[i][0]])
            for tmp1 in range(len(nbr)):
                c.append([center[nbr[tmp1][3]][1], center[nbr[tmp1][3]][0]])
            #Draw(img, c, wrong_file+'neighbor/', 'neighbor', i)
        
        add = False
        if up != 10 and down != 10 and left != 10 and right != 10:
            add = True
            u, c = np.unique(dir_num, return_counts=True)
            for tmp2 in c:
                if tmp2 > 1:
                    add = False
                    break
        if add:
            detect_hash[(up, left, pattern_detect[i], right, down)] = (center[i][0], center[i][1])
            tmp = (up, left, pattern_detect[i], right, down)
            if tmp not in pattern_hash:
                c = []
                c.append([center[i][1], center[i][0]])
                for tmp1 in range(len(dir_num)):
                    c.append([center[int(dir_num[tmp1])][1], center[int(dir_num[tmp1])][0]])
                if draw_wrong:
                    Draw(img, c, wrong_file, 'detect_wrong', i)

            
    print('detect neighbor number : '+str(len(detect_hash)))
    #print(detect_hash)
    #print(pattern_hash)
    project = []
    for k in detect_hash.keys():
        if k in pattern_hash:
            project.append([pattern_hash[k][0], pattern_hash[k][1], detect_hash[k][0], detect_hash[k][1]])
    project = np.array(project)
    #print((project))
    print('match pattern number : '+str(len(project)))
    #print('number of pattern that has same neighbor : '+str(same_neighbor))
    np.savetxt(dic+'project.txt', project)
      
    '''
    with open(dic+'output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(project)'''

dic = '0316/'
# for i in range(6):
#     match(dic, str(i))