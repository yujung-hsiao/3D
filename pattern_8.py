import numpy as np
import os
import cv2
import random

def pattern(img, x, y, width, length, thick, num):
    width = int(width/2)
    length = int(length/2)
    thick = int(thick/2)
    if num == 0:
        img[x-thick:x+thick, y-width:y+width, :] = 255
        img[x-thick:x-thick+2*length, y-width:y-width+2*thick, :] = 255
    elif num == 1:
        img[x-thick:x+thick, y-width:y+width, :] = 255
        img[x-thick:x-thick+2*length, y+width-2*thick:y+width, :] = 255
    elif num == 2:
        img[x-thick:x+thick, y-width:y+width, :] = 255
        img[x+thick-2*length:x+thick, y-width:y-width+2*thick, :] = 255
    elif num == 3:
        img[x-thick:x+thick, y-width:y+width, :] = 255
        img[x+thick-2*length:x+thick, y+width-2*thick:y+width, :] = 255
    elif num == 4:
        img[x-width:x+width, y-thick:y+thick, :] = 255
        img[x-width:x-width+2*thick, y-thick:y-thick+2*length, :] = 255
    elif num == 5:
        img[x-width:x+width, y-thick:y+thick, :] = 255
        img[x-width:x-width+2*thick, y+thick-2*length:y+thick, :] = 255
    elif num == 6:
        img[x-width:x+width, y-thick:y+thick, :] = 255
        img[x+width-2*thick:x+width, y-thick:y-thick+2*length, :] = 255
    elif num == 7:
        img[x-width:x+width, y-thick:y+thick, :] = 255
        img[x+width-2*thick:x+width, y+thick-2*length:y+thick, :] = 255
    elif num == 8:
        img[x-width:x+width, y-width:y+width, :] = 255



square = 5  # half size of pattern 
# gap = int(square * 3/5) # gap between two pattern
gap = 3
# num_w = 80
# num_h = 50
# img_w = num_w*2*(square + gap) # width of image size
# img_h = num_h*2*(square + gap) # height of image size
img_w, img_h = 1280, 800
num_w, num_h = int(img_w/(2*(square + gap))), int(img_h/(2*(square + gap)))

img = np.zeros((img_h, img_w, 3), np.uint8)

width = square*2
length = int(width*2/3)
thick = int(width*0.4)
# width = 10
# length = 6
# thick = 4
# pattern(img, gap+square, gap+square, width, length, thick, 0)
# cv2.imwrite('tmp.jpg', img)
'''
# write information file
# a = 5
# dic = 'pattern_8/test_'+str(a)+'/'
# os.makedirs(dic)
# f = open(dic+'information.txt', 'w')
# f.write('square :'+str(square*2)+"\n")
# f.write('gap :'+str(gap*2)+"\n")
# f.write('width :'+str(width)+"\n")
# f.write('length :'+str(length)+"\n")
# f.write('thick :'+str(thick)+"\n")
# f.close()

# for cnn data project
for p in range(8):
    img = np.zeros((img_h, img_w, 3), np.uint8)
    for i in range(gap+square, img_h, 2*(square+gap)):
        for j in range(gap+square, img_w, 2*(square+gap)):
            pattern(img, i, j, width, length, thick, p)
    #cv2.imshow('show', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(dic+str(p)+'.jpg', img) 

'''
# for normal project

# hash table
# k = 0
# do_again = True
# hash_table = {}
# num = np.zeros([num_h, num_w], np.uint8)
# while(do_again):
#     k += 1
#     hash_table.clear()
#     same = False
#     # for i in range(num_h):
#     #     for j in range(num_w):
#     #         num[i,j] = random.randint(0, 7)
#     num = np.random.randint(0, 7, size=(num_h, num_w))  
            
#     for i in range(1, num_h-1, 1):
#         for j in range(1, num_w-1, 1):
#             tmp = (num[i-1][j], num[i][j-1], num[i][j], num[i][j+1], num[i+1][j])
#             if tmp in hash_table:
#                 same = True
#                 break
#             hash_table[tmp] = (i, j)
#         if same is True:
#             break
#     if same is False:
#         do_again = False
num = np.loadtxt('pic/pattern_64_512.txt')
num = num[:num_h, :num_w]
hash_table = {}
cnt = 0
for i in range(1, num_h-1, 1):
    for j in range(1, num_w-1, 1):
        tmp = (num[i-1][j], num[i][j-1], num[i][j], num[i][j+1], num[i+1][j])
        hash_table[tmp] = (i, j)

# save hash table and its center pixel
folder = 'pic/0518/'
name = folder+'test'
if os.path.exists(folder) == False:
    os.makedirs(folder)

hash_file = open(name+'_hash.txt', 'w')
for i in hash_table:
    w = gap/2 + square + hash_table[i][0]*(2*square+gap)
    h = gap/2 + square + hash_table[i][1]*(2*square+gap)
    hash_table[i] = (w, h)
    hash_file.write(str(i)+" : "+str(hash_table[i])+"\n")
hash_file.close()


# folder = 'pic/0504/'
name = folder+'test'
fp = open(name+'_ans.txt', 'w')
# num = np.loadtxt(name+'_ans.txt', int)
img[:, :, :] = 0
r = 0
for i in range(gap+square, img_h, 2*(square+gap)):
    c = 0
    for j in range(gap+square, img_w, 2*(square+gap)):
        # print(r, c)
        pattern(img, i, j, width, length, thick, num[r, c])
        fp.write(str(num[r, c])+' ')
        c += 1
    r += 1
    fp.write('\n')

# fp.close()
cv2.imshow('show',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#name = 'test'
cv2.imwrite(f'{name}_gap{str(gap*2)}_size{str(square*2)}.png', img)
