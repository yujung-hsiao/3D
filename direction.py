import cv2
import numpy as np

def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()

def FindCenter(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    return np.array([x_bar, y_bar], int)
 
def eigvec_length(image):
    M = cv2.moments(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cX = int(M["m10"]/M['m00'])
    cY = int(M['m01']/M['m00'])
    u11 = (M['m11'] - cX *  M['m01']) / M['m00']
    u20 = (M['m20'] - cX *  M['m10']) / M['m00']
    u02 = (M['m02'] - cY *  M['m01']) / M['m00']
    cov = np.array([[u20, u11], [u11, u02]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_idx]
    #print(eigvals)
    #print(np.sqrt(eigvals))
    #eigvecs = eigvecs[sorted_idx]
    #long_length = 2 * np.sqrt(eigvals[0]) * eigvecs[:, 0] / np.hypot(*eigvecs[:, 0])
    #short_length = 2 * np.sqrt(eigvals[-1]) * eigvecs[:, -1] / np.hypot(*eigvecs[:, -1])
  
    return np.sqrt(eigvals)

def show(a):
    for i in range(len(a)):
        print(' '.join(str(val) for val in a[i]))
    print('\n')

def vector(data, d):
    rec = np.where(data>0)[d]
    cutpoint = int(min(rec) + (max(rec) - min(rec))/2)
    if d == 0:
        c1 = FindCenter(data[:cutpoint, :])
        c2 = FindCenter(data[cutpoint:, :])
        c2 += [0, cutpoint]
        
    else:
        c1 = FindCenter(data[:, :cutpoint])
        c2 = FindCenter(data[:, cutpoint:])
        c2 += [cutpoint, 0]
    #return np.array((c1, c2))
    return (c2 - c1)


def Compute_Direction(dic):
    '''
        Computer the direction of pattern
    '''
    # print(dic)
    img_name = []
    pattern = []
    c = []
    fp = open(dic+'predict_direction.txt', 'w')
    with open(dic+'predict.txt') as f:
        for line in f.readlines():
            n, p, x, y = line.split()
            img_name.append(n)
            pattern.append(p)
            x, y = float(x), float(y)
            c.append([int(x),int(y)])
    #dic = 'pattern_8/'
    for i in range(len(img_name)):
        #print(img_name[i])
        image = cv2.imread(dic+img_name[i])
        long_length, short_length = eigvec_length(image)
        img = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        #img = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/128, int)

        center = FindCenter(img)
        # print(i, center)
        num = int(pattern[i])
        num = i
        if num == 2 or num == 3:
            img1 = img[center[1]:, :]
            img2 = img[:, :center[0]]
        elif num == 5 or num == 7:
            img1 = img[:center[1], :]
            img2 = img[:, center[0]:]
        else:
            img1 = img[:center[1], :]
            img2 = img[:, :center[0]]
        #show(img1)
        #show(img2)
        
        ver = vector(img1, 1)
        hor = vector(img2, 0)
        #print(up_d, left_d)
        
        ver = ver / np.linalg.norm(ver)
        hor = hor / np.linalg.norm(hor)
        #print(v1, v2)
        
        '''
        if num < 4:
            ver *= long_length
            hor *= short_length
        else:
            ver *= short_length
            hor *= long_length'''
        
        ver = np.array(np.round(ver), int)
        hor = np.array(np.round(hor), int)
        fp.write(pattern[i]+' '+str(ver[0])+' '+str(ver[1])+' '+str(hor[0])+' '+str(hor[1])+' '+str(c[i][0])+' '+str(c[i][1])+'\n')
        
    fp.close()

# Compute_Direction('0725/test1/fps180_white/test_06_cut/')