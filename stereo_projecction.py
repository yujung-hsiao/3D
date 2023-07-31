import numpy as np
import math
import cv2

def show(a):
    for i in range(len(a)):
        print(' '.join(str(int(val)) for val in a[i]))
    print('\n')

def project_to_sphere(plane_x, plane_y, sphere_R):
    sphere_x = 4 * sphere_R**2 * plane_x / (4*sphere_R**2 + plane_x**2 + plane_y**2)
    sphere_y = 4 * sphere_R**2 * plane_y / (4*sphere_R**2 + plane_x**2 + plane_y**2)
    sphere_z = sphere_R*(-4*sphere_R**2 + plane_x**2 + plane_y**2) / (4*sphere_R**2 + plane_x**2 + plane_y**2)
    return (sphere_x, sphere_y, sphere_z)

def project_to_camera(sphere, camera, angle):
    angle = [math.pi/180*angle[0], math.pi/180*angle[1], math.pi/180*angle[2], ]
    dif = np.array(sphere) - np.array(camera)
    dx = math.cos(angle[1]) * (math.sin(angle[2])*dif[1] + math.cos(angle[2])*dif[0]) - math.sin(angle[1])*dif[2]
    dy = math.sin(angle[0]) * (math.cos(angle[1])*dif[2] + math.sin(angle[1])*(math.sin(angle[2])*dif[1] + math.cos(angle[2])*dif[0])) + math.cos(angle[0])*(math.cos(angle[2])*dif[1] - math.sin(angle[2])*dif[0])
    dz = math.cos(angle[0]) * (math.cos(angle[1])*dif[2] + math.sin(angle[1])*(math.sin(angle[2])*dif[1] + math.cos(angle[2])*dif[0])) - math.sin(angle[0])*(math.cos(angle[2])*dif[1] - math.sin(angle[2])*dif[0])
    capture_x = (dx - camera[0])*(camera[2]/dz)
    capture_y = (dy-camera[1])*(camera[2]/dz)
    return (capture_x, capture_y)

img = cv2.imread('pic/test.jpg', 0)
img = np.array(img/128, int)
# show(img)
h, w = img.shape
tmp = np.where(img>0)
translation = []
for i in range(len(tmp[0])):
    x, y, z = project_to_sphere(-(tmp[0][i]-int(w/2)), -(tmp[1][i]-int(h/2)), 100)
    c_x, c_y = project_to_camera([x,y,z], [-50,-100,-500], [0,0,0])
    translation.append([c_x, c_y])
translation = np.array(translation)
l_x = np.max(translation[:, 0]) - np.min(translation[:, 0])+20
l_y = np.max(translation[:, 1]) - np.min(translation[:, 1])+20
p = np.zeros((int(l_x), int(l_y), 3))
for t in translation:
    p[int(t[0]-np.min(translation[:, 0])+10), int(t[1]-np.min(translation[:, 1])+10), :] = [255,255,255]

kernel = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(p, kernel, iterations=1)
kernel_e = np.ones((1, 1), np.uint8)
erosion = cv2.erode(dilation, kernel_e, iterations=1)
# show(p)
# cv2.imshow('show', erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('stereo/7.jpg', erosion)