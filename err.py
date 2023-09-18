import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
from scipy import stats
from ransac_circle import *
import math

def sphere_fitting(pos_xyz):
    # add columns to construct matrix A
    A = np.ones((pos_xyz.shape[0], 4))
    A[:, 0:3] = pos_xyz

    #construct f
    f = np.sum(np.multiply(pos_xyz, pos_xyz), axis=1)

    sol, residules, rank, singval = np.linalg.lstsq(A, f)
    

    radius = math.sqrt((sol[0]*sol[0]/4.0) + (sol[1]*sol[1]/4.0) + (sol[2]*sol[2]/4.0) + sol[3])
    
    return radius, sol[0]/2.0, sol[1]/2.0, sol[2]/2.0

def dif(a1, a2):
    if a2.shape[0] == 0:
        return a1
    a1r = a1.view([('', a1.dtype)]*a1.shape[1])
    a2r = a2.view([('', a2.dtype)]*a2.shape[1])
    return np.setdiff1d(a1r, a2r).view(a1.dtype).reshape(-1,a1.shape[1])

def plane_fit(points):
    mean_point = np.mean(points, axis=0)
    sub_point = points - mean_point
    cov = np.matmul(sub_point.T, sub_point)
    U, S, Vh = np.linalg.svd(cov)
    # print(U, S)
    index = np.where(S==min(S))
    normal = U[:, index]
    normal = np.reshape(normal, (3,1))
    mean_point = np.reshape(mean_point, (1,3))
    return mean_point, normal, U

def projectToPlane(point, mpt, normal):
    pro = np.matmul((np.eye(3) - np.matmul(normal , normal.T)), np.reshape((point-mpt), (3,1)))
    return pro

def transformPoint(point, R):
    trans_p = np.matmul(np.linalg.inv(R), point)
    return trans_p



dic = '0913/s10/t3'
pr = []
for i in range(0, 90):
    if not os.path.exists(f'{dic}/3D_{i}.ply'):
        continue
    pcd = o3d.io.read_point_cloud(f'{dic}/3D_{i}.ply')

    p = np.asarray(pcd.points)
    r, x, y, z = sphere_fitting(p)
    pr.append([x,y,z])

pr = np.asarray(pr)




outlier = []
#x
x_zscore = np.abs(stats.zscore(pr[:, 0]))

for i in range(len(np.where(x_zscore > 2)[0])):
    outlier.append(pr[np.where(x_zscore > 2)[0][i]])

y_zscore = np.abs(stats.zscore(pr[:, 1]))
for i in range(len(np.where(y_zscore > 2)[0])):
    outlier.append(pr[np.where(y_zscore > 2)[0][i]])

z_zscore = np.abs(stats.zscore(pr[:, 2]))
for i in range(len(np.where(z_zscore > 2)[0])):
    outlier.append(pr[np.where(z_zscore > 2)[0][i]])
outlier = np.asarray(outlier)
# print(outlier.shape)

clean_point = dif(pr, outlier)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(clean_point[:, 0], clean_point[:,1], clean_point[:,2], color='blue', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('route with raw data')
plt.show()



m, normal, U = plane_fit(clean_point)

d = -m.dot(normal)
plane_error = 0
p_pro = []
for i in clean_point:
    tmp = projectToPlane(i, m, normal)
    p_pro.append(tmp)
    plane_error += ((tmp[0][0]+m[0][0]-i[0])**2+(tmp[1][0]+m[0][1]-i[1])**2+(tmp[2][0]+m[0][2]-i[2])**2)**0.5
p_pro = np.asarray(p_pro)
plane_error /= len(clean_point)
print(f'plane fitting error : {plane_error}')
tran_p = []
for i in clean_point:
    cc = [i[0]-m[0][0], i[1]-m[0][1], i[2]-m[0][2]]
    tran_p.append(transformPoint(cc,U))
tran_p = np.asarray(tran_p)

tran_p2 = []
for i in p_pro:
    cc = [i[0][0], i[1][0], i[2][0]]
    tran_p2.append(transformPoint(cc,U))
tran_p2 = np.asarray(tran_p2)


plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(clean_point[:,0], clean_point[:,1], clean_point[:,2], color='b', label='raw data')
# ax.scatter(p_pro[:,0], p_pro[:,1], p_pro[:,2], color='r', label='projected point')
# ax.scatter(tran_p[:, 0], tran_p[:, 1], tran_p[:,2], color='y', label='transformed point')

X,Y = np.meshgrid(np.arange(min(clean_point[:,0])-2, max(clean_point[:,0])+2),
                  np.arange(min(clean_point[:,1])-2, max(clean_point[:,1])+2))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = (-normal[0][0] * X[r,c] - normal[1][0] * Y[r,c] - d)*1/normal[2][0]
ax.plot_surface(X,Y,Z, alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.title('fit plane')
plt.show()

plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(p_pro[:,0], p_pro[:,1], p_pro[:,2], color='r', label='projected point')
plt.show()


plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(tran_p[:,0], tran_p[:,1], 0, color='b', label='raw data')
ax.scatter(tran_p2[:,0], tran_p2[:,1], 0, color='r', label='project data')
xx, yy = np.meshgrid(range(-20,20), range(-20, 20))
z = np.zeros((40,40))


# plot the plane
ax.plot_surface(xx, yy, z, alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.title('tranform to the plane')
plt.show()


fig = plt.figure()
ax = plt.subplot(111)
tran_p2[:,1] *= -1
ax.scatter(tran_p2[:,0], tran_p2[:,1], color='r', label='project data')
ransac = RANSAC(tran_p2[:,0], tran_p2[:,1], 50)
	
# execute ransac algorithm
ransac.execute_ransac()

# get best model from ransac
a, b, r = ransac.best_model[0], ransac.best_model[1], ransac.best_model[2]
print(f'center:{a, b}, radius:{r}')
# show result

circle = plt.Circle((a, b), radius=r, fill=False)
plt.gca().add_patch(circle)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('equal')
plt.title('fit circle')
plt.show()

cir_error = 0
for i in tran_p2:
    cir_error += abs((i[0]-a)**2+(i[1]-b)**2-r**2)

cir_error = cir_error**0.5
print(len(tran_p2))
print(f'circle fitting error : {cir_error}')