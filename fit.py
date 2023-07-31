import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")

def sphere_fitting(pos_xyz):
    # add columns to construct matrix A
    A = np.ones((pos_xyz.shape[0], 4))
    A[:, 0:3] = pos_xyz

    #construct f
    f = np.sum(np.multiply(pos_xyz, pos_xyz), axis=1)

    sol, residules, rank, singval = np.linalg.lstsq(A, f)
    

    radius = math.sqrt((sol[0]*sol[0]/4.0) + (sol[1]*sol[1]/4.0) + (sol[2]*sol[2]/4.0) + sol[3])
    
    return radius, sol[0]/2.0, sol[1]/2.0, sol[2]/2.0


def create_sphere_points(radius, x0, y0, z0, n=72):
    sp = np.linspace(0, 2.0*np.pi, num=n)
    nx = sp.shape[0]
    u = np.repeat(sp, nx)
    v = np.tile(sp, nx)
    x = x0 + np.cos(u)*np.sin(v)*radius
    y = y0 + np.sin(u)*np.sin(v)*radius
    z = z0 + np.cos(v)*radius
    return x, y, z


pcd = o3d.io.read_point_cloud('0720/3D_0.ply')


p = np.asarray(pcd.points)

#find the fitting sphere
r, x, y, z = sphere_fitting(p)
print(f'radius: {r}\ncenter: ({x}, {y}, {z})')

xs, ys, zs = create_sphere_points(r, x, y, z, 30)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, s=1, c='g', label='create')
ax.scatter(p[:,0], p[:,1],p[:,2], s=1, c='b', label='reconstruct')
plt.legend()
plt.show() 