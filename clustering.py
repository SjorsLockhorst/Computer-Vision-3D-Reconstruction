import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def Kmeans_clustering(voxels_to_draw): # rescaled and rearanged axis for openGL, nu lijst tuples x, y, z
    voxels_array = np.array(voxels_to_draw)
    x_y_points = voxels_array[:,:2]
    Z = np.float32(x_y_points)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(Z,4,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    
    A = Z[label.ravel()==0]
    B = Z[label.ravel()==1]
    C = Z[label.ravel()==2]
    D = Z[label.ravel()==3] 
    # Plot the data
    plt.scatter(A[:,0],A[:,1])
    plt.scatter(B[:,0],B[:,1],c = 'r')
    plt.scatter(C[:,0],C[:,1],c = 'g')
    plt.scatter(D[:,0],D[:,1],c = 'b')
    plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.show()
    return ret, label, center


points_per_cluster = 10

cluster_1 = np.random.randn(points_per_cluster, 3) + [0, 0, 0]
cluster_2 = np.random.randn(points_per_cluster, 3) + [5, 5, 5]
cluster_3 = np.random.randn(points_per_cluster, 3) + [5, 0, 0]
cluster_4 = np.random.randn(points_per_cluster, 3) + [0, 5, 0]
cluster_1 = np.array(cluster_1, dtype=int)
cluster_2 = np.array(cluster_2, dtype=int)
cluster_3 = np.array(cluster_3, dtype=int)
cluster_4 = np.array(cluster_4, dtype=int)
points = np.concatenate([cluster_1, cluster_2, cluster_3, cluster_4], axis=0)

ret, label, center = Kmeans_clustering(points)

