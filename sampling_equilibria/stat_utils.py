'''
Created on Oct 7, 2020

@author: Atrisha
'''

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
import math
    
def get_cov_matrix(majorAxis, minorAxis, phi, N_sd):
    
    majorAxis = majorAxis/(2*N_sd)
    minorAxis = minorAxis/(2*N_sd)
    varX1 = (majorAxis**2 * np.cos(phi)**2) + (minorAxis**2 * np.sin(phi)**2)
    varX2 = (majorAxis**2 * np.sin(phi)**2) + (minorAxis**2 * np.cos(phi)**2)
    cov21 = (majorAxis**2 - minorAxis**2) * np.sin(phi) * np.cos(phi) 
    #cov12 = (minorAxis**2 - majorAxis**2) * np.sin(phi) * np.cos(phi) 
    cov12 = cov21
    
    
    
    #Parameters to set
    mu_x = 0
    variance_x = varX1
    
    mu_y = 0
    variance_y = varX2
    
    #Create grid and multivariate normal
    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal([mu_x, mu_y], [[variance_x, cov12], [cov21, variance_y]])
    
    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    #plt.show()
    return  [[variance_x, cov12], [cov21, variance_y]]

'''https://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python'''
def mvee(points, tol = 0.001):
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    """
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * la.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = u*points
    A = la.inv(points.T*np.diag(u)*points - c.T*c)/d    
    return np.asarray(A), np.squeeze(np.asarray(c))


points = np.random.multivariate_normal([0,0],get_cov_matrix(10, 2, np.pi/6, 2),size = 100).tolist()



def construct_min_bounding_ellipse(points):
    plt.figure()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.plot([x[0] for x in points],[x[1] for x in points],'.')
    A,C = mvee(points)
    u, s, vh = la.svd(A)
    phi_hat = np.rad2deg(np.arccos(vh[0,0]))
    min_axs = 1/math.sqrt(s[0])
    maj_axs = 1/math.sqrt(s[1])
    get_cov_matrix(maj_axs, min_axs, phi_hat, 2)
    plt.show()
    return (maj_axs, min_axs, phi_hat)
f=1