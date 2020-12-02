'''
Created on Oct 7, 2020

@author: Atrisha
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import numpy.linalg as la
from scipy.linalg import sqrtm
import math
from itertools import combinations
import visualizer
import constants
log = constants.common_logger

def square_distance(x,y): return sum([(xi-yi)**2 for xi, yi in zip(x,y)]) 

def get_maximal_points(points):
    max_square_distance = 0
    for pair in combinations(points,2):
        if square_distance(*pair) > max_square_distance:
            max_square_distance = square_distance(*pair)
            max_pair = pair
    return (max_pair,max_square_distance)
    
def get_cov_matrix(majorAxis, minorAxis, phi, N_sd, center = (0,0)):
    
    majorAxis = majorAxis/(2*N_sd)
    minorAxis = minorAxis/(2*N_sd)
    varX1 = (majorAxis**2 * np.cos(phi)**2) + (minorAxis**2 * np.sin(phi)**2)
    varX2 = (majorAxis**2 * np.sin(phi)**2) + (minorAxis**2 * np.cos(phi)**2)
    cov21 = (majorAxis**2 - minorAxis**2) * np.sin(phi) * np.cos(phi) 
    #cov12 = (minorAxis**2 - majorAxis**2) * np.sin(phi) * np.cos(phi) 
    cov12 = cov21
    
    
    
    #Parameters to set
    variance_x = varX1
    variance_y = varX2
    
    #Create grid and multivariate normal
    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
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
        step_size /= 5
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = u*points
    A = la.inv(points.T*np.diag(u)*points - c.T*c)/d    
    return np.asarray(A), np.squeeze(np.asarray(c))


def construct_min_bounding_ellipse(points):
    points = list(set(points))
    A,C = mvee(points)
    u, s, vh = la.svd(A)
    
    phi_hat = np.arccos(vh[0,0]) if vh[0,0] == vh[1,1] else np.arccos(vh[0,1])  
    min_axs = 1/math.sqrt(s[0])*2
    maj_axs = 1/math.sqrt(s[1])*2
    get_cov_matrix(maj_axs, min_axs, phi_hat, 2,tuple(C))
    return (maj_axs, min_axs, phi_hat, tuple(C))

def get_distribution_params(points):
    points = list(set(points))
    try:
        
        #plt.title(k)
        #ax = plt.gca()
        #visualizer.visualizer.plot_traffic_regions()
        #plt.plot([x[0] for x in points],[x[1] for x in points],'o')
        '''
        A,C = mvee(points)
        u, s, vh = la.svd(A)
        phi_hat = np.arccos(vh[0,0]) if vh[0,0] == vh[1,1] else np.arccos(vh[0,1])  
        log.info(phi_hat)
        min_axs = 1/math.sqrt(s[0])*2
        maj_axs = 1/math.sqrt(s[1])*2
        plt.plot([x[0] for x in points],[x[1] for x in points],'o')
        el = Ellipse(xy=C, width=maj_axs, height=min_axs, angle=np.rad2deg(phi_hat), fill=False, color='blue')
        ax.add_patch(el)
        '''
        if len(points) > 3:
            A,C = mvee(points)
            u, s, vh = la.svd(A)
            phi_hat = np.arccos(vh[0,0]) if vh[0,0] == vh[1,1] else np.arccos(vh[0,1])  
            min_axs = 1/math.sqrt(s[0])*2
            maj_axs = 1/math.sqrt(s[1])*2
        else:
            maximal_pts, maj_axs = get_maximal_points(points)
            C = (np.mean([maximal_pts[0][0],maximal_pts[1][0]]), np.mean([maximal_pts[0][1],maximal_pts[1][1]]))
            phi_hat = math.atan((maximal_pts[1][1]-maximal_pts[0][1])/( maximal_pts[1][0] - maximal_pts[0][0]))
            #log.info(str(maximal_pts))
            #log.info(phi_hat)
            min_axs = min(maj_axs/2,1)
            #el = Ellipse(xy=C, width=maj_axs, height=min_axs, angle=np.rad2deg(phi_hat), fill=False, color='green')
            #ax.add_patch(el) 
        #plt.show()  
    except np.linalg.LinAlgError:
        maximal_pts, maj_axs = get_maximal_points(points)
        C = (np.mean([maximal_pts[0][0],maximal_pts[1][0]]), np.mean([maximal_pts[0][1],maximal_pts[1][1]]))
        phi_hat = math.atan(maximal_pts[1][1]-maximal_pts[0][1]/ maximal_pts[1][0] - maximal_pts[0][0])
        min_axs = min(maj_axs/2,1)
    mean = tuple(C)
    cov = np.asarray(get_cov_matrix(maj_axs, min_axs, phi_hat, 2,tuple(C)))
    return (mean,cov)

def calc_bhattacharya_distance_normal(mu1,sigma1,mu2,sigma2):
    return (0.25*np.log(0.25*((sigma1**2/sigma2**2)+(sigma2**2/sigma1**2)+2))) + (0.25*(((mu1-mu2)**2)/( (sigma1**2)+(sigma2**2) )))

def calc_hellinger_distance_multivariate(dist_1,dist_2):
    mu_1, sigma_1, mu_2, sigma_2 = np.asarray(dist_1[0]), np.asarray(dist_1[1]), np.asarray(dist_2[0]), np.asarray(dist_2[1])
    sigma = np.true_divide(sigma_1 + sigma_2, 2)
    a = np.linalg.inv(sigma)
    b = np.transpose(mu_1 - mu_2)
    c = (mu_1 - mu_2)
    d = np.matmul(a , b)
    e = np.matmul(c , d)
    d_b_1 =  (1/8) * (np.transpose(mu_1 - mu_2) @ np.linalg.inv(sigma) @ (mu_1 - mu_2)) 
    d_b_2 =  0.5 * np.log( np.linalg.det(sigma) / np.sqrt(np.linalg.det(sigma_1)*np.linalg.det(sigma_2)) )
    d_b = d_b_1 + d_b_2 
    bc = np.exp(-d_b)
    hellinger_dist = np.sqrt(1-bc) 
    return hellinger_dist   


def calc_wasserstein_distance_multivariate(dist_1,dist_2):
    mu_1, sigma_1, mu_2, sigma_2 = np.asarray(dist_1[0]), np.asarray(dist_1[1]), np.asarray(dist_2[0]), np.asarray(dist_2[1])
    mean_part = np.linalg.norm(mu_1-mu_2)**2
    try:
        trace_part = np.trace( sigma_1 + sigma_2 - (2*sqrtm(sigma_1@sigma_2)) )
    except ValueError:
        trace_part = 0 
    wasserstein_dist = math.sqrt( mean_part + trace_part)
    return wasserstein_dist