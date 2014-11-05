#!/usr/bin/env python

#Duncan Campbell
#April 8, 2014
#Yale University
#test to see if algorithm can recover two separate clusters of points

#load packages
import numpy as np
import h5py
import matplotlib.pyplot as plt
from halotools.mock_observables.spatial.kdtrees.ckdtree import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import *

def main():

    N1 = 20
    N2 = 20
    
    x1 = np.random.normal(0,1,N1)
    y1 = np.random.normal(0,1,N1)
    z1 = np.random.normal(0,1,N1)
    
    x2 = np.random.normal(5,1,N2)
    y2 = np.random.normal(5,1,N2)
    z2 = np.random.normal(5,1,N2)
    
    x = np.hstack((x1,x2))
    y = np.hstack((y1,y2))
    z = np.hstack((z1,z2))
    
    data = np.vstack((x,y,z)).T
    
    plt.figure()
    plt.plot(data[:,0],data[:,1],'.')
    plt.show()
    
    tree = cKDTree(data)
    
    result = tree.sparse_distance_matrix(tree, 2)
    print result.nnz
    
    print result.col
    print result.row
    
    labels = connected_components(result)
    print labels[1]
    
    plt.figure()
    for label in np.unique(labels[1]):
        selection = (labels[1]==label)
        print selection
        plt.plot(data[:,0][selection],data[:,1][selection],'o')
    
    x_coords, y_coords = get_graph_segments(data[:,0:2], result)
    
    plt.plot(x_coords, y_coords, '-k')
    plt.show()
    
    N = result.getnnz(axis=1) + result.getnnz(axis=0)
    c_result = csc_matrix(result)
    r_result = csr_matrix(result)
    
    plt.figure()
    for i in range(0,len(data)):
        d1 = np.sort(result.getrow(i).data)
        d2 = np.sort(result.getcol(i).data)
        d = np.sort(np.hstack((0.0,d1,d2)))
        n = np.cumsum(np.zeros(N[i]+1)+1)
        plt.plot(d,n)
    plt.show()    
    

from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture import GMM
def get_graph_segments(X, G):
    """Get graph segments for plotting a 2D graph

    Parameters
    ----------
    X : array_like
        the data, of shape [n_samples, 2]
    G : array_like or sparse graph
        the [n_samples, n_samples] matrix encoding the graph of connectinons
        on X

    Returns
    -------
    x_coords, y_coords : ndarrays
        the x and y coordinates for plotting the graph.  They are of size
        [2, n_links], and can be visualized using
        ``plt.plot(x_coords, y_coords, '-k')``
    """
    X = np.asarray(X)
    if (X.ndim != 2) or (X.shape[1] != 2):
        raise ValueError('shape of X should be (n_samples, 2)')

    n_samples = X.shape[0]

    G = sparse.coo_matrix(G)
    A = X[G.row].T
    B = X[G.col].T

    x_coords = np.vstack([A[0], B[0]])
    y_coords = np.vstack([A[1], B[1]])

    return x_coords, y_coords


if __name__ == '__main__':
    main()