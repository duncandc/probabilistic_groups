#Duncan Campbell
#November, 2014
#Yale University


from __future__ import division, print_function

import numpy as np
import custom_utilities as cu
import halotools.mock_observables
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from halotools.mock_observables.spatial.kdtrees.ckdtree import cKDTree
from scipy.sparse.csgraph import connected_components
import igraph
import build_perfect_groups
import sys

def main():
    
    catalogue = 'Mr19_age_distribution_matching_mock_dist_obs'
    filepath_mock = cu.get_output_path()+'processed_data/hearin_mocks/custom_catalogues/'
    savepath = cu.get_output_path()+'processed_data/hearin_mocks/custom_catalogues/'
    f = h5py.File(filepath_mock+catalogue+'.hdf5', 'r') #open catalogue file
    mock = f.get(catalogue)
    
    #identify true groups first
    group_IDs = build_perfect_groups.get_group_IDs(mock['ID_host'], mock['ID_halo']) 
    perfect_groups = build_perfect_groups.sparse_group_matrix(group_IDs)
    
    #need z-coordinate in length units 
    H0 = 100.0
    c = 299792.458
    zz = mock['redshift']*c/H0
    
    '''
    plt.figure()
    plt.plot(mock['x'],zz,'.', alpha=0.1, ms=2)
    plt.show()
    '''
    
    data = np.vstack((mock['x'],mock['y'],zz)).T
    
    tree = cKDTree(data)
    
    ngal = len(mock)/(250.0**3.0)
    b = 0.2*ngal**(-1.0/3.0)
    print("b={0}".format(b))
    
    '''
    sparse_d_matrix = tree.sparse_distance_matrix(tree, b, period=np.array([250,250,250]))
    csr_sparse_d_matrix = sparse_d_matrix.tocsr()
    '''
    
    ngal = len(mock)/(250.0**3.0)
    b_perp = 0.14*ngal**(-1.0/3.0)
    b_para = 0.75*ngal**(-1.0/3.0)
    b = max(b_perp,b_para)
    los = np.array([0,0,1])
    sparse_d_matrix = tree.sparse_distance_matrix_custom(tree, b, period=np.array([250,250,250]), los=los, b_para=b_para, b_perp=b_perp)
    csr_sparse_d_matrix = sparse_d_matrix.tocsr()
    
    labels = connected_components(sparse_d_matrix)[1]
    print("number of groups: {0}".format(len(np.unique(labels))))
    
    multiplicity = np.unique(labels, return_counts=True)[1]
    multiplicity = np.sort(multiplicity)
    print("multiplicity: {0}".format(multiplicity))
    
    g = scipy_to_igraph(csr_sparse_d_matrix, directed=False)
    degree = g.degree_distribution()
    print("degree distribution: {0}".format(degree))
    
    ind = np.argsort(np.unique(labels, return_counts=True)[1])
    biggest_group = ind[-10]
    biggest_group = np.where(labels==biggest_group)[0]
    print(biggest_group)
    
    mean_x = np.mean(data[:,0][biggest_group])
    mean_y = np.mean(data[:,1][biggest_group])
    mean_z = np.mean(data[:,2][biggest_group])
    
    selection_x = (data[:,0]>(mean_x-10)) & (data[:,0]<(mean_x+10))
    selection_y = (data[:,1]>(mean_y-10)) & (data[:,1]<(mean_y+10))
    selection_z = (data[:,2]>(mean_z-10)) & (data[:,2]<(mean_z+10))
    selection = (selection_x & selection_y) & selection_z
    
    #x_coords, y_coords, z_coords = get_graph_segments(data,sparse_d_matrix)
    
    fig = plt.figure()
    ax = fig.add_subplot(111,  projection='3d')
    ax.plot(data[:,0][selection],data[:,1][selection],data[:,2][selection],'.', color='black')
    ax.plot(data[:,0][biggest_group],data[:,1][biggest_group],data[:,2][biggest_group],'o')
    #ax.plot(x_coords, y_coords, z_coords, '-k')
    ax.set_xlim([mean_x-10,mean_x+10])
    ax.set_ylim([mean_y-10,mean_y+10])
    ax.set_zlim([mean_z-10,mean_z+10])
    plt.show()
    
    
def scipy_to_igraph(matrix, directed=True):
    sources, targets = matrix.nonzero()
    weights = matrix[sources, targets]
    return igraph.Graph(zip(sources, targets), directed=directed, edge_attrs={'weight': weights})    

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
    from scipy import sparse
    
    X = np.asarray(X)
    '''
    if (X.ndim != 2) or (X.shape[1] != 2):
        raise ValueError('shape of X should be (n_samples, 2)')
    '''

    n_samples = X.shape[0]

    G = sparse.coo_matrix(G)
    A = X[G.row].T
    B = X[G.col].T
    print(np.shape(A))

    x_coords = np.vstack([A[0], B[0]])
    y_coords = np.vstack([A[1], B[1]])
    z_coords = np.vstack([A[2], B[2]])

    return x_coords, y_coords, z_coords    
    
if __name__ == '__main__':
    main() 