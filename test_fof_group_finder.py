#Duncan Campbell
#November, 2014
#Yale University


from __future__ import division, print_function

import numpy as np
import custom_utilities as cu
import halotools.mock_observables
import h5py
import matplotlib.pyplot as plt
from halotools.mock_observables.spatial.kdtrees.ckdtree import cKDTree
from scipy.sparse.csgraph import connected_components
import igraph

def main():
    
    catalogue = 'Mr19_age_distribution_matching_mock_dist_obs'
    filepath_mock = cu.get_output_path()+'processed_data/hearin_mocks/custom_catalogues/'
    savepath = cu.get_output_path()+'processed_data/hearin_mocks/custom_catalogues/'
    f = h5py.File(filepath_mock+catalogue+'.hdf5', 'r') #open catalogue file
    mock = f.get(catalogue)
    
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
    print(data)
    
    tree = cKDTree(data)
    
    ngal = len(mock)/(250.0**3.0)
    b = 0.2*ngal**(-1.0/3.0)
    print(b)
    
    sparse_d_matrix = tree.sparse_distance_matrix(tree, b, period=np.array([250,250,250]))
    csr_sparse_d_matrix = sparse_d_matrix.tocsr()
    
    labels = connected_components(sparse_d_matrix)[1]
    print("number of groups: {0}".format(len(np.unique(labels))))
    
    g = scipy_to_igraph(csr_sparse_d_matrix, directed=False)
    
def scipy_to_igraph(matrix, directed=True):
    sources, targets = matrix.nonzero()
    weights = matrix[sources, targets]
    return igraph.Graph(zip(sources, targets), directed=directed, edge_attrs={'weight': weights})    
    
    
if __name__ == '__main__':
    main() 