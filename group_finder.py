#Duncan Campbell
#November, 2014
#Yale University

from __future__ import division, print_function

import numpy as np
import igraph


def fof_link(mock, b_para=0.75, b_perp=0.14, los=np.array([0,0,1]), period=np.array([250,250,255]),\
             Lbox=250.0):
    
    from halotools.mock_observables.spatial.kdtrees.ckdtree import cKDTree
    
    #need z-coordinate in length units 
    H0 = 100.0
    c = 299792.458
    zz = mock['redshift']*c/H0
    data = np.vstack((mock['x'],mock['y'],zz)).T
    tree = cKDTree(data)
    
    ngal = len(mock)/(250.0**3.0)
    b_perp = 0.14*ngal**(-1.0/3.0)
    b_para = 0.75*ngal**(-1.0/3.0)
    b = max(b_perp,b_para)

    sparse_d_matrix = tree.sparse_distance_matrix_custom(\
        tree, b, period=period, los=los, b_para=b_para, b_perp=b_perp)
    
    csr_sparse_d_matrix = sparse_d_matrix.tocsr()
    
    return _scipy_to_igraph(csr_sparse_d_matrix)


def fof_conditional_link(mock, w1, w2, b_para=0.75, b_perp=0.14,los=np.array([0,0,1]),\
                         period=np.array([250,250,255]), Lbox=250.0):
    
    from halotools.mock_observables.spatial.kdtrees.ckdtree import cKDTree
    
    #need z-coordinate in length units 
    H0 = 100.0
    c = 299792.458
    zz = mock['redshift']*c/H0
    data = np.vstack((mock['x'],mock['y'],zz)).T
    tree = cKDTree(data)
    
    ngal = len(mock)/(250.0**3.0)
    b_perp = 0.14*ngal**(-1.0/3.0)
    b_para = 0.75*ngal**(-1.0/3.0)
    b = max(b_perp,b_para)

    sparse_d_matrix = tree.sparse_distance_matrix_custom(tree, b, period=period, los=los,\
                          b_para=b_para, b_perp=b_perp, sweights=w1, oweights=w2)
    
    csr_sparse_d_matrix = sparse_d_matrix.tocsr()
    
    return _scipy_to_igraph(csr_sparse_d_matrix)


def gal_group_IDs(g):
    clusters = g.clusters()
    group_IDs = clusters.membership
    
    return group_IDs


def group_IDs(g):
    clusters = g.clusters()
    group_IDs = np.unique(clusters.membership)
    
    return group_IDs


def group_multiplicity(g):
    clusters = g.clusters()
    m = clusters.sizes()
    
    return m


def n_groups(g):
    clusters = g.clusters()
    n = clusters.n
    
    return n


def gal_degree(g):
    degree = g.degree()
    
    return degree


def gal_betweenness(g):
    result = g.betweenness()
    
    return result


def open_mock():
    
    import custom_utilities as cu
    import h5py
    
    catalogue = 'Mr19_age_distribution_matching_mock_dist_obs'
    filepath_mock = cu.get_output_path()+'processed_data/hearin_mocks/custom_catalogues/'
    savepath = cu.get_output_path()+'processed_data/hearin_mocks/custom_catalogues/'
    f = h5py.File(filepath_mock+catalogue+'.hdf5', 'r') #open catalogue file
    mock = f.get(catalogue)
    
    return mock


def _scipy_to_igraph(matrix, directed=False):
    sources, targets = matrix.nonzero()
    weights = matrix[sources, targets]
    return igraph.Graph(zip(sources, targets), n=matrix.shape[0],\
                        directed=directed, edge_attrs={'weight': weights})
    
    