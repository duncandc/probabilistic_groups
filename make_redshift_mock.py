#Duncan Campbell
#November, 2014
#Yale University


from __future__ import division, print_function

import numpy as np
import custom_utilities as cu
import halotools.mock_observables
import h5py

def main():
    
    catalogue = 'Mr19_age_distribution_matching_mock'
    filepath_mock = cu.get_output_path()+'processed_data/hearin_mocks/custom_catalogues/'
    savepath = cu.get_output_path()+'processed_data/hearin_mocks/custom_catalogues/'
    f = h5py.File(filepath_mock+catalogue+'.hdf5', 'r') #open catalogue file
    mock = f.get(catalogue)
    
    #from astropy.constants import c
    c = 299792.458 #speed of light in km/s
    from astropy import cosmology
    from scipy.interpolate import interp1d
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    
    #get the peculiar velocity component along the line of sight direction
    v_los = mock['Vz']
    
    #compute cosmological redshift
    y = np.linspace(0,1,1000)
    x = cosmo.comoving_distance(y).value
    f = interp1d(x, y, kind='cubic')
    z_cos = f(mock['z'])
    
    #redshift is combination of cosmological and peculiar velocities
    z = z_cos+(v_los/c)*(1.0+z_cos)
    
    #reflect galaxies around redshift PBC
    flip = (z>f(250.0))
    z[flip] = z[flip]-f(250.0)
    flip = (z<0)
    z[flip] = z[flip]+f(250.0)
    
    redshift=z
    
    dtype = [('ID_halo', '<i8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('Vx', '<f8'),\
             ('Vy', '<f8'), ('Vz', '<f8'), ('M_vir', '<f8'), ('V_peak', '<f8'),\
             ('M_r,0.1', '<f8'), ('M_star', '<f8'), ('g-r', '<f8'), ('M_host', '<f8'),\
             ('ID_host', '<i8'), ('r', '<f8'), ('R_proj', '<f8'), ('R200', '<f8'),\
             ('redshift', '<f8')]
    dtype = np.dtype(dtype)
    data = np.recarray((len(mock),), dtype=dtype)
    
    data['ID_halo']=mock['ID_halo']
    data['x']=mock['x']
    data['y']=mock['y']
    data['z']=mock['z']
    data['Vx']=mock['Vx']
    data['Vy']=mock['Vy']
    data['Vz']=mock['Vz']
    data['M_vir']=mock['M_vir']
    data['V_peak']=mock['V_peak']
    data['M_r,0.1']=mock['M_r,0.1']
    data['M_star']=mock['M_star']
    data['g-r']=mock['g-r']
    data['M_host']=mock['M_host']
    data['ID_host']=mock['ID_host']
    data['r']=mock['r']
    data['R_proj']=mock['R_proj']
    data['R200']=mock['R200']
    data['redshift']=redshift
    
    filename = catalogue+'_dist_obs'
    f = h5py.File(savepath+filename+'.hdf5', 'w')
    dset = f.create_dataset(filename, data=data)
    f.close()

if __name__ == '__main__':
    main() 
