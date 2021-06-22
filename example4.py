import numpy as np
import matplotlib.pyplot as plt
import timeit
from tqdm import *
import ndbd as nd
import os
import scipy.interpolate as interp

class data_set:
    def __init__(self,_data,_level):
        self.data = _data
        self.level = _level

        self.prepare_data()


    def prepare_data(self):
        x = self.data[0]
        y = self.data[1]
        z = self.data[2]

        N = 1000
        _x = np.log10(x)
        _y = np.log10(y)

   #     chi2_min = np.min(z)

   #     z = z - chi2_min

        xi = np.linspace(_x.min(),_x.max(),N)
        yi = np.linspace(_y.min(),_y.max(),N)

        zi = interp.griddata((_x,_y),z,(xi[None,:],yi[:,None]))

        cs = plt.contour(10**xi,10**yi,zi,[self.level])
        p = cs.collections[0].get_paths()[0]
        v = p.vertices
        self.X = v[:,0]
        self.Y = v[:,1]

        return 0

stat = nd.Statistics('Xe',4.8 ,10000,10.0,'Hypo')
stat.make_3sigmasens(0.004,1000)

outpath = os.path.expanduser('~/Documents/nu0bb/data/data_files')
file1 = outpath + '/three_sig_Xe_4.8.dat'
data = np.loadtxt(file1)
#data.T[1] = data.T[1]/data.T[0]*0.2


dt = data_set(data.T,0.017)
dt1 = data_set(data.T,0.044)

plt.xscale('log')
plt.yscale('log')
plt.plot(dt.X,dt.Y)
plt.plot(dt1.X,dt1.Y)
plt.show()
