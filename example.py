import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import ndbd as nd
import os
import shutil

def Create_output_folders(_outpath):

    isFile = os.path.isdir(_outpath)


    if(isFile!=True):
        print('Output Folder does not exist..')
        s = [1.0]
        for i in tqdm(s):

            plot_path = _outpath +'/plots'
            data_path = _outpath + '/data_files'
            os.makedirs(plot_path)
            os.makedirs(data_path)

        print('Output folder created with path:',_outpath)
    else:
        print('Output folder already exists at path: ',_outpath)

def Clean_output_folders(_output):

    if(os.path.isdir(_outpath)!=False):
        shutil.rmtree(_outpath)
        print('Output path removed...')
    else:
        print('Path does not exists')



outpath = os.path.expanduser('~/Documents/nu0bb/data/')
Create_output_folders(outpath)
out_data = outpath + '/data_files'
out_plot = outpath + '/plots'

tr_mBB_ih = 0.02
tr_mBB_nh = 0.001

Sample_size = 1000000
M_nu = 4.8
Exp = 8705
bkg = 3.8
nucl = 'Xe'

X = np.array([33.44*np.pi/180.0,8.2*np.pi/180.0,0.0001,7.4,2.39,0,0,4.8])


true_mean_nh = nd.expected_events(tr_mBB_nh,M_nu,nucl,Exp,bkg)
true_mean_ih = nd.expected_events(tr_mBB_ih,M_nu,nucl,Exp,bkg)

stat = nd.Statistics(nucl,M_nu,Exp,bkg)
stat.Calculate_Ntest()

#true_mean = stat.Calculate_ntrue(X,'NH')
print(true_mean_ih)
N_dist_nh = np.random.poisson(true_mean_nh,Sample_size)
T_nh = []

for n in tqdm(N_dist_nh):
    T_nh.append(stat.Test_statistics(n))

T_nh = np.array(T_nh)


N_dist_ih = np.random.poisson(true_mean_ih,Sample_size)
T_ih = []

for n in tqdm(N_dist_ih):
    T_ih.append(stat.Test_statistics(n))

T_ih = np.array(T_ih)

plt.hist(T_nh,bins=50,color='r',alpha=0.5)
plt.hist(T_ih,bins=50,color='g',alpha=0.5)
plt.show()





