import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import ndbd as nd
import os
import shutil




outpath = os.path.expanduser('~/Documents/nu0bb/data/')
nd.Create_output_folders(outpath)
out_data = outpath + '/data_files'
out_plot = outpath + '/plots'

tr_mBB_ih = 0.02
tr_mBB_nh = 0.001

Sample_size = 100000
M_nu = 4.8
Exp = 8705
bkg = 3.8
nucl = 'Xe'

X = np.array([33.44*np.pi/180.0,8.2*np.pi/180.0,0.0001,7.4,2.39,0,0,4.8])

print(nd.mbb(X,'NH'))
print(nd.mbb(X,'IH'))


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

plt.xlabel('T')
plt.hist(T_nh,bins=50,color='r',alpha=0.5)
plt.hist(T_ih,bins=50,color='g',alpha=0.5)
plt.show()

T_med = np.median(T_nh)
sens_ih = nd.edf(T_med,T_ih,'IH')


T_med = np.median(T_ih)
sens_nh = nd.edf(T_med,T_nh,'NH')

print('Median sensitivity for NH true:',sens_nh*100)

print('Median sensitivity for IH true:',sens_ih*100)



