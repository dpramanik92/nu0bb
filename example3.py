import numpy as np
import matplotlib.pyplot as plt
import timeit
from tqdm import *
import ndbd as nd


stat = nd.Statistics('Xe',4.8,8705,3.83,'nEXO')

stat.Calculate_Ntest()

#stat.sensitivity_one(0.15,0.0001,100000)
#stat.discProbability(0.02,100000)

x = np.arange(-3,0.0,0.01)

print(10**x)
y = []
for i in x:
    y.append(stat.discProbability(10**i,10000))
    
y = np.array(y)

plt.xscale('log')
plt.title('nEXO',size=20)
plt.ylabel('Probability of being wrong',size=20)
plt.xlabel(r'$m_{\beta\beta}$ [eV]',size=20)
plt.plot(10**x,y.T[0],c='r',label='IH')
plt.plot(10**x,y.T[1],c='g',label='NH')

plt.show()

