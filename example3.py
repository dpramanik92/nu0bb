import numpy as np
import matplotlib.pyplot as plt
import timeit
from tqdm import *
import ndbd as nd


stat = nd.Statistics('Xe',4.8,8705,3.8,'nEXO')

stat.Calculate_Ntest()
mbb = 0.048

print(mbb,stat.sensitivity_one(0.015,0.004,100000))


